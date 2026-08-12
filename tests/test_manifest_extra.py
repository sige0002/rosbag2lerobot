"""Tests for ``convert --manifest-extra`` (caller-supplied provenance).

The flag lets whoever drives the CLI record its own identifiers — job id,
ticket, operator — inside ``meta/conversion_log.json``, next to the provenance
the tool collects itself. What is pinned here:

  * the file is validated *before* conversion starts (a typo must not cost an
    hour of encoding);
  * built-ins win, so a caller cannot make the manifest lie about the codec,
    the config hash or the frame totals;
  * everything else lands in the manifest verbatim.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from click.testing import CliRunner

from rosbag2lerobot.cli import main
from rosbag2lerobot.manifest import (
    BUILTIN_MANIFEST_KEYS,
    load_manifest_extra,
    strip_builtin_keys,
)

from .conftest import tiny_config_yaml


def _invoke(tmp_path: Path, tiny_bag, *extra: str) -> tuple[int, str, Path]:
    """Convert one tiny bag with *extra* CLI args; return (code, output, out)."""
    tiny_bag(name="bags/ep0")
    cfg_path = tiny_config_yaml(tmp_path / "c.yaml")
    out = tmp_path / "out"
    result = CliRunner().invoke(
        main,
        [
            "convert",
            "--config",
            str(cfg_path),
            "--bags",
            str(tmp_path / "bags"),
            "--output",
            str(out),
            *extra,
        ],
    )
    return result.exit_code, result.output, out


# ---------------------------------------------------------------------------
# Unit
# ---------------------------------------------------------------------------


class TestLoadManifestExtra:
    def test_reads_a_json_object(self, tmp_path: Path) -> None:
        path = tmp_path / "extra.json"
        path.write_text(json.dumps({"job_id": "j-1", "nested": {"a": [1, 2]}}))
        assert load_manifest_extra(path) == {"job_id": "j-1", "nested": {"a": [1, 2]}}

    def test_invalid_json_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "extra.json"
        path.write_text("{not json")
        try:
            load_manifest_extra(path)
        except ValueError as exc:
            assert "not valid JSON" in str(exc)
            assert str(path) in str(exc)
        else:  # pragma: no cover - the call must raise
            raise AssertionError("expected ValueError")

    def test_non_object_root_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "extra.json"
        path.write_text(json.dumps(["a", "list"]))
        try:
            load_manifest_extra(path)
        except ValueError as exc:
            assert "JSON object" in str(exc)
        else:  # pragma: no cover - the call must raise
            raise AssertionError("expected ValueError")


class TestStripBuiltinKeys:
    def test_passes_through_unreserved_keys(self) -> None:
        filtered, dropped = strip_builtin_keys({"job_id": "j-1", "operator": "kim"})
        assert filtered == {"job_id": "j-1", "operator": "kim"}
        assert dropped == []

    def test_drops_reserved_keys_and_reports_them(self) -> None:
        filtered, dropped = strip_builtin_keys(
            {"codec": "fake", "config_sha256": "0" * 64, "job_id": "j-1"}
        )
        assert filtered == {"job_id": "j-1"}
        assert dropped == ["codec", "config_sha256"]

    def test_covers_both_writer_and_cli_owned_fields(self) -> None:
        # The guarantee is over the whole manifest, not just the CLI's half.
        assert {"codec", "episode_lengths", "total_frames"} <= BUILTIN_MANIFEST_KEYS
        assert {"inputs", "run_timestamp", "config_snapshot"} <= BUILTIN_MANIFEST_KEYS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestConvertManifestExtra:
    def test_fields_land_in_the_conversion_log(self, tmp_path: Path, tiny_bag) -> None:
        extra = tmp_path / "extra.json"
        extra.write_text(json.dumps({"job_id": "j-42", "labels": ["a", "b"]}))

        code, output, out = _invoke(tmp_path, tiny_bag, "--manifest-extra", str(extra))

        assert code == 0, output
        log = json.loads((out / "meta" / "conversion_log.json").read_text())
        assert log["job_id"] == "j-42"
        assert log["labels"] == ["a", "b"]
        # ...alongside, not instead of, the provenance the tool collects.
        assert log["total_episodes"] == 1
        assert len(log["config_sha256"]) == 64

    def test_builtin_keys_win(self, tmp_path: Path, tiny_bag, caplog) -> None:
        extra = tmp_path / "extra.json"
        extra.write_text(
            json.dumps(
                {
                    "codec": "not-the-real-codec",
                    "total_frames": 999999,
                    "config_sha256": "f" * 64,
                    "job_id": "j-42",
                }
            )
        )

        with caplog.at_level(logging.WARNING, logger="rosbag2lerobot"):
            code, output, out = _invoke(
                tmp_path, tiny_bag, "--manifest-extra", str(extra)
            )

        assert code == 0, output
        log = json.loads((out / "meta" / "conversion_log.json").read_text())
        assert log["codec"] != "not-the-real-codec"
        assert log["total_frames"] != 999999
        assert log["config_sha256"] != "f" * 64
        # The non-colliding key still gets through, and the drop is announced.
        assert log["job_id"] == "j-42"
        assert "ignoring 3 key(s)" in caplog.text
        assert "codec, config_sha256, total_frames" in caplog.text

    def test_invalid_json_fails_before_conversion(
        self, tmp_path: Path, tiny_bag
    ) -> None:
        extra = tmp_path / "extra.json"
        extra.write_text("{oops")

        code, output, out = _invoke(tmp_path, tiny_bag, "--manifest-extra", str(extra))

        assert code != 0
        assert "not valid JSON" in output
        # Nothing was written: the failure is up-front, not half way through.
        assert not out.exists()

    def test_missing_file_is_rejected(self, tmp_path: Path, tiny_bag) -> None:
        code, output, out = _invoke(
            tmp_path, tiny_bag, "--manifest-extra", str(tmp_path / "nope.json")
        )
        assert code != 0
        assert "does not exist" in output
