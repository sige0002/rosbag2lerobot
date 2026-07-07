"""CLI ``--json`` structured-output tests (⑬).

Exercises the all-verb ``--json`` flag added to the report verbs. Where a real
dataset under ``output/`` or a real bag under ``bagdata/`` is required the test
skips when the data is absent (both trees are gitignored).

Run with:  uv run pytest -m integration tests/test_cli_json.py -q

What is asserted:
  * ``--json`` writes a JSON object to stdout (``output`` starts with ``{``),
  * the documented top-level keys are present and ``verdict`` is OK/FAIL,
  * the human summary is suppressed under ``--json``,
  * ``--json`` stdout matches the ``--json-out FILE`` contents (back-compat),
  * ``validate-config --json`` exposes ``results.verdict``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from rosbag2lerobot.cli import main

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
BAGDATA_DIR = PROJECT_ROOT / "bagdata"

SAMPLE_001 = BAGDATA_DIR / "bag" / "sample_001"
REAL_BAGS = BAGDATA_DIR / "airoa-moma-mcap"
HSR_CONFIG = PROJECT_ROOT / "configs" / "hsr.yaml"


def _first_dataset() -> Path | None:
    """Return the first finalized dataset under ``output/``, or ``None``."""
    if not OUTPUT_DIR.is_dir():
        return None
    for child in sorted(OUTPUT_DIR.iterdir()):
        if (child / "meta" / "info.json").exists():
            return child
    return None


def _require_dataset() -> Path:
    ds = _first_dataset()
    if ds is None:
        pytest.skip("no finalized dataset under output/")
    return ds


@pytest.mark.integration
def test_validate_dataset_json(tmp_path: Path) -> None:
    ds = _require_dataset()
    result = CliRunner().invoke(
        main, ["validate-dataset", "--dataset", str(ds), "--json"]
    )
    assert result.exit_code in (0, 1), result.output
    # Human summary suppressed -> stdout is a single JSON object.
    assert result.output.lstrip().startswith("{")
    parsed = json.loads(result.output)
    for key in ("dataset", "issues", "n_errors", "n_warnings", "verdict"):
        assert key in parsed
    assert parsed["verdict"] in {"OK", "FAIL"}

    # Back-compat parity: --json stdout == --json-out FILE contents.
    out_file = tmp_path / "report.json"
    file_result = CliRunner().invoke(
        main,
        ["validate-dataset", "--dataset", str(ds), "--json-out", str(out_file)],
    )
    assert file_result.exit_code in (0, 1), file_result.output
    assert json.loads(out_file.read_text()) == parsed


@pytest.mark.integration
def test_audit_timestamps_json() -> None:
    ds = _require_dataset()
    result = CliRunner().invoke(
        main, ["audit-timestamps", "--dataset", str(ds), "--json"]
    )
    assert result.exit_code in (0, 1), result.output
    assert result.output.lstrip().startswith("{")
    parsed = json.loads(result.output)
    for key in ("dataset", "video_keys", "results", "verdict"):
        assert key in parsed
    assert parsed["verdict"] in {"OK", "FAIL"}


@pytest.mark.integration
def test_quality_report_json() -> None:
    ds = _require_dataset()
    result = CliRunner().invoke(
        main, ["quality-report", "--dataset", str(ds), "--json"]
    )
    assert result.exit_code in (0, 1), result.output
    assert result.output.lstrip().startswith("{")
    parsed = json.loads(result.output)
    for key in ("dataset", "features", "videos", "score", "verdict"):
        assert key in parsed
    assert parsed["verdict"] in {"OK", "FAIL"}


@pytest.mark.integration
def test_convert_json_is_indented(tmp_path: Path) -> None:
    """`convert --json` emits an indented JSON object (matches the report verbs).

    The emitted summary must parse as one JSON object and be pretty-printed
    (``indent=2``), i.e. span multiple lines, consistent with ``_emit_report``.
    """
    if not REAL_BAGS.is_dir():
        pytest.skip(f"real bags not present: {REAL_BAGS}")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    out = tmp_path / "ds"
    result = CliRunner().invoke(
        main,
        [
            "convert",
            "--config",
            str(HSR_CONFIG),
            "--bags",
            str(REAL_BAGS),
            "--output",
            str(out),
            "--json",
            "--max-episodes",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    # --quiet/--json suppress chatter, so the whole stdout is the JSON object.
    payload = json.loads(result.output)
    assert payload["n_success"] == 1
    assert "total_frames" in payload
    # Indented output spans multiple lines (one key per line under indent=2).
    assert "\n  " in result.output


@pytest.mark.integration
def test_validate_config_json(tmp_path: Path) -> None:
    """`validate-config --json` against a real bag exposes results.verdict."""
    if not SAMPLE_001.exists() or not (SAMPLE_001 / "metadata.yaml").exists():
        pytest.skip(f"real RealMan bag not available at {SAMPLE_001}")

    # Scaffold a config from the bag first (it has no shipped config).
    out = tmp_path / "config.yaml"
    scaffold = CliRunner().invoke(
        main,
        [
            "scaffold",
            "--bags",
            str(SAMPLE_001),
            "-o",
            str(out),
            "--robot-type",
            "realman_dual",
            "--task",
            "manipulation",
            "--no-validate",
        ],
    )
    assert scaffold.exit_code == 0, scaffold.output

    result = CliRunner().invoke(
        main,
        ["validate-config", "--config", str(out), "--bags", str(SAMPLE_001), "--json"],
    )
    assert result.exit_code in (0, 1), result.output
    assert result.output.lstrip().startswith("{")
    parsed = json.loads(result.output)
    assert "results" in parsed
    assert parsed["results"]["verdict"] in {"OK", "FAIL"}
