"""E2E + unit tests for the conversion manifest (⑥, plan.md D-2).

Unit tests prove :func:`rosbag2lerobot.manifest.build_manifest` is pure (an injected
``run_timestamp`` round-trips unchanged) and that :func:`sha256_of_path`
matches a direct ``hashlib`` digest. The integration test converts the real
``bagdata/airoa-moma-mcap`` bags (7 episodes) and asserts the
``meta/conversion_log.json`` provenance fields.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

import rosbag2lerobot
from rosbag2lerobot.cli import main
from rosbag2lerobot.manifest import ManifestInput, build_manifest, sha256_of_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_BAGS = PROJECT_ROOT / "bagdata" / "airoa-moma-mcap"
HSR_CONFIG = PROJECT_ROOT / "configs" / "hsr.yaml"


# ---------------------------------------------------------------------------
# Unit tests (pure / I/O-isolated)
# ---------------------------------------------------------------------------


def test_build_manifest_is_pure() -> None:
    manifest = build_manifest(
        inputs=[
            ManifestInput(
                path="/bags/a", sha256="ab" * 32, frame_count=10, processing_time_s=1.5
            )
        ],
        codec="libx264",
        ffmpeg_preset=None,
        ffmpeg_crf=None,
        total_episodes=1,
        total_frames=10,
        fps=10,
        config_snapshot="robot_type: x\n",
        config_sha256="cd" * 32,
        rosbag2lerobot_version="9.9.9",
        ffmpeg_version="ffmpeg version test",
        run_timestamp="FIXED",
    )
    # Injected timestamp passes through verbatim -> no datetime.now() inside.
    assert manifest["run_timestamp"] == "FIXED"
    assert manifest["rosbag2lerobot_version"] == "9.9.9"
    assert manifest["total_episodes"] == 1
    assert manifest["total_frames"] == 10
    assert manifest["config_sha256"] == "cd" * 32
    assert manifest["inputs"][0]["frame_count"] == 10
    # JSON-serializable.
    json.dumps(manifest)


def test_sha256_of_path_matches_hashlib_for_file(tmp_path: Path) -> None:
    f = tmp_path / "blob.bin"
    payload = b"some bytes \x00\x01\x02"
    f.write_bytes(payload)
    assert sha256_of_path(f) == hashlib.sha256(payload).hexdigest()


def test_sha256_of_path_ignores_metadata_yaml(tmp_path: Path) -> None:
    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "data.mcap").write_bytes(b"payload")
    digest_a = sha256_of_path(bag)
    # Adding metadata.yaml must not change the digest.
    (bag / "metadata.yaml").write_text("version: 9\n")
    assert sha256_of_path(bag) == digest_a


def test_sha256_of_path_is_deterministic(tmp_path: Path) -> None:
    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "a.bin").write_bytes(b"aaa")
    (bag / "b.bin").write_bytes(b"bbb")
    assert sha256_of_path(bag) == sha256_of_path(bag)
    assert len(sha256_of_path(bag)) == 64


# ---------------------------------------------------------------------------
# Integration (real bags)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_conversion_log_real_bags(tmp_path: Path) -> None:
    if not REAL_BAGS.is_dir():
        pytest.skip(f"real bags not present: {REAL_BAGS}")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    out = tmp_path / "ds"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "convert",
            "--config",
            str(HSR_CONFIG),
            "--bags",
            str(REAL_BAGS),
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output

    log = json.loads((out / "meta" / "conversion_log.json").read_text())
    info = json.loads((out / "meta" / "info.json").read_text())

    assert log["rosbag2lerobot_version"] == rosbag2lerobot.__version__
    assert log["total_episodes"] == 7
    assert info["total_episodes"] == 7
    assert len(log["inputs"]) == 7

    for inp in log["inputs"]:
        assert len(inp["sha256"]) == 64
        int(inp["sha256"], 16)  # valid hex
        assert inp["frame_count"] > 0

    assert sum(i["frame_count"] for i in log["inputs"]) == info["total_frames"]

    expected_cfg_sha = hashlib.sha256(HSR_CONFIG.read_bytes()).hexdigest()
    assert log["config_sha256"] == expected_cfg_sha
    # Full config text is embedded.
    assert "robot_type" in log["config_snapshot"]
