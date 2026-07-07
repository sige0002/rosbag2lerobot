"""Tests for :mod:`rosbag2lerobot.preview` (read-only static HTML preview report).

Fast unit tests cover the pure :func:`build_preview_html` builder with
synthetic dicts (no I/O). Integration tests render the real dataset under
``output/airoa_moma_mcap_hsr`` and exercise the CLI on a tiny dataset built
with :class:`DatasetWriter` (the same pattern as ``tests/test_quality.py``).
No test performs a network call.
"""

from __future__ import annotations

import base64
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from rosbag2lerobot.cli import main
from rosbag2lerobot.preview import build_preview_html, generate_preview

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_DATASET = PROJECT_ROOT / "output" / "airoa_moma_mcap_hsr"


# ---------------------------------------------------------------------------
# Pure-function unit test (no I/O)
# ---------------------------------------------------------------------------


def _dummy_b64() -> str:
    """A tiny valid base64 JPEG-ish blob (content irrelevant for the builder)."""
    return base64.b64encode(b"\xff\xd8\xff\xe0dummy").decode("ascii")


def test_build_preview_html_offline() -> None:
    info = {
        "robot_type": "hsr",
        "fps": 10,
        "total_episodes": 3,
        "total_frames": 4934,
        "total_tasks": 1,
        "codebase_version": "v3.0",
    }
    stats = {
        "observation.state": {
            "min": [-1.0, 0.0],
            "max": [1.0, 2.0],
            "mean": [0.0, 1.0],
            "std": [0.5, 0.5],
            "q50": [0.0, 1.0],
        },
    }
    quality = {
        "verdict": "OK",
        "score": 0.9876,
        "score_threshold": 0.95,
        "features": [
            {
                "feature": "observation.state",
                "n_null": 0,
                "n_nan": 0,
                "null_rate": 0.0,
                "n_out_of_range": 0,
                "oor_rate": 0.0,
            }
        ],
        "videos": [
            {
                "video_key": "observation.images.cam",
                "expected_frames": 12,
                "mp4_frames": 12,
                "frame_mismatch": 0,
                "n_freeze": 0,
                "freeze_rate": 0.0,
            }
        ],
    }
    frames_b64 = {
        "observation.images.cam": [_dummy_b64(), _dummy_b64()],
        "observation.images.hand": [_dummy_b64()],
    }

    html = build_preview_html(info, stats, quality, frames_b64)

    # Self-contained doc.
    assert html.startswith("<!DOCTYPE html>")
    assert "<table" in html
    # Score is rendered.
    assert "0.9876" in html
    # Inline base64 images.
    assert "data:image/jpeg;base64" in html
    # Each video key present.
    assert "observation.images.cam" in html
    assert "observation.images.hand" in html
    # Quality verdict badge + numeric stats feature row.
    assert "OK" in html
    assert "observation.state" in html


def test_build_preview_html_fail_verdict() -> None:
    info = {"robot_type": "hsr"}
    quality = {
        "verdict": "FAIL",
        "score": 0.10,
        "score_threshold": 0.95,
        "features": [],
        "videos": [],
    }
    html = build_preview_html(info, {}, quality, {})
    assert "FAIL" in html
    assert "badge fail" in html


# ---------------------------------------------------------------------------
# Integration (real data)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_generate_preview_real() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not available")
    if not REAL_DATASET.is_dir():
        pytest.skip(f"real dataset not present: {REAL_DATASET}")

    html = generate_preview(REAL_DATASET, n_frames=2)

    # Summary content.
    assert "hsr" in html
    assert "4934" in html
    # Both camera keys present.
    assert "observation.images.head_rgb" in html
    assert "observation.images.hand" in html
    # At least one embedded image + the quality score.
    assert "data:image/jpeg;base64" in html
    assert "score" in html
    # Truly self-contained: no external assets / scripts.
    assert "http://" not in html
    assert "https://" not in html
    assert "<script src=" not in html


# ---------------------------------------------------------------------------
# Tiny-dataset CLI test (reuses test_quality.py's DatasetWriter pattern)
# ---------------------------------------------------------------------------


def _features(video_keys: list[str], shape: tuple[int, int, int]) -> dict[str, Any]:
    feats: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
        "action": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for k in video_keys:
        feats[k] = {
            "dtype": "video",
            "shape": list(shape),
            "names": ["height", "width", "channels"],
        }
    return feats


def _write_tiny_dataset(out_dir: Path) -> None:
    from rosbag2lerobot.writer import DatasetWriter

    shape = (32, 32, 3)
    vkey = "observation.images.cam"
    writer = DatasetWriter(
        out_dir,
        {"robot_type": "regression"},
        _features([vkey], shape),
        fps=10,
        video_codec="libx264",
    )
    rng = np.random.default_rng(1)
    for ep_len in (5, 7):
        for i in range(ep_len):
            frame: dict[str, Any] = {
                "observation.state": np.array([float(i), 1.0], dtype=np.float32),
                "action": np.array([float(i), 1.0], dtype=np.float32),
                "task": "t",
                vkey: Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8)),
            }
            writer.add_frame(frame)
        writer.save_episode()
    writer.finalize()


@pytest.mark.integration
def test_cli_preview_writes_file(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not available")
    dataset_dir = tmp_path / "ds"
    _write_tiny_dataset(dataset_dir)

    out_html = tmp_path / "p.html"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "preview",
            "--dataset",
            str(dataset_dir),
            "-o",
            str(out_html),
            "--n-frames",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_html.is_file()
    html = out_html.read_text()
    assert "data:image/jpeg;base64" in html
    assert "observation.images.cam" in html
