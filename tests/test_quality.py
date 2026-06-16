"""Tests for :mod:`bagel.quality` (P0-5 quality-report).

Fast unit tests cover the pure metric functions (:func:`count_freeze_frames`,
:func:`count_out_of_range`) and a CLI round-trip on a tiny real dataset built
with :class:`DatasetWriter`. An integration test runs the full quality report
against ``output/airoa_moma_mcap_hsr``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from bagel.cli import main
from bagel.quality import (
    QualityReport,
    compute_quality_report,
    count_freeze_frames,
    count_out_of_range,
)
from bagel.writer import DatasetWriter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_DATASET = PROJECT_ROOT / "output" / "airoa_moma_mcap_hsr"


# ---------------------------------------------------------------------------
# Pure-function unit tests (no I/O, no ffmpeg)
# ---------------------------------------------------------------------------


def test_count_freeze_frames_all_identical() -> None:
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    # [a, a, a] -> two consecutive frozen pairs.
    assert count_freeze_frames([a, a, a], std_eps=1e-3) == 2


def test_count_freeze_frames_distinct() -> None:
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (4, 4, 3), dtype=np.uint8) for _ in range(3)]
    assert count_freeze_frames(frames, std_eps=1e-3) == 0


def test_count_freeze_frames_single_frame() -> None:
    a = np.zeros((2, 2, 3), dtype=np.uint8)
    assert count_freeze_frames([a], std_eps=1e-3) == 0
    assert count_freeze_frames([], std_eps=1e-3) == 0


def test_count_out_of_range_basic() -> None:
    values = np.array([[0.0, 5.0], [-1.0, 2.0], [3.0, 11.0]], dtype=np.float64)
    lo = np.array([0.0, 0.0])
    hi = np.array([2.0, 10.0])
    # bounds are per-dim: dim0 [0,2], dim1 [0,10].
    # row0 [0.0, 5.0]  -> both in range          -> 0
    # row1 [-1.0, 2.0] -> -1.0 < 0 (dim0)        -> 1
    # row2 [3.0, 11.0] -> 3.0 > 2, 11.0 > 10     -> 2
    assert count_out_of_range(values, lo, hi, tol=0.0) == 3


def test_count_out_of_range_tolerance() -> None:
    values = np.array([[2.5]], dtype=np.float64)
    lo = np.array([0.0])
    hi = np.array([2.0])
    assert count_out_of_range(values, lo, hi, tol=0.0) == 1
    assert count_out_of_range(values, lo, hi, tol=1.0) == 0


def test_count_out_of_range_excludes_nan() -> None:
    values = np.array([[np.nan], [100.0]], dtype=np.float64)
    lo = np.array([0.0])
    hi = np.array([2.0])
    # NaN not counted; only 100.0 is out of range.
    assert count_out_of_range(values, lo, hi, tol=0.0) == 1


def test_align_bounds_scalar_broadcast() -> None:
    from bagel.quality import _align_bounds, count_out_of_range

    # Scalar (0-d) bounds broadcast to the column width and then count OOR.
    lo, hi = _align_bounds(np.asarray(0.0), np.asarray(2.0), n_dims=3, key="f")
    assert lo is not None and hi is not None
    assert lo.shape == (3,) and hi.shape == (3,)
    values = np.array([[1.0, 5.0, -1.0]], dtype=np.float64)
    # 5.0 > 2 and -1.0 < 0 -> 2 out of range.
    assert count_out_of_range(values, lo, hi, tol=0.0) == 2


def test_align_bounds_length1_broadcast() -> None:
    from bagel.quality import _align_bounds

    lo, hi = _align_bounds(np.asarray([0.0]), np.asarray([2.0]), n_dims=4, key="f")
    assert lo is not None and hi is not None
    assert lo.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert hi.tolist() == [2.0, 2.0, 2.0, 2.0]


def test_align_bounds_dim_mismatch_skips(caplog) -> None:
    import logging

    from bagel.quality import _align_bounds

    # A 7-d bound against a 6-d column genuinely mismatches: do not silently
    # pass; return (None, None) and record a warning.
    with caplog.at_level(logging.WARNING):
        lo, hi = _align_bounds(
            np.arange(7.0), np.arange(7.0) + 10, n_dims=6, key="pose"
        )
    assert lo is None and hi is None
    assert any("dim mismatch" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Helpers for the tiny-dataset CLI test
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


@pytest.fixture(autouse=True)
def _require_ffmpeg_ffprobe() -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe not available")


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    _write_tiny_dataset(tmp_path)
    return tmp_path


def test_compute_quality_report_clean(tiny_dataset: Path) -> None:
    report = compute_quality_report(tiny_dataset)
    assert report.verdict == "OK", report.to_dict()
    assert report.exit_code == 0
    assert report.score == pytest.approx(1.0)
    # No nulls / NaNs in synthetic data.
    for f in report.features:
        assert f.n_null == 0
        assert f.n_nan == 0
    # Video reconciliation: mp4 == expected, no mismatch.
    assert report.videos
    for v in report.videos:
        assert v.frame_mismatch == 0
        assert v.mp4_frames == v.expected_frames


def test_cli_quality_report_writes_parseable_json(
    tiny_dataset: Path, tmp_path: Path
) -> None:
    out_json = tmp_path / "report.json"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "quality-report",
            "--dataset",
            str(tiny_dataset),
            "-o",
            str(out_json),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_json.read_text())
    assert payload["verdict"] == "OK"
    assert "score" in payload
    assert "features" in payload
    assert "videos" in payload
    assert payload["weights"]  # default weights documented + emitted.


# ---------------------------------------------------------------------------
# Integration (real data)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_quality_report_real_dataset() -> None:
    if not REAL_DATASET.is_dir():
        pytest.skip(f"real dataset not present: {REAL_DATASET}")
    report: QualityReport = compute_quality_report(REAL_DATASET)

    assert report.verdict == "OK", report.to_dict()
    assert report.exit_code == 0

    for f in report.features:
        assert f.null_rate == 0.0, f.feature
        assert f.n_nan == 0, f.feature

    assert len(report.videos) == 2
    for v in report.videos:
        assert v.frame_mismatch == 0, v.video_key
        assert v.mp4_frames == 4934, v.video_key
        assert v.expected_frames == 4934, v.video_key
        # Freeze frames are reported (a valid, non-negative count), not
        # required to be nonzero.
        assert v.n_freeze >= 0
        assert v.freeze_rate >= 0.0
