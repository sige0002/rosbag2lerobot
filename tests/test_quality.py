"""Tests for :mod:`rosbag2lerobot.quality` (P0-5 quality-report).

Fast unit tests cover the pure metric functions (:func:`count_freeze_frames`,
:func:`count_out_of_range`) and a CLI round-trip on a tiny real dataset built
with :class:`DatasetWriter`. An integration test runs the full quality report
against ``output/airoa_moma_mcap_hsr``.
"""

from __future__ import annotations

import gc
import itertools
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from click.testing import CliRunner
from PIL import Image

from rosbag2lerobot.cli import main
from rosbag2lerobot.quality import (
    QualityReport,
    _column_to_2d,
    _decode_video_frames,
    compute_quality_report,
    count_freeze_frames,
    count_out_of_range,
)
from rosbag2lerobot.writer import DatasetWriter

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
    from rosbag2lerobot.quality import _align_bounds, count_out_of_range

    # Scalar (0-d) bounds broadcast to the column width and then count OOR.
    lo, hi = _align_bounds(np.asarray(0.0), np.asarray(2.0), n_dims=3, key="f")
    assert lo is not None and hi is not None
    assert lo.shape == (3,) and hi.shape == (3,)
    values = np.array([[1.0, 5.0, -1.0]], dtype=np.float64)
    # 5.0 > 2 and -1.0 < 0 -> 2 out of range.
    assert count_out_of_range(values, lo, hi, tol=0.0) == 2


def test_align_bounds_length1_broadcast() -> None:
    from rosbag2lerobot.quality import _align_bounds

    lo, hi = _align_bounds(np.asarray([0.0]), np.asarray([2.0]), n_dims=4, key="f")
    assert lo is not None and hi is not None
    assert lo.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert hi.tolist() == [2.0, 2.0, 2.0, 2.0]


def test_align_bounds_dim_mismatch_skips(caplog) -> None:
    import logging

    from rosbag2lerobot.quality import _align_bounds

    # A 7-d bound against a 6-d column genuinely mismatches: do not silently
    # pass; return (None, None) and record a warning.
    with caplog.at_level(logging.WARNING):
        lo, hi = _align_bounds(
            np.arange(7.0), np.arange(7.0) + 10, n_dims=6, key="pose"
        )
    assert lo is None and hi is None
    assert any("dim mismatch" in r.message for r in caplog.records)


def _column_to_2d_legacy(col: pa.ChunkedArray) -> tuple[np.ndarray, int]:
    """Pre-vectorization reference impl of ``_column_to_2d`` (semantics oracle).

    Mirrors the original per-element Python ``float()`` path so the vectorized
    implementation can be asserted byte-identical against it.
    """
    n_null = col.null_count
    if pa.types.is_fixed_size_list(col.type) or pa.types.is_list(col.type):
        dim = col.type.list_size if pa.types.is_fixed_size_list(col.type) else None
        pylist = col.to_pylist()
        rows: list[list[float]] = []
        for item in pylist:
            if item is None:
                rows.append([float("nan")] * (dim if dim else 1))
            else:
                rows.append([float("nan") if v is None else float(v) for v in item])
        values = np.asarray(rows, dtype=np.float64) if rows else np.empty((0, dim or 1))
    else:
        np_arr = col.to_numpy(zero_copy_only=False).astype(np.float64)
        values = np_arr.reshape(-1, 1)
    return values, int(n_null)


def _assert_column_2d_equal(
    got: tuple[np.ndarray, int], want: tuple[np.ndarray, int]
) -> None:
    gv, gn = got
    wv, wn = want
    assert gn == wn
    assert gv.dtype == wv.dtype == np.float64
    assert gv.shape == wv.shape
    g_nan = np.isnan(gv)
    # NaN positions must match exactly (row nulls + per-element nulls).
    assert np.array_equal(g_nan, np.isnan(wv))
    # Finite entries equal (NaN-aware allclose on the non-NaN cells).
    assert np.allclose(gv[~g_nan], wv[~g_nan], rtol=0, atol=0)


def test_column_to_2d_fixed_size_list_row_and_elem_nulls() -> None:
    # fixed_size_list<float32, 3> with a fully-null row AND per-element nulls.
    col = pa.chunked_array(
        [
            pa.array(
                [[1.5, 2.5, 3.5], None, [4.0, None, 6.25]],
                type=pa.list_(pa.float32(), 3),
            )
        ]
    )
    _assert_column_2d_equal(_column_to_2d(col), _column_to_2d_legacy(col))


def test_column_to_2d_fixed_size_list_multichunk() -> None:
    # Multiple chunks must combine and reshape identically.
    c1 = pa.array([[1.0, 2.0]], type=pa.list_(pa.float32(), 2))
    c2 = pa.array([None, [3.0, None]], type=pa.list_(pa.float32(), 2))
    col = pa.chunked_array([c1, c2])
    _assert_column_2d_equal(_column_to_2d(col), _column_to_2d_legacy(col))


def test_column_to_2d_fixed_size_list_empty() -> None:
    col = pa.chunked_array([pa.array([], type=pa.list_(pa.float32(), 4))])
    got, n = _column_to_2d(col)
    assert n == 0
    assert got.shape == (0, 4)
    assert got.dtype == np.float64


def test_column_to_2d_scalar_int64() -> None:
    col = pa.chunked_array([pa.array([1, 2, 3, None], type=pa.int64())])
    _assert_column_2d_equal(_column_to_2d(col), _column_to_2d_legacy(col))


def test_video_feature_keys_filters_and_orders() -> None:
    from rosbag2lerobot.validation import video_feature_keys

    info = {
        "features": {
            "observation.state": {"dtype": "float32", "shape": [2]},
            "observation.images.cam_b": {"dtype": "video", "shape": [3, 4, 3]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "observation.images.cam_a": {"dtype": "video", "shape": [3, 4, 3]},
            # Non-dict feature spec must be skipped, not crash.
            "bogus": "not-a-dict",
        }
    }
    # Only dtype=="video" dict specs, in declaration order.
    assert video_feature_keys(info) == [
        "observation.images.cam_b",
        "observation.images.cam_a",
    ]
    # Missing/empty features -> empty list.
    assert video_feature_keys({}) == []
    assert video_feature_keys({"features": {}}) == []


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


def test_compute_quality_report_preloaded_info_stats_equivalent(
    tiny_dataset: Path,
) -> None:
    # Passing already-loaded info/stats must yield an identical report dict to
    # the default path that reads both JSON files from disk.
    info = json.loads((tiny_dataset / "meta" / "info.json").read_text())
    stats = json.loads((tiny_dataset / "meta" / "stats.json").read_text())

    from_disk = compute_quality_report(tiny_dataset).to_dict()
    preloaded = compute_quality_report(tiny_dataset, info=info, stats=stats).to_dict()

    assert preloaded == from_disk


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


# ---------------------------------------------------------------------------
# Corrupt-input decoding (regression: an unread stderr pipe deadlocks ffmpeg)
# ---------------------------------------------------------------------------


def _encode_testsrc(path: Path, duration_s: int = 20) -> None:
    """Write a small, valid h264 mp4 at ``path`` using ffmpeg's test source."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=320x240:rate=30:duration={duration_s}",
            "-c:v",
            "libx264",
            "-crf",
            "30",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


@pytest.mark.slow
def test_decode_of_a_corrupt_video_fails_instead_of_hanging(tmp_path: Path) -> None:
    """A damaged mp4 must raise, not wedge.

    ``_decode_video_frames`` reads stdout frame by frame and only inspects
    stderr after the process exits. A corrupt video emits >100 KiB of decode
    errors even at ``-loglevel error``, so with stderr on a pipe ffmpeg
    blocks in write() once the ~64 KiB buffer fills, stops producing stdout,
    and both sides wait forever. Driven from a worker thread so the failure
    mode is a timeout rather than a hung test session.
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    video = tmp_path / "corrupt.mp4"
    _encode_testsrc(video)
    data = bytearray(video.read_bytes())
    # Overwrite the middle of the stream, leaving the container header intact
    # so ffmpeg opens the file and then chokes on the frames themselves.
    start = 2000
    data[start : start + 40000] = os.urandom(min(40000, len(data) - start))
    video.write_bytes(bytes(data))

    # The test only exercises the deadlock if this corruption is loud enough
    # to fill a pipe buffer, so assert that premise rather than assume it —
    # a quieter future ffmpeg would otherwise let a re-broken decoder pass.
    # subprocess.run reads both streams concurrently, so it cannot wedge.
    probe = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert len(probe.stderr) > 64 * 1024, (
        f"the corrupt clip emits only {len(probe.stderr)} B of stderr, below "
        "the pipe buffer this test is meant to overflow"
    )

    outcome: list[str] = []

    def _drive() -> None:
        try:
            for _ in _decode_video_frames(video):
                pass
            outcome.append("completed")
        except RuntimeError:
            outcome.append("raised")
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assert
            outcome.append(f"unexpected: {exc!r}")

    t = threading.Thread(target=_drive, name="decode-corrupt", daemon=True)
    t.start()
    t.join(timeout=120.0)
    assert not t.is_alive(), (
        "decoding a corrupt video deadlocked: ffmpeg's stderr is not being "
        "drained, so it blocked in write() and stopped producing frames"
    )
    # Either outcome is acceptable — ffmpeg may salvage enough frames to exit
    # 0, or bail out with an error. What matters is that it finished.
    assert outcome and not outcome[0].startswith("unexpected"), outcome


@pytest.mark.slow
def test_abandoning_the_decode_early_reports_no_failure(tmp_path: Path) -> None:
    """Stopping early must not turn a healthy video into an error.

    Closing the generator (explicitly, or by dropping it after a ``break``)
    closes ffmpeg's stdout, which kills ffmpeg with EPIPE. That exit status is
    a consequence of our own teardown, so it must not be reported as a decode
    failure — otherwise a partial consumer sees "decode failed (returncode=224)
    ... Broken pipe" for a perfectly good file.
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    video = tmp_path / "good.mp4"
    _encode_testsrc(video, duration_s=5)

    gen = _decode_video_frames(video)
    frames = list(itertools.islice(gen, 3))
    assert len(frames) == 3
    gen.close()  # must not raise

    # The same shape as a caller that breaks out of a for-loop and drops the
    # reference: CPython closes the generator when it is collected.
    gen2 = _decode_video_frames(video)
    for _ in gen2:
        break
    del gen2
    gc.collect()


@pytest.mark.slow
def test_full_decode_of_a_healthy_video_yields_every_frame(tmp_path: Path) -> None:
    """The non-abandoned path is unchanged: all frames, no error."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    video = tmp_path / "good.mp4"
    _encode_testsrc(video, duration_s=2)
    frames = list(_decode_video_frames(video))

    # 2s @ 30fps. Kept as a range: the point is that the decode runs to
    # completion, not that a future ffmpeg emits exactly 60 frames for a
    # 2-second testsrc.
    assert 58 <= len(frames) <= 62, len(frames)
    assert frames[0].shape == (240, 320, 3)


def test_a_nonzero_exit_still_raises_with_its_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Suppressing the abandoned case must not suppress real failures.

    Only ffmpeg is stubbed; the ffprobe call that sizes the video still runs
    for real, so this exercises the decode path exactly as production hits it.
    """
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not available")

    video = tmp_path / "good.mp4"
    _encode_testsrc(video, duration_s=1)

    stub = tmp_path / "stub_ffmpeg.py"
    stub.write_text(
        'import sys\nsys.stderr.write("stub: decoder exploded\\n")\nsys.exit(7)\n'
    )
    real_popen = subprocess.Popen

    def _fake_popen(cmd: Any, *args: Any, **kwargs: Any) -> subprocess.Popen:
        argv = [cmd] if isinstance(cmd, str) else list(cmd)
        if not argv or Path(str(argv[0])).name != "ffmpeg":
            return real_popen(cmd, *args, **kwargs)
        return real_popen([sys.executable, str(stub)], *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)

    with pytest.raises(RuntimeError) as excinfo:
        list(_decode_video_frames(video))
    assert "returncode=7" in str(excinfo.value)
    assert "stub: decoder exploded" in str(excinfo.value)
