"""Tests for video ↔ metadata reconciliation (LeRobot reference-condition check).

Exercises :func:`rosbag2lerobot.video_reconciliation.validate_video_metadata` in
both fast and strict modes. Datasets are built synthetically (info.json +
episodes parquet + data parquet + real mp4s encoded to an exact frame count) so
every status can be provoked deterministically, plus an end-to-end check
against real :class:`DatasetWriter` output.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rosbag2lerobot import video_reconciliation
from rosbag2lerobot.video_reconciliation import (
    SetupError,
    validate_video_metadata,
)

_DEFAULT_VIDEO_PATH = (
    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
)
_W = 16
_H = 16
_FPS = 10


@pytest.fixture(autouse=True)
def _require_ffmpeg_and_ffprobe() -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe not available")


# ---------------------------------------------------------------------------
# Synthetic dataset builders
# ---------------------------------------------------------------------------


def _make_mp4(path: Path, n_frames: int, fps: int = _FPS, noise: bool = False) -> None:
    """Encode an mp4 with exactly *n_frames* frames (mirrors the writer's args).

    ``noise=True`` fills frames with random pixels so the mdat box dominates
    the file size — required by the truncation tests, where cutting the tail
    must remove *frames* rather than the whole (tiny) stream.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = bytearray()
    if noise:
        rng = np.random.default_rng(0)
        for _ in range(n_frames):
            payload += rng.integers(0, 256, _W * _H * 3, dtype=np.uint8).tobytes()
    else:
        frame = bytearray(b"\x00" * (_W * _H * 3))
        for i in range(n_frames):
            frame[0] = i % 256  # vary a pixel so frames stay distinct
            payload += frame
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{_W}x{_H}", "-r", str(fps),
        "-i", "pipe:0", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ]  # fmt: skip
    subprocess.run(
        cmd,
        input=bytes(payload),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def _write_info(dataset_dir: Path, fps: int, video_keys: list[str]) -> None:
    features: dict[str, Any] = {
        vk: {
            "dtype": "video",
            "shape": [_H, _W, 3],
            "names": ["height", "width", "channels"],
        }
        for vk in video_keys
    }
    info = {
        "codebase_version": "v3.0",
        "fps": fps,
        "video_path": _DEFAULT_VIDEO_PATH,
        "features": features,
    }
    path = dataset_dir / "meta" / "info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(info, fh)


def _write_episodes(
    dataset_dir: Path,
    episodes: list[dict[str, Any]],
    drop_columns: tuple[str, ...] = (),
) -> None:
    """Write meta/episodes parquet from a compact spec.

    Each episode dict: ``{"episode_index", "length", "videos": {vk: {"chunk",
    "file", "from_ts", "to_ts"} | None}}``. ``drop_columns`` removes whole
    columns (to provoke MISSING_REQUIRED_COLUMN).
    """
    video_keys: list[str] = []
    for ep in episodes:
        for vk in ep["videos"]:
            if vk not in video_keys:
                video_keys.append(vk)

    arrays: dict[str, pa.Array] = {
        "episode_index": pa.array(
            [ep["episode_index"] for ep in episodes], type=pa.int64()
        ),
        "length": pa.array([ep["length"] for ep in episodes], type=pa.int64()),
    }
    for vk in video_keys:
        for suffix, field_key, typ in (
            ("chunk_index", "chunk", pa.int64()),
            ("file_index", "file", pa.int64()),
            ("from_timestamp", "from_ts", pa.float64()),
            ("to_timestamp", "to_ts", pa.float64()),
        ):
            col = f"videos/{vk}/{suffix}"
            if col in drop_columns:
                continue
            vals = [
                None if ep["videos"].get(vk) is None else ep["videos"][vk][field_key]
                for ep in episodes
            ]
            arrays[col] = pa.array(vals, type=typ)

    path = dataset_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(arrays), path)


def _write_data(
    dataset_dir: Path,
    episodes: list[dict[str, Any]],
    fps: int = _FPS,
    override_rows: Optional[dict[int, dict[str, list[float]]]] = None,
) -> None:
    """Write data parquet rows: index sequential, timestamp = i/fps per episode.

    ``override_rows[ep_idx]`` may replace ``{"index": [...], "timestamp": [...]}``
    for one episode (to provoke strict-mode data anomalies). An episode listed
    with ``"skip_data": True`` gets no rows (MISSING_EPISODE_DATA).
    """
    ep_col: list[int] = []
    idx_col: list[int] = []
    ts_col: list[float] = []
    g = 0
    for ep in episodes:
        ep_idx = ep["episode_index"]
        if ep.get("skip_data"):
            continue
        ov = (override_rows or {}).get(ep_idx)
        if ov is not None:
            n = len(ov["timestamp"])
            ep_col += [ep_idx] * n
            idx_col += [int(x) for x in ov["index"]]
            ts_col += [float(x) for x in ov["timestamp"]]
            g += n
            continue
        n = ep.get("data_rows", ep["length"])
        for i in range(n):
            ep_col.append(ep_idx)
            idx_col.append(g)
            ts_col.append(i / fps)
            g += 1

    table = pa.table(
        {
            "episode_index": pa.array(ep_col, type=pa.int64()),
            "index": pa.array(idx_col, type=pa.int64()),
            "timestamp": pa.array(ts_col, type=pa.float32()),
        }
    )
    path = dataset_dir / "data" / "chunk-000" / "file-000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _build_dataset(
    dataset_dir: Path,
    episodes: list[dict[str, Any]],
    mp4_frames: dict[tuple[str, int, int], int],
    fps: int = _FPS,
    mp4_fps: Optional[int] = None,
    override_rows: Optional[dict[int, dict[str, list[float]]]] = None,
    drop_columns: tuple[str, ...] = (),
) -> None:
    """Build a full synthetic dataset (info + episodes + data + mp4s)."""
    video_keys: list[str] = []
    for ep in episodes:
        for vk in ep["videos"]:
            if vk not in video_keys:
                video_keys.append(vk)
    _write_info(dataset_dir, fps, video_keys)
    _write_episodes(dataset_dir, episodes, drop_columns=drop_columns)
    _write_data(dataset_dir, episodes, fps=fps, override_rows=override_rows)
    for (vk, ck, fk), n in mp4_frames.items():
        mp4 = dataset_dir / "videos" / vk / f"chunk-{ck:03d}" / f"file-{fk:03d}.mp4"
        _make_mp4(mp4, n, fps=mp4_fps or fps)


_VK = "observation.images.cam"


def _two_episode_spec(vk: str = _VK, **ep1_extra: Any) -> list[dict[str, Any]]:
    """Two consecutive 10-frame episodes packed into one 20-frame mp4."""
    return [
        {
            "episode_index": 0,
            "length": 10,
            "videos": {vk: {"chunk": 0, "file": 0, "from_ts": 0.0, "to_ts": 1.0}},
        },
        {
            "episode_index": 1,
            "length": 10,
            "videos": {vk: {"chunk": 0, "file": 0, "from_ts": 1.0, "to_ts": 2.0}},
            **ep1_extra,
        },
    ]


def _statuses(report: Any) -> set[str]:
    return {e.status for e in report.errors}


# ---------------------------------------------------------------------------
# Fast mode
# ---------------------------------------------------------------------------


def test_fast_ok(tmp_path: Path) -> None:
    _build_dataset(tmp_path, _two_episode_spec(), {(_VK, 0, 0): 20})
    report = validate_video_metadata(tmp_path)
    assert report.verdict == "OK"
    assert report.exit_code == 0
    assert report.mode == "fast"
    assert report.videos_checked == 1
    assert report.episodes_checked == 2
    assert report.mappings_checked == 2
    assert report.rows_checked == 0  # fast mode checks extremes only
    assert report.total_errors == 0 and report.total_warnings == 0


def test_fast_frame_index_out_of_range(tmp_path: Path) -> None:
    # mp4 has 18 frames; ep1's max row ts = 0.9, shifted = 1.9 -> frame 19.
    _build_dataset(tmp_path, _two_episode_spec(), {(_VK, 0, 0): 18})
    report = validate_video_metadata(tmp_path)
    assert report.verdict == "ERROR"
    assert report.exit_code == 1
    errs = [e for e in report.errors if e.status == "FRAME_INDEX_OUT_OF_RANGE"]
    assert len(errs) == 1
    err = errs[0]
    assert err.episode_index == 1
    assert err.video_key == _VK
    assert err.requested_frame == 19
    assert err.video_frame_count == 18
    assert err.max_valid_frame == 17
    assert err.overflow == 2
    assert err.dataset_index == 19  # last global row
    assert err.row_timestamp == pytest.approx(0.9)
    assert err.shifted_timestamp == pytest.approx(1.9)
    assert err.video_average_fps == pytest.approx(10.0)


def test_fast_frame_index_negative(tmp_path: Path) -> None:
    episodes = [
        {
            "episode_index": 0,
            "length": 10,
            "videos": {_VK: {"chunk": 0, "file": 0, "from_ts": -0.5, "to_ts": 0.5}},
        }
    ]
    _build_dataset(tmp_path, episodes, {(_VK, 0, 0): 20})
    report = validate_video_metadata(tmp_path)
    assert "FRAME_INDEX_NEGATIVE" in _statuses(report)
    err = next(e for e in report.errors if e.status == "FRAME_INDEX_NEGATIVE")
    assert err.requested_frame == -5


def test_fast_video_missing(tmp_path: Path) -> None:
    _build_dataset(tmp_path, _two_episode_spec(), {})  # no mp4 written
    report = validate_video_metadata(tmp_path)
    assert report.verdict == "ERROR"
    assert _statuses(report) == {"VIDEO_MISSING"}
    assert report.total_errors == 2  # per referencing episode


def test_fast_video_unreadable(tmp_path: Path) -> None:
    _build_dataset(tmp_path, _two_episode_spec(), {})
    mp4 = tmp_path / "videos" / _VK / "chunk-000" / "file-000.mp4"
    mp4.parent.mkdir(parents=True, exist_ok=True)
    mp4.write_bytes(b"not a real mp4 file")
    report = validate_video_metadata(tmp_path)
    assert _statuses(report) == {"VIDEO_UNREADABLE"}


def test_fast_episode_length_mismatch(tmp_path: Path) -> None:
    spec = _two_episode_spec(data_rows=8)  # ep1: length=10 but 8 data rows
    _build_dataset(tmp_path, spec, {(_VK, 0, 0): 20})
    report = validate_video_metadata(tmp_path)
    assert "EPISODE_LENGTH_MISMATCH" in _statuses(report)
    err = next(e for e in report.errors if e.status == "EPISODE_LENGTH_MISMATCH")
    assert err.episode_index == 1


def test_fast_missing_episode_data(tmp_path: Path) -> None:
    spec = _two_episode_spec(skip_data=True)  # ep1 has no data rows
    _build_dataset(tmp_path, spec, {(_VK, 0, 0): 20})
    report = validate_video_metadata(tmp_path)
    assert "MISSING_EPISODE_DATA" in _statuses(report)


def test_fast_invalid_timestamp(tmp_path: Path) -> None:
    override = {1: {"index": list(range(10, 20)), "timestamp": [float("nan")] * 10}}
    _build_dataset(
        tmp_path, _two_episode_spec(), {(_VK, 0, 0): 20}, override_rows=override
    )
    report = validate_video_metadata(tmp_path)
    assert "INVALID_TIMESTAMP" in _statuses(report)


def test_fast_missing_metadata_value(tmp_path: Path) -> None:
    # ep1 present but from_timestamp null while the other 3 are set.
    episodes = _two_episode_spec()
    episodes[1]["videos"][_VK]["from_ts"] = None
    _build_dataset(tmp_path, episodes, {(_VK, 0, 0): 20})
    report = validate_video_metadata(tmp_path)
    assert "MISSING_METADATA_VALUE" in _statuses(report)


def test_fast_missing_required_column(tmp_path: Path) -> None:
    _build_dataset(
        tmp_path,
        _two_episode_spec(),
        {(_VK, 0, 0): 20},
        drop_columns=(f"videos/{_VK}/to_timestamp",),
    )
    report = validate_video_metadata(tmp_path)
    assert "MISSING_REQUIRED_COLUMN" in _statuses(report)
    err = next(e for e in report.errors if e.status == "MISSING_REQUIRED_COLUMN")
    assert err.video_key == _VK
    assert "to_timestamp" in err.detail


def test_fast_all_null_mapping_is_skipped(tmp_path: Path) -> None:
    # ep1 has no video at all (all four columns null) -> legal, no error.
    episodes = _two_episode_spec()
    episodes[1]["videos"][_VK] = None
    _build_dataset(tmp_path, episodes, {(_VK, 0, 0): 20})
    report = validate_video_metadata(tmp_path)
    assert report.verdict == "OK"
    assert report.mappings_checked == 1


def test_warning_fps_mismatch(tmp_path: Path) -> None:
    # info fps=10 but mp4 encoded at 30fps (60 frames): range still OK
    # (max requested = round(0.9*30)=27 < 60) -> warning only, verdict OK.
    _build_dataset(
        tmp_path,
        [_two_episode_spec()[0]],
        {(_VK, 0, 0): 60},
        mp4_fps=30,
    )
    report = validate_video_metadata(tmp_path)
    assert report.verdict == "OK"
    assert {w.status for w in report.warnings} == {"DATASET_VIDEO_FPS_MISMATCH"}


def test_warning_to_timestamp_too_small(tmp_path: Path) -> None:
    episodes = [
        {
            "episode_index": 0,
            "length": 10,
            "videos": {_VK: {"chunk": 0, "file": 0, "from_ts": 0.0, "to_ts": 0.5}},
        }
    ]
    _build_dataset(tmp_path, episodes, {(_VK, 0, 0): 20})
    report = validate_video_metadata(tmp_path)
    assert report.verdict == "OK"  # not used by the frame lookup (§4.6)
    assert {w.status for w in report.warnings} == {"TO_TIMESTAMP_TOO_SMALL"}


def test_ffprobe_called_once_per_file(tmp_path: Path, monkeypatch) -> None:
    _build_dataset(tmp_path, _two_episode_spec(), {(_VK, 0, 0): 20})

    calls: list[Path] = []
    real = video_reconciliation._probe_video

    def _counting(path: Path, strict: bool, full_decode: bool):
        calls.append(path)
        return real(path, strict, full_decode)

    monkeypatch.setattr(video_reconciliation, "_probe_video", _counting)
    report = validate_video_metadata(tmp_path)
    assert report.verdict == "OK"
    assert len(calls) == 1  # two episodes share one mp4


def test_multi_camera_isolates_bad_camera(tmp_path: Path) -> None:
    good = "observation.images.good"
    bad = "observation.images.bad"
    episodes = [
        {
            "episode_index": 0,
            "length": 10,
            "videos": {
                good: {"chunk": 0, "file": 0, "from_ts": 0.0, "to_ts": 1.0},
                bad: {"chunk": 0, "file": 0, "from_ts": 0.0, "to_ts": 1.0},
            },
        }
    ]
    _build_dataset(tmp_path, episodes, {(good, 0, 0): 10, (bad, 0, 0): 8})
    report = validate_video_metadata(tmp_path)
    assert report.verdict == "ERROR"
    assert all(e.video_key == bad for e in report.errors)
    assert _statuses(report) == {"FRAME_INDEX_OUT_OF_RANGE"}


def test_multiple_files_rotation(tmp_path: Path) -> None:
    episodes = _two_episode_spec() + [
        {
            "episode_index": 2,
            "length": 10,
            "videos": {_VK: {"chunk": 0, "file": 1, "from_ts": 0.0, "to_ts": 1.0}},
        }
    ]
    # file-001 short by 2 -> only ep2 flagged; file-000 stays OK.
    _build_dataset(tmp_path, episodes, {(_VK, 0, 0): 20, (_VK, 0, 1): 8})
    report = validate_video_metadata(tmp_path)
    assert report.videos_checked == 2
    errs = report.errors
    assert len(errs) == 1 and errs[0].episode_index == 2
    assert errs[0].overflow == 2


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


def test_strict_ok_and_rows_checked(tmp_path: Path) -> None:
    _build_dataset(tmp_path, _two_episode_spec(), {(_VK, 0, 0): 20})
    report = validate_video_metadata(tmp_path, strict=True)
    assert report.verdict == "OK"
    assert report.mode == "strict"
    assert report.rows_checked == 20
    assert report.tolerance_s == pytest.approx(0.05)


def test_strict_tolerance_error_and_max_errors(tmp_path: Path) -> None:
    # from_ts=0.04 shifts every row 0.04s off the PTS grid; rounding still
    # lands on frame i, so PTS error ~= 0.04 for all 10 rows. With
    # tolerance 0.02 every row fails; max_errors=4 caps the records.
    episodes = [
        {
            "episode_index": 0,
            "length": 10,
            "videos": {_VK: {"chunk": 0, "file": 0, "from_ts": 0.04, "to_ts": 1.04}},
        }
    ]
    _build_dataset(tmp_path, episodes, {(_VK, 0, 0): 20})
    report = validate_video_metadata(
        tmp_path, strict=True, tolerance_s=0.02, max_errors=4
    )
    assert report.verdict == "ERROR"
    assert _statuses(report) == {"FRAME_TIMESTAMP_OUT_OF_TOLERANCE"}
    assert report.total_errors == 10
    assert len(report.errors) == 4
    assert report.truncated is True
    err = report.errors[0]
    assert err.timestamp_error == pytest.approx(0.04, abs=1e-6)
    assert err.tolerance_s == pytest.approx(0.02)
    assert err.loaded_pts is not None


def test_strict_tolerance_pass_with_default(tmp_path: Path) -> None:
    # Same 0.04s offset passes with the default tolerance 0.5/fps = 0.05.
    episodes = [
        {
            "episode_index": 0,
            "length": 10,
            "videos": {_VK: {"chunk": 0, "file": 0, "from_ts": 0.04, "to_ts": 1.04}},
        }
    ]
    _build_dataset(tmp_path, episodes, {(_VK, 0, 0): 20})
    report = validate_video_metadata(tmp_path, strict=True)
    assert report.verdict == "OK"


def test_strict_non_monotonic_timestamp(tmp_path: Path) -> None:
    override = {
        1: {
            "index": list(range(10, 20)),
            "timestamp": [0.0, 0.1, 0.2, 0.15, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }
    }
    _build_dataset(
        tmp_path, _two_episode_spec(), {(_VK, 0, 0): 20}, override_rows=override
    )
    report = validate_video_metadata(tmp_path, strict=True)
    assert "NON_MONOTONIC_TIMESTAMP" in _statuses(report)


def test_strict_duplicate_and_gap_index(tmp_path: Path) -> None:
    override = {
        1: {
            # duplicate global index 5 (also owned by ep0) + a gap 12 -> 14.
            "index": [5, 11, 12, 14, 15, 16, 17, 18, 19, 20],
            "timestamp": [i / _FPS for i in range(10)],
        }
    }
    _build_dataset(
        tmp_path, _two_episode_spec(), {(_VK, 0, 0): 20}, override_rows=override
    )
    report = validate_video_metadata(tmp_path, strict=True)
    statuses = _statuses(report)
    assert "DUPLICATE_INDEX" in statuses
    assert "INVALID_INDEX_SEQUENCE" in statuses


def test_strict_detects_truncated_copy_fast_does_not(tmp_path: Path) -> None:
    """A partially-copied mp4 (header intact, tail cut) passes fast but fails strict."""
    _build_dataset(tmp_path, _two_episode_spec(), {})
    mp4 = tmp_path / "videos" / _VK / "chunk-000" / "file-000.mp4"
    _make_mp4(mp4, 20, noise=True)  # mdat-dominated file
    blob = mp4.read_bytes()
    mp4.write_bytes(blob[: int(len(blob) * 0.8)])  # simulate interrupted copy

    fast = validate_video_metadata(tmp_path)
    # faststart puts moov first, so the lying header still claims 20 frames.
    assert fast.verdict == "OK"

    strict = validate_video_metadata(tmp_path, strict=True)
    assert strict.verdict == "ERROR"
    assert _statuses(strict) & {
        "FRAME_COUNT_MISMATCH",
        "FRAME_INDEX_OUT_OF_RANGE",
        "FRAME_PTS_READ_FAILED",
        "VIDEO_UNREADABLE",
    }


def test_full_decode_failed_on_truncated_file(tmp_path: Path) -> None:
    _build_dataset(tmp_path, _two_episode_spec(), {})
    mp4 = tmp_path / "videos" / _VK / "chunk-000" / "file-000.mp4"
    _make_mp4(mp4, 20, noise=True)  # mdat-dominated file
    blob = mp4.read_bytes()
    mp4.write_bytes(blob[: int(len(blob) * 0.8)])

    report = validate_video_metadata(tmp_path, full_decode=True)
    assert report.mode == "strict"  # --full-decode implies strict
    assert report.full_decode is True
    assert report.verdict == "ERROR"
    # ffmpeg -xerror must flag the broken stream (alongside count mismatches).
    assert "VIDEO_FULL_DECODE_FAILED" in _statuses(report)


def test_real_writer_output_strict_ok(tmp_path: Path) -> None:
    """End-to-end: real DatasetWriter output passes the strict check."""
    from PIL import Image

    from rosbag2lerobot.writer import DatasetWriter

    feats: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": [1], "names": ["x"]},
        "action": {"dtype": "float32", "shape": [1], "names": ["x"]},
        _VK: {
            "dtype": "video",
            "shape": [_H, _W, 3],
            "names": ["height", "width", "channels"],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    writer = DatasetWriter(
        tmp_path, {"robot_type": "t"}, feats, fps=_FPS, video_codec="libx264"
    )
    rng = np.random.default_rng(0)
    for ep_len in (12, 8):
        for i in range(ep_len):
            writer.add_frame(
                {
                    "observation.state": np.array([float(i)], dtype=np.float32),
                    "action": np.array([float(i)], dtype=np.float32),
                    _VK: Image.fromarray(
                        rng.integers(0, 256, (_H, _W, 3), dtype=np.uint8)
                    ),
                    "task": "t",
                }
            )
        writer.save_episode()
    writer.finalize()

    report = validate_video_metadata(tmp_path, strict=True)
    assert report.verdict == "OK", [e.to_dict() for e in report.errors]
    assert report.rows_checked == 20


# ---------------------------------------------------------------------------
# Setup errors / internals
# ---------------------------------------------------------------------------


def test_setup_error_info_json_missing(tmp_path: Path) -> None:
    _write_episodes(tmp_path, _two_episode_spec())
    with pytest.raises(SetupError) as exc:
        validate_video_metadata(tmp_path)
    assert exc.value.code == "INFO_JSON_MISSING"


def test_setup_error_invalid_fps(tmp_path: Path) -> None:
    _build_dataset(tmp_path, _two_episode_spec(), {(_VK, 0, 0): 20})
    info_path = tmp_path / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["fps"] = 0
    info_path.write_text(json.dumps(info))
    with pytest.raises(SetupError) as exc:
        validate_video_metadata(tmp_path)
    assert exc.value.code == "INVALID_DATASET_FPS"


def test_setup_error_episode_metadata_missing(tmp_path: Path) -> None:
    _write_info(tmp_path, _FPS, [_VK])
    _write_data(tmp_path, _two_episode_spec())
    with pytest.raises(SetupError) as exc:
        validate_video_metadata(tmp_path)
    assert exc.value.code == "EPISODE_METADATA_MISSING"


def test_setup_error_data_parquet_missing(tmp_path: Path) -> None:
    _write_info(tmp_path, _FPS, [_VK])
    _write_episodes(tmp_path, _two_episode_spec())
    with pytest.raises(SetupError) as exc:
        validate_video_metadata(tmp_path)
    assert exc.value.code == "DATA_PARQUET_MISSING"


def test_np_rint_matches_python_round() -> None:
    """The vectorized rounding must equal Python round() (§6.3), incl. .5 ties."""
    rng = np.random.default_rng(42)
    vals = np.concatenate(
        [
            rng.uniform(-1e5, 1e5, 10_000),
            np.arange(-100, 100) + 0.5,  # exact half-way ties
            np.array([0.0, -0.0, 0.49999999999, 2.5, 3.5, -2.5, -3.5]),
        ]
    )
    np_rounded = np.rint(vals).astype(np.int64)
    py_rounded = np.array([round(float(v)) for v in vals], dtype=np.int64)
    np.testing.assert_array_equal(np_rounded, py_rounded)
