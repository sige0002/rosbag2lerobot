"""Quality-report generator for LeRobot v3.0 datasets.

Computes a small set of data-quality metrics over a finished dataset and
condenses them into a 0..1 quality score plus a pass/fail verdict. Like
:mod:`rosbag2lerobot.validation` the inspection is read-only.

Metrics (all source-of-truth references are to :mod:`rosbag2lerobot.writer`):

- Per numeric feature null / NaN counts and null rate (data parquet).
- Out-of-range counts, using ``meta/stats.json`` per-feature ``min`` / ``max``
  as the bounds (self-consistency only: stats.json is treated as the
  authoritative range for its own data).
- Freeze frames per video: consecutive decoded frames whose pixel-wise
  difference has ``std <= freeze_std_eps``. The pure counting function
  :func:`count_freeze_frames` is kept free of I/O; mp4 decoding lives in the
  separate :func:`_decode_video_frames` helper.
- Video/data reconciliation: per mp4 file, the decoded frame count
  (``ffprobe -count_frames``) must equal the sum of the parquet ``length``
  values of every episode pointing at that file (the invariant verified in
  ``tests/test_video_frame_alignment.py``).

Design decisions baked in (per the P0-5 spec):

- Freeze frames are *reported only*; they never fail the verdict (the writer
  legitimately pads via zero-order-hold).
- Any nonzero video/data ``frame_mismatch`` is a HARD FAIL regardless of the
  computed score.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rosbag2lerobot.audit import _collect_episodes_parquet, _discover_video_keys
from rosbag2lerobot.validation import video_feature_keys

logger = logging.getLogger(__name__)

__all__ = [
    "FeatureQuality",
    "VideoReconciliation",
    "QualityReport",
    "count_freeze_frames",
    "count_out_of_range",
    "compute_quality_report",
]


# Default scoring weights. The score is
# ``1 - clamp(w_null*mean_null_rate + w_range*mean_oor_rate
#              + w_freeze*mean_freeze_rate, 0, 1)``.
# Nulls/NaNs are the most damaging (a policy cannot train on them), so they
# carry the largest weight; out-of-range values are suspicious but may be
# legitimate extrema; freeze frames are informational (the writer pads
# legitimately) so they carry the smallest weight.
_DEFAULT_W_NULL = 0.5
_DEFAULT_W_RANGE = 0.3
_DEFAULT_W_FREEZE = 0.2

# Bytes of a failed decode's stderr kept for the error message.
_STDERR_TAIL_MAXLEN = 16 * 1024


@dataclass
class FeatureQuality:
    """Per numeric feature quality metrics.

    Attributes:
        feature: Feature key (data parquet column name).
        n_values: Total number of (flattened) scalar values inspected.
        n_null: Number of arrow nulls (``null_count``).
        n_nan: Number of NaN values among the non-null entries.
        null_rate: ``(n_null + n_nan) / n_values`` (0 when ``n_values == 0``).
        n_out_of_range: Count of non-null, non-NaN values outside
            ``[min - tol, max + tol]`` from stats.json.
        oor_rate: ``n_out_of_range / n_values`` (0 when ``n_values == 0``).
    """

    feature: str
    n_values: int
    n_null: int
    n_nan: int
    null_rate: float
    n_out_of_range: int
    oor_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VideoReconciliation:
    """Per video-key mp4/data frame-count reconciliation + freeze metrics.

    Attributes:
        video_key: The LeRobot video feature key.
        expected_frames: Sum of episode ``length`` values across all mp4
            files for this key (== data frames for this key).
        mp4_frames: Sum of decoded mp4 frame counts across all files.
        frame_mismatch: ``mp4_frames - expected_frames``. Nonzero is a HARD
            FAIL.
        n_freeze: Number of consecutive frozen frame pairs detected (summed
            over sampled mp4 files). Reported only; never fails.
        freeze_rate: ``n_freeze / mp4_frames`` (0 when ``mp4_frames == 0``).
    """

    video_key: str
    expected_frames: int
    mp4_frames: int
    frame_mismatch: int
    n_freeze: int
    freeze_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualityReport:
    """Aggregate quality report for a dataset."""

    dataset: str
    features: list[FeatureQuality] = field(default_factory=list)
    videos: list[VideoReconciliation] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    score: float = 1.0
    score_threshold: float = 0.95
    verdict: str = "OK"
    exit_code: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "features": [f.to_dict() for f in self.features],
            "videos": [v.to_dict() for v in self.videos],
            "weights": dict(self.weights),
            "score": self.score,
            "score_threshold": self.score_threshold,
            "verdict": self.verdict,
            "exit_code": int(self.exit_code),
        }


# ---------------------------------------------------------------------------
# Pure metric functions (no I/O)
# ---------------------------------------------------------------------------


def count_freeze_frames(frames: Iterable[np.ndarray], std_eps: float) -> int:
    """Count consecutive frame pairs that are (near-)identical.

    A pair ``(prev, curr)`` is counted when ``std(curr - prev) <= std_eps``.
    This is intentionally pure: callers decode the video into an iterable of
    frames and pass it in, so the counting logic is unit-testable without
    ffmpeg.

    Args:
        frames: Iterable of decoded frames as numeric ndarrays (any shape,
            but all frames must broadcast to the same shape).
        std_eps: Maximum per-pair difference standard deviation for a pair to
            be considered frozen.

    Returns:
        The number of frozen consecutive pairs. ``0`` for fewer than two
        frames.
    """
    n_freeze = 0
    prev: np.ndarray | None = None
    for frame in frames:
        cur = np.asarray(frame, dtype=np.float64)
        if prev is not None:
            if float(np.std(cur - prev)) <= std_eps:
                n_freeze += 1
        prev = cur
    return n_freeze


def count_out_of_range(
    values: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    tol: float,
) -> int:
    """Count values outside ``[lo - tol, hi + tol]`` per dimension.

    NaN values are excluded (they are accounted for separately as nulls).

    Args:
        values: 2D array ``(n_rows, n_dims)`` of feature values.
        lo: Per-dimension lower bounds (length ``n_dims``).
        hi: Per-dimension upper bounds (length ``n_dims``).
        tol: Absolute tolerance added to both bounds.

    Returns:
        The number of out-of-range scalar values.
    """
    if values.size == 0:
        return 0
    finite = ~np.isnan(values)
    below = (values < (lo - tol)) & finite
    above = (values > (hi + tol)) & finite
    return int(np.count_nonzero(below | above))


def _align_bounds(
    lo: np.ndarray,
    hi: np.ndarray,
    n_dims: int,
    key: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Coerce stats.json ``min``/``max`` bounds to per-dimension arrays.

    stats.json stores ``min``/``max`` as per-dimension lists, but a scalar
    bound (length-1 or 0-d) is a legitimate degenerate case for a 1-d feature
    and is broadcast to the column width here. A genuine dimension mismatch
    (e.g. a 7-d bound for a 6-d column) is logged and the out-of-range check is
    skipped for that file rather than silently passing as in-range.

    Args:
        lo: Per-feature lower bound array from stats.json.
        hi: Per-feature upper bound array from stats.json.
        n_dims: The data column width to align the bounds to.
        key: Feature key, for the mismatch log message.

    Returns:
        ``(lo, hi)`` shaped ``(n_dims,)`` on success, or ``(None, None)`` when
        the bounds cannot be aligned to ``n_dims``.
    """
    lo = np.atleast_1d(lo)
    hi = np.atleast_1d(hi)
    if lo.shape == (n_dims,) and hi.shape == (n_dims,):
        return lo, hi
    # Broadcast a scalar (length-1) bound to the column dimension.
    if lo.size == 1 and hi.size == 1:
        return np.full(n_dims, lo.reshape(-1)[0]), np.full(n_dims, hi.reshape(-1)[0])
    logger.warning(
        "quality-report: stats.json bound dim mismatch for %r "
        "(min dim=%d, max dim=%d, column dim=%d); skipping out-of-range count",
        key,
        lo.size,
        hi.size,
        n_dims,
    )
    return None, None


# ---------------------------------------------------------------------------
# I/O helpers (kept out of the pure functions)
# ---------------------------------------------------------------------------


def _ffprobe_count_frames(path: Path) -> int:
    """Return the authoritative decoded frame count of an mp4 (``-count_frames``)."""
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        stdin=subprocess.DEVNULL,
    )
    stream = json.loads(out)["streams"][0]
    return int(stream["nb_read_frames"])


def _video_dimensions(path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` of the first video stream via ffprobe."""
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        stdin=subprocess.DEVNULL,
    )
    stream = json.loads(out)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def _tail_text(f: IO[bytes]) -> str:
    """Return the last :data:`_STDERR_TAIL_MAXLEN` bytes of ``f`` as text.

    A damaged video produces error lines by the hundred; only the tail is
    worth putting in an exception message.
    """
    try:
        size = f.seek(0, os.SEEK_END)
        f.seek(max(0, size - _STDERR_TAIL_MAXLEN))
        return f.read().decode(errors="replace")
    except OSError:  # pragma: no cover - the temp file is ours and seekable
        return ""


def _decode_video_frames(path: Path) -> Iterable[np.ndarray]:
    """Yield decoded RGB frames of ``path`` as ``(H, W, 3)`` uint8 ndarrays.

    Decodes to raw ``rgb24`` via ffmpeg ``-f rawvideo -pix_fmt rgb24 pipe:1``
    and slices the byte stream into per-frame arrays. Kept separate from
    :func:`count_freeze_frames` so the latter stays I/O-free.

    stderr goes to a temp file rather than a pipe: this loop reads stdout
    frame by frame and only looks at stderr after the process exits, so a
    piped stderr would deadlock on a damaged video — the one input a quality
    check is most likely to be pointed at. A corrupt mp4 emits well over
    100 KiB of decode errors even at ``-loglevel error``, far past the
    ~64 KiB pipe buffer at which ffmpeg would block in write() and stop
    producing the stdout this loop is waiting on.
    """
    width, height = _video_dimensions(path)
    frame_size = width * height * 3
    with tempfile.TemporaryFile() as errfile:
        proc = subprocess.Popen(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=errfile,
        )
        abandoned = False
        try:
            assert proc.stdout is not None
            while True:
                buf = proc.stdout.read(frame_size)
                if not buf or len(buf) < frame_size:
                    break
                yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
        except GeneratorExit:
            # The consumer stopped early (an explicit close(), or the
            # generator being collected after a break). Closing stdout below
            # kills ffmpeg with EPIPE — that is our own teardown, not a decode
            # failure, and reporting it would turn a healthy video into a
            # spurious error raised from close() (or, under GC, into an
            # "Exception ignored in generator" with the diagnostic lost).
            abandoned = True
            raise
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            if abandoned and proc.poll() is None:
                # Don't wait out a decode whose output nobody will read.
                proc.kill()
            ret = proc.wait()
            if ret != 0 and not abandoned:
                raise RuntimeError(
                    f"ffmpeg decode failed for {path} (returncode={ret}): "
                    f"{_tail_text(errfile)}"
                )


def _load_stats(dataset_dir: Path) -> dict[str, Any]:
    """Load ``meta/stats.json`` (raise on missing/unreadable — a setup error)."""
    path = dataset_dir / "meta" / "stats.json"
    with open(path) as fh:
        return json.load(fh)


def _read_info(dataset_dir: Path) -> dict[str, Any]:
    """Load ``meta/info.json`` (raise on missing/unreadable — a setup error)."""
    path = dataset_dir / "meta" / "info.json"
    with open(path) as fh:
        return json.load(fh)


def _numeric_feature_keys(info: dict[str, Any]) -> list[str]:
    """Return non-video feature keys in info.json declaration order.

    Includes the numeric bookkeeping columns (timestamp/frame_index/...);
    those have stats.json entries and are legitimate quality targets.
    """
    return [
        k
        for k, v in info.get("features", {}).items()
        if isinstance(v, dict) and v.get("dtype") != "video"
    ]


def _column_to_2d(col: pa.ChunkedArray) -> tuple[np.ndarray, int]:
    """Flatten a data-parquet column to a 2D float array and report null count.

    Handles both scalar int64 columns and ``fixed_size_list<float32>``
    columns. Arrow nulls are converted to NaN in the returned array so the
    NaN mask covers them too, but the *null* count is taken from arrow's
    ``null_count`` before conversion. Returns ``(values_2d, n_null)``.

    The fixed_size_list path is vectorized through arrow: ``combine_chunks().values``
    yields the flat child array of length ``n_rows * list_size`` with both
    row-level and per-element nulls already marked, and arrow's
    ``to_numpy(zero_copy_only=False)`` maps every such null to ``NaN``. This
    avoids the per-scalar ``float()`` round-trip (≈1.4M calls on a 14-d feature
    over 100k frames) while producing a byte-identical result. The plain
    variable-length ``list`` branch keeps the per-element fallback (rosbag2lerobot's
    writer only emits fixed_size_list; this is defensive).
    """
    n_null = col.null_count
    if pa.types.is_fixed_size_list(col.type):
        list_size = col.type.list_size
        if col.length() == 0:
            values = np.empty((0, list_size), dtype=np.float64)
        else:
            child = col.combine_chunks().values
            flat = child.to_numpy(zero_copy_only=False).astype(np.float64)
            values = flat.reshape(-1, list_size)
    elif pa.types.is_list(col.type):
        pylist = col.to_pylist()
        rows: list[list[float]] = []
        for item in pylist:
            if item is None:
                rows.append([float("nan")])
            else:
                rows.append([float("nan") if v is None else float(v) for v in item])
        values = np.asarray(rows, dtype=np.float64) if rows else np.empty((0, 1))
    else:
        np_arr = col.to_numpy(zero_copy_only=False).astype(np.float64)
        values = np_arr.reshape(-1, 1)
    return values, int(n_null)


def _compute_feature_quality(
    dataset_dir: Path,
    info: dict[str, Any],
    stats: dict[str, Any],
    range_tol: float,
) -> list[FeatureQuality]:
    """Compute per numeric feature null/NaN/out-of-range metrics."""
    data_files = sorted((dataset_dir / "data").rglob("*.parquet"))
    feature_keys = _numeric_feature_keys(info)

    # Accumulators per feature.
    acc_values: dict[str, int] = defaultdict(int)
    acc_null: dict[str, int] = defaultdict(int)
    acc_nan: dict[str, int] = defaultdict(int)
    acc_oor: dict[str, int] = defaultdict(int)

    # Pre-extract per-feature bounds from stats.json.
    bounds: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
    for key in feature_keys:
        s = stats.get(key)
        if s is not None and "min" in s and "max" in s:
            bounds[key] = (
                np.asarray(s["min"], dtype=np.float64),
                np.asarray(s["max"], dtype=np.float64),
            )
        else:
            bounds[key] = None

    for path in data_files:
        table = pq.read_table(path)
        present = set(table.column_names)
        for key in feature_keys:
            if key not in present:
                continue
            col = table.column(key)
            values, n_null = _column_to_2d(col)
            n_dims = values.shape[1] if values.ndim == 2 else 1
            acc_values[key] += values.shape[0] * n_dims
            acc_null[key] += n_null
            # NaN among non-null cells. Null rows were filled with NaN above,
            # so subtract the per-row null contribution to avoid double count.
            n_nan_total = int(np.count_nonzero(np.isnan(values)))
            acc_nan[key] += n_nan_total - n_null * n_dims
            b = bounds.get(key)
            if b is not None:
                lo, hi = _align_bounds(b[0], b[1], n_dims, key)
                if lo is not None and hi is not None:
                    acc_oor[key] += count_out_of_range(values, lo, hi, range_tol)

    results: list[FeatureQuality] = []
    for key in feature_keys:
        n_values = acc_values[key]
        n_null = acc_null[key]
        n_nan = max(acc_nan[key], 0)
        n_oor = acc_oor[key]
        null_rate = (n_null + n_nan) / n_values if n_values else 0.0
        oor_rate = n_oor / n_values if n_values else 0.0
        results.append(
            FeatureQuality(
                feature=key,
                n_values=n_values,
                n_null=n_null,
                n_nan=n_nan,
                null_rate=null_rate,
                n_out_of_range=n_oor,
                oor_rate=oor_rate,
            )
        )
    return results


def _video_file_groups(
    dataset_dir: Path,
    video_keys: list[str],
) -> dict[str, dict[tuple[int, int], int]]:
    """Group episode lengths by ``(videos/<vk>/chunk_index, file_index)`` per key.

    Mirrors ``tests/test_video_frame_alignment.py``: every mp4 file's expected
    frame count is the sum of the ``length`` values of episodes pointing at it.

    Returns ``{video_key: {(chunk, file): expected_frames}}``.
    """
    parquet_paths = _collect_episodes_parquet(dataset_dir)
    tables = [pq.read_table(p) for p in parquet_paths]
    merged = pa.concat_tables(tables, promote_options="default")
    lengths = [int(x) for x in merged.column("length").to_pylist()]

    groups: dict[str, dict[tuple[int, int], int]] = {vk: {} for vk in video_keys}
    for vk in video_keys:
        chunk_col = merged.column(f"videos/{vk}/chunk_index").to_pylist()
        file_col = merged.column(f"videos/{vk}/file_index").to_pylist()
        per_file: dict[tuple[int, int], int] = defaultdict(int)
        for length, ck, fk in zip(lengths, chunk_col, file_col):
            per_file[(int(ck), int(fk))] += length
        groups[vk] = dict(per_file)
    return groups


def _compute_video_reconciliation(
    dataset_dir: Path,
    info: dict[str, Any],
    freeze_std_eps: float,
    sample_video: bool,
) -> list[VideoReconciliation]:
    """Reconcile mp4 frame counts with parquet lengths + count freeze frames."""
    video_keys = video_feature_keys(info)
    if not video_keys:
        return []

    # Restrict to keys actually present in the episodes parquet.
    parquet_paths = _collect_episodes_parquet(dataset_dir)
    sample_cols = pq.read_schema(parquet_paths[0]).names
    present_vkeys = [
        vk for vk in video_keys if vk in set(_discover_video_keys(sample_cols))
    ]

    groups = _video_file_groups(dataset_dir, present_vkeys)

    results: list[VideoReconciliation] = []
    for vk in present_vkeys:
        expected_total = 0
        mp4_total = 0
        freeze_total = 0
        for (ck, fk), expected in sorted(groups[vk].items()):
            mp4 = dataset_dir / "videos" / vk / f"chunk-{ck:03d}" / f"file-{fk:03d}.mp4"
            expected_total += expected
            if not mp4.is_file():
                # Missing mp4 surfaces as a frame_mismatch (mp4 contributes 0).
                logger.warning("quality-report: missing mp4 %s", mp4)
                continue
            mp4_total += _ffprobe_count_frames(mp4)
            if sample_video:
                freeze_total += count_freeze_frames(
                    _decode_video_frames(mp4),
                    freeze_std_eps,
                )

        freeze_rate = freeze_total / mp4_total if mp4_total else 0.0
        results.append(
            VideoReconciliation(
                video_key=vk,
                expected_frames=expected_total,
                mp4_frames=mp4_total,
                frame_mismatch=mp4_total - expected_total,
                n_freeze=freeze_total,
                freeze_rate=freeze_rate,
            )
        )
    return results


def _compute_score(
    features: list[FeatureQuality],
    videos: list[VideoReconciliation],
    weights: dict[str, float],
) -> float:
    """Combine the metric rates into a 0..1 quality score."""
    mean_null = float(np.mean([f.null_rate for f in features])) if features else 0.0
    mean_oor = float(np.mean([f.oor_rate for f in features])) if features else 0.0
    mean_freeze = float(np.mean([v.freeze_rate for v in videos])) if videos else 0.0
    penalty = (
        weights["w_null"] * mean_null
        + weights["w_range"] * mean_oor
        + weights["w_freeze"] * mean_freeze
    )
    penalty = min(max(penalty, 0.0), 1.0)
    return 1.0 - penalty


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_quality_report(
    dataset_dir: Path,
    freeze_std_eps: float = 1e-3,
    range_tol: float = 0.0,
    sample_video: bool = True,
    score_threshold: float = 0.95,
    info: dict[str, Any] | None = None,
    stats: dict[str, Any] | None = None,
) -> QualityReport:
    """Compute a :class:`QualityReport` for a generated LeRobot v3.0 dataset.

    Reads the data parquet, ``meta/stats.json``, ``meta/info.json``, and the
    episode/mp4 layout to derive null/NaN, out-of-range, freeze, and
    video/data-reconciliation metrics, then folds them into a 0..1 score.

    Verdict policy: ``FAIL`` if any video key has a nonzero ``frame_mismatch``
    (hard fail, independent of score) **or** the score is below
    ``score_threshold``. Freeze frames are reported only and never fail.

    Args:
        dataset_dir: Root of a LeRobot v3.0 dataset.
        freeze_std_eps: Per-pair std threshold for freeze-frame detection.
        range_tol: Absolute tolerance added to stats.json min/max bounds.
        sample_video: When ``True`` (default), decode mp4s to count freeze
            frames. When ``False``, freeze metrics are skipped (0) but the
            cheap ffprobe-based reconciliation still runs.
        score_threshold: Minimum score for an ``OK`` verdict.
        info: Pre-loaded ``meta/info.json`` contents. When ``None`` (default),
            it is read from disk; pass it to avoid re-reading when the caller
            already has it (e.g. :func:`rosbag2lerobot.preview.generate_preview`).
        stats: Pre-loaded ``meta/stats.json`` contents. When ``None``
            (default), it is read from disk (see ``info``).

    Returns:
        A populated :class:`QualityReport` with ``verdict`` / ``exit_code``
        already resolved.
    """
    dataset_dir = Path(dataset_dir)
    if info is None:
        info = _read_info(dataset_dir)
    if stats is None:
        stats = _load_stats(dataset_dir)

    weights = {
        "w_null": _DEFAULT_W_NULL,
        "w_range": _DEFAULT_W_RANGE,
        "w_freeze": _DEFAULT_W_FREEZE,
    }

    features = _compute_feature_quality(dataset_dir, info, stats, range_tol)
    videos = _compute_video_reconciliation(
        dataset_dir, info, freeze_std_eps, sample_video
    )

    score = _compute_score(features, videos, weights)

    hard_fail = any(v.frame_mismatch != 0 for v in videos)
    verdict_fail = hard_fail or score < score_threshold

    return QualityReport(
        dataset=str(dataset_dir),
        features=features,
        videos=videos,
        weights=weights,
        score=score,
        score_threshold=score_threshold,
        verdict="FAIL" if verdict_fail else "OK",
        exit_code=1 if verdict_fail else 0,
    )
