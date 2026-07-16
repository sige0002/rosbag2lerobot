"""Video ↔ metadata reconciliation for LeRobot v3.0 datasets (torch-free).

Reproduces, with ffprobe/ffmpeg + pyarrow + numpy only, the video frame lookup
that LeRobot performs at training time, so that failures like::

    IndexError: Invalid frame index=43507 for streamIndex=0;
    must be less than 43226

are caught *before* training starts.

What LeRobot actually computes (verified against ``lerobot_dataset.py`` /
``dataset_reader.py`` / ``video_utils.py``): for a data row with episode-relative
``timestamp`` and its episode's ``videos/<key>/from_timestamp``, the TorchCodec
backend requests::

    requested_frame = round((from_timestamp + row_timestamp) * average_fps)

where ``average_fps`` is the *video decoder's* fps (ffprobe ``avg_frame_rate``
here), NOT ``info.json``'s ``fps``. The row is loadable iff::

    0 <= requested_frame < video_num_frames

and, after decoding, the loaded frame's PTS must satisfy
``abs(loaded_pts - (from_timestamp + row_timestamp)) < tolerance_s`` (strict
inequality; LeRobot raises ``FrameTimestampError`` otherwise).

Two modes:

- **fast** (default): per episode x video key, only the min/max row timestamps
  are range-checked against the mp4 frame count taken from the container header
  (``nb_frames``; falls back to a ``-count_frames`` scan when the header lacks
  it). Catches end-of-video overruns instantly.
- **strict**: additionally fetches every frame's PTS
  (``best_effort_timestamp_time``, falling back to ``pts_time``) and validates
  *every* data row: frame range, PTS tolerance, index continuity, timestamp
  monotonicity, and header/decoded frame-count agreement. ``full_decode=True``
  additionally runs ``ffmpeg -xerror`` to the end of each stream.

Everything is vectorised with numpy; the only Python-level per-row work is
formatting the (capped) error records. Video probing is parallelised across
files with a thread pool (subprocess waits release the GIL), and each mp4 is
probed exactly once regardless of how many episodes reference it.

Scope guarantee (by design): this checks *LeRobot's video reference conditions
via FFmpeg*. It does not guarantee that ``lerobot-train`` will succeed, nor
cover TorchCodec- or DataLoader-worker-specific failures.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rosbag2lerobot.audit import _collect_episodes_parquet

logger = logging.getLogger(__name__)

__all__ = [
    "SetupError",
    "VideoMetadataError",
    "VideoMetadataReport",
    "validate_video_metadata",
]

# Default video path template (matches DatasetWriter._write_info_json). Only
# used when ``info.json`` does not carry an explicit ``video_path`` key.
_DEFAULT_VIDEO_PATH = (
    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
)

# The four per-video-key columns an episodes parquet must provide.
_VIDEO_COL_SUFFIXES = ("chunk_index", "file_index", "from_timestamp", "to_timestamp")

# Warn when |info.json fps - video avg_frame_rate| exceeds this (§4.6).
_FPS_MISMATCH_EPS = 1e-3

# Max worker threads for parallel ffprobe. Probing is subprocess-bound, so a
# small pool is enough to overlap multi-camera decode scans.
_PROBE_MAX_WORKERS = 4


class SetupError(Exception):
    """The dataset cannot be checked at all (CLI exit code 2).

    Attributes:
        code: Machine-readable reason (``INFO_JSON_MISSING``,
            ``INVALID_DATASET_FPS``, ``EPISODE_METADATA_MISSING``,
            ``DATA_PARQUET_MISSING``, ``MISSING_REQUIRED_COLUMN``,
            ``FFPROBE_NOT_FOUND``, ``FFMPEG_NOT_FOUND``).
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class VideoMetadataError:
    """One video ↔ metadata inconsistency (or warning).

    Field availability depends on ``status``; unavailable fields are ``None``.
    ``severity`` is ``"error"`` (fails the verdict) or ``"warning"``.
    """

    status: str
    severity: str
    episode_index: Optional[int]
    video_key: Optional[str]
    video_path: Optional[str]
    dataset_index: Optional[int] = None
    row_timestamp: Optional[float] = None
    from_timestamp: Optional[float] = None
    shifted_timestamp: Optional[float] = None
    video_average_fps: Optional[float] = None
    requested_frame: Optional[int] = None
    video_frame_count: Optional[int] = None
    max_valid_frame: Optional[int] = None
    overflow: Optional[int] = None
    loaded_pts: Optional[float] = None
    timestamp_error: Optional[float] = None
    tolerance_s: Optional[float] = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "severity": self.severity,
            "episode_index": self.episode_index,
            "video_key": self.video_key,
            "video_path": self.video_path,
            "dataset_index": self.dataset_index,
            "row_timestamp": self.row_timestamp,
            "from_timestamp": self.from_timestamp,
            "shifted_timestamp": self.shifted_timestamp,
            "video_average_fps": self.video_average_fps,
            "requested_frame": self.requested_frame,
            "video_frame_count": self.video_frame_count,
            "max_valid_frame": self.max_valid_frame,
            "overflow": self.overflow,
            "loaded_pts": self.loaded_pts,
            "timestamp_error": self.timestamp_error,
            "tolerance_s": self.tolerance_s,
            "detail": self.detail,
        }


@dataclass
class VideoMetadataReport:
    """Top-level result of :func:`validate_video_metadata`."""

    dataset: str
    mode: str  # "fast" | "strict"
    tolerance_s: Optional[float]  # None in fast mode
    full_decode: bool
    videos_checked: int
    episodes_checked: int
    mappings_checked: int
    rows_checked: int  # 0 in fast mode
    errors: list[VideoMetadataError] = field(default_factory=list)
    warnings: list[VideoMetadataError] = field(default_factory=list)
    total_errors: int = 0  # includes records dropped by max_errors
    total_warnings: int = 0
    truncated: bool = False

    @property
    def verdict(self) -> str:
        return "OK" if self.total_errors == 0 else "ERROR"

    @property
    def exit_code(self) -> int:
        return 0 if self.total_errors == 0 else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "mode": self.mode,
            "tolerance_s": self.tolerance_s,
            "full_decode": self.full_decode,
            "verdict": self.verdict,
            "exit_code": self.exit_code,
            "videos_checked": self.videos_checked,
            "episodes_checked": self.episodes_checked,
            "mappings_checked": self.mappings_checked,
            "rows_checked": self.rows_checked,
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
            "truncated": self.truncated,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
        }


# ---------------------------------------------------------------------------
# ffprobe / ffmpeg helpers (the only external processes used)
# ---------------------------------------------------------------------------


@dataclass
class _VideoInfo:
    """Cached per-mp4 probe result."""

    status: str  # OK | VIDEO_MISSING | VIDEO_UNREADABLE | INVALID_VIDEO_FPS | FRAME_PTS_READ_FAILED
    avg_fps: Optional[float] = None
    n_frames: Optional[int] = None  # authoritative bound used for range checks
    header_frames: Optional[int] = None  # container nb_frames (may lie)
    pts: Optional[np.ndarray] = None  # strict only: per-frame PTS, presentation order
    detail: str = ""
    decode_ok: Optional[bool] = None  # full-decode result (None = not run)
    decode_detail: str = ""


def _parse_rate(rate: Optional[str]) -> Optional[float]:
    """Parse ffprobe's ``avg_frame_rate`` (``"30000/1001"``) into a float fps."""
    if not rate or rate == "N/A":
        return None
    num_s, _, den_s = rate.partition("/")
    try:
        num = float(num_s)
        den = float(den_s) if den_s else 1.0
    except ValueError:
        return None
    if den == 0 or not math.isfinite(num / den) or num / den <= 0:
        return None
    return num / den


def _parse_int(val: Any) -> Optional[int]:
    if val in (None, "", "N/A"):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _ffprobe(args: list[str], path: Path, *, text: bool = True) -> Optional[str]:
    """Run ffprobe with *args* on *path*; return stdout or ``None`` on failure."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", *args, str(path)]
    try:
        return subprocess.check_output(
            cmd,
            text=text,
            stdin=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, OSError):
        return None


def _probe_header(path: Path) -> tuple[Optional[float], Optional[int]]:
    """Return ``(avg_fps, header_nb_frames)`` from the container header (no decode)."""
    out = _ffprobe(
        ["-show_entries", "stream=avg_frame_rate,nb_frames", "-of", "json"], path
    )
    if out is None:
        return None, None
    try:
        streams = json.loads(out).get("streams") or []
    except json.JSONDecodeError:
        return None, None
    if not streams:
        return None, None
    stream = streams[0]
    return _parse_rate(stream.get("avg_frame_rate")), _parse_int(
        stream.get("nb_frames")
    )


def _count_frames_decoded(path: Path) -> Optional[int]:
    """Authoritative decoded frame count (``-count_frames``; decodes every packet)."""
    out = _ffprobe(
        [
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
        ],
        path,
    )
    if out is None:
        return None
    return _parse_int(out.strip())


def _read_frame_pts(path: Path) -> tuple[Optional[np.ndarray], int]:
    """Return ``(pts_array, n_missing)`` for every frame of *path*.

    Decodes the stream once (``-show_entries frame=...``) and parses the
    self-describing ``compact=nk=0`` output. ``best_effort_timestamp_time`` is
    preferred; ``pts_time`` is the per-frame fallback (§5.2). Frames providing
    neither become NaN and are counted in ``n_missing``.

    Frames are emitted by ffprobe in decoder-output (= presentation) order,
    matching how TorchCodec indexes ``get_frames_at``.
    """
    out = _ffprobe(
        [
            "-show_entries",
            "frame=best_effort_timestamp_time,pts_time",
            "-of",
            "compact=p=0:nk=0",
        ],
        path,
    )
    if out is None:
        return None, 0

    lines = out.split()
    pts = np.full(len(lines), np.nan, dtype=np.float64)
    n_missing = 0
    for i, line in enumerate(lines):
        # Typical line: "pts_time=0.100000|best_effort_timestamp_time=0.100000"
        best = None
        fallback = None
        for tok in line.split("|"):
            if not tok:
                continue
            key, _, val = tok.partition("=")
            if val in ("", "N/A"):
                continue
            if key == "best_effort_timestamp_time":
                best = val
                break  # preferred value found; no need to scan further
            if key == "pts_time":
                fallback = val
        chosen = best if best is not None else fallback
        if chosen is None:
            n_missing += 1
            continue
        try:
            pts[i] = float(chosen)
        except ValueError:
            n_missing += 1
    return pts, n_missing


def _full_decode_check(path: Path) -> tuple[bool, str]:
    """Decode the whole stream with ``ffmpeg -xerror``; return ``(ok, detail)``."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-xerror",
        "-nostdin",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        return False, f"ffmpeg could not run: {exc}"
    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode(errors="replace").strip()
        return False, stderr.splitlines()[-1] if stderr else f"exit {proc.returncode}"
    return True, ""


def _probe_video(path: Path, strict: bool, full_decode: bool) -> _VideoInfo:
    """Probe one mp4. See module docstring for the fast/strict split."""
    if not path.is_file():
        return _VideoInfo(status="VIDEO_MISSING", detail=f"mp4 not found: {path}")

    avg_fps, header_frames = _probe_header(path)

    if not strict:
        n = header_frames
        if n is None or n <= 0:
            # Header lacks nb_frames — fall back to the authoritative decode
            # count for this one file so fast mode never spuriously fails.
            n = _count_frames_decoded(path)
        if n is None or n <= 0:
            return _VideoInfo(
                status="VIDEO_UNREADABLE",
                avg_fps=avg_fps,
                detail="no frame count via header nb_frames nor -count_frames",
            )
        if avg_fps is None:
            return _VideoInfo(
                status="INVALID_VIDEO_FPS",
                n_frames=n,
                header_frames=header_frames,
                detail="avg_frame_rate missing/invalid in stream header",
            )
        return _VideoInfo(
            status="OK", avg_fps=avg_fps, n_frames=n, header_frames=header_frames
        )

    # Strict: decode once for the per-frame PTS list. len(pts) IS the decoded
    # frame count (identical to nb_read_frames by construction — same decode),
    # so no separate -count_frames pass is needed.
    pts, n_missing = _read_frame_pts(path)
    if pts is None or pts.size == 0:
        return _VideoInfo(
            status="VIDEO_UNREADABLE",
            avg_fps=avg_fps,
            header_frames=header_frames,
            detail="ffprobe could not read any frames",
        )
    info = _VideoInfo(
        status="OK",
        avg_fps=avg_fps,
        n_frames=int(pts.size),
        header_frames=header_frames,
        pts=pts,
    )
    if n_missing > 0:
        info.status = "FRAME_PTS_READ_FAILED"
        info.detail = f"{n_missing}/{pts.size} frames had no usable PTS"
    elif avg_fps is None:
        info.status = "INVALID_VIDEO_FPS"
        info.detail = "avg_frame_rate missing/invalid in stream header"

    if full_decode:
        info.decode_ok, info.decode_detail = _full_decode_check(path)
    return info


# ---------------------------------------------------------------------------
# Metadata / data loading (pyarrow -> numpy, no per-row Python)
# ---------------------------------------------------------------------------


def _read_info(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "meta" / "info.json"
    if not path.is_file():
        raise SetupError("INFO_JSON_MISSING", f"info.json not found: {path}")
    try:
        with open(path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise SetupError("INFO_JSON_MISSING", f"info.json unreadable: {exc}") from exc


def _load_episodes(dataset_dir: Path) -> pa.Table:
    try:
        paths = _collect_episodes_parquet(dataset_dir)
    except FileNotFoundError as exc:
        raise SetupError("EPISODE_METADATA_MISSING", str(exc)) from exc
    try:
        tables = [pq.read_table(p) for p in paths]
        merged = pa.concat_tables(tables, promote_options="default")
    except (pa.ArrowInvalid, OSError) as exc:
        raise SetupError(
            "EPISODE_METADATA_MISSING", f"episodes parquet unreadable: {exc}"
        ) from exc
    for col in ("episode_index", "length"):
        if col not in merged.column_names:
            raise SetupError(
                "MISSING_REQUIRED_COLUMN",
                f"episodes parquet is missing required column {col!r}",
            )
    return merged.sort_by("episode_index")


@dataclass
class _EpisodeData:
    """One episode's data rows, sorted by ``index``.

    ``idx_f`` / ``ts_f`` / ``ts_f_max`` are the finite-timestamp views computed
    once here so the per-(episode x video key) loop doesn't re-mask the same
    arrays for every camera. When every timestamp is finite they alias ``idx`` /
    ``ts`` directly (no copy); downstream code only reads them.
    """

    idx: np.ndarray  # int64 dataset indices
    ts: np.ndarray  # float64 episode-relative timestamps
    finite: np.ndarray  # bool mask over ts
    n_finite: int
    idx_f: np.ndarray  # idx restricted to finite-ts rows
    ts_f: np.ndarray  # ts restricted to finite rows
    ts_f_max: float  # max(ts_f); nan when n_finite == 0


def _load_data(dataset_dir: Path) -> dict[int, _EpisodeData]:
    """Load ``data/**/*.parquet`` into per-episode numpy arrays.

    Reads only the three needed columns and groups by episode with one
    lexsort — no per-row Python.
    """
    data_dir = dataset_dir / "data"
    files = sorted(data_dir.rglob("*.parquet")) if data_dir.is_dir() else []
    if not files:
        raise SetupError(
            "DATA_PARQUET_MISSING", f"no data parquet files under {data_dir}"
        )
    needed = ["episode_index", "index", "timestamp"]
    tables = []
    for p in files:
        # One ParquetFile handle per shard: schema check + column read share a
        # single footer parse instead of opening the file twice.
        try:
            pf = pq.ParquetFile(p)
            schema_names = pf.schema_arrow.names
        except (pa.ArrowInvalid, OSError) as exc:
            raise SetupError(
                "DATA_PARQUET_MISSING", f"data parquet unreadable: {p}: {exc}"
            ) from exc
        missing = [c for c in needed if c not in schema_names]
        if missing:
            raise SetupError(
                "MISSING_REQUIRED_COLUMN",
                f"data parquet {p} is missing required columns {missing}",
            )
        tables.append(pf.read(columns=needed))
    merged = pa.concat_tables(tables)

    ep = merged.column("episode_index").to_numpy(zero_copy_only=False)
    idx = merged.column("index").to_numpy(zero_copy_only=False)
    ts = (
        merged.column("timestamp")
        .to_numpy(zero_copy_only=False)
        .astype(np.float64, copy=False)
    )
    ep = np.asarray(ep, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)

    # Single lexsort groups rows by (episode, index) in one C-level pass.
    order = np.lexsort((idx, ep))
    ep, idx, ts = ep[order], idx[order], ts[order]
    boundaries = np.flatnonzero(np.diff(ep)) + 1
    ep_starts = np.concatenate(([0], boundaries))
    ep_ends = np.concatenate((boundaries, [ep.size]))

    out: dict[int, _EpisodeData] = {}
    for s, e in zip(ep_starts.tolist(), ep_ends.tolist()):
        idx_slice = idx[s:e]
        ts_slice = ts[s:e]
        finite = np.isfinite(ts_slice)
        n_finite = int(np.count_nonzero(finite))
        if n_finite == ts_slice.size:
            idx_f, ts_f = idx_slice, ts_slice  # all finite: alias, no copy
        else:
            idx_f, ts_f = idx_slice[finite], ts_slice[finite]
        out[int(ep[s])] = _EpisodeData(
            idx=idx_slice,
            ts=ts_slice,
            finite=finite,
            n_finite=n_finite,
            idx_f=idx_f,
            ts_f=ts_f,
            ts_f_max=float(np.max(ts_f)) if n_finite else math.nan,
        )
    return out


def _discover_video_keys(columns: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Extract video keys from ALL ``videos/`` columns (§6.5).

    Returns ``(keys_in_order, {key: missing_required_suffixes})`` so callers can
    flag keys whose 4-column set is incomplete instead of silently skipping.
    """
    seen: dict[str, set[str]] = {}
    order: list[str] = []
    for col in columns:
        if not col.startswith("videos/"):
            continue
        body = col[len("videos/") :]
        vkey, _, suffix = body.rpartition("/")
        if not vkey:
            continue
        if vkey not in seen:
            seen[vkey] = set()
            order.append(vkey)
        seen[vkey].add(suffix)
    missing = {
        vk: [s for s in _VIDEO_COL_SUFFIXES if s not in seen[vk]] for vk in order
    }
    return order, missing


# ---------------------------------------------------------------------------
# Issue collection with a hard cap (§6.6)
# ---------------------------------------------------------------------------


class _IssueSink:
    """Collects errors/warnings, keeping at most ``cap`` records of each."""

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.errors: list[VideoMetadataError] = []
        self.warnings: list[VideoMetadataError] = []
        self.total_errors = 0
        self.total_warnings = 0

    def error(self, issue: VideoMetadataError) -> None:
        self.total_errors += 1
        if len(self.errors) < self.cap:
            self.errors.append(issue)

    def warning(self, issue: VideoMetadataError) -> None:
        self.total_warnings += 1
        if len(self.warnings) < self.cap:
            self.warnings.append(issue)

    @property
    def truncated(self) -> bool:
        return self.total_errors > len(self.errors) or self.total_warnings > len(
            self.warnings
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_video_metadata(
    dataset_dir: Path,
    strict: bool = False,
    tolerance_s: Optional[float] = None,
    full_decode: bool = False,
    max_errors: int = 50,
) -> VideoMetadataReport:
    """Check LeRobot's video reference conditions via FFmpeg.

    Args:
        dataset_dir: Root of a generated LeRobot v3.0 dataset.
        strict: ``False`` = fast mode (min/max row per mapping, header frame
            count). ``True`` = every data row is validated against per-frame
            PTS (see module docstring).
        tolerance_s: PTS tolerance for strict mode. ``None`` defaults to
            ``0.5 / info.fps``. LeRobot's own training default is ``1e-4``;
            pass that to reproduce training behavior exactly.
        full_decode: Strict mode only — additionally decode every stream to
            its end with ``ffmpeg -xerror`` (catches mid-stream corruption).
        max_errors: Cap on *recorded* error and warning records (totals are
            still counted; ``report.truncated`` flags the cut).

    Returns:
        A :class:`VideoMetadataReport`. ``exit_code`` is 0 (consistent) or 1
        (inconsistencies found, including missing/unreadable mp4s).

    Raises:
        SetupError: The check cannot run at all (exit code 2): missing
            ``info.json`` / episodes metadata / data parquet, invalid dataset
            fps, missing required columns, or ffprobe/ffmpeg not installed.
    """
    dataset_dir = Path(dataset_dir)
    if shutil.which("ffprobe") is None:
        raise SetupError(
            "FFPROBE_NOT_FOUND",
            "ffprobe not found. Please install ffmpeg to check videos.",
        )
    if full_decode and shutil.which("ffmpeg") is None:
        raise SetupError(
            "FFMPEG_NOT_FOUND",
            "ffmpeg not found. --full-decode needs the ffmpeg binary.",
        )
    if full_decode and not strict:
        strict = True  # §5.6: full decode is a strict-mode extension.

    info = _read_info(dataset_dir)
    fps = info.get("fps")
    if (
        not isinstance(fps, (int, float))
        or isinstance(fps, bool)
        or not math.isfinite(float(fps))
        or fps <= 0
    ):
        raise SetupError("INVALID_DATASET_FPS", f"info.json has invalid fps: {fps!r}")
    fps = float(fps)
    tol = float(tolerance_s) if tolerance_s is not None else 0.5 / fps
    video_path_tmpl = info.get("video_path", _DEFAULT_VIDEO_PATH)

    episodes = _load_episodes(dataset_dir)
    data = _load_data(dataset_dir)

    columns = episodes.column_names
    vkeys, vkey_missing_cols = _discover_video_keys(columns)

    ep_index = episodes.column("episode_index").to_numpy(zero_copy_only=False)
    ep_index = np.asarray(ep_index, dtype=np.int64)
    ep_length = np.asarray(
        episodes.column("length").to_numpy(zero_copy_only=False), dtype=np.float64
    )
    n_eps = ep_index.size

    # Per-vkey metadata columns as float64 (nulls -> NaN uniformly).
    vcol: dict[str, dict[str, np.ndarray]] = {}
    for vk in vkeys:
        if vkey_missing_cols[vk]:
            continue
        vcol[vk] = {
            suf: np.asarray(
                episodes.column(f"videos/{vk}/{suf}").to_numpy(zero_copy_only=False),
                dtype=np.float64,
            )
            for suf in _VIDEO_COL_SUFFIXES
        }

    sink = _IssueSink(max_errors)

    # ---- Per-video-key column completeness (§6.5) ----
    for vk in vkeys:
        if vkey_missing_cols[vk]:
            sink.error(
                VideoMetadataError(
                    status="MISSING_REQUIRED_COLUMN",
                    severity="error",
                    episode_index=None,
                    video_key=vk,
                    video_path=None,
                    detail=(
                        f"episodes parquet lacks videos/{vk}/"
                        f"{{{', '.join(vkey_missing_cols[vk])}}}"
                    ),
                )
            )

    checked_vkeys = [vk for vk in vkeys if not vkey_missing_cols[vk]]

    # ---- Prefetch: probe every referenced mp4 exactly once, in parallel ----
    unique_paths: dict[Path, str] = {}  # abs path -> rel path (first seen)
    mapping_path: dict[str, list[Optional[str]]] = {}  # vk -> per-ep-row rel path
    for vk in checked_vkeys:
        cols = vcol[vk]
        ck_arr, fk_arr = cols["chunk_index"], cols["file_index"]
        from_arr, to_arr = cols["from_timestamp"], cols["to_timestamp"]
        null_mask = (
            np.isnan(ck_arr) | np.isnan(fk_arr) | np.isnan(from_arr) | np.isnan(to_arr)
        ).tolist()
        # Per-vkey list indexed by episode row (cheaper than a tuple-keyed
        # dict) + memoised path formatting: many episodes share one mp4.
        vk_paths: list[Optional[str]] = [None] * n_eps
        fmt_cache: dict[tuple[int, int], str] = {}
        for i in range(n_eps):
            if null_mask[i]:
                continue
            ck_fk = (int(ck_arr[i]), int(fk_arr[i]))
            rel = fmt_cache.get(ck_fk)
            if rel is None:
                rel = video_path_tmpl.format(
                    video_key=vk, chunk_index=ck_fk[0], file_index=ck_fk[1]
                )
                fmt_cache[ck_fk] = rel
                unique_paths.setdefault(dataset_dir / rel, rel)
            vk_paths[i] = rel
        mapping_path[vk] = vk_paths

    probe_cache: dict[Path, _VideoInfo] = {}
    if unique_paths:
        paths = list(unique_paths)
        # Fast-mode probes are header-only (startup-bound): a small pool is
        # plenty. Strict/full-decode probes decode whole streams (CPU-bound in
        # the child process), so scale the pool to the machine.
        if strict:
            workers = min(os.cpu_count() or _PROBE_MAX_WORKERS, len(paths))
        else:
            workers = min(_PROBE_MAX_WORKERS, len(paths))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for path, vinfo in zip(
                paths,
                pool.map(lambda p: _probe_video(p, strict, full_decode), paths),
            ):
                probe_cache[path] = vinfo

    # ---- Global index duplicates (strict, §5.5) ----
    if strict and data:
        all_idx = np.concatenate([d.idx for d in data.values()])
        uniq, counts = np.unique(all_idx, return_counts=True)
        for dup_idx in uniq[counts > 1][:max_errors].tolist():
            sink.error(
                VideoMetadataError(
                    status="DUPLICATE_INDEX",
                    severity="error",
                    episode_index=None,
                    video_key=None,
                    video_path=None,
                    dataset_index=int(dup_idx),
                    detail="dataset index appears in more than one data row",
                )
            )

    # ---- Per-episode data checks ----
    episodes_with_issue_skip: set[int] = set()  # episodes whose ts are unusable
    for i in range(n_eps):
        ep_i = int(ep_index[i])
        d = data.get(ep_i)
        if d is None or d.idx.size == 0:
            sink.error(
                VideoMetadataError(
                    status="MISSING_EPISODE_DATA",
                    severity="error",
                    episode_index=ep_i,
                    video_key=None,
                    video_path=None,
                    detail="episode has metadata but no data parquet rows",
                )
            )
            episodes_with_issue_skip.add(ep_i)
            continue

        if not math.isnan(ep_length[i]) and int(ep_length[i]) != d.idx.size:
            sink.error(
                VideoMetadataError(
                    status="EPISODE_LENGTH_MISMATCH",
                    severity="error",
                    episode_index=ep_i,
                    video_key=None,
                    video_path=None,
                    detail=(
                        f"episodes.length={int(ep_length[i])} but data parquet has "
                        f"{d.idx.size} rows"
                    ),
                )
            )

        n_bad_ts = d.ts.size - d.n_finite
        if n_bad_ts:
            first_bad = int(np.flatnonzero(~d.finite)[0])
            sink.error(
                VideoMetadataError(
                    status="INVALID_TIMESTAMP",
                    severity="error",
                    episode_index=ep_i,
                    video_key=None,
                    video_path=None,
                    dataset_index=int(d.idx[first_bad]),
                    detail=f"{n_bad_ts} non-finite timestamp value(s) in data rows",
                )
            )
            if d.n_finite == 0:
                episodes_with_issue_skip.add(ep_i)
                continue

        if strict:
            if not np.all(np.diff(d.idx) == 1):
                gap_pos = int(np.flatnonzero(np.diff(d.idx) != 1)[0])
                sink.error(
                    VideoMetadataError(
                        status="INVALID_INDEX_SEQUENCE",
                        severity="error",
                        episode_index=ep_i,
                        video_key=None,
                        video_path=None,
                        dataset_index=int(d.idx[gap_pos]),
                        detail=(
                            f"index jumps {int(d.idx[gap_pos])} -> "
                            f"{int(d.idx[gap_pos + 1])} within the episode"
                        ),
                    )
                )
            if n_bad_ts == 0 and d.ts.size > 1 and not np.all(np.diff(d.ts) >= 0):
                drop_pos = int(np.flatnonzero(np.diff(d.ts) < 0)[0])
                sink.error(
                    VideoMetadataError(
                        status="NON_MONOTONIC_TIMESTAMP",
                        severity="error",
                        episode_index=ep_i,
                        video_key=None,
                        video_path=None,
                        dataset_index=int(d.idx[drop_pos + 1]),
                        row_timestamp=float(d.ts[drop_pos + 1]),
                        detail=(
                            f"timestamp decreases {d.ts[drop_pos]:.6f} -> "
                            f"{d.ts[drop_pos + 1]:.6f}"
                        ),
                    )
                )

    # ---- Per-mapping (episode x video key) checks ----
    mappings_checked = 0
    rows_checked = 0
    fps_warned: set[Path] = set()
    decode_reported: set[Path] = set()

    for vk in checked_vkeys:
        cols = vcol[vk]
        from_arr, to_arr = cols["from_timestamp"], cols["to_timestamp"]
        vk_paths = mapping_path[vk]

        for i in range(n_eps):
            ep_i = int(ep_index[i])
            rel = vk_paths[i]
            if rel is None:
                # Nulls: either "no video for this episode/camera" (all four
                # null — writer-legal, skip silently) or partially null
                # (broken metadata — MISSING_METADATA_VALUE).
                ck = cols["chunk_index"][i]
                fk = cols["file_index"][i]
                nulls = [
                    math.isnan(ck),
                    math.isnan(fk),
                    math.isnan(from_arr[i]),
                    math.isnan(to_arr[i]),
                ]
                if not all(nulls):
                    sink.error(
                        VideoMetadataError(
                            status="MISSING_METADATA_VALUE",
                            severity="error",
                            episode_index=ep_i,
                            video_key=vk,
                            video_path=None,
                            detail=(
                                "one of chunk_index/file_index/from_timestamp/"
                                "to_timestamp is null"
                            ),
                        )
                    )
                continue

            mappings_checked += 1
            abs_path = dataset_dir / rel
            vinfo = probe_cache[abs_path]
            from_ts = float(from_arr[i])

            # Video-level problems (reported per referencing episode).
            if vinfo.status in ("VIDEO_MISSING", "VIDEO_UNREADABLE"):
                sink.error(
                    VideoMetadataError(
                        status=vinfo.status,
                        severity="error",
                        episode_index=ep_i,
                        video_key=vk,
                        video_path=rel,
                        from_timestamp=from_ts,
                        detail=vinfo.detail,
                    )
                )
                continue
            if vinfo.status in ("INVALID_VIDEO_FPS", "FRAME_PTS_READ_FAILED"):
                sink.error(
                    VideoMetadataError(
                        status=vinfo.status,
                        severity="error",
                        episode_index=ep_i,
                        video_key=vk,
                        video_path=rel,
                        from_timestamp=from_ts,
                        video_frame_count=vinfo.n_frames,
                        detail=vinfo.detail,
                    )
                )
                continue

            avg_fps = float(vinfo.avg_fps)  # type: ignore[arg-type]
            n_frames = int(vinfo.n_frames)  # type: ignore[arg-type]

            # Once-per-file reports.
            if abs_path not in fps_warned:
                fps_warned.add(abs_path)
                if abs(avg_fps - fps) > _FPS_MISMATCH_EPS:
                    sink.warning(
                        VideoMetadataError(
                            status="DATASET_VIDEO_FPS_MISMATCH",
                            severity="warning",
                            episode_index=ep_i,
                            video_key=vk,
                            video_path=rel,
                            video_average_fps=avg_fps,
                            detail=f"info.json fps={fps} but avg_frame_rate={avg_fps}",
                        )
                    )
                if (
                    strict
                    and vinfo.header_frames is not None
                    and (vinfo.header_frames != n_frames)
                ):
                    sink.error(
                        VideoMetadataError(
                            status="FRAME_COUNT_MISMATCH",
                            severity="error",
                            episode_index=ep_i,
                            video_key=vk,
                            video_path=rel,
                            video_frame_count=n_frames,
                            detail=(
                                f"container header claims {vinfo.header_frames} "
                                f"frames but only {n_frames} decode "
                                "(truncated / partially-copied file?)"
                            ),
                        )
                    )
            if vinfo.decode_ok is False and abs_path not in decode_reported:
                decode_reported.add(abs_path)
                sink.error(
                    VideoMetadataError(
                        status="VIDEO_FULL_DECODE_FAILED",
                        severity="error",
                        episode_index=ep_i,
                        video_key=vk,
                        video_path=rel,
                        video_frame_count=n_frames,
                        detail=vinfo.decode_detail,
                    )
                )

            d = data.get(ep_i)
            if d is None or ep_i in episodes_with_issue_skip:
                continue

            # Finite-row views precomputed once per episode in _load_data
            # (shared across every camera; aliases, not copies, when all
            # timestamps are finite).
            ts_f = d.ts_f
            idx_f = d.idx_f

            # ``np.rint`` implements round-half-to-even on the double values —
            # identical results to Python's built-in round() (§6.3), verified
            # by test_np_rint_matches_python_round.
            if not strict:
                # Fast: check only the extreme rows (min/max timestamp).
                # rows_checked intentionally stays 0 in this mode.
                lo_pos = int(np.argmin(ts_f))
                hi_pos = int(np.argmax(ts_f))
                for pos in {lo_pos, hi_pos}:
                    row_ts = float(ts_f[pos])
                    shifted = from_ts + row_ts
                    req = int(round(shifted * avg_fps))
                    _check_frame_range(
                        sink,
                        ep_i,
                        vk,
                        rel,
                        int(idx_f[pos]),
                        row_ts,
                        from_ts,
                        shifted,
                        avg_fps,
                        req,
                        n_frames,
                    )
            else:
                shifted = from_ts + ts_f
                req = np.rint(shifted * avg_fps).astype(np.int64)
                rows_checked += int(ts_f.size)

                # Evaluate each comparison once; reuse for the ok partition.
                neg_mask = req < 0
                oob_mask = req >= n_frames
                neg = np.flatnonzero(neg_mask)
                oob = np.flatnonzero(oob_mask)
                for pos in neg.tolist():
                    _check_frame_range(
                        sink,
                        ep_i,
                        vk,
                        rel,
                        int(idx_f[pos]),
                        float(ts_f[pos]),
                        from_ts,
                        float(shifted[pos]),
                        avg_fps,
                        int(req[pos]),
                        n_frames,
                    )
                for pos in oob.tolist():
                    _check_frame_range(
                        sink,
                        ep_i,
                        vk,
                        rel,
                        int(idx_f[pos]),
                        float(ts_f[pos]),
                        from_ts,
                        float(shifted[pos]),
                        avg_fps,
                        int(req[pos]),
                        n_frames,
                    )

                ok_pos = np.flatnonzero(~(neg_mask | oob_mask))
                if ok_pos.size:
                    pts = vinfo.pts  # type: ignore[union-attr]
                    loaded = pts[req[ok_pos]]
                    err = np.abs(loaded - shifted[ok_pos])
                    # LeRobot passes iff error < tolerance (strict inequality).
                    bad = np.flatnonzero(err >= tol)
                    for b in bad.tolist():
                        pos = int(ok_pos[b])
                        sink.error(
                            VideoMetadataError(
                                status="FRAME_TIMESTAMP_OUT_OF_TOLERANCE",
                                severity="error",
                                episode_index=ep_i,
                                video_key=vk,
                                video_path=rel,
                                dataset_index=int(idx_f[pos]),
                                row_timestamp=float(ts_f[pos]),
                                from_timestamp=from_ts,
                                shifted_timestamp=float(shifted[pos]),
                                video_average_fps=avg_fps,
                                requested_frame=int(req[pos]),
                                video_frame_count=n_frames,
                                max_valid_frame=n_frames - 1,
                                loaded_pts=float(loaded[b]),
                                timestamp_error=float(err[b]),
                                tolerance_s=tol,
                                detail=(
                                    "|loaded PTS - requested ts| >= tolerance "
                                    "(LeRobot would raise FrameTimestampError)"
                                ),
                            )
                        )

            # ---- to_timestamp advisory (§4.6: warning only) ----
            needed_end = from_ts + d.ts_f_max
            to_ts = float(to_arr[i])
            if to_ts < needed_end:
                sink.warning(
                    VideoMetadataError(
                        status="TO_TIMESTAMP_TOO_SMALL",
                        severity="warning",
                        episode_index=ep_i,
                        video_key=vk,
                        video_path=rel,
                        from_timestamp=from_ts,
                        detail=(
                            f"to_timestamp={to_ts:.6f} < from_timestamp + max row "
                            f"timestamp = {needed_end:.6f} (not used by the frame "
                            "lookup, but metadata is inconsistent)"
                        ),
                    )
                )

    return VideoMetadataReport(
        dataset=str(dataset_dir),
        mode="strict" if strict else "fast",
        tolerance_s=tol if strict else None,
        full_decode=full_decode,
        videos_checked=len(probe_cache),
        episodes_checked=int(n_eps),
        mappings_checked=mappings_checked,
        rows_checked=rows_checked,
        errors=sink.errors,
        warnings=sink.warnings,
        total_errors=sink.total_errors,
        total_warnings=sink.total_warnings,
        truncated=sink.truncated,
    )


def _check_frame_range(
    sink: _IssueSink,
    ep_i: int,
    vk: str,
    rel: str,
    dataset_index: int,
    row_ts: float,
    from_ts: float,
    shifted: float,
    avg_fps: float,
    req: int,
    n_frames: int,
) -> None:
    """Record FRAME_INDEX_NEGATIVE / FRAME_INDEX_OUT_OF_RANGE when applicable."""
    if 0 <= req < n_frames:
        return
    status = "FRAME_INDEX_NEGATIVE" if req < 0 else "FRAME_INDEX_OUT_OF_RANGE"
    sink.error(
        VideoMetadataError(
            status=status,
            severity="error",
            episode_index=ep_i,
            video_key=vk,
            video_path=rel,
            dataset_index=dataset_index,
            row_timestamp=row_ts,
            from_timestamp=from_ts,
            shifted_timestamp=shifted,
            video_average_fps=avg_fps,
            requested_frame=req,
            video_frame_count=n_frames,
            max_valid_frame=n_frames - 1,
            overflow=req - (n_frames - 1) if req >= n_frames else None,
            detail=(
                f"requested frame {req} outside [0, {n_frames}) "
                "(LeRobot/TorchCodec would raise IndexError)"
            ),
        )
    )
