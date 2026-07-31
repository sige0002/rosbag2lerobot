"""Dataset writer for LeRobot Dataset v3.0 format.

Buffers frames, encodes videos, writes parquet files, and generates
metadata conforming to the LeRobot v3.0 specification.

The main class ``DatasetWriter`` manages:

- Size-aggregated parquet data files (``data/chunk-XXX/file-XXX.parquet``).
  Multiple episodes are appended into the same parquet file until the file
  grows past ``_DATA_FILES_SIZE_IN_MB``; at that point the writer rotates to
  a fresh file index.
- Size-aggregated per-camera MP4 files: one long-lived ffmpeg encoder per
  camera streams consecutive episodes directly into the current output mp4;
  the file rotates once it grows past ``_VIDEO_FILES_SIZE_IN_MB`` (checked at
  episode boundaries). Every frame is encoded exactly once.
- Episode metadata parquet files grouped by the data file they belong to.
- ``meta/info.json``, ``meta/stats.json``, and ``meta/tasks.parquet``.
  In ``tasks.parquet``, task strings are stored as the (unnamed) pandas
  Index and ``task_index`` is the only regular column, matching the
  physical schema used by public LeRobot v3 datasets on the Hub.

Typical usage::

    writer = DatasetWriter(output_dir, config, features, fps=30)
    for frame in frames:
        writer.add_frame(frame)
    writer.save_episode()
    writer.finalize()
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import queue
import subprocess
import threading
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from rosbag2lerobot.config import RobotConfig, compute_splits
from rosbag2lerobot.stats import StatsComputer
from rosbag2lerobot.task_spec import (
    SubtaskSpan,
    subtask_for_timestamp,
    validate_subtask_coverage,
)

logger = logging.getLogger(__name__)

# LeRobot v3.0 constants
_CODEBASE_VERSION = "v3.0"
_CHUNKS_SIZE = 1000
_DATA_FILES_SIZE_IN_MB = 100
_VIDEO_FILES_SIZE_IN_MB = 200

# Precision for rounding video file timestamps (LeRobot PR #3239 correspondence).
# With integer-driven frame durations (ep_len / fps) the theoretical value is
# exact in rational arithmetic; rounding to 6 decimals eliminates float
# accumulation drift across ep boundaries while staying well below 1 frame at
# typical robotics fps (1e-6 s vs >= 1e-3 s frame period).
_TIMESTAMP_ROUND_DECIMALS = 6

# Queue depth for the per-camera feeder threads that stream raw frames into
# the ffmpeg stdin pipe. Small enough to bound peak memory, large enough to
# keep the encoder fed while the main thread is building the next frame.
_IMAGE_FEED_QUEUE_MAXSIZE = 32

# Timeout (seconds) for waiting on an ffmpeg encoder to drain and exit.
_ENCODER_WAIT_TIMEOUT = 300

# Timeout (seconds) for tearing an encoder down on a failure path. Nothing is
# being drained there — the process is killed first — so this only bounds how
# long a wedged child or feeder can delay the error propagating.
_ENCODER_TEARDOWN_TIMEOUT = 10

# Timeout (seconds) for waiting on a stderr drain thread to observe EOF.
# Only ever waited on after the encoder has been reaped, so EOF is already
# pending; the bound exists so a wedged reader cannot hang the error path.
_STDERR_DRAIN_TIMEOUT = 10

# Read size (bytes) for the per-encoder stderr drain threads.
_STDERR_READ_CHUNK = 8192

# Bytes of ffmpeg stderr retained per encoder for diagnostics. ffmpeg writes
# to stderr for the whole life of the encoder, so it must be drained
# unconditionally — an unread pipe stalls ffmpeg inside write() once the OS
# buffer (typically 64 KiB) fills. Only the tail is kept so a long
# multi-episode conversion cannot grow this without bound; the tail is what
# matters anyway, since ffmpeg reports the fatal error last.
_STDERR_TAIL_MAXLEN = 16 * 1024

# Mapping from ffmpeg encoder name to the LeRobot ``info.json`` ``video.codec``
# label. Unknown codecs fall back to the raw ffmpeg string.
_CODEC_LABEL_MAP: dict[str, str] = {
    "libx264": "h264",
    "libx264rgb": "h264",
    "h264_nvenc": "h264",
    "libx265": "h265",
    "hevc_nvenc": "h265",
    "libsvtav1": "av1",
    "libaom-av1": "av1",
    "av1_nvenc": "av1",
}


def _advance_chunk_file(chunk_idx: int, file_idx: int) -> tuple[int, int]:
    """Advance (chunk_idx, file_idx) by one file slot.

    When ``file_idx`` reaches the last slot in the chunk, roll over to the
    next chunk with ``file_idx = 0``. Otherwise, increment ``file_idx``.
    """
    if file_idx >= _CHUNKS_SIZE - 1:
        return chunk_idx + 1, 0
    return chunk_idx, file_idx + 1


def _codec_label(codec: str) -> str:
    """Return the LeRobot ``video.codec`` label for a given ffmpeg encoder."""
    return _CODEC_LABEL_MAP.get(codec, codec)


def _build_codec_args(
    codec: str,
    preset: str | None,
    crf: int | None,
) -> list[str]:
    """Build codec-specific ffmpeg args for ``-c:v <codec>`` + preset/rate-control.

    The returned list is a suffix of ffmpeg options starting with ``-c:v``,
    suitable for appending to a base command. CPU encoders use ``-crf`` for
    rate-control, NVENC encoders use ``-rc vbr -cq``. ``-threads 0`` is added
    to CPU encoders to let ffmpeg pick a reasonable worker count.

    Args:
        codec: ffmpeg encoder name (``libx264``, ``libsvtav1``, ``h264_nvenc``,
            ``hevc_nvenc``, ``av1_nvenc``, ...).
        preset: explicit preset override, or ``None`` to use the codec default.
        crf: explicit quality override. Mapped to ``-crf`` for CPU encoders
            and ``-cq`` for NVENC encoders. ``None`` to use the codec default.

    Returns:
        A list like ``["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-threads", "0"]``.
    """
    args: list[str] = ["-c:v", codec]

    if codec == "libx264":
        args += ["-preset", preset or "veryfast"]
        args += ["-crf", str(crf if crf is not None else 23)]
        args += ["-threads", "0"]
    elif codec == "libsvtav1":
        args += ["-preset", preset or "8"]
        args += ["-crf", str(crf if crf is not None else 30)]
        args += ["-threads", "0"]
    elif codec in ("h264_nvenc", "hevc_nvenc", "av1_nvenc"):
        args += ["-preset", preset or "p4"]
        args += ["-rc", "vbr"]
        args += ["-cq", str(crf if crf is not None else 25)]
        args += ["-tune", "hq"]
    else:
        logger.warning(
            "Unknown codec %r; falling back to generic CPU settings "
            "(preset=medium, crf=23).",
            codec,
        )
        args += ["-preset", preset or "medium"]
        args += ["-crf", str(crf if crf is not None else 23)]
        args += ["-threads", "0"]

    return args


class _StderrTail:
    """Bounded, thread-safe tail buffer for a subprocess's stderr.

    Written by the drain thread, read by whichever thread reports the
    failure, so every access is taken under a lock: the reporting thread
    joins the drain thread first, but that join is bounded and may time out.
    """

    def __init__(self, maxlen: int = _STDERR_TAIL_MAXLEN) -> None:
        self._maxlen = maxlen
        self._lock = threading.Lock()
        self._buf = bytearray()

    def append(self, chunk: bytes) -> None:
        """Append ``chunk``, discarding everything past ``maxlen`` bytes."""
        with self._lock:
            self._buf.extend(chunk)
            excess = len(self._buf) - self._maxlen
            if excess > 0:
                del self._buf[:excess]

    def text(self) -> str:
        """Return the retained tail decoded for a log line / error message."""
        with self._lock:
            return bytes(self._buf).decode(errors="replace")


class DatasetWriter:
    """Writes frames into a LeRobot v3.0 dataset directory.

    Usage::

        writer = DatasetWriter(output_dir, config, features, fps=30)
        for frame in frames:
            writer.add_frame(frame)
        writer.save_episode()
        writer.finalize()
    """

    def __init__(
        self,
        output_dir: str | Path,
        config: dict[str, Any],
        features: dict[str, dict[str, Any]],
        fps: int,
        repo_id: str | None = None,
        video_codec: str = "libx264",
        ffmpeg_preset: str | None = None,
        ffmpeg_crf: int | None = None,
        has_subtasks: bool = False,
        manifest_extra: dict[str, Any] | None = None,
        splits: dict[str, float] | None = None,
    ) -> None:
        """Initialize the dataset writer.

        Args:
            output_dir: Root directory for the output dataset.
            config: Conversion configuration dict (robot_type, etc.).
            features: Feature specification dict matching info.json ``features`` schema.
                      Keys like ``observation.state``, ``observation.images.right_wrist``, etc.
            fps: Frames per second for the dataset.
            repo_id: Optional Hugging Face repo id.
            video_codec: Video codec for encoding (e.g. ``"libx264"``, ``"libsvtav1"``,
                ``"h264_nvenc"``).
            ffmpeg_preset: Explicit ffmpeg ``-preset`` override. ``None`` uses the
                per-codec default chosen by :func:`_build_codec_args`.
            ffmpeg_crf: Explicit quality override. Mapped to ``-crf`` for CPU
                encoders and ``-cq`` for NVENC encoders. ``None`` uses the per-
                codec default.
            manifest_extra: Optional provenance fields (inputs, config snapshot,
                versions, run timestamp) merged into ``meta/conversion_log.json``.
                ``None`` still writes the writer-owned subset.
            splits: ``{split_name: ratio}`` train/val/test fractions for the
                ``info.json`` ``splits`` partition. ``None`` falls back to the
                legacy single ``{"train": "0:N"}`` split.
        """
        self.output_dir = Path(output_dir)
        self.config = config
        self.features = features
        self.fps = fps
        self.repo_id = repo_id
        self.video_codec = video_codec
        self._ffmpeg_preset = ffmpeg_preset
        self._ffmpeg_crf = ffmpeg_crf
        self.has_subtasks = has_subtasks
        self._manifest_extra = manifest_extra
        # ``None`` (or default {"train": 1.0}) reproduces the legacy single
        # split exactly; compute_splits handles the byte-identical fallback.
        self._splits = splits if splits is not None else {"train": 1.0}

        # State tracking
        self._global_index: int = 0
        self._episode_index: int = 0
        self._frame_index: int = 0
        self._episode_frames: list[dict[str, Any]] = []
        self._episodes_meta: list[dict[str, Any]] = []
        self._tasks: dict[str, int] = {}  # task_str -> task_index

        # Subtask state (populated only when has_subtasks=True). ``_subtasks``
        # is the global subtask_str -> subtask_index mapping (parallel to
        # ``_tasks``). ``_current_ep_subtasks`` is the span list attached to
        # the current episode's first frame; when empty the episode has no
        # subtasks. Frames missing their span fall back to ``_fill_index``.
        self._subtasks: dict[str, int] = {}
        self._current_ep_subtasks: list[SubtaskSpan] = []
        self._video_keys: list[str] = self._detect_video_keys()
        self._stats = StatsComputer()
        # Per-episode statistics accumulator, reset after each episode. Feeds
        # the ``stats/<feature>/<stat>`` columns in meta/episodes parquet,
        # mirroring lerobot-record's per-episode stats.
        self._episode_stats = StatsComputer()

        # Size-based chunk/file rotation — data parquet
        self._data_chunk_idx: int = 0
        self._data_file_idx: int = 0
        self._data_pq_writer: pq.ParquetWriter | None = None
        self._data_current_file_bytes: int = 0

        # Size-based chunk/file rotation — per-video-key state.
        #
        # Each camera owns one long-lived ffmpeg encoder that streams the
        # frames of consecutive episodes directly into the current target
        # mp4 (``videos/<key>/chunk-XXX/file-XXX.mp4``). When the file has
        # grown past ``_VIDEO_FILES_SIZE_IN_MB`` at an episode boundary the
        # encoder is closed and the next episode starts a fresh file
        # (lerobot-record semantics). Compared to the previous design
        # (per-episode staging clips concatenated via an ffmpeg re-encode),
        # every frame is encoded exactly once and no intermediate clips hit
        # the disk.
        self._video_chunk_idx: dict[str, int] = {k: 0 for k in self._video_keys}
        self._video_file_idx: dict[str, int] = {k: 0 for k in self._video_keys}
        self._video_file_duration: dict[str, float] = {k: 0.0 for k in self._video_keys}

        # Streaming encoder state. Each active camera owns one
        # subprocess.Popen writing to the current target mp4, a background
        # feeder thread, a bounded queue providing backpressure between
        # add_frame() and the ffmpeg pipe, and a background stderr drain
        # thread keeping that pipe from filling up (see _open_encoder).
        self._image_encoders: dict[str, subprocess.Popen[bytes]] = {}
        self._image_feeders: dict[str, threading.Thread] = {}
        self._image_feed_queues: dict[str, queue.Queue[bytes | None]] = {}
        self._image_stderr_readers: dict[str, threading.Thread] = {}
        self._image_stderr_tails: dict[str, _StderrTail] = {}
        self._video_target_paths: dict[str, Path] = {}
        self._image_shapes: dict[str, tuple[int, int]] = {}
        self._image_last_frame: dict[str, bytes | None] = {
            k: None for k in self._video_keys
        }
        self._image_frame_counts: dict[str, int] = {k: 0 for k in self._video_keys}
        self._image_feeder_errors: dict[str, BaseException] = {}

        # Log the effective codec configuration once at construction time.
        if self._video_keys:
            logger.info(
                "Video encoder: codec=%s preset=%s crf=%s",
                self.video_codec,
                self._ffmpeg_preset if self._ffmpeg_preset is not None else "<default>",
                self._ffmpeg_crf if self._ffmpeg_crf is not None else "<default>",
            )

        # Create directory structure
        self._ensure_dirs()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_frame(self, frame_dict: dict[str, Any]) -> None:
        """Add a single frame to the current episode.

        Args:
            frame_dict: Mapping of feature keys to values.
                Expected keys: feature names from ``self.features`` plus
                an optional ``"task"`` key (str) and, on the first frame of
                an episode only, an optional ``"_episode_subtasks"`` list
                of :class:`SubtaskSpan` covering the episode's duration.
        """
        # The first frame of each episode may carry subtask spans that
        # cover the full episode. Capture them before extracting other
        # keys so downstream logic can compute ``subtask_index``.
        ep_subtasks = frame_dict.pop("_episode_subtasks", None)
        if ep_subtasks is not None:
            self._current_ep_subtasks = list(ep_subtasks)

        # Extract or default task
        task_str = frame_dict.pop("task", "default_task")
        if task_str not in self._tasks:
            self._tasks[task_str] = len(self._tasks)
        task_index = self._tasks[task_str]

        timestamp = float(self._frame_index) / self.fps

        record: dict[str, Any] = {
            "index": self._global_index,
            "timestamp": np.float32(timestamp),
            "frame_index": self._frame_index,
            "episode_index": self._episode_index,
            "task_index": task_index,
        }

        # Copy feature data + feed stats + stream images into per-camera ffmpeg
        # encoders. We MUST keep each camera's encoded frame count in lockstep
        # with the number of added frames, otherwise the encoded mp4 ends up
        # shorter than the data parquet and LeRobot's frame-time lookups fail
        # (query timestamp exceeds video duration). When a frame has no image
        # we pad: prefer zero-order hold from the most recent image; if none
        # has been seen yet in this episode, use a black frame matching the
        # feature's declared shape. Padded frames are intentionally excluded
        # from the running image stats.
        for key, feat_spec in self.features.items():
            if key in (
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
            ):
                continue
            dtype = feat_spec.get("dtype", "float32")
            value = frame_dict.get(key)

            if dtype == "video":
                self._feed_video_frame(key, value, feat_spec)
            else:
                if value is None:
                    continue
                arr = np.asarray(value, dtype=np.float32).ravel()
                record[key] = arr
                self._feed_stats(key, arr)

        # Feed bookkeeping scalars to the stats accumulators so that the
        # info-level features (timestamp/frame_index/episode_index/index/
        # task_index) receive per-episode and global statistics, matching
        # lerobot-record's stats.json and meta/episodes layout.
        for bkey in (
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ):
            self._feed_stats(bkey, np.asarray([record[bkey]], dtype=np.float32))

        self._episode_frames.append(record)
        self._global_index += 1
        self._frame_index += 1

    def _feed_stats(self, key: str, arr: np.ndarray) -> None:
        """Feed one frame of *arr* to both the global and per-episode stats."""
        self._stats.add_frame(key, arr)
        self._episode_stats.add_frame(key, arr)

    def save_episode(self) -> None:
        """Finalize the current episode: append to data parquet + append to videos."""
        if not self._episode_frames:
            logger.warning("save_episode called with no frames; skipping.")
            return

        ep_len = len(self._episode_frames)
        ep_idx = self._episode_index

        # Note: the ``min_length`` episode-length filter (⑨) is applied
        # upstream in the CLI episode iterators, so short episodes never reach
        # the writer. Every episode that arrives here is written.

        # Determine tasks for this episode
        ep_task_indices = {f["task_index"] for f in self._episode_frames}
        ep_task_strs = [t for t, ti in self._tasks.items() if ti in ep_task_indices]

        # Resolve subtask_index per frame. Coverage is validated against the
        # final (post-trim) episode duration. If this writer was configured
        # without subtasks globally, we still accept spans silently here —
        # but the column will be absent from info.json / data parquet (since
        # ``has_subtasks`` gates the writes). Conversely, when
        # ``has_subtasks=True`` but this episode has no spans, every frame
        # gets subtask_index = -1 (the "no subtask" sentinel).
        ep_subtask_strs: list[str] = []
        if self.has_subtasks:
            if self._current_ep_subtasks:
                episode_duration = ep_len / self.fps
                validate_subtask_coverage(
                    self._current_ep_subtasks,
                    episode_duration,
                    context=f"episode {ep_idx}",
                )
                for record in self._episode_frames:
                    ts = float(record["timestamp"])
                    name = subtask_for_timestamp(self._current_ep_subtasks, ts)
                    if name not in self._subtasks:
                        self._subtasks[name] = len(self._subtasks)
                    record["subtask_index"] = self._subtasks[name]
                ep_subtask_strs = [span.subtask for span in self._current_ep_subtasks]
            else:
                for record in self._episode_frames:
                    record["subtask_index"] = -1

        dataset_from = self._episode_frames[0]["index"]
        dataset_to = self._episode_frames[-1]["index"] + 1

        # ---- Append data parquet at the current (chunk, file) slot ----
        data_chunk_idx = self._data_chunk_idx
        data_file_idx = self._data_file_idx
        data_path = self._data_parquet_path(data_chunk_idx, data_file_idx)
        self._append_data_parquet(data_path, self._episode_frames)

        # Rotate data file if its on-disk size has passed the threshold.
        if data_path.stat().st_size >= _DATA_FILES_SIZE_IN_MB * 1024 * 1024:
            self._close_data_writer()
            self._data_chunk_idx, self._data_file_idx = _advance_chunk_file(
                self._data_chunk_idx,
                self._data_file_idx,
            )
            self._data_current_file_bytes = 0
        else:
            self._data_current_file_bytes = data_path.stat().st_size

        # ---- Register this episode's video segment per camera ----
        # Wait for each camera's queued frames to reach ffmpeg (this also
        # surfaces encoder failures at episode granularity), then record the
        # segment's (chunk, file, from_ts, to_ts) and rotate the output file
        # if it has grown past the size threshold.
        video_meta: dict[str, dict[str, Any]] = {}
        for vkey in self._video_keys:
            if self._image_frame_counts[vkey] == 0:
                continue
            self._drain_video_queue(vkey)
            video_meta[vkey] = self._register_episode_video(vkey, ep_len)

        # ---- Episode metadata (column order mirrors lerobot-record) ----
        ep_meta: dict[str, Any] = {
            "episode_index": ep_idx,
            "tasks": ep_task_strs,
            "length": ep_len,
            "data/chunk_index": data_chunk_idx,
            "data/file_index": data_file_idx,
            "dataset_from_index": dataset_from,
            "dataset_to_index": dataset_to,
        }
        if self.has_subtasks:
            ep_meta["subtasks"] = ep_subtask_strs
        for vkey, vm in video_meta.items():
            ep_meta[f"videos/{vkey}/chunk_index"] = vm["chunk_index"]
            ep_meta[f"videos/{vkey}/file_index"] = vm["file_index"]
            ep_meta[f"videos/{vkey}/from_timestamp"] = vm["from_timestamp"]
            ep_meta[f"videos/{vkey}/to_timestamp"] = vm["to_timestamp"]

        # ---- Per-episode statistics (stats/<feature>/<stat> columns) ----
        ep_meta.update(self._format_episode_stats(self._episode_stats.compute()))
        self._episodes_meta.append(ep_meta)

        # Reset per-episode state and advance the episode counter.
        self._reset_episode_state()
        self._episode_index += 1

    def _reset_episode_state(self) -> None:
        """Clear all per-episode buffers (frames, stats, subtasks, image state).

        Called at the end of a successful :meth:`save_episode`. The episode
        counter (``_episode_index``) is intentionally NOT touched here so the
        caller advances it after a successful save.
        """
        self._episode_frames.clear()
        self._episode_stats = StatsComputer()
        self._frame_index = 0
        self._current_ep_subtasks = []
        for vkey in self._video_keys:
            self._image_last_frame[vkey] = None
            self._image_frame_counts[vkey] = 0

    def finalize(self) -> None:
        """Write all metadata files after all episodes are saved."""
        # Flush any unsaved episode
        if self._episode_frames:
            self.save_episode()

        # Close the data parquet writer before reading metadata off disk.
        self._close_data_writer()

        # Finalize every still-open per-camera encoder (in normal flow each
        # camera has one open unless a size rotation just closed it).
        for vkey in list(self._image_encoders):
            self._close_video_encoder(vkey)

        self._write_tasks_parquet()
        if self.has_subtasks and self._subtasks:
            self._write_subtasks_parquet()
        self._write_episodes_parquet()
        stats = self._stats.compute()
        self._write_stats_json(stats)
        self._write_info_json()
        self._write_conversion_log()

        logger.info(
            "Dataset finalized: %d episodes, %d total frames at %s",
            self._episode_index,
            self._global_index,
            self.output_dir,
        )

    # ------------------------------------------------------------------
    # Directory / path helpers
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        """Create the directory skeleton: data/, meta/episodes/, videos/<key>/."""
        (self.output_dir / "data").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
        for vkey in self._video_keys:
            (self.output_dir / "videos" / vkey).mkdir(parents=True, exist_ok=True)

    def _detect_video_keys(self) -> list[str]:
        """Return feature keys whose dtype is ``"video"`` (i.e. camera streams)."""
        return [k for k, v in self.features.items() if v.get("dtype") == "video"]

    @staticmethod
    def _chunk_file(base_dir: Path, chunk_idx: int, file_idx: int, ext: str) -> Path:
        """Return ``base/chunk-XXX/file-XXX.ext`` after ensuring the directory exists."""
        chunk_dir = base_dir / f"chunk-{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        return chunk_dir / f"file-{file_idx:03d}.{ext}"

    def _data_parquet_path(self, chunk_idx: int, file_idx: int) -> Path:
        return self._chunk_file(
            self.output_dir / "data", chunk_idx, file_idx, "parquet"
        )

    def _video_path(self, video_key: str, chunk_idx: int, file_idx: int) -> Path:
        return self._chunk_file(
            self.output_dir / "videos" / video_key,
            chunk_idx,
            file_idx,
            "mp4",
        )

    def _episodes_parquet_path(self, chunk_idx: int, file_idx: int) -> Path:
        return self._chunk_file(
            self.output_dir / "meta" / "episodes",
            chunk_idx,
            file_idx,
            "parquet",
        )

    # ------------------------------------------------------------------
    # Parquet writing
    # ------------------------------------------------------------------

    def _build_data_table(self, frames: list[dict[str, Any]]) -> pa.Table:
        """Build a pyarrow Table for one episode's worth of frames."""
        reserved = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
        if self.has_subtasks:
            reserved.add("subtask_index")

        columns: dict[str, list[Any]] = {
            "index": [],
            "timestamp": [],
            "frame_index": [],
            "episode_index": [],
            "task_index": [],
        }
        if self.has_subtasks:
            columns["subtask_index"] = []

        feature_keys = [
            k
            for k, v in self.features.items()
            if k not in reserved and v.get("dtype") != "video"
        ]
        for k in feature_keys:
            columns[k] = []

        for frame in frames:
            columns["index"].append(frame["index"])
            columns["timestamp"].append(frame["timestamp"])
            columns["frame_index"].append(frame["frame_index"])
            columns["episode_index"].append(frame["episode_index"])
            columns["task_index"].append(frame["task_index"])
            if self.has_subtasks:
                columns["subtask_index"].append(frame.get("subtask_index", -1))
            for k in feature_keys:
                val = frame.get(k)
                if val is None:
                    feat_spec = self.features[k]
                    if feat_spec.get("dtype") == "float32":
                        dim = feat_spec.get("shape", [1])[0]
                        columns[k].append([float("nan")] * dim)
                    else:
                        columns[k].append(None)
                elif isinstance(val, np.ndarray):
                    columns[k].append(val.tolist())
                else:
                    columns[k].append(val)

        pa_columns: dict[str, pa.Array] = {
            "index": pa.array(columns["index"], type=pa.int64()),
            "timestamp": pa.array(columns["timestamp"], type=pa.float32()),
            "frame_index": pa.array(columns["frame_index"], type=pa.int64()),
            "episode_index": pa.array(columns["episode_index"], type=pa.int64()),
            "task_index": pa.array(columns["task_index"], type=pa.int64()),
        }
        if self.has_subtasks:
            pa_columns["subtask_index"] = pa.array(
                columns["subtask_index"], type=pa.int64()
            )
        for k in feature_keys:
            feat_spec = self.features[k]
            dtype = feat_spec.get("dtype", "float32")
            if dtype == "float32":
                shape = feat_spec.get("shape", [1])
                dim = shape[0] if shape else 1
                pa_columns[k] = pa.array(
                    columns[k],
                    type=pa.list_(pa.float32(), dim),
                )
            elif dtype == "int64":
                pa_columns[k] = pa.array(columns[k], type=pa.int64())
            else:
                pa_columns[k] = pa.array(columns[k])

        return pa.table(pa_columns)

    def _append_data_parquet(self, path: Path, frames: list[dict[str, Any]]) -> None:
        """Append one episode's frames to the current data parquet file.

        Opens a long-lived ``pq.ParquetWriter`` on first use (or after a
        rotation) and writes the episode as a new row group via
        ``write_table``. The writer is closed either on rotation (size
        threshold crossed) or in ``finalize``.
        """
        table = self._build_data_table(frames)
        if self._data_pq_writer is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._data_pq_writer = pq.ParquetWriter(
                path,
                schema=table.schema,
                compression="snappy",
                use_dictionary=True,
            )
        self._data_pq_writer.write_table(table)

    def _close_data_writer(self) -> None:
        """Close the data ParquetWriter if one is open."""
        if self._data_pq_writer is not None:
            self._data_pq_writer.close()
            self._data_pq_writer = None

    def _write_tasks_parquet(self) -> None:
        """Write ``meta/tasks.parquet`` in LeRobot v3 layout.

        The task string is stored as the pandas Index *named* ``task`` (so it
        materializes as a ``task`` column with ``index_columns=["task"]`` in
        the parquet metadata); the single ``task_index`` column holds the
        int64 id. This matches lerobot-record output (e.g. the columns
        ``[task_index, task]``).
        """
        tasks_sorted = sorted(self._tasks.items(), key=lambda kv: kv[1])
        task_strs = [t for t, _ in tasks_sorted]
        task_indices = [i for _, i in tasks_sorted]

        # ``dtype=object`` on the index pins the physical arrow type to
        # ``string`` (not ``large_string``), matching the reference v3
        # datasets on the Hub regardless of the local pandas major version.
        df = pd.DataFrame(
            {"task_index": pd.array(task_indices, dtype="int64")},
            index=pd.Index(task_strs, dtype=object, name="task"),
        )
        path = self.output_dir / "meta" / "tasks.parquet"
        df.to_parquet(path, engine="pyarrow", compression="snappy")

    def _write_subtasks_parquet(self) -> None:
        """Write ``meta/subtasks.parquet`` mirroring the tasks.parquet layout.

        Only called when at least one episode supplied subtask spans. The
        subtask string is the (unnamed) pandas Index; ``subtask_index`` is
        the int64 id. Matches the schema documented in LeRobot's dataset
        subtask guide.
        """
        subtasks_sorted = sorted(self._subtasks.items(), key=lambda kv: kv[1])
        subtask_strs = [s for s, _ in subtasks_sorted]
        subtask_indices = [i for _, i in subtasks_sorted]

        df = pd.DataFrame(
            {"subtask_index": pd.array(subtask_indices, dtype="int64")},
            index=pd.Index(subtask_strs, dtype=object),
        )
        path = self.output_dir / "meta" / "subtasks.parquet"
        df.to_parquet(path, engine="pyarrow", compression="snappy")

    # Order of per-episode statistics emitted into the episodes parquet.
    _STAT_ORDER = (
        "min",
        "max",
        "mean",
        "std",
        "count",
        "q01",
        "q10",
        "q50",
        "q90",
        "q99",
    )

    def _format_episode_stats(
        self, ep_stats: dict[str, dict[str, list[float]]]
    ) -> dict[str, Any]:
        """Flatten a per-episode stats dict into ``stats/<feature>/<stat>`` columns.

        Numeric features yield per-dimension lists; image (video) features are
        nested to ``[C, 1, 1]``; ``count`` is always a single-element list.
        This mirrors lerobot-record's meta/episodes stats layout.
        """
        cols: dict[str, Any] = {}
        for key in self.features:
            stats = ep_stats.get(key)
            if stats is None:
                continue
            count = int(stats["count"][0]) if stats.get("count") else 0
            for stat in self._STAT_ORDER:
                col = f"stats/{key}/{stat}"
                if stat == "count":
                    cols[col] = [count]
                else:
                    cols[col] = self._shape_stat_value(key, stat, stats[stat])
        return cols

    def _shape_stat_value(self, key: str, stat: str, vals: list[float]) -> Any:
        """Shape a per-dimension stat list to the LeRobot v3.0 layout.

        Image (video) features are nested to ``[C, 1, 1]``; integer features
        keep integer min/max; everything else is a flat per-dimension
        ``list[float]``.  Shared by the per-episode stats columns and the
        global ``meta/stats.json``.
        """
        is_image = self.features.get(key, {}).get("dtype") == "video"
        is_int = self.features.get(key, {}).get("dtype") == "int64"
        if is_image:
            return [[[float(v)]] for v in vals]
        if is_int and stat in ("min", "max"):
            return [int(round(v)) for v in vals]
        return [float(v) for v in vals]

    def _stats_column_type(self, col: str) -> pa.DataType:
        """Return the pyarrow type for a ``stats/<feature>/<stat>`` column."""
        _, feature, stat = col.split("/", 2)
        dtype = self.features.get(feature, {}).get("dtype", "float32")
        if stat == "count":
            return pa.list_(pa.int64())
        if dtype == "video":
            return pa.list_(pa.list_(pa.list_(pa.float64())))
        if stat in ("min", "max") and dtype == "int64":
            return pa.list_(pa.int64())
        return pa.list_(pa.float64())

    def _write_episodes_parquet(self) -> None:
        """Write ``meta/episodes/chunk-XXX/file-XXX.parquet``.

        Episodes that share the same ``(data/chunk_index, data/file_index)``
        are grouped into one parquet file with one row per episode.
        """
        groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for ep_meta in self._episodes_meta:
            key = (ep_meta["data/chunk_index"], ep_meta["data/file_index"])
            groups[key].append(ep_meta)

        for (chunk_idx, file_idx), metas in groups.items():
            path = self._episodes_parquet_path(chunk_idx, file_idx)

            # Preserve key insertion order across the whole group
            all_keys: list[str] = []
            seen: set[str] = set()
            for m in metas:
                for k in m.keys():
                    if k not in seen:
                        seen.add(k)
                        all_keys.append(k)

            arrays: dict[str, pa.Array] = {}
            for k in all_keys:
                vals = [m.get(k) for m in metas]
                if k.startswith("stats/"):
                    arr = pa.array(vals, type=self._stats_column_type(k))
                elif k in ("tasks", "subtasks"):
                    arr = pa.array(vals, type=pa.list_(pa.string()))
                elif k in ("episode_index", "length"):
                    arr = pa.array(vals, type=pa.int64())
                elif "index" in k and "dataset" in k:
                    arr = pa.array(vals, type=pa.int64())
                elif "chunk_index" in k or "file_index" in k:
                    arr = pa.array(vals, type=pa.int64())
                elif "timestamp" in k:
                    arr = pa.array(vals, type=pa.float64())
                else:
                    arr = pa.array(vals)
                arrays[k] = arr

            # Self-referential pointers to this episodes parquet file.
            nrows = len(metas)
            arrays["meta/episodes/chunk_index"] = pa.array(
                [chunk_idx] * nrows, type=pa.int64()
            )
            arrays["meta/episodes/file_index"] = pa.array(
                [file_idx] * nrows, type=pa.int64()
            )

            table = pa.table(arrays)
            pq.write_table(table, path, compression="snappy")

    # ------------------------------------------------------------------
    # Video encoding & concatenation
    # ------------------------------------------------------------------

    def _feed_video_frame(
        self,
        vkey: str,
        value: Any,
        feat_spec: dict[str, Any],
    ) -> None:
        """Feed a single video frame for ``vkey`` into the streaming encoder.

        Handles:

        - Lazy encoder startup on the first real image of an output file
          (shape inferred from it). When the first frame is ``None`` but we
          have a declared shape, we still start the encoder using the
          declared shape and pad with black.
        - RGB uint8 numpy arrays from the decoder fast path (no PIL round
          trip) as well as PIL Images / other arrays.
        - Zero-order-hold padding from the last image when a frame is missing.
        - Black-frame padding when no image has been seen yet in the episode.
        - Feeding statistics only for real (non-padded) frames.
        """
        if value is not None:
            if (
                isinstance(value, np.ndarray)
                and value.dtype == np.uint8
                and value.ndim == 3
                and value.shape[2] == 3
            ):
                # RGB uint8 array from the decoder fast path.
                height, width = value.shape[:2]
                stats_arr: np.ndarray = value
                frame_bytes = value.tobytes()
            else:
                img: Image.Image = (
                    value if isinstance(value, Image.Image) else Image.fromarray(value)
                )
                width, height = img.size
                rgb = img if img.mode == "RGB" else img.convert("RGB")
                frame_bytes = np.asarray(rgb, dtype=np.uint8).tobytes()
                stats_arr = np.asarray(img)
            self._ensure_encoder(vkey, width, height)
            self._feed_stats(vkey, stats_arr)
        else:
            # Padding path. Prefer the last real frame bytes; otherwise
            # produce black at the declared shape and start the encoder with
            # that shape so the parquet/video frame counts stay in lockstep.
            last = self._image_last_frame.get(vkey)
            if last is not None:
                frame_bytes = last
            else:
                shape = feat_spec.get("shape", [480, 640, 3])
                h, w = int(shape[0]), int(shape[1])
                c = int(shape[2]) if len(shape) >= 3 else 3
                black = np.zeros((h, w, c), dtype=np.uint8)
                frame_bytes = black.tobytes()
                self._ensure_encoder(vkey, w, h)

        self._image_last_frame[vkey] = frame_bytes
        self._image_feed_queues[vkey].put(frame_bytes)
        self._image_frame_counts[vkey] += 1

    def _ensure_encoder(self, vkey: str, width: int, height: int) -> None:
        """Open the streaming encoder for ``vkey`` if none is active.

        When an encoder is already open but the incoming episode's frame size
        differs (checked at episode start only), the current output file is
        finalized and a fresh file is started at the new size — mixing sizes
        within one rawvideo stream would corrupt the mp4.
        """
        if vkey in self._image_encoders:
            if (
                self._image_shapes[vkey] != (width, height)
                and self._image_frame_counts[vkey] == 0
            ):
                self._close_video_encoder(vkey)
                self._advance_video_file(vkey)
            else:
                return
        target = self._video_path(
            vkey,
            self._video_chunk_idx[vkey],
            self._video_file_idx[vkey],
        )
        self._video_target_paths[vkey] = target
        self._open_encoder(vkey, width, height, target)

    def _open_encoder(
        self,
        vkey: str,
        width: int,
        height: int,
        target_path: Path,
    ) -> None:
        """Start an ffmpeg subprocess + feeder thread writing ``target_path``.

        The codec arguments go through :func:`_build_codec_args`, matching
        the historical per-episode encoder; ``+faststart`` keeps the moov
        atom at the front like the previous concatenated output files.

        Two background threads serve the process: the feeder pushes frames
        into its stdin, and the drain thread consumes its stderr. The latter
        is not optional — see the comment on ``_stderr_reader`` below.
        """
        target_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            # Keep stderr quiet: the banner and the per-frame progress lines
            # are pure noise here, and every byte of them has to be drained.
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostats",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
            *_build_codec_args(self.video_codec, self._ffmpeg_preset, self._ffmpeg_crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(target_path),
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg to encode videos."
            ) from exc

        tail = _StderrTail()

        def _stderr_reader() -> None:
            # ffmpeg keeps writing to stderr for as long as it runs. With
            # stderr=PIPE and nobody reading it, the OS pipe buffer fills and
            # ffmpeg blocks inside its own write() — permanently. That looks
            # exactly like a hang: 0% CPU, no progress, easily mistaken for
            # an OOM kill. So drain unconditionally, keeping only the tail
            # for diagnostics. Exits on EOF, i.e. when ffmpeg exits.
            stream = proc.stderr
            if stream is None:
                return
            try:
                while True:
                    chunk = stream.read1(_STDERR_READ_CHUNK)
                    if not chunk:
                        break
                    tail.append(chunk)
            except (OSError, ValueError):
                # Pipe torn down under us (e.g. the process was killed);
                # whatever was already captured is still worth reporting.
                pass
            finally:
                try:
                    stream.close()
                except Exception:  # a failed close must not mask the exit status
                    logger.debug(
                        "closing ffmpeg stderr for %s failed", vkey, exc_info=True
                    )

        r = threading.Thread(
            target=_stderr_reader, name=f"ffmpeg-stderr-{vkey}", daemon=True
        )
        r.start()

        q: queue.Queue[bytes | None] = queue.Queue(maxsize=_IMAGE_FEED_QUEUE_MAXSIZE)

        def _feeder() -> None:
            # On a write failure the loop keeps consuming (and task_done-ing)
            # items so queue.join()/put() in the main thread never deadlock;
            # the recorded error surfaces via _drain_video_queue / close.
            failed = False
            while True:
                item = q.get()
                if item is None:
                    q.task_done()
                    break
                if not failed:
                    try:
                        assert proc.stdin is not None
                        proc.stdin.write(item)
                    except BaseException as exc:  # noqa: BLE001
                        self._image_feeder_errors[vkey] = exc
                        failed = True
                q.task_done()
            try:
                if proc.stdin is not None and not proc.stdin.closed:
                    proc.stdin.close()
            except BrokenPipeError:
                pass

        t = threading.Thread(target=_feeder, name=f"ffmpeg-feeder-{vkey}", daemon=True)
        t.start()

        self._image_encoders[vkey] = proc
        self._image_feeders[vkey] = t
        self._image_feed_queues[vkey] = q
        self._image_stderr_readers[vkey] = r
        self._image_stderr_tails[vkey] = tail
        self._image_shapes[vkey] = (width, height)

    def _encoder_stderr_text(self, vkey: str) -> str:
        """Return the retained tail of ``vkey``'s ffmpeg stderr as text.

        Callers must have reaped the process first: EOF is what ends the
        drain thread, so joining it before that would just burn the timeout
        and report a truncated tail.
        """
        reader = self._image_stderr_readers.get(vkey)
        if reader is not None:
            reader.join(timeout=_STDERR_DRAIN_TIMEOUT)
        tail = self._image_stderr_tails.get(vkey)
        return tail.text() if tail is not None else ""

    def _drain_video_queue(self, vkey: str) -> None:
        """Block until every queued frame for ``vkey`` reached ffmpeg's stdin.

        Called at episode boundaries so (a) encoder failures surface with the
        same per-episode granularity as the previous per-episode encoders and
        (b) the on-disk size check in :meth:`_register_episode_video` sees an
        up-to-date file.
        """
        q = self._image_feed_queues.get(vkey)
        if q is not None:
            q.join()
        err = self._image_feeder_errors.get(vkey)
        if err is not None:
            self._raise_encoder_failure(vkey, err)

    def _discard_encoder(self, vkey: str) -> None:
        """Tear ``vkey``'s encoder down without finalizing its output file.

        The failure paths raise past :meth:`_close_video_encoder`, so nothing
        else would ever reap the process: an encoder that is still alive would
        hold its stdin pipe (and its feeder thread would stay parked on the
        queue) until the interpreter exits. The output mp4 is unusable at this
        point, so there is nothing to salvage — just stop the process, release
        the threads, and drop the per-encoder state.

        Deliberately *not* a resumable teardown: the (chunk, file) pointer is
        left where it was and ``_video_file_duration`` keeps the dead file's
        accumulated duration, so a future caller that caught the failure and
        kept going would reopen onto the same target path (``-y`` overwrites
        the partial mp4) with timestamps continuing from a file that no longer
        exists — and this episode's rows are already in the data parquet by
        then. Advance the file slot and reconcile the parquet before making
        any such salvage flow possible.
        """
        proc = self._image_encoders.pop(vkey, None)
        if proc is None:
            return
        if proc.poll() is None:
            # Killing closes ffmpeg's end of the pipe, so a feeder parked in
            # write() comes back with EPIPE instead of blocking forever.
            proc.kill()
        try:
            proc.wait(timeout=_ENCODER_TEARDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:  # pragma: no cover - kill() ignored
            logger.warning("ffmpeg for %s did not exit after kill()", vkey)
        q = self._image_feed_queues.pop(vkey, None)
        if q is not None:
            # The feeder is in drain mode after a failure, so the sentinel is
            # consumed promptly rather than sitting behind a blocked write.
            q.put(None)
        feeder = self._image_feeders.pop(vkey, None)
        if feeder is not None:
            feeder.join(timeout=_ENCODER_TEARDOWN_TIMEOUT)
        reader = self._image_stderr_readers.pop(vkey, None)
        if reader is not None:
            reader.join(timeout=_STDERR_DRAIN_TIMEOUT)
        self._image_stderr_tails.pop(vkey, None)
        self._image_feeder_errors.pop(vkey, None)
        self._image_shapes.pop(vkey, None)
        self._video_target_paths.pop(vkey, None)

    def _raise_encoder_failure(self, vkey: str, err: BaseException) -> None:
        """Raise a RuntimeError describing a failed encoder for ``vkey``.

        Both branches tear the encoder down first — this raise unwinds past
        :meth:`_close_video_encoder`, so it is the last chance to reap.
        """
        proc = self._image_encoders.get(vkey)
        if isinstance(err, BrokenPipeError) and proc is not None:
            # The feeder hit a closed pipe: ffmpeg itself died. Surface its
            # exit code and stderr instead of the secondary pipe error.
            try:
                ret = proc.wait(timeout=_ENCODER_WAIT_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                ret = proc.wait()
            stderr_text = self._encoder_stderr_text(vkey)
            logger.error(
                "ffmpeg failed for %s (returncode=%d): %s",
                vkey,
                ret,
                stderr_text,
            )
            self._discard_encoder(vkey)
            raise RuntimeError(
                f"ffmpeg exited with code {ret} for {vkey}: {stderr_text}"
            ) from err
        # Any other feeder failure (an OSError mid-write, say) leaves ffmpeg
        # alive and waiting on a pipe nobody will write to again.
        self._discard_encoder(vkey)
        raise RuntimeError(f"ffmpeg feeder for {vkey} failed: {err}") from err

    def _register_episode_video(self, vkey: str, ep_len: int) -> dict[str, Any]:
        """Record this episode's segment inside the current output mp4.

        Returns the episode's ``(chunk_index, file_index, from_ts, to_ts)``
        and advances the cumulative file duration. Afterwards, if the output
        file has grown past ``_VIDEO_FILES_SIZE_IN_MB``, the encoder is
        closed and the (chunk, file) pointer rotates so the next episode
        starts a fresh file (lerobot-record rotates after exceeding the
        threshold too).
        """
        # Frame-index driven duration: exact in rational arithmetic, so the
        # round() below only trims the float representation error that would
        # otherwise accumulate across episodes.
        ep_duration = float(ep_len) / self.fps
        from_ts = round(self._video_file_duration[vkey], _TIMESTAMP_ROUND_DECIMALS)
        to_ts = round(from_ts + ep_duration, _TIMESTAMP_ROUND_DECIMALS)
        # Carry the rounded value forward so the *next* ep sees a clean
        # boundary rather than re-introducing the rounding error.
        self._video_file_duration[vkey] = to_ts

        meta = {
            "chunk_index": self._video_chunk_idx[vkey],
            "file_index": self._video_file_idx[vkey],
            "from_timestamp": from_ts,
            "to_timestamp": to_ts,
        }

        target = self._video_target_paths.get(vkey)
        if target is not None:
            try:
                size = target.stat().st_size
            except FileNotFoundError:
                size = 0
            if size >= _VIDEO_FILES_SIZE_IN_MB * 1024 * 1024:
                self._close_video_encoder(vkey)
                self._advance_video_file(vkey)

        return meta

    def _advance_video_file(self, vkey: str) -> None:
        """Move ``vkey`` to the next (chunk, file) slot and reset its duration."""
        self._video_chunk_idx[vkey], self._video_file_idx[vkey] = _advance_chunk_file(
            self._video_chunk_idx[vkey],
            self._video_file_idx[vkey],
        )
        self._video_file_duration[vkey] = 0.0

    def _close_video_encoder(self, vkey: str) -> None:
        """Finalize the current output mp4 for ``vkey``.

        Queues the feeder sentinel, waits for ffmpeg to drain and exit,
        checks the exit status and feeder errors, normalises the file
        permissions to rw-rw-rw- (so other users / containers can read the
        produced mp4), and clears the per-encoder state so the next frame
        opens a fresh file.
        """
        proc = self._image_encoders.pop(vkey, None)
        if proc is None:
            return
        q = self._image_feed_queues.pop(vkey)
        q.put(None)
        t = self._image_feeders.pop(vkey)
        t.join(timeout=_ENCODER_WAIT_TIMEOUT)

        try:
            ret = proc.wait(timeout=_ENCODER_WAIT_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            ret = proc.wait()

        # The process is reaped, so its stderr is at EOF and the drain thread
        # is about to finish; collect the tail before dropping its state. The
        # drain thread owns the pipe and closes it on the way out.
        stderr_text = self._encoder_stderr_text(vkey)
        self._image_stderr_readers.pop(vkey, None)
        self._image_stderr_tails.pop(vkey, None)

        if ret != 0:
            logger.error(
                "ffmpeg failed for %s (returncode=%d): %s",
                vkey,
                ret,
                stderr_text,
            )
            raise RuntimeError(
                f"ffmpeg exited with code {ret} for {vkey}: {stderr_text}"
            )
        err = self._image_feeder_errors.pop(vkey, None)
        if err is not None and not isinstance(err, BrokenPipeError):
            raise RuntimeError(f"ffmpeg feeder for {vkey} failed: {err}") from err

        self._image_shapes.pop(vkey, None)
        target = self._video_target_paths.pop(vkey, None)
        if target is not None and target.exists():
            os.chmod(target, 0o666)

    # ------------------------------------------------------------------
    # Metadata JSON
    # ------------------------------------------------------------------

    def _write_stats_json(self, stats: dict[str, dict[str, list[float]]]) -> None:
        path = self.output_dir / "meta" / "stats.json"
        shaped = {key: self._shape_feature_stats(key, st) for key, st in stats.items()}
        with open(path, "w") as f:
            json.dump(shaped, f, indent=2)

    def _shape_feature_stats(
        self, key: str, stats: dict[str, list[float]]
    ) -> dict[str, Any]:
        """Shape one feature's stats dict to the LeRobot v3.0 ``stats.json`` layout.

        ``count`` becomes a single-element list; image (video) stats are nested
        to ``[C, 1, 1]`` per channel (see :meth:`_shape_stat_value`).
        """
        out: dict[str, Any] = {}
        count = int(stats["count"][0]) if stats.get("count") else 0
        for stat in self._STAT_ORDER:
            if stat not in stats:
                continue
            if stat == "count":
                out[stat] = [count]
            else:
                out[stat] = self._shape_stat_value(key, stat, stats[stat])
        return out

    def _write_info_json(self) -> None:
        """Write ``meta/info.json`` with dataset-level metadata and feature schema."""
        total_episodes = self._episode_index
        total_frames = self._global_index
        total_tasks = len(self._tasks)

        info: dict[str, Any] = {
            "codebase_version": _CODEBASE_VERSION,
            "robot_type": self.config.get("robot_type", "unknown"),
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": total_tasks,
            "chunks_size": _CHUNKS_SIZE,
            "data_files_size_in_mb": _DATA_FILES_SIZE_IN_MB,
            "video_files_size_in_mb": _VIDEO_FILES_SIZE_IN_MB,
            "fps": self.fps,
            "splits": compute_splits(total_episodes, self._splits),
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": self.features,
        }

        if self.has_subtasks and self._subtasks:
            info["total_subtasks"] = len(self._subtasks)

        if self.repo_id:
            info["repo_id"] = self.repo_id

        path = self.output_dir / "meta" / "info.json"
        with open(path, "w") as f:
            json.dump(info, f, indent=2)

    def _write_conversion_log(self) -> None:
        """Write ``meta/conversion_log.json`` (conversion manifest, plan.md D-2).

        Merges the writer-owned encode/output facts (codec/preset/crf, episode
        and frame totals, fps, per-episode lengths) with the caller-supplied
        provenance in ``self._manifest_extra`` (inputs, config snapshot,
        versions, run timestamp). When ``manifest_extra`` is ``None`` only the
        writer-owned subset is written. The output is additive/optional and is
        not part of the structural validation contract.
        """
        log: dict[str, Any] = {
            "codec": self.video_codec,
            "codec_label": _codec_label(self.video_codec),
            "ffmpeg_preset": self._ffmpeg_preset,
            "ffmpeg_crf": self._ffmpeg_crf,
            "fps": self.fps,
            "total_episodes": self._episode_index,
            "total_frames": self._global_index,
            "episode_lengths": [m["length"] for m in self._episodes_meta],
        }
        if self._manifest_extra:
            log.update(self._manifest_extra)

        path = self.output_dir / "meta" / "conversion_log.json"
        with open(path, "w") as f:
            json.dump(log, f, indent=2)


# ---------------------------------------------------------------------------
# Module-level bridge function (called by CLI)
# ---------------------------------------------------------------------------


# Default ``-crf`` / ``-cq`` per encoder, mirroring ``_build_codec_args``.
def _default_crf(codec: str) -> int:
    if codec == "libsvtav1":
        return 30
    if codec in ("h264_nvenc", "hevc_nvenc", "av1_nvenc"):
        return 25
    return 23  # libx264 and generic fallback


def _build_features(
    config: RobotConfig,
    video_codec: str = "libx264",
    has_subtasks: bool = False,
    ffmpeg_preset: str | None = None,
    ffmpeg_crf: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the LeRobot v3.0 ``features`` dict from a RobotConfig.

    Each observation/action becomes either a ``"video"`` feature (for images)
    or a ``"float32"`` feature (for numeric data). The bookkeeping columns
    (timestamp, frame_index, episode_index, index, task_index) are appended
    *after* the real features and carry ``names=None`` -- matching
    lerobot-record's ``info.json`` ordering and schema. When
    ``has_subtasks=True`` an additional ``subtask_index`` column is declared.

    Numeric feature names come from the config's ``names:`` list when given,
    otherwise per-dimension names (``<key>_0`` ...) are filled in once the
    shape is known (see :func:`write_dataset`).

    Args:
        config: Validated RobotConfig instance.
        video_codec: ffmpeg encoder name used to encode videos. Affects the
            ``video.codec`` label and the video ``info`` block in info.json.
        ffmpeg_preset: Effective ffmpeg ``-preset`` (``None`` = codec default).
        ffmpeg_crf: Effective quality (``None`` = codec default).

    Returns:
        Feature specification dict suitable for ``DatasetWriter.features``.
    """
    codec_label = _codec_label(video_codec)
    crf_value = ffmpeg_crf if ffmpeg_crf is not None else _default_crf(video_codec)

    features: dict[str, dict[str, Any]] = {}

    # Real features first (mirrors lerobot-record ordering).
    for fm in config.observations + config.actions:
        if fm.is_image:
            h, w = (fm.image_size[0], fm.image_size[1]) if fm.image_size else (480, 640)
            c = fm.image_size[2] if (fm.image_size and len(fm.image_size) == 3) else 3
            # compressedDepth トピックは深度マップとして記録する（4a: 8bit動画に正規化）。
            is_depth = fm.topic.lower().endswith("compresseddepth")
            features[fm.key] = {
                "dtype": "video",
                "shape": [h, w, c],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": h,
                    "video.width": w,
                    "video.codec": codec_label,
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": is_depth,
                    "video.fps": config.fps,
                    "video.channels": c,
                    "has_audio": False,
                    "video.g": 2,
                    "video.crf": crf_value,
                    "video.preset": ffmpeg_preset,
                    "video.fast_decode": 0,
                    "video.video_backend": "pyav",
                    "video.extra_options": {},
                },
            }
        else:
            # Numeric feature. Shape is inferred from the first frame later;
            # per-dimension names are finalized once the shape is known.
            features[fm.key] = {
                "dtype": "float32",
                "shape": [1],
                "names": list(fm.names) if fm.names else None,
            }

    # Bookkeeping columns last, with names=None (matching lerobot-record).
    for bkey, bdtype in (
        ("timestamp", "float32"),
        ("frame_index", "int64"),
        ("episode_index", "int64"),
        ("index", "int64"),
        ("task_index", "int64"),
    ):
        features[bkey] = {"dtype": bdtype, "shape": [1], "names": None}

    if has_subtasks:
        features["subtask_index"] = {"dtype": "int64", "shape": [1], "names": None}

    return features


def write_dataset(
    episodes: Iterable[list[dict]],
    config: RobotConfig,
    output_dir: Path | str,
    video_codec: str = "libx264",
    repo_id: str | None = None,
    ffmpeg_preset: str | None = None,
    ffmpeg_crf: int | None = None,
    has_subtasks: bool = False,
    manifest_extra: dict[str, Any] | None = None,
) -> None:
    """Convert a stream of episode frame-lists into a LeRobot v3.0 dataset.

    This is the top-level entry point called by the CLI. ``episodes`` is
    consumed lazily so callers may pass a generator that produces one
    episode at a time without materializing the full dataset in memory.

    Shape inference, optional-feature filtering, and sub-key merging are
    performed on the **first** episode. Subsequent episodes must use the
    same schema (same keys, same per-key array shapes). This matches the
    prior behavior where all downstream structure was determined once,
    while avoiding the O(total_frames) memory footprint of the old
    multi-pass scan over ``list[list[dict]]``.

    Args:
        episodes: Iterable of episodes, each a list of frame dicts.
        config: RobotConfig instance.
        output_dir: Output directory path.
        video_codec: ffmpeg encoder name for video encoding.
        repo_id: Optional HuggingFace repo ID.
        ffmpeg_preset: Explicit ffmpeg ``-preset`` override (``None`` = codec default).
        ffmpeg_crf: Explicit quality override (``None`` = codec default).
        manifest_extra: Optional provenance fields forwarded to the writer for
            ``meta/conversion_log.json`` (inputs, config snapshot, versions,
            run timestamp). ``None`` writes only the writer-owned subset.
    """
    output_dir = Path(output_dir)

    # --- First-pass: buffer episode-0 for shape inference ---
    episodes_iter = iter(episodes)
    try:
        first_ep = next(episodes_iter)
    except StopIteration:
        logger.warning("No episodes to write; dataset will be empty.")
        return

    features = _build_features(
        config,
        video_codec=video_codec,
        has_subtasks=has_subtasks,
        ffmpeg_preset=ffmpeg_preset,
        ffmpeg_crf=ffmpeg_crf,
    )

    # Infer shapes from the first episode only. Later episodes are assumed
    # to use the same schema; a mismatched shape would surface as a parquet
    # type error in _append_data_parquet, which is acceptable since the
    # input pipeline guarantees one RobotConfig per conversion run.
    pending_keys = {
        k
        for k, v in features.items()
        if v.get("dtype") == "float32" and k not in ("timestamp",)
    }
    for frame in first_ep:
        if not pending_keys:
            break
        for key in list(pending_keys):
            val = frame.get(key)
            if val is None:
                continue
            if isinstance(val, np.ndarray):
                features[key]["shape"] = list(val.shape)
            elif hasattr(val, "__len__"):
                features[key]["shape"] = [len(val)]
            pending_keys.discard(key)

    # Filter optional features using the first episode's keys. Features not
    # present in ep-0 are treated as absent for the whole dataset; this
    # preserves the legacy "drop optional features that never have data"
    # behavior while avoiding a full pre-scan over every episode. The
    # trade-off is intentional: spot-sparse optional features that first
    # appear in ep-N>0 will be dropped. In practice the CLI's per-episode
    # decode produces a consistent key set, so this is safe.
    keys_with_data: set[str] = set()
    for frame in first_ep:
        for k, v in frame.items():
            if v is not None:
                keys_with_data.add(k)
    reserved_meta = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    if has_subtasks:
        reserved_meta.add("subtask_index")
    features = {
        k: v for k, v in features.items() if k in keys_with_data or k in reserved_meta
    }

    # Merge sub-keyed state/action into the canonical LeRobot keys.
    #
    # LeRobot policies expect exactly "observation.state" (robot_state_feature) and
    # "action" (action_feature). Sub-keyed configs like "observation.state.right_arm" /
    # "action.left_arm" must be concatenated into single vectors before writing.
    #
    # observation.state.<part>  →  "observation.state"  (concatenated, FeatureType.STATE)
    # action.<part>             →  "action"             (concatenated, FeatureType.ACTION)
    #
    # Preserve config declaration order: `features` was populated in config
    # order at _build_features, and both the merged "names" list and the
    # per-frame concatenation below must use the same order to stay aligned.
    state_subkeys = [k for k in features if k.startswith("observation.state.")]
    action_subkeys = [k for k in features if k.startswith("action.") and k != "action"]

    # Save shapes for zero-filling absent optional sub-keys during frame merging
    subkey_shapes: dict[str, int] = {}

    if state_subkeys:
        merged_dim = sum(features[k]["shape"][0] for k in state_subkeys)
        merged_names: list[str] = []
        for k in state_subkeys:
            part = k.rsplit(".", 1)[-1]
            dim = features[k]["shape"][0]
            merged_names.extend(f"{part}_{i}" for i in range(dim))
            subkey_shapes[k] = dim
        features["observation.state"] = {
            "dtype": "float32",
            "shape": [merged_dim],
            "names": merged_names,
        }
        for k in state_subkeys:
            del features[k]

    if action_subkeys:
        merged_dim = sum(features[k]["shape"][0] for k in action_subkeys)
        merged_names = []
        for k in action_subkeys:
            part = k.rsplit(".", 1)[-1]
            dim = features[k]["shape"][0]
            merged_names.extend(f"{part}_{i}" for i in range(dim))
            subkey_shapes[k] = dim
        features["action"] = {
            "dtype": "float32",
            "shape": [merged_dim],
            "names": merged_names,
        }
        for k in action_subkeys:
            del features[k]

    # Finalize per-dimension names for numeric features whose names were not
    # supplied via config (or do not match the inferred shape). Bookkeeping
    # columns keep names=None. This matches lerobot-record, where
    # len(names) == shape[0] for every numeric feature.
    _bookkeeping = {
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "subtask_index",
    }
    for k, v in features.items():
        if k in _bookkeeping or v.get("dtype") != "float32":
            continue
        dim = v["shape"][0] if v.get("shape") else 1
        names = v.get("names")
        if not names or len(names) != dim:
            v["names"] = [f"{k}_{i}" for i in range(dim)]

    # Keep bookkeeping columns last in info.json. Sub-keyed action/state are
    # merged and re-inserted above, which can push them after the bookkeeping
    # keys; restore the "real features first, bookkeeping last" order.
    _bk_order = [
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "subtask_index",
    ]
    reordered = {k: v for k, v in features.items() if k not in _bookkeeping}
    for bk in _bk_order:
        if bk in features:
            reordered[bk] = features[bk]
    features = reordered

    # Build a plain dict for the writer's config parameter
    writer_config: dict[str, Any] = {
        "robot_type": config.robot_type,
        "task": config.task,
    }

    writer = DatasetWriter(
        output_dir=output_dir,
        config=writer_config,
        features=features,
        fps=config.fps,
        repo_id=repo_id,
        video_codec=video_codec,
        ffmpeg_preset=ffmpeg_preset,
        ffmpeg_crf=ffmpeg_crf,
        has_subtasks=has_subtasks,
        manifest_extra=manifest_extra,
        splits=config.split.ratios,
    )

    # --- Stream first_ep + remaining episodes into the writer ---
    for episode_frames in itertools.chain([first_ep], episodes_iter):
        for frame in episode_frames:
            if "task" not in frame:
                frame["task"] = config.task

            # Concatenate sub-state arrays → "observation.state"
            if state_subkeys:
                parts = []
                for k in state_subkeys:
                    val = frame.pop(k, None)
                    if val is None:
                        val = np.zeros(subkey_shapes[k], dtype=np.float32)
                    parts.append(np.asarray(val, dtype=np.float32).ravel())
                frame["observation.state"] = np.concatenate(parts)

            # Concatenate sub-action arrays → "action"
            if action_subkeys:
                parts = []
                for k in action_subkeys:
                    val = frame.pop(k, None)
                    if val is None:
                        val = np.zeros(subkey_shapes[k], dtype=np.float32)
                    parts.append(np.asarray(val, dtype=np.float32).ravel())
                frame["action"] = np.concatenate(parts)

            writer.add_frame(frame)
        writer.save_episode()

    writer.finalize()
