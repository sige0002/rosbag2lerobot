"""``convert`` command: convert ROS2 rosbags to a LeRobot v3.0 dataset."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import click

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from rosbag2lerobot import __version__ as _ROSBAG2LEROBOT_VERSION
from rosbag2lerobot.config import (
    load_config,
    RobotConfig,
    FeatureMapping,
)
from rosbag2lerobot.decoders import decode, decode_array
from rosbag2lerobot.jobmeta import EpisodeResult, JobSummary, dir_bytes
from rosbag2lerobot.manifest import (
    ManifestInput,
    ffmpeg_version,
    load_manifest_extra,
    sha256_of_path,
    strip_builtin_keys,
)
from rosbag2lerobot.progress import (
    PROGRESS_FILENAME,
    ProgressReporter,
    bag_message_count,
)
from rosbag2lerobot.reader import BagReader, discover_bags, extract_header_stamp_ns
from rosbag2lerobot.resampler import Resampler, trim_to_valid_range
from rosbag2lerobot.task_spec import SubtaskSpan, resolve_task
from rosbag2lerobot.timestamps import NS_PER_MS, StampSkewError, format_skew_error
from rosbag2lerobot.transforms import TransformLookup, quat_xyzw_to_euler
from rosbag2lerobot.cli._common import logger, _detect_nvenc, _make_progress


@click.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to robot_config.yaml.",
)
@click.option(
    "--bags",
    "bags_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to bag directory or parent directory containing multiple bags.",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(),
    help="Output directory for the LeRobot dataset.",
)
@click.option("--task", default=None, help="Override task name from config.")
@click.option("--fps", default=None, type=int, help="Override FPS from config.")
@click.option("--max-episodes", default=None, type=int, help="Max episodes to convert.")
@click.option("--workers", default=1, type=int, help="Number of parallel workers.")
@click.option(
    "--video-codec",
    default="auto",
    help="Video codec (default: auto - uses h264_nvenc if NVENC available, else libx264).",
)
@click.option(
    "--gpu/--no-gpu",
    default=None,
    help="Force GPU (NVENC) on/off. Default: auto-detect.",
)
@click.option(
    "--ffmpeg-preset",
    default=None,
    help="Override ffmpeg preset (e.g. veryfast, p4, 8).",
)
@click.option(
    "--ffmpeg-crf",
    default=None,
    type=int,
    help="Override quality (CRF for CPU codecs, -cq for NVENC).",
)
@click.option(
    "--dry-run", is_flag=True, help="Validate config and bags without writing."
)
@click.option("--repo-id", default=None, help="HuggingFace repo ID for the dataset.")
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help=(
        "Safely re-run into an existing --output: no-op if the dataset is "
        "already complete, clean-restart if a previous run crashed before "
        "finalizing. (Skipping already-converted episodes is a P1 feature.)"
    ),
)
@click.option(
    "--json",
    "json_summary",
    is_flag=True,
    default=False,
    help="Emit the job summary as JSON to stdout (suppresses human Done. logs).",
)
@click.option(
    "--quiet",
    is_flag=True,
    default=False,
    help="Suppress the progress bar and INFO chatter.",
)
@click.option(
    "--skip-failed",
    is_flag=True,
    default=False,
    help=(
        "Record per-episode failures and continue (dataset finalizes from "
        "good episodes). Default: a worker exception aborts the run."
    ),
)
@click.option(
    "--manifest-extra",
    "manifest_extra_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "JSON file whose object is merged into meta/conversion_log.json "
        "(e.g. your own job/ticket ids). Keys that collide with the fields "
        "rosbag2lerobot writes itself are ignored: built-ins win."
    ),
)
def convert(
    config_path: str,
    bags_path: str,
    output_path: str,
    task: Optional[str],
    fps: Optional[int],
    max_episodes: Optional[int],
    workers: int,
    video_codec: str,
    gpu: Optional[bool],
    ffmpeg_preset: Optional[str],
    ffmpeg_crf: Optional[int],
    dry_run: bool,
    repo_id: Optional[str],
    resume: bool,
    json_summary: bool,
    quiet: bool,
    skip_failed: bool,
    manifest_extra_path: Optional[str],
) -> None:
    """Convert ROS2 rosbags to a LeRobot v3.0 dataset.

    Each bag directory is treated as one episode. The pipeline loads the
    YAML config, discovers bags, reads and decodes messages, resamples to
    the target FPS, and writes parquet + video + metadata files.
    """
    # 1. Load config
    cfg = load_config(config_path)

    # Caller-supplied manifest fields are validated up-front: a typo in the
    # JSON should fail before any bag is read, not after an hour of encoding.
    user_manifest_extra: dict[str, Any] = {}
    if manifest_extra_path is not None:
        try:
            user_manifest_extra = load_manifest_extra(manifest_extra_path)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        user_manifest_extra, dropped = strip_builtin_keys(user_manifest_extra)
        if dropped:
            logger.warning(
                "--manifest-extra: ignoring %d key(s) reserved by rosbag2lerobot: %s",
                len(dropped),
                ", ".join(dropped),
            )

    # --quiet / --json suppress rosbag2lerobot's INFO chatter so the progress bar (or
    # the emitted JSON summary) is the only output. Errors still surface.
    if quiet or json_summary:
        logging.getLogger("rosbag2lerobot").setLevel(logging.WARNING)

    # Apply CLI overrides
    if task is not None:
        cfg.task = task  # type: ignore[misc]
    if fps is not None:
        cfg.fps = fps  # type: ignore[misc]
    if repo_id is not None:
        cfg.repo_id = repo_id  # type: ignore[misc]

    # Resolve video codec (handles --video-codec auto, --gpu/--no-gpu).
    effective_codec = video_codec
    use_gpu = gpu  # --gpu/--no-gpu, None=auto

    if effective_codec == "auto":
        nvenc_available = _detect_nvenc()
        if use_gpu is False:
            effective_codec = "libx264"
        elif use_gpu is True:
            if not nvenc_available:
                raise click.UsageError(
                    "--gpu specified but NVENC encoders not found in ffmpeg"
                )
            effective_codec = "h264_nvenc"
        else:  # auto
            effective_codec = "h264_nvenc" if nvenc_available else "libx264"
    elif use_gpu is not None:
        # Explicit codec + --gpu/--no-gpu - validate consistency.
        is_nvenc = effective_codec.endswith("_nvenc")
        if use_gpu and not is_nvenc:
            logger.warning(
                "--gpu specified but codec %s is not NVENC",
                effective_codec,
            )
        if not use_gpu and is_nvenc:
            raise click.UsageError(f"--no-gpu conflicts with codec {effective_codec}")

    logger.info(
        "Selected codec: %s (preset=%s, crf=%s)",
        effective_codec,
        ffmpeg_preset,
        ffmpeg_crf,
    )

    logger.info("Robot type : %s", cfg.robot_type)
    logger.info("FPS        : %d", cfg.fps)
    logger.info("Task       : %s", cfg.task)
    logger.info("Policy     : %s", cfg.resampling.default_policy)

    # 2. Discover bags
    bag_paths = discover_bags(bags_path)
    if max_episodes is not None:
        bag_paths = bag_paths[:max_episodes]
    logger.info("Found %d bag(s) to process", len(bag_paths))

    if dry_run:
        _dry_run_report(cfg, bag_paths)
        return

    # Output-dir guard / safe re-run. True work-skipping resume is a P1 item;
    # here we only prevent clobbering an existing dataset and recover cleanly
    # from a crashed (non-finalized) output.
    output_dir = Path(output_path)
    if not _prepare_output_dir(output_dir, resume):
        return

    # 3. Pre-resolve task.json per bag. Doing this up-front fails fast on
    # schema errors and lets the writer decide once whether to materialize
    # subtask-related outputs (subtask_index column, subtasks.parquet).
    bag_specs = _resolve_bag_specs(bag_paths, cfg.task)
    has_subtasks = any(subtasks for _, subtasks in bag_specs)
    logger.info(
        "task.json pre-scan: %d bag(s), has_subtasks=%s",
        len(bag_specs),
        has_subtasks,
    )

    # 4. Build resampler
    resampler = Resampler(
        fps=cfg.fps,
        policy=cfg.resampling.default_policy,
        tolerance_ms=cfg.resampling.tolerance_ms,
    )

    # --- Job summary + manifest accumulators (⑥ + ⑧) -------------------
    # The progress bar advances and per-episode results accrue via a single
    # callback fired by the episode iterators. ``manifest_inputs`` is a live
    # list reference handed to the writer; the writer reads it back when it
    # writes meta/conversion_log.json inside finalize(), by which point the
    # generator has been fully drained and the list is complete + index-sorted.
    job_summary = JobSummary()
    # Hash the input bags on background threads so the (potentially large)
    # sequential read overlaps with the conversion itself instead of delaying
    # its start. Each future is resolved in _on_episode_done, by which point
    # the hash has usually finished; hashing errors surface there (previously
    # they surfaced before conversion started).
    sha_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bag-sha256")
    bag_sha256_futures = {bp: sha_pool.submit(sha256_of_path, bp) for bp in bag_paths}
    manifest_inputs: list[dict[str, Any]] = []
    progress = _make_progress(len(bag_paths), disable=quiet or json_summary)
    # Without a bar (piped stdout, --quiet, --json) the run would otherwise be
    # silent between episodes, so plain progress lines take its place. They go
    # through the logger, which --quiet / --json have already turned down to
    # WARNING — those two ask for silence, and meta/progress.json still carries
    # the same numbers for anything watching the run.
    heartbeat = ProgressReporter(
        output_dir / "meta" / PROGRESS_FILENAME,
        len(bag_paths),
        log_fn=logger.info if progress is None else None,
    )

    # Wall clock is started here (before the callback closes over it) so each
    # incremental checkpoint can record the elapsed time so far. The
    # authoritative final write below recomputes it identically.
    wall_start = time.monotonic()
    summary_path = output_dir / "meta" / "job_summary.json"

    def _checkpoint_summary() -> None:
        """Persist a partial job_summary.json after each episode.

        Lets an external watcher poll real ``done/total`` progress
        (n_success + n_failed vs the up-front bag count) while a conversion is
        still running. The final, post-finalize write in :func:`convert`
        overwrites this with the byte-equivalent authoritative summary, so this
        checkpoint never affects the completed dataset.
        """
        # input/output byte sizes are still 0 here (filled at finalize); the BE
        # only needs n_success / n_failed for the done/total ratio.
        partial = job_summary.to_dict(wall_time_s=time.monotonic() - wall_start)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace so a concurrent poller never reads a half-written file.
        tmp = summary_path.with_suffix(".json.partial")
        with open(tmp, "w") as fh:
            json.dump(partial, fh, indent=2)
        tmp.replace(summary_path)

    def _on_episode_done(result: EpisodeResult) -> None:
        job_summary.add(result)
        if result.success:
            sha_future = bag_sha256_futures.get(Path(result.bag_path))
            manifest_inputs.append(
                ManifestInput(
                    path=result.bag_path,
                    sha256=sha_future.result() if sha_future is not None else "",
                    frame_count=result.n_frames,
                    processing_time_s=result.processing_time_s,
                ).to_dict()
            )
            manifest_inputs.sort(key=lambda d: d["path"])
        _checkpoint_summary()
        if progress is not None:
            progress.update(1)

    config_snapshot = Path(config_path).read_text()
    config_sha256 = hashlib.sha256(config_snapshot.encode("utf-8")).hexdigest()
    # Caller-supplied fields go in first so the built-ins below always win
    # (colliding keys were already dropped by strip_builtin_keys).
    manifest_extra: dict[str, Any] = {
        **user_manifest_extra,
        "inputs": manifest_inputs,
        "config_snapshot": config_snapshot,
        "config_sha256": config_sha256,
        "rosbag2lerobot_version": _ROSBAG2LEROBOT_VERSION,
        "ffmpeg_version": ffmpeg_version(),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # 5. Process each bag (= 1 episode) — generator-based pipeline so the
    # writer sees episodes one at a time instead of the old "materialize all
    # then write" path that blew up memory on large datasets (T11).
    output_dir.mkdir(parents=True, exist_ok=True)

    if workers and workers > 1 and len(bag_paths) > 1:
        episodes_iter: Iterator[list[dict]] = _iter_episodes_parallel(
            bag_paths,
            cfg,
            resampler,
            workers,
            bag_specs,
            on_episode_done=_on_episode_done,
            skip_failed=skip_failed,
            progress=heartbeat,
        )
    else:
        episodes_iter = _iter_episodes_serial(
            bag_paths,
            cfg,
            resampler,
            bag_specs,
            on_episode_done=_on_episode_done,
            skip_failed=skip_failed,
            progress=heartbeat,
        )

    # 6. Write dataset
    try:
        from rosbag2lerobot.writer import write_dataset

        write_dataset(
            episodes=episodes_iter,
            config=cfg,
            output_dir=output_dir,
            video_codec=effective_codec,
            repo_id=cfg.repo_id,
            ffmpeg_preset=ffmpeg_preset,
            ffmpeg_crf=ffmpeg_crf,
            has_subtasks=has_subtasks,
            manifest_extra=manifest_extra,
        )
    except StampSkewError as exc:
        # Without --skip-failed a bad clock aborts the run. Re-raise as a
        # ClickException so the operator gets the actionable message instead of
        # a traceback; the partial output stays on disk for inspection and can
        # be cleaned up by re-running with --resume.
        raise click.ClickException(str(exc)) from exc
    except ImportError:
        # Fallback: materialize the generator so the summary writer can
        # report frame totals. This path should only fire in test/dev
        # environments where the writer module is intentionally absent.
        all_episodes = list(episodes_iter)
        logger.warning(
            "Writer module not available. Processed %d episodes with %d total frames.",
            len(all_episodes),
            sum(len(ep) for ep in all_episodes),
        )
        _store_episode_summary(all_episodes, output_dir)
    finally:
        # Skipped/failed episodes never resolve their hash future; drop those
        # instead of blocking on them.
        sha_pool.shutdown(wait=False, cancel_futures=True)
        if progress is not None:
            progress.close()

    # 7. Job summary (⑧). Compute output size, finalize wall time, persist to
    # meta/job_summary.json, and emit JSON to stdout when --json is set.
    wall_time_s = time.monotonic() - wall_start
    job_summary.input_bytes = sum(dir_bytes(bp) for bp in bag_paths)
    job_summary.output_bytes = dir_bytes(output_dir)
    summary_dict = job_summary.to_dict(wall_time_s=wall_time_s)

    # Authoritative final write (byte-equivalent to the pre-checkpoint
    # behavior). ``summary_path`` was set up-front for the incremental
    # checkpoints written by ``_on_episode_done``; this overwrites the last one.
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump(summary_dict, fh, indent=2)

    # The heartbeat is transient run state: the run is over, and job_summary
    # (plus meta/info.json) is now the record of it. A progress.json left
    # behind therefore means the run died, and says where.
    heartbeat.remove()

    if json_summary:
        click.echo(json.dumps(summary_dict, indent=2))
        return

    logger.info("Done.")


def _split_selector(selector: str) -> list[str] | None:
    """Split a comma-separated selector string, or return None if empty."""
    return selector.split(",") if selector else None


def _build_decoder_config(fm: FeatureMapping) -> dict[str, Any]:
    """Collect per-feature options that the decoder needs at call time."""
    cfg: dict[str, Any] = {}
    if fm.image_size is not None:
        cfg["image_size"] = fm.image_size
    if fm.unit_conversion != 1.0:
        cfg["unit_conversion"] = fm.unit_conversion
    return cfg


class _LazyImage:
    """Deferred image decode for the resampling pipeline.

    Image topics typically run faster than the target FPS, so most camera
    messages are never picked by the resampler. Wrapping the raw message in
    this thin holder lets :func:`_process_episode` postpone the expensive
    JPEG/raw decode until after resample + trim, when only the frames that
    actually survive are materialized. The same instance may be shared by
    several frames (hold policy); the decode runs once and the result is
    shared, exactly like the previous eager path.
    """

    __slots__ = ("_msg", "_msg_type", "_selector", "_config", "_value")

    def __init__(
        self,
        msg_type: str,
        msg: object,
        selector: list[str] | None,
        config: dict[str, Any],
    ) -> None:
        self._msg_type = msg_type
        self._msg = msg
        self._selector = selector
        self._config = config
        self._value = None

    def materialize(self) -> object:
        """Decode the wrapped message (once) and return the image value."""
        if self._msg is not None:
            self._value = decode_array(
                msg_type=self._msg_type,
                deserialized_msg=self._msg,
                selector=self._selector,
                config=self._config,
            )
            self._msg = None  # release the raw message buffer
        return self._value


def _process_bag_entry(
    args: tuple[int, Path, RobotConfig, dict[str, Any], str, list[SubtaskSpan]],
) -> tuple[int, list[dict], str, int, float]:
    """Worker-side entry point for :func:`_iter_episodes_parallel`.

    Rebuilds a local ``Resampler`` from the serialized config dict (resampler
    instances themselves can be pickled but we serialize the config to stay
    resilient to dataclass changes). Task and subtasks are already resolved
    by the caller (pre-scan in :func:`convert`) and passed in directly.

    Returns the episode index, decoded frames, resolved task, the worker
    process id (mapped to a stable ordinal by the parent), and the wall time
    spent decoding/resampling this bag.
    """
    import os
    import time

    ep_idx, bag_path, cfg, resampler_kwargs, resolved_task, subtasks = args
    resampler = Resampler(**resampler_kwargs)
    started = time.monotonic()
    frames = _process_episode(bag_path, cfg, resampler)
    elapsed = time.monotonic() - started
    _tag_episode(frames, resolved_task, subtasks)
    return ep_idx, frames, resolved_task, os.getpid(), elapsed


def _resolve_bag_specs(
    bag_paths: list[Path],
    fallback_task: str,
) -> list[tuple[str, list[SubtaskSpan]]]:
    """Resolve ``(task, subtasks)`` for every bag up-front.

    Parses each ``<bag>/task.json`` eagerly so malformed files abort the run
    before any conversion work begins. The returned list is aligned with
    ``bag_paths`` by index.
    """
    import json as _json  # local import keeps the module-level import surface small

    specs: list[tuple[str, list[SubtaskSpan]]] = []
    for bag in bag_paths:
        try:
            resolved, subtasks = resolve_task(bag, fallback_task)
        except (_json.JSONDecodeError, ValueError) as exc:
            raise click.UsageError(f"Invalid task.json in {bag}: {exc}") from exc
        specs.append((resolved, subtasks))
    return specs


def _tag_episode(
    frames: list[dict],
    task: str,
    subtasks: list[SubtaskSpan],
) -> None:
    """Stamp each frame with ``task`` and embed ``subtasks`` on the first frame.

    The writer pops ``_episode_subtasks`` from the first frame it sees for an
    episode and uses it to validate coverage and compute per-frame
    ``subtask_index``. Downstream code should not rely on ``_episode_subtasks``
    remaining on the frame after ``add_frame``.
    """
    for frame in frames:
        frame["task"] = task
    if frames:
        frames[0]["_episode_subtasks"] = subtasks


def _iter_episodes_serial(
    bag_paths: list[Path],
    cfg: RobotConfig,
    resampler: Resampler,
    bag_specs: list[tuple[str, list[SubtaskSpan]]],
    on_episode_done: Optional[Callable[[EpisodeResult], None]] = None,
    skip_failed: bool = False,
    progress: Optional[ProgressReporter] = None,
) -> Iterator[list[dict]]:
    """Yield processed episodes one-at-a-time in ``bag_paths`` order.

    Each bag is decoded, resampled, and tagged with its pre-resolved task
    and subtask spans before being yielded. This is the streaming
    counterpart to the old ``all_episodes = []`` accumulator: the writer
    only holds one episode in memory at a time, which is critical for
    large-scale conversions where total frame memory would otherwise
    exceed tens of GB.

    ``on_episode_done`` (if given) is invoked with an :class:`EpisodeResult`
    after each bag completes (success or, when ``skip_failed`` is set,
    failure). With ``skip_failed`` enabled a per-bag exception is recorded and
    that episode is skipped so the dataset still finalizes from good episodes;
    with it disabled (default) the exception propagates and aborts the run,
    preserving the legacy behavior.

    ``progress`` (if given) receives per-message updates for the whole
    episode, so ``meta/progress.json`` advances continuously on this path.

    Episodes shorter than ``cfg.split.min_length`` are filtered here at the
    producer: they are never yielded to the writer and never reported via
    ``on_episode_done``, so dataset totals (stats.json / job_summary /
    conversion_log) only ever reflect the episodes actually written.
    """
    import time

    min_length = cfg.split.min_length
    read_topics = _read_topics(cfg) if progress is not None else []
    for ep_idx, bag_path in enumerate(bag_paths):
        resolved_task, subtasks = bag_specs[ep_idx]
        logger.info("Episode %d: %s", ep_idx, bag_path)
        if progress is not None:
            progress.start_episode(ep_idx, bag_message_count(bag_path, read_topics))
        started = time.monotonic()
        try:
            frames = _process_episode(bag_path, cfg, resampler, progress=progress)
        except Exception as exc:  # noqa: BLE001 - guarded by skip_failed
            if not skip_failed:
                raise
            elapsed = time.monotonic() - started
            logger.warning("  Episode %d failed: %s", ep_idx, exc)
            if on_episode_done is not None:
                on_episode_done(
                    EpisodeResult(
                        index=ep_idx,
                        bag_path=str(bag_path),
                        worker=0,
                        success=False,
                        n_frames=0,
                        processing_time_s=elapsed,
                        error=str(exc),
                    )
                )
            continue
        elapsed = time.monotonic() - started
        if progress is not None:
            progress.finish_episode()
        _tag_episode(frames, resolved_task, subtasks)
        # Episode-length filter (⑨). Drop short episodes before they reach the
        # writer or the job/manifest accounting so they are absent everywhere.
        if min_length > 0 and len(frames) < min_length:
            logger.info(
                "  Episode %d dropped: %d frames < min_length %d",
                ep_idx,
                len(frames),
                min_length,
            )
            continue
        logger.info(
            "  -> %d frames (%.1f s) [task=%r, subtasks=%d]",
            len(frames),
            len(frames) / cfg.fps if frames else 0,
            resolved_task,
            len(subtasks),
        )
        if on_episode_done is not None:
            on_episode_done(
                EpisodeResult(
                    index=ep_idx,
                    bag_path=str(bag_path),
                    worker=0,
                    success=True,
                    n_frames=len(frames),
                    processing_time_s=elapsed,
                    error=None,
                )
            )
        yield frames


def _iter_episodes_parallel(
    bag_paths: list[Path],
    cfg: RobotConfig,
    resampler: Resampler,
    workers: int,
    bag_specs: list[tuple[str, list[SubtaskSpan]]],
    on_episode_done: Optional[Callable[[EpisodeResult], None]] = None,
    skip_failed: bool = False,
    progress: Optional[ProgressReporter] = None,
) -> Iterator[list[dict]]:
    """Yield processed episodes in ``bag_paths`` order using a process pool.

    Workers receive jobs and return results via
    :func:`concurrent.futures.as_completed`, so completion order is
    non-deterministic. We buffer out-of-order results keyed by their
    original bag index and drain any contiguous prefix starting at
    ``next_idx`` each iteration. This preserves the same deterministic
    output ordering as the serial path while allowing bags to decode
    concurrently.

    Memory footprint is bounded by the size of the "gap" between the
    slowest in-flight worker and the fastest-completing follow-up bag;
    in the worst case that equals ``min(workers, len(bag_paths))`` eps,
    which is still far smaller than materializing the full dataset.

    ``on_episode_done`` (if given) is invoked with an :class:`EpisodeResult`
    as each future completes. The worker process id returned by
    :func:`_process_bag_entry` is mapped to a stable 0-based ordinal so the
    summary reports a small, deterministic worker set. With ``skip_failed`` a
    worker exception is recorded as a failed result and the episode is skipped;
    without it the exception propagates and aborts the run (legacy behavior).

    Episodes shorter than ``cfg.split.min_length`` are filtered here at the
    producer (mirroring the serial path): they are never yielded to the writer
    and never reported via ``on_episode_done``, so the dropped index is simply
    hopped over in the contiguous-prefix drain.

    ``progress`` (if given) is updated once per completed episode rather than
    per message: the decoding happens in worker processes that cannot share
    this reporter, and several episodes are in flight at once.
    """
    resampler_kwargs: dict[str, Any] = {
        "fps": resampler.fps,
        "policy": resampler.policy,
        "tolerance_ms": resampler.tolerance_ms,
    }

    jobs = [
        (idx, bag, cfg, resampler_kwargs, bag_specs[idx][0], bag_specs[idx][1])
        for idx, bag in enumerate(bag_paths)
    ]

    effective_workers = min(workers, len(bag_paths))
    logger.info(
        "Processing %d bag(s) with %d worker(s)",
        len(bag_paths),
        effective_workers,
    )

    min_length = cfg.split.min_length
    read_topics = _read_topics(cfg) if progress is not None else []
    pending: dict[int, list[dict]] = {}
    # Indices that will never enter ``pending`` (worker failure or a producer
    # min_length drop); the drain hops over them to keep the prefix contiguous.
    skipped_idx: set[int] = set()
    next_idx = 0
    # Map worker pid -> stable 0-based ordinal in first-seen order.
    pid_ordinals: dict[int, int] = {}

    def _ordinal(pid: int) -> int:
        return pid_ordinals.setdefault(pid, len(pid_ordinals))

    def _drain() -> Iterator[list[dict]]:
        """Yield the contiguous ready prefix, hopping over skipped indices."""
        nonlocal next_idx
        while next_idx in pending or next_idx in skipped_idx:
            if next_idx in skipped_idx:
                next_idx += 1
                continue
            yield pending.pop(next_idx)
            next_idx += 1

    with ProcessPoolExecutor(max_workers=effective_workers) as pool:
        futures = {pool.submit(_process_bag_entry, job): job[0] for job in jobs}
        for fut in as_completed(futures):
            job_idx = futures[fut]
            try:
                ep_idx, frames, resolved, pid, elapsed = fut.result()
            except Exception as exc:  # noqa: BLE001 - guarded by skip_failed
                if not skip_failed:
                    raise
                logger.warning("  Episode %d failed: %s", job_idx, exc)
                skipped_idx.add(job_idx)
                if on_episode_done is not None:
                    on_episode_done(
                        EpisodeResult(
                            index=job_idx,
                            bag_path=str(bag_paths[job_idx]),
                            worker=0,
                            success=False,
                            n_frames=0,
                            processing_time_s=0.0,
                            error=str(exc),
                        )
                    )
                yield from _drain()
                continue
            # Episode-length filter (⑨). Drop short episodes before the writer
            # or job/manifest accounting see them; hop over the index in drain.
            if min_length > 0 and len(frames) < min_length:
                logger.info(
                    "Episode %d dropped: %d frames < min_length %d",
                    ep_idx,
                    len(frames),
                    min_length,
                )
                skipped_idx.add(ep_idx)
                yield from _drain()
                continue
            if progress is not None:
                progress.episode_completed(
                    ep_idx,
                    bag_message_count(bag_paths[ep_idx], read_topics),
                )
            logger.info(
                "Episode %d: %s -> %d frames (%.1f s) [task=%r]",
                ep_idx,
                bag_paths[ep_idx],
                len(frames),
                len(frames) / cfg.fps if frames else 0,
                resolved,
            )
            if on_episode_done is not None:
                on_episode_done(
                    EpisodeResult(
                        index=ep_idx,
                        bag_path=str(bag_paths[ep_idx]),
                        worker=_ordinal(pid),
                        success=True,
                        n_frames=len(frames),
                        processing_time_s=elapsed,
                        error=None,
                    )
                )
            pending[ep_idx] = frames

            # Drain the contiguous prefix starting at next_idx so that
            # episodes flow to the writer in original bag order as soon
            # as they are ready (rather than waiting for the slowest).
            yield from _drain()

    # Safety net: flush any trailing buffered episodes. In normal flow
    # the drain inside the loop already handles everything, but if the
    # executor shutdown races with a late completion we still want the
    # remaining episodes to reach the writer.
    yield from _drain()


def _required_window(
    messages: list[tuple[str, int, object]],
    required_keys: list[str],
) -> Optional[tuple[int, int]]:
    """Compute the intersection time window of the required features.

    Aggregates the min (first) and max (last) adopted timestamp for every
    key in *required_keys* in a single O(N) pass over *messages*. The window
    is ``[max(per-key min), min(per-key max)]`` — the span where every
    required feature has at least one message on both sides.

    Args:
        messages: ``(feature_key, ts_ns, value)`` tuples (tagged with the
            adopted timestamp). Need not be sorted.
        required_keys: Non-optional feature keys that must overlap.

    Returns:
        ``(win_start_ns, win_end_ns)`` on success, or ``None`` when the
        episode should be treated as empty (a required key has no messages,
        or the required spans do not overlap). A warning is logged in both
        empty cases.
    """
    if not required_keys:
        # Nothing to align to; fall back to the message span.
        if not messages:
            return None
        return messages[0][1], messages[-1][1]

    mins: dict[str, int] = {}
    maxs: dict[str, int] = {}
    required = set(required_keys)
    for key, ts_ns, _value in messages:
        if key not in required:
            continue
        cur_min = mins.get(key)
        if cur_min is None or ts_ns < cur_min:
            mins[key] = ts_ns
        cur_max = maxs.get(key)
        if cur_max is None or ts_ns > cur_max:
            maxs[key] = ts_ns

    missing = [k for k in required_keys if k not in mins]
    if missing:
        logger.warning(
            "  align_to_required: empty episode; required feature(s) had no "
            "messages: %s",
            missing,
        )
        return None

    win_start = max(mins.values())
    win_end = min(maxs.values())
    if win_end < win_start:
        logger.warning(
            "  align_to_required: empty episode; required features do not "
            "overlap in time (win_start=%d > win_end=%d ns)",
            win_start,
            win_end,
        )
        return None

    return win_start, win_end


def _tf_feature_value(fm: FeatureMapping, pose7: object) -> object:
    """Map a 7-vector ``[tx,ty,tz,qx,qy,qz,qw]`` to this TF feature's output.

    The default contract returns the 7-vector unchanged. When ``fm.selector``
    names a euler convention (``orientation.euler_xyz`` / ``euler_zyx`` or the
    bare ``euler_xyz`` form) the quaternion is replaced by euler angles
    (radians), yielding the 6-vector ``[tx,ty,tz,roll,pitch,yaw]``.

    Args:
        fm: The TF feature mapping (``is_tf_feature`` is True).
        pose7: The ``np.ndarray`` returned by :meth:`TransformLookup.lookup`.

    Returns:
        The feature value (7-vector by default, 6-vector for euler selectors).
    """
    import numpy as np

    pose7 = np.asarray(pose7, dtype=np.float32)
    sel = fm.selector
    last = sel.rsplit(".", 1)[-1] if sel else ""
    if not last.startswith("euler_"):
        return pose7
    convention = last[len("euler_") :]
    roll, pitch, yaw = quat_xyzw_to_euler(
        float(pose7[3]),
        float(pose7[4]),
        float(pose7[5]),
        float(pose7[6]),
        convention=convention,
    )
    return np.array([pose7[0], pose7[1], pose7[2], roll, pitch, yaw], dtype=np.float32)


def _read_topics(cfg: RobotConfig) -> list[str]:
    """Return the topics read from a bag for one episode.

    The configured feature topics plus the TF sources that any TF feature
    samples from. Shared with the progress heartbeat so its message total
    covers exactly the topics the reader will iterate.
    """
    topics = list(cfg.all_topics)
    for fm in cfg.observations + cfg.actions:
        if not fm.is_tf_feature:
            continue
        for topic in (fm.tf_topic, fm.tf_static_topic):
            if topic not in topics:
                topics.append(topic)
    return topics


def _process_episode(
    bag_path: Path,
    cfg: RobotConfig,
    resampler: Resampler,
    progress: Optional[ProgressReporter] = None,
) -> list[dict]:
    """Read one rosbag and produce resampled fixed-fps frames.

    Args:
        bag_path: Path to a single bag directory.
        cfg: Validated robot configuration.
        resampler: Configured Resampler instance.
        progress: Optional heartbeat reporter, already positioned on this
            episode by the caller; advanced once per message read.

    Returns:
        List of frame dicts, each containing all feature keys plus
        ``frame_index`` and ``timestamp``.

    Raises:
        StampSkewError: If a message's header stamp diverges from its bag
            receive time by more than
            ``timestamps.max_header_receive_skew_ms`` (see
            :mod:`rosbag2lerobot.timestamps`).
    """
    with BagReader(bag_path, cfg) as reader:
        topic_to_fms = cfg.topic_to_features
        global_delay = cfg.resampling.max_stamp_delay_ms
        # Timestamp integrity guard: the largest header/receive divergence we
        # will convert rather than fail on. None disables the check.
        skew_limit_ms = cfg.timestamps.max_header_receive_skew_ms
        skew_limit_ns = None if skew_limit_ms is None else skew_limit_ms * NS_PER_MS

        # TF features (frame_from/frame_to set) are sampled off the output frame
        # grid from /tf + /tf_static rather than a single topic. When present we
        # also read the TF source topics in this same pass and accumulate a
        # TransformLookup. With no TF features, behaviour/perf is unchanged.
        tf_features = [fm for fm in cfg.observations + cfg.actions if fm.is_tf_feature]
        tf_lookup: Optional[TransformLookup] = None
        tf_dynamic_topics: set[str] = set()
        tf_static_topics: set[str] = set()
        read_topics = _read_topics(cfg)
        if tf_features:
            tf_lookup = TransformLookup()
            for fm in tf_features:
                tf_dynamic_topics.add(fm.tf_topic)
                tf_static_topics.add(fm.tf_static_topic)

        # Collect and decode messages referenced by the config. The adopted
        # timestamp per feature follows ``stamp_source`` (header vs. bag
        # receive time); stale latched messages are dropped *before* decode
        # (decode is the expensive step) when their header lags the receive
        # time beyond the effective ``max_stamp_delay_ms`` threshold.
        messages: list[tuple[str, int, object]] = []
        stale_dropped = 0
        for topic, recv_ns, raw_msg in reader.iter_messages(topics=read_topics):
            if progress is not None:
                progress.advance()
            if tf_lookup is not None:
                if topic in tf_static_topics:
                    tf_lookup.add_static(raw_msg)
                if topic in tf_dynamic_topics:
                    tf_lookup.add_dynamic(raw_msg)
            header_ns = extract_header_stamp_ns(raw_msg)
            skew_ns = None if header_ns is None else abs(recv_ns - header_ns)
            for fm in topic_to_fms.get(topic, []):
                # TF features are not decoded per-message; they are sampled off
                # the output frame grid from the accumulated TransformLookup below.
                if fm.is_tf_feature:
                    continue
                # (B) Per-message stale drop. Effective threshold is the
                # per-feature override, else the global default.
                thr = (
                    fm.max_stamp_delay_ms
                    if fm.max_stamp_delay_ms is not None
                    else global_delay
                )
                if thr is not None and skew_ns is not None and skew_ns > thr * 1e6:
                    stale_dropped += 1
                    continue

                # (B') Timestamp integrity guard. Only messages that survived
                # the configured stale drop reach here, and only header-stamped
                # features can be corrupted by a divergent header stamp — for
                # stamp_source: receive the header is never adopted.
                if (
                    skew_limit_ns is not None
                    and skew_ns is not None
                    and skew_ns > skew_limit_ns
                    and fm.stamp_source == "header"
                ):
                    raise StampSkewError(
                        format_skew_error(
                            bag_path=bag_path,
                            topic=topic,
                            feature_key=fm.key,
                            header_ns=header_ns,  # type: ignore[arg-type]
                            receive_ns=recv_ns,
                            threshold_ms=skew_limit_ms,  # type: ignore[arg-type]
                        )
                    )

                # (A) Adopted timestamp: header when requested and present,
                # otherwise the bag receive time.
                ts = (
                    header_ns
                    if (fm.stamp_source == "header" and header_ns is not None)
                    else recv_ns
                )

                # Image features are decoded lazily: most camera messages are
                # never picked by the resampler (topic rate > target fps), so
                # the expensive decode is deferred until after resample+trim
                # (see the materialization pass below).
                if fm.is_image:
                    messages.append(
                        (
                            fm.key,
                            ts,
                            _LazyImage(
                                fm.msg_type,
                                raw_msg,
                                _split_selector(fm.selector),
                                _build_decoder_config(fm),
                            ),
                        )
                    )
                    continue

                decoded_value = decode(
                    msg_type=fm.msg_type,
                    deserialized_msg=raw_msg,
                    selector=_split_selector(fm.selector),
                    config=_build_decoder_config(fm),
                )
                messages.append((fm.key, ts, decoded_value))

        if stale_dropped:
            logger.info(
                "  dropped %d stale message(s) (header lag > max_stamp_delay_ms)",
                stale_dropped,
            )

        # Sort by adopted timestamp (header times can reorder vs. receive order)
        messages.sort(key=lambda m: m[1])

        feature_keys = cfg.observation_keys + cfg.action_keys

        # (C) Determine the resample window. When aligning to the required
        # features, clip to the intersection of every required feature's
        # [first, last] adopted-timestamp span so the grid only covers the
        # range where all required features actually have data. Otherwise use
        # the bag's full time range (legacy behaviour).
        #
        # TF feature keys are excluded from the required-window: they have no
        # per-message timestamps (samples are generated on the grid below to
        # cover the whole window), so they must not constrain it.
        if cfg.resampling.align_to_required:
            tf_keys = {fm.key for fm in tf_features}
            required_for_window = [
                k for k in cfg.required_feature_keys if k not in tf_keys
            ]
            window = _required_window(messages, required_for_window)
            if window is None:
                return []
            start_ns, end_ns = window
            logger.debug("  align_to_required window: [%d, %d] ns", start_ns, end_ns)
        else:
            start_ns, end_ns = reader.get_time_range()

        # (C') Sample TF features on the output frame grid. The grid matches the
        # resampler's own arange (frame_period_ns = int(1e9/fps), n_frames =
        # ceil(duration_s*fps)) so each generated sample lands exactly on a frame
        # time. Appended before resample so the resampler treats them opaquely.
        if tf_features and tf_lookup is not None:
            import math as _math

            frame_period_ns = int(1e9 / cfg.fps)
            duration_s = (end_ns - start_ns) / 1e9
            n_frames = max(1, int(_math.ceil(duration_s * cfg.fps)))
            for fm in tf_features:
                for i in range(n_frames):
                    t = start_ns + i * frame_period_ns
                    pose7 = tf_lookup.lookup(fm.frame_to, fm.frame_from, t)
                    messages.append((fm.key, t, _tf_feature_value(fm, pose7)))
            messages.sort(key=lambda m: m[1])

        frames = resampler.resample(
            messages=messages,
            feature_keys=feature_keys,
            start_ns=start_ns,
            end_ns=end_ns,
        )

    # LeRobot v3.0 rejects frames with missing required features. Trim the
    # episode to the range where every non-optional feature has data so the
    # resulting parquet has no nulls for declared features. Optional
    # features may still be sparse within the retained range; the writer
    # fills those for schema compatibility.
    if cfg.resampling.trim_to_valid and frames:
        required_keys = cfg.required_feature_keys
        raw_len = len(frames)
        frames = trim_to_valid_range(frames, required_keys, cfg.fps)
        dropped = raw_len - len(frames)
        if dropped > 0:
            logger.info(
                "  trim_to_valid: dropped %d frames (%.2f s) with missing required features",
                dropped,
                dropped / cfg.fps,
            )
        if not frames:
            logger.warning(
                "  trim_to_valid produced an empty episode; one or more "
                "required features had no valid frames. Required keys: %s",
                required_keys,
            )

    # Materialize lazily-decoded images for the surviving frames only. A
    # shared _LazyImage (hold policy) decodes once; frames dropped by the
    # resample window / trim are never decoded at all.
    image_keys = [fm.key for fm in cfg.observations + cfg.actions if fm.is_image]
    if image_keys and frames:
        for frame in frames:
            for k in image_keys:
                v = frame.get(k)
                if type(v) is _LazyImage:
                    frame[k] = v.materialize()

    return frames


def _prepare_output_dir(output_dir: Path, resume: bool) -> bool:
    """Guard the output directory against accidental overwrite / corruption.

    True work-skipping resume (reusing already-converted episodes) is a P1
    feature; this only provides safe re-run semantics for P0:

    - Non-existent or empty directory: proceed normally.
    - Non-empty without ``--resume``: abort with a ``UsageError`` so an
      existing dataset is never silently mixed into / corrupted.
    - ``--resume`` on a finalized dataset (``meta/info.json`` present):
      no-op — the conversion already completed.
    - ``--resume`` on a partial / crashed output (no ``meta/info.json``):
      wipe the partial artifacts and reconvert from scratch, guaranteeing a
      correct dataset without manual cleanup.

    Args:
        output_dir: Target dataset directory.
        resume: Value of the ``--resume`` flag.

    Returns:
        ``True`` if conversion should proceed, ``False`` if the dataset is
        already complete and nothing needs to be done.

    Raises:
        click.UsageError: If the directory is non-empty and ``--resume`` was
            not given.
    """
    if not output_dir.exists() or not any(output_dir.iterdir()):
        return True

    info_json = output_dir / "meta" / "info.json"
    if not resume:
        raise click.UsageError(
            f"Output directory {output_dir} is not empty. Pass --resume to "
            "re-run into it, or choose a fresh --output."
        )

    if info_json.exists():
        click.secho(
            f"Dataset at {output_dir} is already complete; nothing to do "
            "(use a fresh --output to reconvert).",
            fg="green",
        )
        return False

    click.secho(
        f"--resume: previous run at {output_dir} did not finalize "
        "(no meta/info.json). Cleaning partial output and reconverting from "
        "scratch. [skipping already-converted episodes is a P1 feature]",
        fg="yellow",
    )
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    return True


def _dry_run_report(cfg: RobotConfig, bag_paths: list[Path]) -> None:
    """Print a summary of what would be converted."""
    click.echo(f"\n{'=' * 60}")
    click.echo("DRY RUN – no data will be written")
    click.echo(f"{'=' * 60}\n")
    click.echo(f"Robot type : {cfg.robot_type}")
    click.echo(f"FPS        : {cfg.fps}")
    click.echo(f"Task       : {cfg.task}")
    click.echo(f"Policy     : {cfg.resampling.default_policy}")
    click.echo(f"Tolerance  : {cfg.resampling.tolerance_ms} ms")
    click.echo(f"\nObservations ({len(cfg.observations)}):")
    for fm in cfg.observations:
        click.echo(f"  {fm.key:40s} <- {fm.topic} [{fm.msg_type}]")
    click.echo(f"\nActions ({len(cfg.actions)}):")
    for fm in cfg.actions:
        click.echo(f"  {fm.key:40s} <- {fm.topic} [{fm.msg_type}]")
    click.echo(f"\nBags to process ({len(bag_paths)}):")
    for bp in bag_paths:
        click.echo(f"  {bp}")

    # Try to open each bag and report topic info
    for bp in bag_paths:
        try:
            with BagReader(bp, cfg) as reader:
                start_ns, end_ns = reader.get_time_range()
                duration_s = (end_ns - start_ns) / 1e9
                click.echo(f"\n  {bp.name}: {duration_s:.1f}s")
                for topic, info in reader.get_topics_info().items():
                    click.echo(
                        f"    {topic:50s} {info.msg_type:40s} ({info.count} msgs)"
                    )
        except Exception as exc:
            click.echo(f"\n  {bp.name}: ERROR – {exc}")

    click.echo("")


def _store_episode_summary(
    all_episodes: list[list[dict]],
    output_dir: Path,
) -> None:
    """Write a minimal JSON summary when the full writer module is unavailable.

    Args:
        all_episodes: List of episode frame lists.
        output_dir: Directory where ``conversion_summary.json`` will be written.
    """
    import json

    summary = {
        "num_episodes": len(all_episodes),
        "frames_per_episode": [len(ep) for ep in all_episodes],
        "total_frames": sum(len(ep) for ep in all_episodes),
    }
    summary_path = output_dir / "conversion_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Summary written to %s", summary_path)
