"""CLI entry point for rosbag2lerobot.

Provides the following Click commands:

- ``convert``          -- Convert one or more ROS2 rosbags to a LeRobot v3.0
  dataset. Shows a tqdm ETA progress bar by default and writes
  ``meta/conversion_log.json`` (provenance) plus ``meta/job_summary.json``
  (run statistics). ``--json`` emits the job summary to stdout, ``--quiet``
  suppresses the progress bar / INFO chatter, and ``--skip-failed`` records a
  failed bag and continues instead of aborting the run. Supports ``--resume``
  as a safe re-run guard: converting into a non-empty ``--output`` without it
  aborts, and with it a crashed (non-finalized) output is wiped and rebuilt
  while a finalized one is left untouched.
- ``inspect``          -- Display topics, message counts, and time ranges of
  rosbags (with ``--fps-stats`` / ``--suggest-image-size`` diagnostics).
- ``scaffold``         -- Auto-generate a starter ``robot_config.yaml`` from an
  unknown robot's bag and (unless ``--no-validate``) validate it.
- ``validate-config``  -- Validate a YAML config against a rosbag's contents.
- ``validate-dataset`` -- Validate that a generated dataset conforms to the
  LeRobot Dataset v3.0 structure.
- ``quality-report``   -- Score the data quality of a generated dataset.
- ``audit-timestamps`` -- Audit timestamp continuity of a generated dataset.
- ``validate-msg``     -- Check a ``.msg`` file for syntactic correctness.
- ``preview``          -- Write a self-contained static HTML preview report
  (summary, quality score, sample frames, numeric stats) for a dataset.
- ``push-to-hub``      -- Upload a generated dataset to the HuggingFace Hub and
  generate a dataset card (opt-in; ``--dry-run`` plans the upload only).
- ``to-mcap``          -- Convert ROS1 ``.bag`` recordings to ROS2 MCAP bags.

All report commands (``validate-config`` / ``validate-dataset`` /
``quality-report`` / ``audit-timestamps`` / ``inspect`` / ``validate-msg`` /
``to-mcap``) accept ``--json`` to emit their report dict to stdout instead of
the human-readable summary.

Usage::

    rosbag2lerobot convert --config my_config.yaml --bags /bags/ --output /out/
    rosbag2lerobot scaffold --bags /bags/ -o robot_config.yaml
    rosbag2lerobot inspect --bags /bags/
    rosbag2lerobot validate-dataset --dataset /out/
    rosbag2lerobot quality-report --dataset /out/
    rosbag2lerobot validate-msg --msg msgs/MyType.msg
    rosbag2lerobot preview --dataset /out/
    rosbag2lerobot push-to-hub --dataset /out/ --dry-run
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import click

if TYPE_CHECKING:
    from rosbag2lerobot.diagnostics import ValidationReport

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from rosbag2lerobot import __version__ as _ROSBAG2LEROBOT_VERSION
from rosbag2lerobot.bagconvert import (
    DEFAULT_DST_VERSION,
    convert_to_mcap,
    discover_ros1_bags,
    output_name,
)
from rosbag2lerobot.config import (
    config_to_yaml,
    load_config,
    RobotConfig,
    FeatureMapping,
    ResamplingConfig,
)
from rosbag2lerobot.decoders import decode, decode_array
from rosbag2lerobot.jobmeta import EpisodeResult, JobSummary, dir_bytes
from rosbag2lerobot.manifest import ManifestInput, ffmpeg_version, sha256_of_path
from rosbag2lerobot.reader import BagReader, discover_bags, extract_header_stamp_ns
from rosbag2lerobot.resampler import Resampler, trim_to_valid_range
from rosbag2lerobot.task_spec import SubtaskSpan, resolve_task
from rosbag2lerobot.transforms import TransformLookup, quat_xyzw_to_euler


logger = logging.getLogger("rosbag2lerobot")


def _detect_nvenc() -> bool:
    """Return True if ffmpeg has at least one ``*_nvenc`` encoder available.

    Implemented as a pure function that shells out to ``ffmpeg -encoders``
    and scans the stdout for the NVENC encoder names. The subprocess call
    is isolated so unit tests can mock :func:`subprocess.run`.

    Returns:
        ``True`` if any of ``h264_nvenc``, ``hevc_nvenc``, or
        ``av1_nvenc`` appears in ffmpeg's encoder list; ``False`` when
        ffmpeg is missing, times out, or reports no NVENC encoder.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            stdin=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return any(
        enc in result.stdout for enc in ("h264_nvenc", "hevc_nvenc", "av1_nvenc")
    )


def _setup_logging(verbose: bool = False) -> None:
    """Configure root logger format and level.

    Args:
        verbose: If True, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _make_progress(total: int, disable: bool) -> Any:
    """Return a tqdm progress bar over *total* episodes, or ``None``.

    Args:
        total: Number of episodes to track.
        disable: When True, returns ``None`` (no bar) — used for ``--quiet``
            and ``--json`` so machine-readable output stays uncluttered.

    Returns:
        A configured ``tqdm`` instance, or ``None`` when *disable* is set.
    """
    if disable:
        return None
    from tqdm import tqdm

    return tqdm(total=total, unit="ep", desc="convert")


def _emit_report(
    payload: dict[str, Any],
    *,
    json_stdout: bool,
    json_out: Optional[str],
    human_fn: Callable[[dict[str, Any]], None],
) -> None:
    """Emit a report verb's *payload* per the uniform output precedence.

    Precedence (independent of one another):

    - ``--json`` (``json_stdout``): print ``json.dumps(payload, indent=2)`` to
      stdout and SUPPRESS the human summary. Logging stays on stderr so the
      stdout JSON is clean for machine consumers.
    - ``--json-out`` / ``-o`` FILE (``json_out``): write the payload as JSON to
      the file. This is the back-compat P0 file flag; it is independent of
      ``--json`` (both may be set: file is written AND stdout JSON is emitted).
    - Neither / file-only: render the human summary via *human_fn*.

    Args:
        payload: JSON-serializable report dict.
        json_stdout: Value of the verb's ``--json`` flag.
        json_out: Value of the verb's existing ``--json-out`` / ``-o`` FILE
            flag, or ``None`` when the verb has none / it was not set.
        human_fn: Callback that renders the human summary from *payload*.
    """
    if json_out is not None:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w") as fh:
            json.dump(payload, fh, indent=2)

    if json_stdout:
        click.echo(json.dumps(payload, indent=2))
        return

    if json_out is not None:
        click.echo(f"Wrote JSON report to {json_out}")
    human_fn(payload)


# ---------------------------------------------------------------------------
# Click group
# ---------------------------------------------------------------------------


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(verbose: bool) -> None:
    """rosbag2lerobot – convert ROS2 rosbags to LeRobot Dataset v3.0."""
    _setup_logging(verbose)


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


@main.command()
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
) -> None:
    """Convert ROS2 rosbags to a LeRobot v3.0 dataset.

    Each bag directory is treated as one episode. The pipeline loads the
    YAML config, discovers bags, reads and decodes messages, resamples to
    the target FPS, and writes parquet + video + metadata files.
    """
    # 1. Load config
    cfg = load_config(config_path)

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
    manifest_extra: dict[str, Any] = {
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
        )
    else:
        episodes_iter = _iter_episodes_serial(
            bag_paths,
            cfg,
            resampler,
            bag_specs,
            on_episode_done=_on_episode_done,
            skip_failed=skip_failed,
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

    Episodes shorter than ``cfg.split.min_length`` are filtered here at the
    producer: they are never yielded to the writer and never reported via
    ``on_episode_done``, so dataset totals (stats.json / job_summary /
    conversion_log) only ever reflect the episodes actually written.
    """
    import time

    min_length = cfg.split.min_length
    for ep_idx, bag_path in enumerate(bag_paths):
        resolved_task, subtasks = bag_specs[ep_idx]
        logger.info("Episode %d: %s", ep_idx, bag_path)
        started = time.monotonic()
        try:
            frames = _process_episode(bag_path, cfg, resampler)
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


def _process_episode(
    bag_path: Path,
    cfg: RobotConfig,
    resampler: Resampler,
) -> list[dict]:
    """Read one rosbag and produce resampled fixed-fps frames.

    Args:
        bag_path: Path to a single bag directory.
        cfg: Validated robot configuration.
        resampler: Configured Resampler instance.

    Returns:
        List of frame dicts, each containing all feature keys plus
        ``frame_index`` and ``timestamp``.
    """
    with BagReader(bag_path, cfg) as reader:
        topic_to_fms = cfg.topic_to_features
        global_delay = cfg.resampling.max_stamp_delay_ms

        # TF features (frame_from/frame_to set) are sampled off the output frame
        # grid from /tf + /tf_static rather than a single topic. When present we
        # also read the TF source topics in this same pass and accumulate a
        # TransformLookup. With no TF features, behaviour/perf is unchanged.
        tf_features = [fm for fm in cfg.observations + cfg.actions if fm.is_tf_feature]
        tf_lookup: Optional[TransformLookup] = None
        tf_dynamic_topics: set[str] = set()
        tf_static_topics: set[str] = set()
        read_topics = list(cfg.all_topics)
        if tf_features:
            tf_lookup = TransformLookup()
            for fm in tf_features:
                tf_dynamic_topics.add(fm.tf_topic)
                tf_static_topics.add(fm.tf_static_topic)
            for t in sorted(tf_dynamic_topics | tf_static_topics):
                if t not in read_topics:
                    read_topics.append(t)

        # Collect and decode messages referenced by the config. The adopted
        # timestamp per feature follows ``stamp_source`` (header vs. bag
        # receive time); stale latched messages are dropped *before* decode
        # (decode is the expensive step) when their header lags the receive
        # time beyond the effective ``max_stamp_delay_ms`` threshold.
        messages: list[tuple[str, int, object]] = []
        stale_dropped = 0
        for topic, recv_ns, raw_msg in reader.iter_messages(topics=read_topics):
            if tf_lookup is not None:
                if topic in tf_static_topics:
                    tf_lookup.add_static(raw_msg)
                if topic in tf_dynamic_topics:
                    tf_lookup.add_dynamic(raw_msg)
            header_ns = extract_header_stamp_ns(raw_msg)
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
                if (
                    thr is not None
                    and header_ns is not None
                    and abs(recv_ns - header_ns) > thr * 1e6
                ):
                    stale_dropped += 1
                    continue

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


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--config",
    "config_path",
    required=False,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to robot_config.yaml (optional for inspect).",
)
@click.option(
    "--bags",
    "bags_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to bag directory or parent directory.",
)
@click.option(
    "--fps-stats/--no-fps-stats",
    default=False,
    help="Compute per-topic FPS, head/tail lag, and gap statistics.",
)
@click.option(
    "--topics",
    default=None,
    type=str,
    help="Comma-separated topic filter for --fps-stats (default: all topics).",
)
@click.option(
    "--gap-threshold-ms",
    default=200.0,
    type=float,
    show_default=True,
    help="Inter-arrival gaps above this (ms) are reported under --fps-stats.",
)
@click.option(
    "--head",
    "head_n",
    default=5,
    type=int,
    show_default=True,
    help="How many head sample intervals to include in --fps-stats output.",
)
@click.option(
    "--json-out",
    "json_out",
    default=None,
    type=click.Path(dir_okay=False),
    help="If set, write --fps-stats / --suggest-image-size output as JSON.",
)
@click.option(
    "--json",
    "json_stdout",
    is_flag=True,
    default=False,
    help="Emit the report dict as JSON to stdout (suppresses the human summary).",
)
@click.option(
    "--suggest-image-size",
    "suggest_image_size",
    is_flag=True,
    default=False,
    help="Decode sample frames and compare shapes against YAML image_size.",
)
@click.option(
    "--samples",
    default=5,
    type=int,
    show_default=True,
    help="Number of samples to decode for --suggest-image-size.",
)
def inspect(
    config_path: Optional[str],
    bags_path: str,
    fps_stats: bool,
    topics: Optional[str],
    gap_threshold_ms: float,
    head_n: int,
    json_out: Optional[str],
    json_stdout: bool,
    suggest_image_size: bool,
    samples: int,
) -> None:
    """Inspect rosbag(s): show topics, message counts, and time ranges.

    With ``--fps-stats`` also report per-topic FPS, head/tail lag, and
    gaps. With ``--suggest-image-size`` and a ``--config``, decode the
    first few samples of each image topic and compare the shape against
    the YAML ``image_size``.
    """
    # Default (no extra flags): keep the legacy plain-text output intact.
    if not fps_stats and not suggest_image_size:
        _inspect_legacy(bags_path)
        return

    bag_paths = discover_bags(bags_path)
    topics_filter = _split_selector(topics) if topics else None

    cfg: Optional[RobotConfig] = None
    if config_path is not None:
        cfg = load_config(config_path)

    if suggest_image_size and cfg is None:
        raise click.UsageError(
            "--suggest-image-size requires --config to know which topics are images."
        )

    output: dict[str, Any] = {"bags": []}
    for bp in bag_paths:
        bag_entry: dict[str, Any] = {"bag": str(bp)}
        if fps_stats:
            bag_entry.update(
                _run_fps_stats(bp, cfg, topics_filter, gap_threshold_ms, head_n)
            )
        if suggest_image_size and cfg is not None:
            bag_entry["image_shape_check"] = _run_image_shape_check(bp, cfg, samples)
        output["bags"].append(bag_entry)

    _emit_report(
        output,
        json_stdout=json_stdout,
        json_out=json_out,
        human_fn=lambda p: _print_inspect_human(
            p, show_fps=fps_stats, show_shape=suggest_image_size
        ),
    )


def _inspect_legacy(bags_path: str) -> None:
    """Backwards-compatible plain-text inspect, used when no new flags are set."""
    from rosbags.rosbag2 import Reader

    bag_paths = discover_bags(bags_path)
    click.echo(f"Found {len(bag_paths)} bag(s)\n")

    for bp in bag_paths:
        click.echo(f"Bag: {bp}")
        try:
            reader = Reader(bp)
            reader.open()

            start_ns = reader.start_time
            end_ns = start_ns + reader.duration
            duration_s = reader.duration / 1e9
            click.echo(f"  Duration : {duration_s:.2f} s")
            click.echo(f"  Start    : {start_ns} ns")
            click.echo(f"  End      : {end_ns} ns")
            click.echo("  Topics:")

            for conn in reader.connections:
                click.echo(
                    f"    {conn.topic:50s} {conn.msgtype:40s} ({conn.msgcount} msgs)"
                )

            reader.close()
        except Exception as exc:
            click.echo(f"  ERROR: {exc}")

        click.echo("")


def _collect_topic_fps_reports(
    reader: BagReader,
    topics: list[str],
    gap_threshold_ms: float,
    head_n: int,
) -> list[dict[str, Any]]:
    """Collect per-topic FPS reports for *topics* in a single timestamp pass.

    Reads raw timestamps (no CDR deserialize) for every requested topic and
    runs :func:`compute_topic_fps_report` on each. Shared by
    :func:`_run_fps_stats` (full per-topic report) and
    :func:`_measure_topic_fps` (which reads back ``report['fps']['mean']``).

    Args:
        reader: An open :class:`BagReader`.
        topics: Topics to measure (in the order they are returned).
        gap_threshold_ms: Inter-arrival gap threshold passed through to
            :func:`compute_topic_fps_report`.
        head_n: Number of head sample intervals to include in each report.

    Returns:
        One report dict per topic, aligned with *topics* by index. The
        topic name is *not* set on the report (callers add ``"name"`` when
        they need it).
    """
    import numpy as np

    from rosbag2lerobot.diagnostics import compute_topic_fps_report

    start_ns, end_ns = reader.get_time_range()
    topics_info = reader.get_topics_info()

    # Collect timestamps per topic in a single pass.
    ts_by_topic: dict[str, list[int]] = {t: [] for t in topics}
    for topic, ts_ns, _msg in _iter_raw_timestamps(reader, topics):
        if topic in ts_by_topic:
            ts_by_topic[topic].append(ts_ns)

    reports: list[dict[str, Any]] = []
    for topic in topics:
        info = topics_info[topic]
        ts_array = np.asarray(ts_by_topic[topic], dtype=np.int64)
        ts_array.sort()
        reports.append(
            compute_topic_fps_report(
                ts_ns=ts_array,
                bag_start_ns=start_ns,
                bag_end_ns=end_ns,
                msg_type=info.msg_type,
                msg_count=info.count,
                gap_threshold_ms=gap_threshold_ms,
                head_n=head_n,
            )
        )
    return reports


def _run_fps_stats(
    bag_path: Path,
    cfg: Optional[RobotConfig],
    topics_filter: Optional[list[str]],
    gap_threshold_ms: float,
    head_n: int,
) -> dict[str, Any]:
    """Build the FPS report section for one bag."""
    # A minimal stub config is enough for BagReader when the user did
    # not supply --config; we only need topic metadata / iter_messages.
    stub_cfg = cfg if cfg is not None else _build_stub_config(bag_path)

    entry: dict[str, Any] = {"topics": []}
    try:
        with BagReader(bag_path, stub_cfg) as reader:
            start_ns, end_ns = reader.get_time_range()
            entry["duration_s"] = float((end_ns - start_ns) / 1e9)
            entry["start_ns"] = int(start_ns)
            entry["end_ns"] = int(end_ns)

            topics_info = reader.get_topics_info()
            wanted_topics = (
                [t for t in topics_info if t in set(topics_filter)]
                if topics_filter
                else list(topics_info.keys())
            )

            reports = _collect_topic_fps_reports(
                reader, wanted_topics, gap_threshold_ms, head_n
            )
            for topic, report in zip(wanted_topics, reports):
                report["name"] = topic
                entry["topics"].append(report)
    except Exception as exc:
        entry["error"] = str(exc)
    return entry


def _iter_raw_timestamps(
    reader: "BagReader",
    wanted_topics: list[str],
):
    """Yield ``(topic, ts_ns, None)`` without deserializing payloads.

    ``BagReader.iter_messages`` deserializes the CDR payload which is
    wasted work when we only need timestamps. We access the underlying
    :mod:`rosbags` reader directly here so FPS stats stay cheap even on
    image-heavy bags.
    """
    connections = reader._reader.connections  # type: ignore[attr-defined]
    if wanted_topics:
        topic_set = set(wanted_topics)
        connections = [c for c in connections if c.topic in topic_set]
    for conn, ts_ns, _raw in reader._reader.messages(connections=connections):  # type: ignore[attr-defined]
        yield conn.topic, int(ts_ns), None


def _build_stub_config(bag_path: Path) -> RobotConfig:
    """Construct a throwaway RobotConfig so BagReader can open a bag.

    BagReader requires a config to compare the bag against, but for pure
    topic-iteration use we don't care about features — we just need a
    valid dataclass. The stub declares a single dummy feature.
    """
    return RobotConfig(
        robot_type="stub",
        fps=1,
        task="stub",
        observations=[
            FeatureMapping(
                key="observation.state",
                topic="__stub__",
                msg_type="sensor_msgs/msg/JointState",
                optional=True,
            ),
        ],
        actions=[
            FeatureMapping(
                key="action",
                topic="__stub_action__",
                msg_type="sensor_msgs/msg/JointState",
                optional=True,
            ),
        ],
    )


def _run_image_shape_check(
    bag_path: Path,
    cfg: RobotConfig,
    samples: int,
) -> list[dict[str, Any]]:
    """Detect per-image-topic shape and compare with YAML image_size."""
    from rosbag2lerobot.diagnostics import (
        _normalize_yaml_image_size,
        detect_image_shape,
    )

    results: list[dict[str, Any]] = []
    with BagReader(bag_path, cfg) as reader:
        for fm in cfg.image_features:
            detected = detect_image_shape(reader, fm, samples)
            yaml_norm = _normalize_yaml_image_size(fm.image_size)
            mismatch = (
                detected is not None
                and yaml_norm is not None
                and list(detected) != yaml_norm
            )
            results.append(
                {
                    "key": fm.key,
                    "topic": fm.topic,
                    "yaml_image_size": (
                        list(fm.image_size) if fm.image_size is not None else None
                    ),
                    "decoded_shape": list(detected) if detected is not None else None,
                    "mismatch": bool(mismatch),
                }
            )
    return results


def _print_inspect_human(
    output: dict[str, Any],
    show_fps: bool,
    show_shape: bool,
) -> None:
    """Render the ``inspect`` report to the terminal."""
    for bag_entry in output["bags"]:
        click.echo(f"Bag: {bag_entry['bag']}")
        if show_fps:
            if "error" in bag_entry:
                click.secho(f"  ERROR: {bag_entry['error']}", fg="red")
            else:
                click.echo(f"  Duration: {bag_entry.get('duration_s', 0.0):.2f} s")
                click.echo(
                    f"  {'TOPIC':50s} {'MSGS':>6s} {'MEAN_FPS':>9s} "
                    f"{'MIN':>6s} {'MAX':>6s} {'HEAD_LAG_MS':>12s} {'GAPS':>5s}"
                )
                for t in bag_entry.get("topics", []):
                    fps = t["fps"]
                    click.echo(
                        f"  {t['name']:50s} {t['msg_count']:>6d} "
                        f"{_fmt(fps.get('mean')):>9s} "
                        f"{_fmt(fps.get('min')):>6s} "
                        f"{_fmt(fps.get('max')):>6s} "
                        f"{_fmt(t.get('head_lag_ms')):>12s} "
                        f"{len(t.get('gaps', [])):>5d}"
                    )
        if show_shape:
            click.echo("  Image shape check:")
            for r in bag_entry.get("image_shape_check", []):
                yaml_s = r["yaml_image_size"]
                dec_s = r["decoded_shape"]
                status = "MISMATCH" if r["mismatch"] else "OK"
                fg = "red" if r["mismatch"] else "green"
                click.echo(f"    {r['key']} ({r['topic']})")
                click.echo(f"      YAML image_size : {yaml_s}")
                click.echo(f"      Decoded shape   : {dec_s}")
                click.secho(f"      Status          : {status}", fg=fg)
        click.echo("")


def _fmt(val: Any) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


# ---------------------------------------------------------------------------
# validate-config
# ---------------------------------------------------------------------------


@main.command("validate-config")
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
    help="Path to a bag directory or parent directory.",
)
@click.option(
    "--samples",
    default=5,
    type=int,
    show_default=True,
    help="Number of image frames to decode per topic for shape check.",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Treat warnings (image shape / unused topics) as failures.",
)
@click.option(
    "--json-out",
    "json_out",
    default=None,
    type=click.Path(dir_okay=False),
    help="If set, write the validation report JSON to this path.",
)
@click.option(
    "--ignore-unused-topics",
    is_flag=True,
    default=False,
    help="Do not report bag topics that the config does not reference.",
)
@click.option(
    "--suggest-fixes",
    is_flag=True,
    default=False,
    help="After the summary, print copy-pasteable image_size diffs for shape mismatches.",
)
@click.option(
    "--json",
    "json_stdout",
    is_flag=True,
    default=False,
    help="Emit the report dict as JSON to stdout (suppresses the human summary).",
)
def validate_config(
    config_path: str,
    bags_path: str,
    samples: int,
    strict: bool,
    json_out: Optional[str],
    ignore_unused_topics: bool,
    suggest_fixes: bool,
    json_stdout: bool,
) -> None:
    """Validate a YAML config against the contents of a rosbag."""
    from rosbag2lerobot.diagnostics import validate_config_against_bag

    cfg = load_config(config_path)
    bag_paths = discover_bags(bags_path)
    # Use the first discovered bag as the primary validation target. CI
    # pipelines typically point at a single representative bag anyway;
    # extending to multi-bag aggregation is out of scope here.
    bag_path = bag_paths[0]

    with BagReader(bag_path, cfg) as reader:
        report = validate_config_against_bag(cfg, reader, samples)

    if ignore_unused_topics:
        report.unused_bag_topics = []

    report.apply_verdict(strict=strict)

    payload = {
        "config": str(config_path),
        "bag": str(bag_path),
        "results": report.to_dict(),
    }

    def _human(p: dict[str, Any]) -> None:
        _print_validation_summary(p)
        if suggest_fixes:
            _print_suggested_fixes(report)

    _emit_report(payload, json_stdout=json_stdout, json_out=json_out, human_fn=_human)

    if report.exit_code != 0:
        sys.exit(report.exit_code)


def _print_validation_summary(payload: dict[str, Any]) -> None:
    """Render the validation report to the terminal."""
    results = payload["results"]
    click.echo(f"Config validation: {payload['config']} <-> {payload['bag']}\n")
    n_err = 0
    n_warn = 0
    n_info = 0

    for t in results["missing_required_topics"]:
        click.secho(f"  [ERROR]   Missing required topic: {t}", fg="red")
        n_err += 1
    for m in results["msg_type_mismatches"]:
        click.secho(f"  [ERROR]   msg_type mismatch on {m['topic']}", fg="red")
        click.echo(f"              YAML: {m['yaml']}")
        click.echo(f"              BAG:  {m['bag']}")
        n_err += 1
    for m in results["image_shape_mismatches"]:
        click.secho(
            f"  [WARN]    Image shape mismatch on {m['key']}",
            fg="yellow",
        )
        click.echo(f"              YAML image_size: {m['yaml']}")
        click.echo(f"              Decoded shape  : {m['decoded']}")
        n_warn += 1
    for t in results["missing_optional_topics"]:
        click.secho(f"  [INFO]    Missing optional topic: {t}", fg="cyan")
        n_info += 1
    if results["unused_bag_topics"]:
        click.secho(
            f"  [INFO]    Unused bag topics ({len(results['unused_bag_topics'])}): "
            f"{', '.join(results['unused_bag_topics'])}",
            fg="cyan",
        )
        n_info += 1

    verdict = results["verdict"]
    color = "green" if verdict == "OK" else "red"
    click.secho(
        f"\nVerdict: {verdict} ({n_err} error, {n_warn} warning, {n_info} info)",
        fg=color,
    )


def _print_suggested_fixes(report: "ValidationReport") -> None:
    """Print copy-pasteable ``image_size`` diffs for image-shape mismatches.

    For every :class:`~rosbag2lerobot.diagnostics.ImageShapeMismatch` the block shows
    the current YAML ``image_size`` and the measured ``[H, W, C]`` as a unified
    diff snippet the user can paste over the offending feature::

        observation.images.front  (/camera/front/image_raw)
        -   image_size: [480, 640, 3]  # current
        +   image_size: [720, 1280, 3]  # measured

    Args:
        report: The validation report whose ``image_shape_mismatches`` drive
            the suggestions. A no-op when that list is empty.
    """
    if not report.image_shape_mismatches:
        return
    click.echo("")
    click.secho("Suggested fixes:", bold=True)
    for m in report.image_shape_mismatches:
        click.echo(f"  {m.key}  ({m.topic})")
        click.secho(f"  -   image_size: {m.yaml}  # current", fg="red")
        click.secho(f"  +   image_size: {m.decoded}  # measured", fg="green")


# ---------------------------------------------------------------------------
# scaffold
# ---------------------------------------------------------------------------

# Image message types that map directly to ``observation.images.*`` features.
_IMAGE_MSG_TYPES = frozenset(
    {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}
)

# Infrastructure topics that are never converted to features.
_INFRA_MSG_TYPES = frozenset({"rcl_interfaces/msg/Log"})
_INFRA_TOPICS = frozenset({"/rosbag/status", "/rosbag/stop"})

# Path segments that carry no disambiguating meaning when building a slug.
_GENERIC_SEGMENTS = frozenset(
    {
        "image_raw",
        "image_rect",
        "image_rect_color",
        "compressed",
        "compressedDepth",
        "joint_states",
        "rm_driver",
        "controller",
        "robot",
        "raw",
    }
)


def _slug_segments(topic: str) -> list[str]:
    """Sanitize a topic into ``[a-z0-9_]`` path segments (leading '/' stripped)."""
    segments: list[str] = []
    for seg in topic.strip("/").split("/"):
        cleaned = "".join(c if c.isalnum() else "_" for c in seg).strip("_").lower()
        if cleaned:
            segments.append(cleaned)
    return segments


def _slug_from_topic(topic: str) -> str:
    """Build a short feature slug from the last 1-2 informative path segments.

    Keeps disambiguators such as ``color`` / ``depth`` while dropping generic
    tail segments like ``image_raw``. Examples::

        /camera/color/image_raw0 -> camera_color_0
        /camera/depth/image_raw0 -> camera_depth_0
        /camera/image_raw0       -> camera_0

    Args:
        topic: ROS2 topic name.

    Returns:
        A sanitized slug consisting of ``[a-z0-9_]`` characters. Never empty
        (falls back to the joined segments / ``"feature"``).
    """
    segs = _slug_segments(topic)
    if not segs:
        return "feature"

    # Split a trailing numeric suffix off the last segment (image_raw0 -> 0).
    last = segs[-1]
    trailing = "".join(c for c in last if c.isdigit())
    core = last[: len(last) - len(trailing)] if trailing else last

    parts: list[str] = []
    # Walk segments except the last, keeping only the informative ones.
    for seg in segs[:-1]:
        if seg in _GENERIC_SEGMENTS:
            continue
        parts.append(seg)
    if core and core not in _GENERIC_SEGMENTS:
        parts.append(core)
    if trailing:
        parts.append(trailing)

    if not parts:
        # Everything was generic; fall back to the last 1-2 raw segments.
        parts = segs[-2:] if len(segs) >= 2 else segs
    return "_".join(parts)


def _dedupe_key(key: str, used: set[str]) -> str:
    """Return *key* (or ``key_2``/``key_3``...) so it is unique within *used*."""
    if key not in used:
        used.add(key)
        return key
    n = 2
    while f"{key}_{n}" in used:
        n += 1
    unique = f"{key}_{n}"
    used.add(unique)
    return unique


def _is_depth_topic(topic: str) -> bool:
    """True for depth image topics (``compressedDepth`` or a ``/depth/`` path)."""
    return topic.endswith("compressedDepth") or "/depth/" in topic


def _is_command_topic(topic: str) -> bool:
    """True when the topic name looks like a command/action candidate."""
    low = topic.lower()
    return any(
        token in low for token in ("_cmd", "command", "command_velocity", "trajectory")
    )


def _measure_topic_fps(reader: BagReader, topics: list[str]) -> dict[str, float]:
    """Return the mean fps per topic via :func:`compute_topic_fps_report`.

    Uses :func:`_iter_raw_timestamps` (no CDR deserialize) so this stays cheap
    on image-heavy bags. Topics whose mean cannot be computed (fewer than two
    timestamps) are omitted from the result.

    Args:
        reader: An open :class:`BagReader`.
        topics: Topics to measure.

    Returns:
        ``{topic: mean_fps}`` for every topic with a computable mean.
    """
    reports = _collect_topic_fps_reports(
        reader, topics, gap_threshold_ms=200.0, head_n=0
    )
    fps_by_topic: dict[str, float] = {}
    for topic, report in zip(topics, reports):
        mean = report["fps"]["mean"]
        if mean is not None:
            fps_by_topic[topic] = float(mean)
    return fps_by_topic


def _pick_target_fps(
    image_topics: list[str],
    state_topics: list[str],
    fps_by_topic: dict[str, float],
) -> int:
    """Choose a target fps: min over image topics, else median of state fps.

    Uses the *mean* fps per topic (max spikes on near-duplicate timestamps).
    Falls back to 30 when nothing measurable is available.

    Args:
        image_topics: Topics mapped as image features.
        state_topics: Topics mapped as numeric state features.
        fps_by_topic: ``{topic: mean_fps}`` from :func:`_measure_topic_fps`.

    Returns:
        A positive integer fps.
    """
    import numpy as np

    img_fps = [fps_by_topic[t] for t in image_topics if t in fps_by_topic]
    if img_fps:
        return max(1, round(min(img_fps)))
    state_fps = [fps_by_topic[t] for t in state_topics if t in fps_by_topic]
    if state_fps:
        return max(1, round(float(np.median(state_fps))))
    return 30


def _arm_suffix(topic: str) -> str | None:
    """Return ``"left"``/``"right"`` if the topic path names an arm, else None."""
    low = topic.lower()
    if "left" in low:
        return "left"
    if "right" in low:
        return "right"
    return None


def _scaffold_from_topics(
    topics_info: dict[str, Any],
    fps_by_topic: dict[str, float],
    image_shapes: dict[str, Optional[tuple[int, int, int]]],
    registered: set[str],
    robot_type: str,
    task: str,
    fps_override: Optional[int],
    min_count: int,
    bag_name: str,
) -> tuple[RobotConfig, dict[str, list[str]], list[str], list[str]]:
    """Build a :class:`RobotConfig` plus comment metadata from bag topic info.

    Implements the scaffold mapping heuristics: pre-filter by count and
    infra topics, map image topics (incl. depth) to ``observation.images.*``,
    map decodable numeric topics to ``observation.state*`` (both arms for
    dual-arm JointState), and collect no-decoder numeric topics and command
    topics as commented-out candidates.

    Args:
        topics_info: ``{topic: TopicInfo}`` from ``BagReader.get_topics_info``.
        fps_by_topic: Measured mean fps per topic.
        image_shapes: ``{topic: (H, W, C) | None}`` consensus image shapes.
        registered: Set of msg_types with a registered decoder.
        robot_type: Value for ``RobotConfig.robot_type``.
        task: Value for ``RobotConfig.task``.
        fps_override: Explicit fps; ``None`` selects it heuristically.
        min_count: Drop topics with a message count below this.
        bag_name: Name of the source bag (for the header comment).

    Returns:
        ``(cfg, obs_annotations, obs_candidates, act_candidates)`` where the
        annotations/candidates feed :func:`config_to_yaml`.
    """
    # 1. Pre-filter: count threshold + infra topics.
    kept: list[tuple[str, Any]] = []
    for topic, info in topics_info.items():
        if info.count < min_count:
            continue
        if info.msg_type in _INFRA_MSG_TYPES or topic in _INFRA_TOPICS:
            continue
        kept.append((topic, info))
    kept.sort(key=lambda x: x[0])

    image_topics: list[str] = []
    state_topics: list[str] = []
    no_decoder: list[tuple[str, Any]] = []
    command: list[tuple[str, Any]] = []

    for topic, info in kept:
        if info.msg_type in _IMAGE_MSG_TYPES:
            image_topics.append(topic)
        elif info.msg_type in registered:
            if _is_command_topic(topic):
                command.append((topic, info))
            else:
                state_topics.append(topic)
        else:
            if _is_command_topic(topic):
                command.append((topic, info))
            else:
                no_decoder.append((topic, info))

    target_fps = (
        fps_override
        if fps_override is not None
        else _pick_target_fps(image_topics, state_topics, fps_by_topic)
    )

    observations: list[FeatureMapping] = []
    obs_annotations: dict[str, list[str]] = {}
    used_keys: set[str] = set()

    def _annot(topic: str, decoder_note: str) -> list[str]:
        notes: list[str] = []
        if topic in fps_by_topic:
            notes.append(f"measured fps: {fps_by_topic[topic]:.2f}")
        notes.append(decoder_note)
        return notes

    # 2. Image features (color/depth/mono), collision-disambiguated by slug.
    for topic in image_topics:
        slug = _slug_from_topic(topic)
        key = _dedupe_key(f"observation.images.{slug}", used_keys)
        shape = image_shapes.get(topic)
        image_size = list(shape) if shape is not None else None
        fm = FeatureMapping(
            key=key,
            topic=topic,
            msg_type=topics_info[topic].msg_type,
            dtype="image",
            image_size=image_size,
            stamp_source="header",
        )
        observations.append(fm)
        notes = _annot(topic, "decoder: builtin")
        if image_size is None:
            notes.append("TODO image_size (shape detection failed)")
        if _is_depth_topic(topic):
            notes.append("depth image (stored as video)")
        obs_annotations[key] = notes

    # 3. Numeric state features with a registered decoder. Dual-arm JointState
    #    (multiple JointState topics) is emitted as observation.state_<arm>.
    jointstate_topics = [
        t
        for t in state_topics
        if topics_info[t].msg_type == "sensor_msgs/msg/JointState"
    ]
    dual_arm_js = len(jointstate_topics) > 1
    first_state_assigned = False
    for topic in state_topics:
        msg_type = topics_info[topic].msg_type
        is_js = msg_type == "sensor_msgs/msg/JointState"
        if is_js and dual_arm_js:
            arm = _arm_suffix(topic)
            base = (
                f"observation.state_{arm}"
                if arm is not None
                else f"observation.state_{_slug_from_topic(topic)}"
            )
            key = _dedupe_key(base, used_keys)
        elif not first_state_assigned:
            key = _dedupe_key("observation.state", used_keys)
            first_state_assigned = True
        else:
            key = _dedupe_key(f"observation.state_{_slug_from_topic(topic)}", used_keys)
        fm = FeatureMapping(
            key=key,
            topic=topic,
            msg_type=msg_type,
            selector="position" if is_js else "",
            dtype="float32",
            stamp_source="header",
        )
        observations.append(fm)
        obs_annotations[key] = _annot(topic, "decoder: builtin")

    # 4. Commented-out no-decoder candidates (numeric topics, no decoder).
    obs_candidates: list[str] = []
    if no_decoder:
        obs_candidates.append("  # ---- candidates without a registered decoder ----")
        obs_candidates.append(
            "  # decoder: NONE — add custom_msgs + user decoder before enabling"
        )
        for topic, info in no_decoder:
            fps_note = (
                f"  ~{fps_by_topic[topic]:.2f} fps" if topic in fps_by_topic else ""
            )
            obs_candidates.append(f"  #   {topic}  [{info.msg_type}]{fps_note}")

    # 5. actions: always empty + commented command candidates to uncomment.
    act_candidates: list[str] = []
    act_candidates.append("  # TODO: actions are never auto-mapped. Uncomment and edit")
    act_candidates.append("  #       a command topic below to define the action space.")
    if command:
        for topic, info in command:
            fps_note = (
                f"  ~{fps_by_topic[topic]:.2f} fps" if topic in fps_by_topic else ""
            )
            note = "" if info.msg_type in registered else "  (decoder: NONE)"
            act_candidates.append(
                f'  #   - key: "action"  # {topic} [{info.msg_type}]{fps_note}{note}'
            )

    cfg = RobotConfig(
        robot_type=robot_type,
        fps=target_fps,
        task=task,
        observations=observations,
        actions=[],
        resampling=ResamplingConfig(),
    )

    header = [
        "Generated by `rosbag2lerobot scaffold` — review before use.",
        f"Source bag: {bag_name} (scaffolded from the first discovered bag only).",
        "Commented '# - key:' blocks are candidates; uncomment + edit to enable.",
    ]
    obs_annotations["__header__"] = header  # carried out-of-band; popped by caller
    return cfg, obs_annotations, obs_candidates, act_candidates


def scaffold_config_yaml(
    bag_path: Path,
    *,
    robot_type: str,
    task: str,
    fps: Optional[int],
    min_count: int,
    samples: int,
) -> str:
    """Build the scaffold ``robot_config.yaml`` text for a single bag.

    Body of the ``scaffold`` verb: open the bag, measure per-topic fps, probe
    image shapes, run the mapping heuristics (:func:`_scaffold_from_topics`),
    and render the YAML (:func:`config.config_to_yaml`).

    Args:
        bag_path: A single bag directory (the caller has already run
            ``discover_bags`` and selected the first bag).
        robot_type: ``robot_type`` value for the generated config.
        task: ``task`` description for the generated config.
        fps: Explicit target fps, or ``None`` to pick it heuristically.
        min_count: Drop topics with fewer than this many messages.
        samples: Image frames to decode per topic for shape detection.

    Returns:
        The rendered YAML text.
    """
    from rosbag2lerobot.decoders import get_registered_types
    from rosbag2lerobot.diagnostics import detect_image_shape

    registered = set(get_registered_types())

    stub_cfg = _build_stub_config(bag_path)
    with BagReader(bag_path, stub_cfg) as reader:
        topics_info = reader.get_topics_info()

        # Measure fps for every kept topic in one cheap pass.
        measurable = [
            t
            for t, info in topics_info.items()
            if info.count >= min_count
            and info.msg_type not in _INFRA_MSG_TYPES
            and t not in _INFRA_TOPICS
        ]
        fps_by_topic = _measure_topic_fps(reader, measurable)

        # Detect image shapes (consensus over `samples` frames).
        image_shapes: dict[str, Optional[tuple[int, int, int]]] = {}
        for topic, info in topics_info.items():
            if info.msg_type in _IMAGE_MSG_TYPES and info.count >= min_count:
                fm = FeatureMapping(
                    key="probe",
                    topic=topic,
                    msg_type=info.msg_type,
                    dtype="image",
                )
                image_shapes[topic] = detect_image_shape(reader, fm, samples)

    cfg, obs_annotations, obs_candidates, act_candidates = _scaffold_from_topics(
        topics_info=topics_info,
        fps_by_topic=fps_by_topic,
        image_shapes=image_shapes,
        registered=registered,
        robot_type=robot_type,
        task=task,
        fps_override=fps,
        min_count=min_count,
        bag_name=bag_path.name,
    )
    header = obs_annotations.pop("__header__", [])

    return config_to_yaml(
        cfg,
        header_lines=header,
        obs_annotations=obs_annotations,
        obs_candidates=obs_candidates,
        act_candidates=act_candidates,
    )


@main.command("scaffold")
@click.option(
    "--bags",
    "bags_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to a bag directory or parent directory.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    default=None,
    type=click.Path(dir_okay=False),
    help="Output config path. Default: print to stdout.",
)
@click.option(
    "--fps",
    default=None,
    type=int,
    help="Target fps. Default: auto (min image fps, else median state fps).",
)
@click.option(
    "--robot-type",
    default="unknown_robot",
    show_default=True,
    help="robot_type value for the generated config.",
)
@click.option(
    "--task",
    default="TODO_describe_task",
    show_default=True,
    help="task description for the generated config.",
)
@click.option(
    "--min-count",
    default=1,
    type=int,
    show_default=True,
    help="Drop topics with fewer than this many messages.",
)
@click.option(
    "--samples",
    default=3,
    type=int,
    show_default=True,
    help="Image frames to decode per topic for shape detection.",
)
@click.option(
    "--no-validate",
    is_flag=True,
    default=False,
    help="Skip the auto validate-config step on the generated config.",
)
def scaffold(
    bags_path: str,
    output_path: Optional[str],
    fps: Optional[int],
    robot_type: str,
    task: str,
    min_count: int,
    samples: int,
    no_validate: bool,
) -> None:
    """Generate a starter robot_config.yaml from a bag's topics.

    Discovers topics in the first bag, maps image and decodable numeric
    topics to LeRobot feature keys, and emits commented-out candidates for
    no-decoder and command topics. The config is validated in memory before
    writing; unless ``--no-validate`` is set, ``validate-config`` is then run
    against the bag to enforce the round-trip / mapping guarantee.
    """
    bag_paths = discover_bags(bags_path)
    bag_path = bag_paths[0]
    if len(bag_paths) > 1:
        logger.info(
            "Found %d bags; scaffolding from the first only: %s",
            len(bag_paths),
            bag_path,
        )

    yaml_text = scaffold_config_yaml(
        bag_path,
        robot_type=robot_type,
        task=task,
        fps=fps,
        min_count=min_count,
        samples=samples,
    )

    if output_path is None:
        click.echo(yaml_text)
    else:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml_text)
        click.secho(f"Wrote scaffold config to {out}", fg="green")

        if not no_validate:
            _scaffold_validate(out, bag_path)


def _scaffold_validate(config_path: Path, bag_path: Path) -> None:
    """Run validate-config on a generated config and print its summary.

    Mirrors the ``validate-config`` command path so the scaffold output is
    immediately checked against the bag it was generated from. Reloading via
    :func:`load_config` also enforces the YAML round-trip guarantee.

    Args:
        config_path: Path to the freshly written scaffold config.
        bag_path: The bag the config was scaffolded from.
    """
    from rosbag2lerobot.diagnostics import validate_config_against_bag

    cfg = load_config(config_path)
    with BagReader(bag_path, cfg) as reader:
        report = validate_config_against_bag(cfg, reader, samples=3)
    report.apply_verdict(strict=False)
    payload = {
        "config": str(config_path),
        "bag": str(bag_path),
        "results": report.to_dict(),
    }
    click.echo("")
    _print_validation_summary(payload)


# ---------------------------------------------------------------------------
# validate-msg
# ---------------------------------------------------------------------------


@main.command("validate-msg")
@click.option(
    "--msg",
    "msg_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a .msg file to validate.",
)
@click.option(
    "--json",
    "json_stdout",
    is_flag=True,
    default=False,
    help="Emit the result dict as JSON to stdout (suppresses the human summary).",
)
def validate_msg(msg_path: str, json_stdout: bool) -> None:
    """Validate a ROS2 .msg file for syntax correctness."""
    from rosbags.typesys import get_types_from_msg

    msg_file = Path(msg_path)
    msg_text = msg_file.read_text()

    # Derive a dummy type name from the filename
    type_name = f"validation_pkg/msg/{msg_file.stem}"

    valid = True
    n_types = 0
    error: Optional[str] = None
    try:
        types = get_types_from_msg(msg_text, type_name)
        n_types = len(types) if types else 0
    except Exception as exc:  # noqa: BLE001 - reported as an invalid result
        valid = False
        error = str(exc)

    payload = {
        "msg": str(msg_file),
        "type_name": type_name,
        "valid": valid,
        "n_types": n_types,
        "error": error,
    }

    def _human(_p: dict[str, Any]) -> None:
        if valid:
            click.echo(f"Valid .msg file: {msg_file.name}")
            click.echo(f"  Registered type: {type_name}")
            if n_types:
                click.echo(f"  Fields defined: {n_types} type(s)")
            click.secho("  OK", fg="green")
        else:
            click.secho(f"  INVALID: {error}", fg="red")

    _emit_report(payload, json_stdout=json_stdout, json_out=None, human_fn=_human)

    if not valid:
        sys.exit(1)


# ---------------------------------------------------------------------------
# audit-timestamps
# ---------------------------------------------------------------------------


@main.command("audit-timestamps")
@click.option(
    "--dataset",
    "dataset_path",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Root directory of a generated LeRobot v3.0 dataset.",
)
@click.option(
    "--max-drift-us",
    default=1.0,
    type=float,
    show_default=True,
    help="Maximum allowed per-row / cumulative drift in microseconds.",
)
@click.option(
    "--json-out",
    default=None,
    type=click.Path(dir_okay=False),
    help="If set, write the audit report as JSON to this path.",
)
@click.option(
    "--json",
    "json_stdout",
    is_flag=True,
    default=False,
    help="Emit the report dict as JSON to stdout (suppresses the human summary).",
)
@click.option(
    "--video-key",
    "video_key",
    default=None,
    help="Audit only this video_key (default: all video_keys present).",
)
def audit_timestamps(
    dataset_path: str,
    max_drift_us: float,
    json_out: Optional[str],
    json_stdout: bool,
    video_key: Optional[str],
) -> None:
    """Audit meta/episodes/*.parquet timestamp continuity for drift.

    Reads every episodes parquet file under the dataset's ``meta/episodes/``
    tree and verifies that ``to_timestamp[i] == from_timestamp[i + 1]`` inside
    each mp4 file and that ``from_timestamp`` only resets to ``0.0`` at mp4
    file boundaries. Exits with status 1 on any violation.
    """
    from rosbag2lerobot.audit import audit_episode_timestamps

    vkeys = [video_key] if video_key else None
    try:
        report = audit_episode_timestamps(
            Path(dataset_path),
            max_drift_us=max_drift_us,
            video_keys=vkeys,
        )
    except (FileNotFoundError, ValueError) as exc:
        click.secho(f"audit-timestamps: {exc}", fg="red")
        sys.exit(2)

    payload = report.to_dict()

    _emit_report(
        payload,
        json_stdout=json_stdout,
        json_out=json_out,
        human_fn=lambda p: _print_audit_summary(p, max_drift_us),
    )

    if report.verdict != "OK":
        sys.exit(report.exit_code)


def _print_audit_summary(payload: dict[str, Any], max_drift_us: float) -> None:
    """Render an AuditReport dict as a compact, colorized CLI summary."""
    click.echo(f"Dataset : {payload['dataset']}")
    click.echo(f"Keys    : {', '.join(payload['video_keys']) or '(none)'}")
    click.echo(f"Max drift threshold: {max_drift_us:.3f} us")
    click.echo("")
    click.echo(
        f"{'VIDEO_KEY':40s} {'EPS':>5s} {'MAX_DRIFT_US':>14s} {'ERRS':>5s}  VERDICT"
    )
    for r in payload["results"]:
        verdict_color = "green" if r["verdict"] == "OK" else "red"
        line = (
            f"{r['video_key']:40s} {r['n_episodes']:>5d} "
            f"{r['max_drift_us']:>14.3f} {len(r['boundary_errors']):>5d}  "
        )
        click.echo(line, nl=False)
        click.secho(r["verdict"], fg=verdict_color)
        for err in r["boundary_errors"][:10]:
            click.secho(
                f"    ep={err['episode_index']:<4d} "
                f"expected={err['expected_from_ts']:.6f} "
                f"actual={err['actual_from_ts']:.6f} "
                f"delta_us={err['delta_us']:+.3f}",
                fg="red",
            )
        if len(r["boundary_errors"]) > 10:
            click.secho(
                f"    ... and {len(r['boundary_errors']) - 10} more",
                fg="red",
            )

    click.echo("")
    fg = "green" if payload["verdict"] == "OK" else "red"
    click.secho(f"Verdict: {payload['verdict']}", fg=fg, bold=True)


# ---------------------------------------------------------------------------
# validate-dataset
# ---------------------------------------------------------------------------


@main.command("validate-dataset")
@click.option(
    "--dataset",
    "dataset_path",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Root directory of a generated LeRobot v3.0 dataset.",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Treat WARN-level issues (extra columns) as failures.",
)
@click.option(
    "--json-out",
    "json_out",
    default=None,
    type=click.Path(dir_okay=False),
    help="If set, write the validation report as JSON to this path.",
)
@click.option(
    "--json",
    "json_stdout",
    is_flag=True,
    default=False,
    help="Emit the report dict as JSON to stdout (suppresses the human summary).",
)
def validate_dataset_cmd(
    dataset_path: str,
    strict: bool,
    json_out: Optional[str],
    json_stdout: bool,
) -> None:
    """Validate the structure of a generated LeRobot v3.0 dataset.

    Checks required files, ``meta/info.json`` keys/values, parquet schemas,
    and episode-count cross-checks. Exits 1 on any ERROR (or any WARN under
    ``--strict``); 2 on a setup error such as an unreadable parquet file.
    """
    import pyarrow.lib as pa_lib

    from rosbag2lerobot.validation import validate_dataset

    try:
        report = validate_dataset(Path(dataset_path))
    except (OSError, ValueError, pa_lib.ArrowInvalid) as exc:
        click.secho(f"validate-dataset: {exc}", fg="red")
        sys.exit(2)

    report.apply_verdict(strict=strict)
    payload = report.to_dict()

    _emit_report(
        payload,
        json_stdout=json_stdout,
        json_out=json_out,
        human_fn=_print_dataset_validation_summary,
    )

    if report.exit_code != 0:
        sys.exit(report.exit_code)


def _print_dataset_validation_summary(payload: dict[str, Any]) -> None:
    """Render a DatasetValidationReport dict as a colorized CLI summary."""
    click.echo(f"Dataset : {payload['dataset']}")
    click.echo("")
    for issue in payload["issues"]:
        color = "red" if issue["severity"] == "ERROR" else "yellow"
        click.secho(
            f"  [{issue['severity']:5s}] {issue['kind']} @ {issue['location']}",
            fg=color,
        )
        click.echo(f"            {issue['message']}")

    verdict = payload["verdict"]
    fg = "green" if verdict == "OK" else "red"
    click.echo("")
    click.secho(
        f"Verdict: {verdict} "
        f"({payload['n_errors']} error, {payload['n_warnings']} warning)",
        fg=fg,
        bold=True,
    )


# ---------------------------------------------------------------------------
# quality-report
# ---------------------------------------------------------------------------


@main.command("quality-report")
@click.option(
    "--dataset",
    "dataset_path",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Root directory of a generated LeRobot v3.0 dataset.",
)
@click.option(
    "-o",
    "--report",
    "report_out",
    default=None,
    type=click.Path(dir_okay=False),
    help="If set, write the quality report as JSON to this path.",
)
@click.option(
    "--freeze-std-eps",
    default=1e-3,
    type=float,
    show_default=True,
    help="Per-pair std threshold for freeze-frame detection.",
)
@click.option(
    "--range-tol",
    default=0.0,
    type=float,
    show_default=True,
    help="Absolute tolerance added to stats.json min/max for out-of-range.",
)
@click.option(
    "--score-threshold",
    default=0.95,
    type=float,
    show_default=True,
    help="Minimum quality score for an OK verdict.",
)
@click.option(
    "--json",
    "json_stdout",
    is_flag=True,
    default=False,
    help="Emit the report dict as JSON to stdout (suppresses the human summary).",
)
def quality_report_cmd(
    dataset_path: str,
    report_out: Optional[str],
    freeze_std_eps: float,
    range_tol: float,
    score_threshold: float,
    json_stdout: bool,
) -> None:
    """Compute a data-quality report for a generated LeRobot v3.0 dataset.

    Reports per-feature null/NaN/out-of-range rates, freeze frames, and
    video/data frame reconciliation, condensed into a 0..1 score. Exits 1
    when the score is below ``--score-threshold`` or any video has a frame
    mismatch; 2 on a setup error (missing/unreadable metadata).
    """
    from rosbag2lerobot.quality import compute_quality_report

    try:
        report = compute_quality_report(
            Path(dataset_path),
            freeze_std_eps=freeze_std_eps,
            range_tol=range_tol,
            score_threshold=score_threshold,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        click.secho(f"quality-report: {exc}", fg="red")
        sys.exit(2)

    payload = report.to_dict()

    _emit_report(
        payload,
        json_stdout=json_stdout,
        json_out=report_out,
        human_fn=_print_quality_summary,
    )

    if report.exit_code != 0:
        sys.exit(report.exit_code)


def _print_quality_summary(payload: dict[str, Any]) -> None:
    """Render a QualityReport dict as a compact, colorized CLI summary."""
    click.echo(f"Dataset : {payload['dataset']}")
    click.echo("")
    click.echo(f"{'FEATURE':40s} {'NULL_RATE':>10s} {'NAN':>8s} {'OOR_RATE':>10s}")
    for f in payload["features"]:
        click.echo(
            f"{f['feature']:40s} {f['null_rate']:>10.4f} "
            f"{f['n_nan']:>8d} {f['oor_rate']:>10.4f}"
        )

    if payload["videos"]:
        click.echo("")
        click.echo(
            f"{'VIDEO_KEY':40s} {'EXPECTED':>9s} {'MP4':>9s} "
            f"{'MISMATCH':>9s} {'FREEZE':>7s}"
        )
        for v in payload["videos"]:
            mismatch_color = "green" if v["frame_mismatch"] == 0 else "red"
            line = (
                f"{v['video_key']:40s} {v['expected_frames']:>9d} "
                f"{v['mp4_frames']:>9d} "
            )
            click.echo(line, nl=False)
            click.secho(f"{v['frame_mismatch']:>9d}", fg=mismatch_color, nl=False)
            click.echo(f" {v['n_freeze']:>7d}")

    click.echo("")
    click.echo(
        f"Score: {payload['score']:.4f} (threshold {payload['score_threshold']:.4f})"
    )
    verdict = payload["verdict"]
    fg = "green" if verdict == "OK" else "red"
    click.secho(f"Verdict: {verdict}", fg=fg, bold=True)


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------


@main.command("preview")
@click.option(
    "--dataset",
    "dataset_path",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Root directory of a generated LeRobot v3.0 dataset.",
)
@click.option(
    "--n-frames",
    default=3,
    type=int,
    show_default=True,
    help="Number of sample frames to embed per video key.",
)
@click.option(
    "-o",
    "--out",
    "out_path",
    default=None,
    type=click.Path(dir_okay=False),
    help="Output HTML path (default: <dataset>/meta/preview.html).",
)
@click.option(
    "--sample-video/--no-sample-video",
    default=False,
    show_default=True,
    help="Decode mp4s to count freeze frames for the quality section.",
)
def preview_cmd(
    dataset_path: str,
    n_frames: int,
    out_path: Optional[str],
    sample_video: bool,
) -> None:
    """Write a self-contained static HTML preview report for a dataset.

    Renders the summary, the quality score and tables, a gallery of sampled
    video frames (inline base64), and the numeric per-feature statistics into
    a single self-contained HTML file (no external assets).
    """
    from rosbag2lerobot.preview import generate_preview

    dataset_dir = Path(dataset_path)
    try:
        html = generate_preview(
            dataset_dir,
            n_frames=n_frames,
            sample_video=sample_video,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        click.secho(f"preview: {exc}", fg="red")
        sys.exit(2)

    out = (
        Path(out_path)
        if out_path is not None
        else dataset_dir / "meta" / "preview.html"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    click.echo(f"Wrote preview to {out}")


# ---------------------------------------------------------------------------
# push-to-hub
# ---------------------------------------------------------------------------


@main.command("push-to-hub")
@click.option(
    "--dataset",
    "dataset_path",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Root directory of a generated LeRobot v3.0 dataset.",
)
@click.option(
    "--repo-id",
    "repo_id",
    default=None,
    help="HuggingFace dataset repo id (default: info.json['repo_id']).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Plan the upload only: print repo_id + file count + card preview.",
)
@click.option(
    "--private",
    is_flag=True,
    default=False,
    help="Create the repo as private (ignored for --dry-run).",
)
@click.option(
    "--token",
    default=None,
    help="HuggingFace auth token (else the ambient login).",
)
@click.option(
    "--card-out",
    "card_out",
    default=None,
    type=click.Path(dir_okay=False),
    help="With --dry-run, also write the generated card to this path.",
)
def push_to_hub_cmd(
    dataset_path: str,
    repo_id: Optional[str],
    dry_run: bool,
    private: bool,
    token: Optional[str],
    card_out: Optional[str],
) -> None:
    """Push a generated dataset to the HuggingFace Hub (with a dataset card).

    ``--repo-id`` falls back to ``info.json['repo_id']``; if neither is set the
    command exits 2. With ``--dry-run`` nothing is uploaded — the planned
    repo_id, file count, and card preview are printed (and the card written to
    ``--card-out`` when given). Without ``--dry-run`` the dataset is uploaded
    and the card is placed at the repo root.
    """
    from rosbag2lerobot.hub import plan_push, push_to_hub
    from rosbag2lerobot.quality import _read_info

    dataset_dir = Path(dataset_path)

    effective_repo_id = repo_id
    if effective_repo_id is None:
        try:
            effective_repo_id = _read_info(dataset_dir).get("repo_id")
        except (OSError, ValueError) as exc:
            click.secho(f"push-to-hub: {exc}", fg="red")
            sys.exit(2)
    if not effective_repo_id:
        click.secho(
            "push-to-hub: no --repo-id given and info.json has no 'repo_id'.",
            fg="red",
        )
        sys.exit(2)

    if dry_run:
        plan = plan_push(dataset_dir, effective_repo_id)
        click.echo(f"[dry-run] repo_id : {plan.repo_id}")
        click.echo(f"[dry-run] files   : {len(plan.files)}")
        if card_out is not None:
            card_path = Path(card_out)
            card_path.parent.mkdir(parents=True, exist_ok=True)
            card_path.write_text(plan.card_text)
            click.echo(f"[dry-run] wrote card to {card_path}")
        click.echo("[dry-run] card preview:")
        click.echo(plan.card_text)
        return

    push_to_hub(
        dataset_dir,
        effective_repo_id,
        private=private,
        token=token,
    )
    click.secho(f"Pushed {dataset_dir} to {effective_repo_id}", fg="green", bold=True)


# ---------------------------------------------------------------------------
# to-mcap
# ---------------------------------------------------------------------------


@main.command("to-mcap")
@click.argument(
    "sources",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    "output_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Output base directory. Each bag is written to <output>/<name>/.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing output bag directories.",
)
@click.option(
    "--dst-version",
    "dst_version",
    default=DEFAULT_DST_VERSION,
    type=int,
    show_default=True,
    help="ROS2 bag format version to write.",
)
@click.option(
    "--json",
    "json_stdout",
    is_flag=True,
    default=False,
    help="Emit the result dict as JSON to stdout (suppresses the human summary).",
)
def to_mcap(
    sources: tuple[Path, ...],
    output_dir: Path,
    overwrite: bool,
    dst_version: int,
    json_stdout: bool,
) -> None:
    """Convert ROS1 .bag recordings to ROS2 MCAP bags.

    rosbag2lerobot itself only reads ROS2 bags (mcap/sqlite3). Use this command to
    pre-convert ROS1 .bag recordings (e.g. the airoa raw dataset) so they
    can be fed to `rosbag2lerobot convert`.

    SOURCES may be .bag files or directories (searched recursively for
    *.bag). Each input bag is written to <output>/<name>/, where <name> is
    the bag file's parent directory name (e.g. .../235210/data.bag ->
    <output>/235210/).
    """
    bags = discover_ros1_bags(list(sources))
    if not bags:
        click.secho("No ROS1 .bag files found in the given sources.", fg="yellow")
        sys.exit(1)

    if not json_stdout:
        click.echo(f"Found {len(bags)} ROS1 bag(s) to convert.")
    converted = 0
    failed = 0
    results: list[dict[str, Any]] = []
    for src in bags:
        dst = output_dir / output_name(src)
        try:
            convert_to_mcap(src, dst, dst_version=dst_version, overwrite=overwrite)
            results.append({"src": str(src), "dst": str(dst), "status": "OK"})
            if not json_stdout:
                click.secho(f"  OK  {src}  ->  {dst}", fg="green")
            converted += 1
        except FileExistsError as exc:
            results.append({"src": str(src), "dst": str(dst), "status": "SKIP"})
            if not json_stdout:
                click.secho(f"  SKIP {exc}", fg="yellow")
            failed += 1
        except Exception as exc:  # noqa: BLE001 - report and continue
            results.append(
                {"src": str(src), "dst": str(dst), "status": "FAIL", "error": str(exc)}
            )
            if not json_stdout:
                click.secho(f"  FAIL {src}: {exc}", fg="red")
            failed += 1

    payload = {
        "output_dir": str(output_dir),
        "results": results,
        "converted": converted,
        "failed": failed,
    }

    def _human(_p: dict[str, Any]) -> None:
        click.echo("")
        click.secho(
            f"Converted {converted}/{len(bags)} bag(s) to MCAP under {output_dir}",
            fg="green" if failed == 0 else "yellow",
            bold=True,
        )

    _emit_report(payload, json_stdout=json_stdout, json_out=None, human_fn=_human)

    if failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
