"""CLI entry point for bagel.

Provides three Click commands:

- ``convert``      -- Convert one or more ROS2 rosbags to a LeRobot v3.0 dataset.
- ``inspect``      -- Display topics, message counts, and time ranges of rosbags.
- ``validate-msg`` -- Check a ``.msg`` file for syntactic correctness.

Usage::

    bagel convert --config my_config.yaml --bags /bags/ --output /out/
    bagel inspect --bags /bags/
    bagel validate-msg --msg msgs/MyType.msg
"""

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

import click

from concurrent.futures import ProcessPoolExecutor, as_completed

from bagel.bagconvert import (
    DEFAULT_DST_VERSION,
    convert_to_mcap,
    discover_ros1_bags,
    output_name,
)
from bagel.config import load_config, RobotConfig, FeatureMapping
from bagel.decoders import decode
from bagel.reader import BagReader, discover_bags, extract_header_stamp_ns
from bagel.resampler import Resampler, trim_to_valid_range
from bagel.task_spec import SubtaskSpan, resolve_task


logger = logging.getLogger("bagel")


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


# ---------------------------------------------------------------------------
# Click group
# ---------------------------------------------------------------------------


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(verbose: bool) -> None:
    """bagel – convert ROS2 rosbags to LeRobot Dataset v3.0."""
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
) -> None:
    """Convert ROS2 rosbags to a LeRobot v3.0 dataset.

    Each bag directory is treated as one episode. The pipeline loads the
    YAML config, discovers bags, reads and decodes messages, resamples to
    the target FPS, and writes parquet + video + metadata files.
    """
    # 1. Load config
    cfg = load_config(config_path)

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

    # 5. Process each bag (= 1 episode) — generator-based pipeline so the
    # writer sees episodes one at a time instead of the old "materialize all
    # then write" path that blew up memory on large datasets (T11).
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if workers and workers > 1 and len(bag_paths) > 1:
        episodes_iter: Iterator[list[dict]] = _iter_episodes_parallel(
            bag_paths,
            cfg,
            resampler,
            workers,
            bag_specs,
        )
    else:
        episodes_iter = _iter_episodes_serial(bag_paths, cfg, resampler, bag_specs)

    # 6. Write dataset
    try:
        from bagel.writer import write_dataset

        write_dataset(
            episodes=episodes_iter,
            config=cfg,
            output_dir=output_dir,
            video_codec=effective_codec,
            repo_id=cfg.repo_id,
            ffmpeg_preset=ffmpeg_preset,
            ffmpeg_crf=ffmpeg_crf,
            has_subtasks=has_subtasks,
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


def _process_bag_entry(
    args: tuple[int, Path, RobotConfig, dict[str, Any], str, list[SubtaskSpan]],
) -> tuple[int, list[dict], str]:
    """Worker-side entry point for :func:`_iter_episodes_parallel`.

    Rebuilds a local ``Resampler`` from the serialized config dict (resampler
    instances themselves can be pickled but we serialize the config to stay
    resilient to dataclass changes). Task and subtasks are already resolved
    by the caller (pre-scan in :func:`convert`) and passed in directly.
    """
    ep_idx, bag_path, cfg, resampler_kwargs, resolved_task, subtasks = args
    resampler = Resampler(**resampler_kwargs)
    frames = _process_episode(bag_path, cfg, resampler)
    _tag_episode(frames, resolved_task, subtasks)
    return ep_idx, frames, resolved_task


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
) -> Iterator[list[dict]]:
    """Yield processed episodes one-at-a-time in ``bag_paths`` order.

    Each bag is decoded, resampled, and tagged with its pre-resolved task
    and subtask spans before being yielded. This is the streaming
    counterpart to the old ``all_episodes = []`` accumulator: the writer
    only holds one episode in memory at a time, which is critical for
    large-scale conversions where total frame memory would otherwise
    exceed tens of GB.
    """
    for ep_idx, bag_path in enumerate(bag_paths):
        resolved_task, subtasks = bag_specs[ep_idx]
        logger.info("Episode %d: %s", ep_idx, bag_path)
        frames = _process_episode(bag_path, cfg, resampler)
        _tag_episode(frames, resolved_task, subtasks)
        logger.info(
            "  -> %d frames (%.1f s) [task=%r, subtasks=%d]",
            len(frames),
            len(frames) / cfg.fps if frames else 0,
            resolved_task,
            len(subtasks),
        )
        yield frames


def _iter_episodes_parallel(
    bag_paths: list[Path],
    cfg: RobotConfig,
    resampler: Resampler,
    workers: int,
    bag_specs: list[tuple[str, list[SubtaskSpan]]],
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

    pending: dict[int, list[dict]] = {}
    next_idx = 0

    with ProcessPoolExecutor(max_workers=effective_workers) as pool:
        futures = {pool.submit(_process_bag_entry, job): job[0] for job in jobs}
        for fut in as_completed(futures):
            ep_idx, frames, resolved = fut.result()
            logger.info(
                "Episode %d: %s -> %d frames (%.1f s) [task=%r]",
                ep_idx,
                bag_paths[ep_idx],
                len(frames),
                len(frames) / cfg.fps if frames else 0,
                resolved,
            )
            pending[ep_idx] = frames

            # Drain the contiguous prefix starting at next_idx so that
            # episodes flow to the writer in original bag order as soon
            # as they are ready (rather than waiting for the slowest).
            while next_idx in pending:
                yield pending.pop(next_idx)
                next_idx += 1

    # Safety net: flush any trailing buffered episodes. In normal flow
    # the drain inside the loop already handles everything, but if the
    # executor shutdown races with a late completion we still want the
    # remaining episodes to reach the writer.
    while next_idx in pending:
        yield pending.pop(next_idx)
        next_idx += 1


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

        # Collect and decode messages referenced by the config. The adopted
        # timestamp per feature follows ``stamp_source`` (header vs. bag
        # receive time); stale latched messages are dropped *before* decode
        # (decode is the expensive step) when their header lags the receive
        # time beyond the effective ``max_stamp_delay_ms`` threshold.
        messages: list[tuple[str, int, object]] = []
        stale_dropped = 0
        for topic, recv_ns, raw_msg in reader.iter_messages(topics=cfg.all_topics):
            header_ns = extract_header_stamp_ns(raw_msg)
            for fm in topic_to_fms.get(topic, []):
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
        if cfg.resampling.align_to_required:
            window = _required_window(messages, cfg.required_feature_keys)
            if window is None:
                return []
            start_ns, end_ns = window
            logger.debug("  align_to_required window: [%d, %d] ns", start_ns, end_ns)
        else:
            start_ns, end_ns = reader.get_time_range()

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

    return frames


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

    if json_out is not None:
        import json

        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w") as fh:
            json.dump(output, fh, indent=2)
        click.echo(f"Wrote JSON report to {json_out}")
    else:
        _print_inspect_human(output, show_fps=fps_stats, show_shape=suggest_image_size)


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


def _run_fps_stats(
    bag_path: Path,
    cfg: Optional[RobotConfig],
    topics_filter: Optional[list[str]],
    gap_threshold_ms: float,
    head_n: int,
) -> dict[str, Any]:
    """Build the FPS report section for one bag."""
    import numpy as np

    from bagel.diagnostics import compute_topic_fps_report

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

            # Collect timestamps per topic in a single pass.
            ts_by_topic: dict[str, list[int]] = {t: [] for t in wanted_topics}
            for topic, ts_ns, _msg in _iter_raw_timestamps(reader, wanted_topics):
                if topic in ts_by_topic:
                    ts_by_topic[topic].append(ts_ns)

            for topic in wanted_topics:
                info = topics_info[topic]
                ts_array = np.asarray(ts_by_topic[topic], dtype=np.int64)
                ts_array.sort()
                report = compute_topic_fps_report(
                    ts_ns=ts_array,
                    bag_start_ns=start_ns,
                    bag_end_ns=end_ns,
                    msg_type=info.msg_type,
                    msg_count=info.count,
                    gap_threshold_ms=gap_threshold_ms,
                    head_n=head_n,
                )
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
    from bagel.diagnostics import (
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
def validate_config(
    config_path: str,
    bags_path: str,
    samples: int,
    strict: bool,
    json_out: Optional[str],
    ignore_unused_topics: bool,
) -> None:
    """Validate a YAML config against the contents of a rosbag."""
    from bagel.diagnostics import validate_config_against_bag

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

    if json_out is not None:
        import json

        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w") as fh:
            json.dump(payload, fh, indent=2)
        click.echo(f"Wrote validation JSON to {json_out}")
    else:
        _print_validation_summary(payload)

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
def validate_msg(msg_path: str) -> None:
    """Validate a ROS2 .msg file for syntax correctness."""
    from rosbags.typesys import get_types_from_msg

    msg_file = Path(msg_path)
    msg_text = msg_file.read_text()

    # Derive a dummy type name from the filename
    type_name = f"validation_pkg/msg/{msg_file.stem}"

    try:
        types = get_types_from_msg(msg_text, type_name)
        click.echo(f"Valid .msg file: {msg_file.name}")
        click.echo(f"  Registered type: {type_name}")
        if types:
            click.echo(f"  Fields defined: {len(types)} type(s)")
        click.secho("  OK", fg="green")
    except Exception as exc:
        click.secho(f"  INVALID: {exc}", fg="red")
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
    "--video-key",
    "video_key",
    default=None,
    help="Audit only this video_key (default: all video_keys present).",
)
def audit_timestamps(
    dataset_path: str,
    max_drift_us: float,
    json_out: Optional[str],
    video_key: Optional[str],
) -> None:
    """Audit meta/episodes/*.parquet timestamp continuity for drift.

    Reads every episodes parquet file under the dataset's ``meta/episodes/``
    tree and verifies that ``to_timestamp[i] == from_timestamp[i + 1]`` inside
    each mp4 file and that ``from_timestamp`` only resets to ``0.0`` at mp4
    file boundaries. Exits with status 1 on any violation.
    """
    import json

    from bagel.audit import audit_episode_timestamps

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

    if json_out is not None:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w") as fh:
            json.dump(payload, fh, indent=2)
        click.echo(f"Wrote audit JSON to {json_out}")

    # Always print a human-readable summary alongside any JSON output.
    _print_audit_summary(payload, max_drift_us)

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
def to_mcap(
    sources: tuple[Path, ...],
    output_dir: Path,
    overwrite: bool,
    dst_version: int,
) -> None:
    """Convert ROS1 .bag recordings to ROS2 MCAP bags.

    bagel itself only reads ROS2 bags (mcap/sqlite3). Use this command to
    pre-convert ROS1 .bag recordings (e.g. the airoa raw dataset) so they
    can be fed to `bagel convert`.

    SOURCES may be .bag files or directories (searched recursively for
    *.bag). Each input bag is written to <output>/<name>/, where <name> is
    the bag file's parent directory name (e.g. .../235210/data.bag ->
    <output>/235210/).
    """
    bags = discover_ros1_bags(list(sources))
    if not bags:
        click.secho("No ROS1 .bag files found in the given sources.", fg="yellow")
        sys.exit(1)

    click.echo(f"Found {len(bags)} ROS1 bag(s) to convert.")
    converted = 0
    failed = 0
    for src in bags:
        dst = output_dir / output_name(src)
        try:
            convert_to_mcap(src, dst, dst_version=dst_version, overwrite=overwrite)
            click.secho(f"  OK  {src}  ->  {dst}", fg="green")
            converted += 1
        except FileExistsError as exc:
            click.secho(f"  SKIP {exc}", fg="yellow")
            failed += 1
        except Exception as exc:  # noqa: BLE001 - report and continue
            click.secho(f"  FAIL {src}: {exc}", fg="red")
            failed += 1

    click.echo("")
    click.secho(
        f"Converted {converted}/{len(bags)} bag(s) to MCAP under {output_dir}",
        fg="green" if failed == 0 else "yellow",
        bold=True,
    )
    if failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
