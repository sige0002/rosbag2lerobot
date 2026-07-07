"""``inspect`` command: topics, message counts, and time ranges of rosbags."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import click

from rosbag2lerobot.config import load_config, RobotConfig, FeatureMapping
from rosbag2lerobot.reader import BagReader, discover_bags
from rosbag2lerobot.cli._common import _emit_report, _fmt
from rosbag2lerobot.cli.convert import _split_selector


@click.command()
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
