"""``scaffold`` command: auto-generate a starter robot_config.yaml from a bag."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import click

from rosbag2lerobot.config import (
    config_to_yaml,
    load_config,
    RobotConfig,
    FeatureMapping,
    ResamplingConfig,
)
from rosbag2lerobot.reader import BagReader, discover_bags
from rosbag2lerobot.cli._common import logger
from rosbag2lerobot.cli.inspect import _build_stub_config, _collect_topic_fps_reports
from rosbag2lerobot.cli.validate_config import _print_validation_summary


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


@click.command("scaffold")
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
