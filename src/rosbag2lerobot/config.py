"""Robot configuration loader and validator for rosbag2lerobot.

Loads ``robot_config.yaml`` files and validates them against the expected
schema for converting ROS2 rosbag data to LeRobot Dataset v3.0 format.

The configuration is represented by a hierarchy of dataclasses:

- ``RobotConfig``      -- top-level configuration (robot_type, fps, etc.)
- ``FeatureMapping``   -- maps a single ROS topic to a dataset feature key
- ``ResamplingConfig`` -- resampling policy and tolerance
- ``CustomMsgDef``     -- reference to a ``.msg`` file for custom types

Public entry point: ``load_config(path)`` returns a validated ``RobotConfig``.
"""

from __future__ import annotations

import copy
import difflib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Unknown-key detection
# ---------------------------------------------------------------------------


def _check_unknown_keys(raw: dict[str, Any], allowed: set[str], context: str) -> None:
    """Raise ``ValueError`` on any key in *raw* not present in *allowed*.

    Unknown keys are usually typos: silently dropping them (the legacy
    behaviour) lets a misspelled option be ignored without warning. For each
    offending key a :func:`difflib.get_close_matches` suggestion is appended
    to the error message when a near match exists.

    Args:
        raw: The raw YAML mapping for a config section.
        allowed: The set of permitted key names for *context* (derived
            dynamically from the corresponding dataclass fields).
        context: Human-readable name of the section, used in the error
            message (e.g. ``"feature mapping"``, ``"resampling"``).

    Raises:
        ValueError: On the first unknown key encountered.
    """
    for key in raw:
        if key in allowed:
            continue
        matches = difflib.get_close_matches(key, sorted(allowed), n=1, cutoff=0.6)
        hint = f" (did you mean: {matches[0]}?)" if matches else ""
        raise ValueError(f"Unknown {context} key: '{key}'{hint}")


# ---------------------------------------------------------------------------
# Dataclass definitions
# ---------------------------------------------------------------------------


@dataclass
class ResamplingConfig:
    """Resampling / time-synchronisation policy.

    Attributes:
        default_policy: One of ``"hold"``, ``"nearest"``, or ``"drop"``.
        tolerance_ms: Maximum time offset (ms) between frame time and the
            nearest message before the value is considered missing.
        trim_to_valid: If True (default), trim each episode to the frame
            range where every non-optional feature has data. This matches
            the LeRobot v3.0 convention of rejecting frames with missing
            required features instead of zero-filling them. Optional
            features may still have missing values within the retained
            range; the writer zero-fills those for schema compatibility.
        max_stamp_delay_ms: Global default stale-message threshold (ms).
            When a message's header timestamp lags its bag receive time by
            more than this, the message is treated as a stale latched value
            (e.g. left over from a TRANSIENT_LOCAL QoS queue) and discarded.
            ``None`` (default) disables the global check; per-feature
            ``FeatureMapping.max_stamp_delay_ms`` overrides this value.
        align_to_required: If True (default), align the output frame grid to
            the required (non-optional) features rather than to the earliest
            message across all topics.
    """

    default_policy: str = "hold"
    tolerance_ms: float = 50.0
    trim_to_valid: bool = True
    max_stamp_delay_ms: Optional[float] = None
    align_to_required: bool = True

    def __post_init__(self) -> None:
        if self.default_policy not in ("hold", "drop", "nearest"):
            raise ValueError(
                f"Invalid resampling policy '{self.default_policy}'. "
                "Must be one of: hold, drop, nearest"
            )
        if self.tolerance_ms < 0:
            raise ValueError("tolerance_ms must be non-negative")
        if self.max_stamp_delay_ms is not None and self.max_stamp_delay_ms < 0:
            raise ValueError("max_stamp_delay_ms must be non-negative")


@dataclass
class TimestampsConfig:
    """Integrity checks on message timestamps (as opposed to resampling policy).

    Attributes:
        max_header_receive_skew_ms: Maximum tolerated divergence (ms) between a
            message's ``header.stamp`` and its bag receive time, checked per
            message. Exceeding it fails the episode instead of producing a
            dataset with garbage timing — the usual cause is an unsynchronised
            clock on the publishing host, which silently shifts every sample
            time by hours or years. ``None`` disables the check. The default
            (60 s) is deliberately generous: it is meant to catch a broken
            clock, not ordinary transport latency.

    The check only applies to features whose ``stamp_source`` is ``"header"``
    (with ``"receive"`` the header stamp is not used, so its divergence cannot
    corrupt anything) and only to messages that actually carry a header stamp.
    Messages already discarded by ``ResamplingConfig.max_stamp_delay_ms`` are
    exempt: dropping stale latched messages is an explicit, configured policy,
    so those messages are handled rather than unexpected.
    """

    max_header_receive_skew_ms: Optional[float] = 60_000.0

    def __post_init__(self) -> None:
        if (
            self.max_header_receive_skew_ms is not None
            and self.max_header_receive_skew_ms < 0
        ):
            raise ValueError("max_header_receive_skew_ms must be non-negative")


@dataclass
class SplitConfig:
    """Train/val/test split ratios and an episode length filter.

    Attributes:
        train: Fraction of episodes assigned to the ``train`` split.
        val: Fraction assigned to the ``val`` split.
        test: Fraction assigned to the ``test`` split.
        min_length: Minimum episode length (frames). Episodes shorter than
            this are discarded by the writer before splits are computed.
            ``0`` (default) keeps every episode.

    The default (``train=1.0``) reproduces the legacy single-split behavior
    byte-for-byte (``info.json`` ``splits`` becomes ``{"train": "0:N"}``).
    """

    train: float = 1.0
    val: float = 0.0
    test: float = 0.0
    min_length: int = 0

    def __post_init__(self) -> None:
        for name in ("train", "val", "test"):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"split '{name}' ratio must be in [0, 1], got {value}")
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"split ratios must sum to 1.0, got {total} "
                f"(train={self.train}, val={self.val}, test={self.test})"
            )
        if self.min_length < 0:
            raise ValueError("split 'min_length' must be non-negative")

    @property
    def ratios(self) -> dict[str, float]:
        """Return the ``{split_name: ratio}`` mapping for :func:`compute_splits`."""
        return {"train": self.train, "val": self.val, "test": self.test}


def compute_splits(total_episodes: int, ratios: dict[str, float]) -> dict[str, str]:
    """Partition ``[0, total_episodes)`` into contiguous ``"a:b"`` ranges.

    Splits are contiguous, gap-free, and non-overlapping, covering the whole
    episode range. Counts use ``round`` for train/val and let ``test`` absorb
    the remainder so the partition is exact. Each count is clamped to the
    episodes still available so the rounded train+val can never exceed ``N``
    (e.g. ``train=val=0.5, N=3`` rounds to ``2+2`` but is clamped to ``2+1``,
    leaving ``n_test=0``): ``n_train = min(round(train*N), N)``,
    ``n_val = min(round(val*N), N - n_train)``, ``n_test = N - n_train -
    n_val``. Zero-width splits are omitted. When *ratios* is the default
    (``train=1.0``) the result is exactly ``{"train": "0:N"}`` — byte-identical
    to the legacy single split.

    Args:
        total_episodes: Number of episodes ``N`` to partition.
        ratios: ``{"train": ..., "val": ..., "test": ...}`` fractions summing
            to ~1.0.

    Returns:
        ``{split_name: "start:end"}`` for every non-empty split, in
        train/val/test order.
    """
    n = total_episodes
    # Clamp each successive count to the episodes still available so a rounded
    # train+val never overruns N (which would force n_test < 0 and produce an
    # invalid partition that validate-dataset rejects).
    n_train = min(round(ratios.get("train", 0.0) * n), n)
    n_val = min(round(ratios.get("val", 0.0) * n), n - n_train)
    n_test = n - n_train - n_val

    counts = [("train", n_train), ("val", n_val), ("test", n_test)]
    splits: dict[str, str] = {}
    start = 0
    for name, count in counts:
        if count <= 0:
            continue
        end = start + count
        splits[name] = f"{start}:{end}"
        start = end
    return splits


@dataclass
class FeatureMapping:
    """A single observation or action mapping from a ROS topic to a dataset key.

    Attributes:
        key: LeRobot feature key, e.g. ``"observation.state"`` or
            ``"observation.images.front"``.
        topic: ROS2 topic name, e.g. ``"/joint_states"``.
        msg_type: Full ROS2 message type, e.g. ``"sensor_msgs/msg/JointState"``.
        selector: Dot-separated field path for sub-field extraction.
        dtype: Output data type (``"float32"``, ``"image"``, etc.).
        image_size: ``[H, W]`` or ``[H, W, C]`` for image features.
        stamp_source: ``"header"`` to use the message header timestamp,
            or ``"receive"`` to use the bag receive time.
        unit_conversion: Numeric multiplier or special string
            (``"rad2deg"``, ``"deg2rad"``).
        optional: If ``True``, the topic is allowed to be missing or have
            0 messages in the bag without raising an error. A warning is
            emitted instead.
        max_stamp_delay_ms: Per-feature stale-message threshold (ms) that
            overrides ``ResamplingConfig.max_stamp_delay_ms`` for this
            feature. When the message header timestamp lags its bag receive
            time by more than this, the message is discarded as a stale
            latched value. ``None`` (default) means this feature is not
            checked individually (the global threshold, if any, still
            applies).
        frame_from: Source TF frame whose pose is looked up. When set (together
            with ``frame_to``), this feature is a *TF feature*: it is sampled
            on the output frame grid from ``/tf`` + ``/tf_static`` instead of a
            single topic. ``topic`` / ``msg_type`` must still be set (use
            ``topic: /tf``, ``msg_type: tf2_msgs/msg/TFMessage``).
        frame_to: Reference TF frame the pose is expressed in. Set together
            with ``frame_from``.
        tf_topic: Dynamic TF topic name (default ``"/tf"``).
        tf_static_topic: Static TF topic name (default ``"/tf_static"``).

    TF feature output contract:
        The default value is the 7-vector ``[tx, ty, tz, qx, qy, qz, qw]``.
        Set ``selector: orientation.euler_xyz`` (or ``euler_xyz`` /
        ``...euler_zyx``) to replace the quaternion with euler angles (radians),
        yielding the 6-vector ``[tx, ty, tz, roll, pitch, yaw]``.
    """

    key: str
    topic: str
    msg_type: str
    selector: str = ""
    dtype: str = "float32"
    image_size: Optional[list[int]] = None
    stamp_source: str = "header"
    unit_conversion: float | str = 1.0
    optional: bool = False
    names: Optional[list[str]] = None
    max_stamp_delay_ms: Optional[float] = None
    frame_from: Optional[str] = None
    frame_to: Optional[str] = None
    tf_topic: str = "/tf"
    tf_static_topic: str = "/tf_static"

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("Feature mapping 'key' must not be empty")
        if not self.topic:
            raise ValueError("Feature mapping 'topic' must not be empty")
        if not self.msg_type:
            raise ValueError("Feature mapping 'msg_type' must not be empty")
        if self.max_stamp_delay_ms is not None and self.max_stamp_delay_ms < 0:
            raise ValueError("max_stamp_delay_ms must be non-negative")
        if (self.frame_from is None) != (self.frame_to is None):
            raise ValueError(
                "frame_from and frame_to must be set together (both or neither)"
            )
        if self.dtype not in (
            "float32",
            "float64",
            "int32",
            "int64",
            "uint8",
            "bool",
            "image",
            "string",
        ):
            raise ValueError(f"Unsupported dtype '{self.dtype}'")
        if self.stamp_source not in ("header", "receive"):
            raise ValueError(
                f"Invalid stamp_source '{self.stamp_source}'. "
                "Must be 'header' or 'receive'"
            )
        if self.image_size is not None:
            if len(self.image_size) not in (2, 3):
                raise ValueError(
                    "image_size must have 2 (H, W) or 3 (H, W, C) elements"
                )

    @property
    def is_image(self) -> bool:
        return self.dtype == "image" or self.image_size is not None

    @property
    def is_tf_feature(self) -> bool:
        """True when this feature is sampled from the TF tree (``frame_from`` set)."""
        return self.frame_from is not None

    @property
    def lerobot_key(self) -> str:
        """Return the full LeRobot v3.0 feature key.

        Convention:
          - observation images  -> observation.images.<name>
          - observation state   -> observation.state
          - action              -> action
        The caller decides the prefix; this just returns ``self.key``.
        """
        return self.key


@dataclass
class CustomMsgDef:
    """Reference to a custom ``.msg`` file to register with the rosbags type system.

    Attributes:
        msg_file: Path to the ``.msg`` file (absolute, or relative to the config).
        package: ROS2 package name for type registration,
            e.g. ``"my_robot_msgs"``.
    """

    msg_file: str
    package: str

    def __post_init__(self) -> None:
        if not self.msg_file:
            raise ValueError("custom_msgs entry 'msg_file' must not be empty")
        if not self.package:
            raise ValueError("custom_msgs entry 'package' must not be empty")


@dataclass
class RobotConfig:
    """Top-level robot configuration.

    Attributes:
        robot_type: Unique identifier for the robot hardware.
        fps: Target frames per second for the output dataset.
        task: Human-readable task description.
        observations: List of observation feature mappings.
        actions: List of action feature mappings.
        repo_id: Optional HuggingFace repo ID.
        custom_msgs: Custom ``.msg`` file references for non-standard types.
        resampling: Resampling policy and tolerance configuration.
        split: Train/val/test split ratios and episode length filter.
        timestamps: Timestamp integrity checks (header vs. receive skew).
    """

    robot_type: str
    fps: int
    task: str
    observations: list[FeatureMapping]
    actions: list[FeatureMapping]
    repo_id: Optional[str] = None
    custom_msgs: list[CustomMsgDef] = field(default_factory=list)
    resampling: ResamplingConfig = field(default_factory=ResamplingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    timestamps: TimestampsConfig = field(default_factory=TimestampsConfig)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if not self.robot_type:
            raise ValueError("robot_type must not be empty")
        if not self.task:
            raise ValueError("task must not be empty")
        self._validate_unique_keys()

    # ----- helpers -----

    def _validate_unique_keys(self) -> None:
        seen: set[str] = set()
        for fm in self.observations + self.actions:
            if fm.key in seen:
                raise ValueError(f"Duplicate feature key: '{fm.key}'")
            seen.add(fm.key)

    @property
    def all_topics(self) -> list[str]:
        """Return a deduplicated list of all ROS topics referenced.

        Order matches the appearance order in the config file
        (observations first, then actions). ``dict.fromkeys`` preserves
        insertion order on Python 3.7+.
        """
        return list(dict.fromkeys(fm.topic for fm in self.observations + self.actions))

    @property
    def optional_topics(self) -> set[str]:
        """Return the set of topics marked as optional."""
        return {fm.topic for fm in self.observations + self.actions if fm.optional}

    @property
    def topic_to_features(self) -> dict[str, list[FeatureMapping]]:
        """Map each topic to the feature mappings that read from it."""
        result: dict[str, list[FeatureMapping]] = {}
        for fm in self.observations + self.actions:
            result.setdefault(fm.topic, []).append(fm)
        return result

    @property
    def observation_keys(self) -> list[str]:
        """Return the feature key strings for all observations."""
        return [fm.key for fm in self.observations]

    @property
    def action_keys(self) -> list[str]:
        """Return the feature key strings for all actions."""
        return [fm.key for fm in self.actions]

    @property
    def required_feature_keys(self) -> list[str]:
        """Feature keys that must have a value at every retained frame.

        Used by trim-to-valid logic: the episode is clipped to the range
        where every key in this list has data. Keys where
        ``optional: true`` is set in the config are excluded — they are
        allowed to have per-frame gaps (the writer fills them for schema
        compatibility).
        """
        return [fm.key for fm in self.observations + self.actions if not fm.optional]

    @property
    def image_features(self) -> list[FeatureMapping]:
        """Return observation mappings that produce images."""
        return [fm for fm in self.observations if fm.is_image]

    @property
    def state_features(self) -> list[FeatureMapping]:
        """Return observation mappings that produce numeric state vectors."""
        return [fm for fm in self.observations if not fm.is_image]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_feature_mapping(raw: dict[str, Any]) -> FeatureMapping:
    """Create a ``FeatureMapping`` from a raw YAML dict."""
    allowed = {f.name for f in fields(FeatureMapping)}
    _check_unknown_keys(raw, allowed, "feature mapping")
    filtered = {k: v for k, v in raw.items() if k in allowed}
    if filtered.get("max_stamp_delay_ms") is not None:
        filtered["max_stamp_delay_ms"] = float(filtered["max_stamp_delay_ms"])
    return FeatureMapping(**filtered)


def _parse_custom_msg(raw: dict[str, Any]) -> CustomMsgDef:
    """Create a ``CustomMsgDef`` from a raw YAML dict."""
    _check_unknown_keys(
        raw, {f.name for f in fields(CustomMsgDef)}, "custom_msgs entry"
    )
    return CustomMsgDef(
        msg_file=raw.get("msg_file", ""),
        package=raw.get("package", ""),
    )


def _parse_resampling(raw: dict[str, Any] | None) -> ResamplingConfig:
    """Create a ``ResamplingConfig`` from a raw YAML dict, using defaults for missing keys."""
    if raw is None:
        return ResamplingConfig()
    _check_unknown_keys(raw, {f.name for f in fields(ResamplingConfig)}, "resampling")
    max_delay = raw.get("max_stamp_delay_ms")
    return ResamplingConfig(
        default_policy=raw.get("default_policy", "hold"),
        tolerance_ms=float(raw.get("tolerance_ms", 50.0)),
        trim_to_valid=bool(raw.get("trim_to_valid", True)),
        max_stamp_delay_ms=None if max_delay is None else float(max_delay),
        align_to_required=bool(raw.get("align_to_required", True)),
    )


def _parse_timestamps(raw: dict[str, Any] | None) -> TimestampsConfig:
    """Create a ``TimestampsConfig`` from a raw YAML dict.

    An absent ``max_header_receive_skew_ms`` key keeps the (enabled) default,
    while an explicit ``null`` disables the check — the two must not collapse
    into the same thing, so the key's presence is tested rather than its value.
    """
    if raw is None:
        return TimestampsConfig()
    _check_unknown_keys(raw, {f.name for f in fields(TimestampsConfig)}, "timestamps")
    if "max_header_receive_skew_ms" not in raw:
        return TimestampsConfig()
    skew = raw["max_header_receive_skew_ms"]
    return TimestampsConfig(
        max_header_receive_skew_ms=None if skew is None else float(skew)
    )


def _parse_split(raw: dict[str, Any] | None) -> SplitConfig:
    """Create a ``SplitConfig`` from a raw YAML dict, using defaults for missing keys."""
    if raw is None:
        return SplitConfig()
    _check_unknown_keys(raw, {f.name for f in fields(SplitConfig)}, "split")
    return SplitConfig(
        train=float(raw.get("train", 1.0)),
        val=float(raw.get("val", 0.0)),
        test=float(raw.get("test", 0.0)),
        min_length=int(raw.get("min_length", 0)),
    )


# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------

_DEFAULT_DUAL_ARM_OBSERVATIONS: list[dict[str, Any]] = [
    {
        "key": "observation.images.right_wrist",
        "topic": "/camera/right_wrist/image_raw/compressed",
        "msg_type": "sensor_msgs/msg/CompressedImage",
        "dtype": "image",
        "image_size": [480, 640, 3],
        "stamp_source": "header",
    },
    {
        "key": "observation.images.left_wrist",
        "topic": "/camera/left_wrist/image_raw/compressed",
        "msg_type": "sensor_msgs/msg/CompressedImage",
        "dtype": "image",
        "image_size": [480, 640, 3],
        "stamp_source": "header",
    },
    {
        "key": "observation.state",
        "topic": "/joint_states",
        "msg_type": "sensor_msgs/msg/JointState",
        "selector": "position",
        "dtype": "float32",
        "stamp_source": "header",
    },
]

_DEFAULT_DUAL_ARM_ACTIONS: list[dict[str, Any]] = [
    {
        "key": "action",
        "topic": "/joint_commands",
        "msg_type": "sensor_msgs/msg/JointState",
        "selector": "position",
        "dtype": "float32",
        "stamp_source": "header",
    },
]


def build_default_config(
    robot_type: str = "dual_arm",
    fps: int = 30,
    task: str = "default_task",
    repo_id: str | None = None,
) -> RobotConfig:
    """Build a sensible default ``RobotConfig`` for common robot setups.

    * ``dual_arm``  – two 7-axis arms, two wrist cameras.
    * ``single_arm`` – one 7-axis arm (right), two wrist cameras.
    """
    obs_raw = copy.deepcopy(_DEFAULT_DUAL_ARM_OBSERVATIONS)
    act_raw = copy.deepcopy(_DEFAULT_DUAL_ARM_ACTIONS)

    if robot_type == "single_arm":
        # Drop the left-wrist camera for single arm
        obs_raw = [o for o in obs_raw if "left_wrist" not in o["key"]]

    return RobotConfig(
        robot_type=robot_type,
        fps=fps,
        task=task,
        repo_id=repo_id,
        observations=[_parse_feature_mapping(o) for o in obs_raw],
        actions=[_parse_feature_mapping(a) for a in act_raw],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> RobotConfig:
    """Load and validate a ``RobotConfig`` from a YAML file.

    Parameters
    ----------
    path:
        Path to a ``robot_config.yaml`` file.

    Returns
    -------
    RobotConfig
        Validated configuration object.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the YAML content is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError("Config YAML root must be a mapping")

    # Reject unknown top-level keys (likely typos). The allowed set is derived
    # from RobotConfig's fields so newly-added top-level fields are included
    # automatically (split, resampling, custom_msgs, repo_id, ...).
    _check_unknown_keys(raw, {f.name for f in fields(RobotConfig)}, "top-level")

    # Required scalars
    robot_type = raw.get("robot_type")
    if not robot_type:
        raise ValueError("Config must specify 'robot_type'")

    fps = raw.get("fps")
    if fps is None:
        raise ValueError("Config must specify 'fps'")
    fps = int(fps)

    task = raw.get("task")
    if not task:
        raise ValueError("Config must specify 'task'")

    repo_id = raw.get("repo_id")

    # Observations & actions
    obs_raw = raw.get("observations")
    act_raw = raw.get("actions")
    if obs_raw is None:
        raise ValueError("Config must specify 'observations'")
    if act_raw is None:
        raise ValueError("Config must specify 'actions'")

    observations = [_parse_feature_mapping(o) for o in obs_raw]
    actions = [_parse_feature_mapping(a) for a in act_raw]

    # Optional sections
    custom_msgs = [_parse_custom_msg(c) for c in raw.get("custom_msgs", [])]
    resampling = _parse_resampling(raw.get("resampling"))
    split = _parse_split(raw.get("split"))
    timestamps = _parse_timestamps(raw.get("timestamps"))

    return RobotConfig(
        robot_type=robot_type,
        fps=fps,
        task=task,
        repo_id=repo_id,
        observations=observations,
        actions=actions,
        custom_msgs=custom_msgs,
        resampling=resampling,
        split=split,
        timestamps=timestamps,
    )


# ---------------------------------------------------------------------------
# YAML serialization (inverse of load_config)
# ---------------------------------------------------------------------------


def _yaml_scalar(value: Any) -> str:
    """Dump a single scalar to its inline YAML form (no document markers).

    ``yaml.safe_dump`` of a bare scalar produces ``"<value>\\n...\\n"``; we
    keep only the first line so the value can be inlined after a ``key:``.
    """
    return yaml.safe_dump(value, default_flow_style=True).splitlines()[0].strip()


def _feature_mapping_to_dict(fm: FeatureMapping) -> dict[str, Any]:
    """Serialize a :class:`FeatureMapping` to a minimal YAML-friendly dict.

    Only fields that differ from the dataclass default are emitted, so the
    generated YAML stays close to the hand-written configs (``key`` /
    ``topic`` / ``msg_type`` always appear; defaults like ``stamp_source:
    header`` are omitted). ``key``, ``topic`` and ``msg_type`` are always
    included even though they have no default.

    Args:
        fm: The feature mapping to serialize.

    Returns:
        Ordered dict suitable for ``yaml.safe_dump`` that round-trips
        through :func:`_parse_feature_mapping`.
    """
    out: dict[str, Any] = {
        "key": fm.key,
        "topic": fm.topic,
        "msg_type": fm.msg_type,
    }
    always = {"key", "topic", "msg_type"}
    for f in fields(fm):
        if f.name in always:
            continue
        value = getattr(fm, f.name)
        if value == f.default:
            continue
        out[f.name] = value
    return out


def _emit_mapping_block(
    entries: list[FeatureMapping],
    annotations: dict[str, list[str]] | None,
) -> list[str]:
    """Render a list of feature mappings as indented YAML list items.

    Each entry is dumped via :func:`yaml.safe_dump` (so the values
    round-trip through :func:`load_config`) and prefixed with any
    annotation comment lines keyed by the feature ``key``.

    Args:
        entries: Feature mappings to emit.
        annotations: Optional ``{feature_key: [comment, ...]}`` map. Comment
            text is emitted as ``# <comment>`` lines immediately above the
            corresponding entry.

    Returns:
        List of YAML text lines (no trailing newline per element).
    """
    annotations = annotations or {}
    lines: list[str] = []
    for fm in entries:
        for comment in annotations.get(fm.key, []):
            lines.append(f"  # {comment}")
        entry_dict = _feature_mapping_to_dict(fm)
        first = True
        for k, v in entry_dict.items():
            # Render list fields (e.g. image_size, names) in flow style so they
            # match the hand-written configs ("[480, 640, 3]"); everything else
            # goes through _yaml_scalar for safe quoting. Both round-trip.
            rendered = (
                yaml.safe_dump(v, default_flow_style=True).splitlines()[0].strip()
                if isinstance(v, list)
                else _yaml_scalar(v)
            )
            prefix = "  - " if first else "    "
            lines.append(f"{prefix}{k}: {rendered}")
            first = False
    return lines


def config_to_yaml(
    cfg: RobotConfig,
    header_lines: list[str] | None = None,
    obs_annotations: dict[str, list[str]] | None = None,
    act_annotations: dict[str, list[str]] | None = None,
    obs_candidates: list[str] | None = None,
    act_candidates: list[str] | None = None,
) -> str:
    """Serialize a :class:`RobotConfig` to a commented ``robot_config.yaml``.

    This is the inverse of :func:`load_config`: the *uncommented* lines of
    the returned text parse back to an equivalent ``RobotConfig`` (the
    round-trip guarantee relied on by ``rosbag2lerobot scaffold``). Comments —
    header notes, per-feature annotations, and commented-out candidate
    blocks — are interleaved by hand because dataclasses carry no comment
    metadata, while the feature *values* go through ``yaml.safe_dump`` so
    they parse cleanly on reload.

    Args:
        cfg: The validated configuration to serialize. Validation (duplicate
            keys, dtypes) has already run via ``RobotConfig.__post_init__``.
        header_lines: Optional comment lines (without the leading ``#``)
            emitted at the very top of the file.
        obs_annotations: ``{key: [comment, ...]}`` annotations placed above
            each observation entry (e.g. measured fps, decoder availability).
        act_annotations: Same as *obs_annotations* but for actions.
        obs_candidates: Pre-rendered comment lines (already including any
            leading ``# ``) emitted after the observation entries — used for
            commented-out no-decoder candidates.
        act_candidates: Pre-rendered comment lines emitted after the
            ``actions:`` key — used for commented-out action candidates.

    Returns:
        The full YAML document as a single string ending in a newline.
    """
    lines: list[str] = []

    for hl in header_lines or []:
        lines.append(f"# {hl}" if hl else "#")
    if header_lines:
        lines.append("")

    # Top-level scalars. Quote string values via safe_dump for safety.
    lines.append(f"robot_type: {_yaml_scalar(cfg.robot_type)}")
    lines.append(f"fps: {cfg.fps}")
    lines.append(f"task: {_yaml_scalar(cfg.task)}")
    if cfg.repo_id is not None:
        lines.append(f"repo_id: {_yaml_scalar(cfg.repo_id)}")
    lines.append("")

    # Observations
    lines.append("observations:")
    lines.extend(_emit_mapping_block(cfg.observations, obs_annotations))
    for cand in obs_candidates or []:
        lines.append(cand)
    lines.append("")

    # Actions — always present (load_config requires the key) even when empty.
    if cfg.actions:
        lines.append("actions:")
        lines.extend(_emit_mapping_block(cfg.actions, act_annotations))
    else:
        lines.append("actions: []")
    for cand in act_candidates or []:
        lines.append(cand)
    lines.append("")

    # Resampling — only emit when it differs from the defaults so the file
    # stays minimal; load_config falls back to ResamplingConfig() otherwise.
    default_res = ResamplingConfig()
    if cfg.resampling != default_res:
        lines.append("resampling:")
        lines.append(f"  default_policy: {cfg.resampling.default_policy}")
        lines.append(f"  tolerance_ms: {cfg.resampling.tolerance_ms}")
        if cfg.resampling.trim_to_valid != default_res.trim_to_valid:
            lines.append(
                f"  trim_to_valid: {str(cfg.resampling.trim_to_valid).lower()}"
            )
        if cfg.resampling.align_to_required != default_res.align_to_required:
            lines.append(
                f"  align_to_required: {str(cfg.resampling.align_to_required).lower()}"
            )
        if cfg.resampling.max_stamp_delay_ms is not None:
            lines.append(f"  max_stamp_delay_ms: {cfg.resampling.max_stamp_delay_ms}")
        lines.append("")

    # Timestamps — same rule: only emitted when it differs from the default, so
    # a scaffolded config stays minimal but a hand-tuned one round-trips.
    if cfg.timestamps != TimestampsConfig():
        lines.append("timestamps:")
        skew = cfg.timestamps.max_header_receive_skew_ms
        lines.append(
            f"  max_header_receive_skew_ms: {'null' if skew is None else skew}"
        )
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"
