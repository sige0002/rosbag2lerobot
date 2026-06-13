"""Robot configuration loader and validator for bagel.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


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

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("Feature mapping 'key' must not be empty")
        if not self.topic:
            raise ValueError("Feature mapping 'topic' must not be empty")
        if not self.msg_type:
            raise ValueError("Feature mapping 'msg_type' must not be empty")
        if self.max_stamp_delay_ms is not None and self.max_stamp_delay_ms < 0:
            raise ValueError("max_stamp_delay_ms must be non-negative")
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
    """

    robot_type: str
    fps: int
    task: str
    observations: list[FeatureMapping]
    actions: list[FeatureMapping]
    repo_id: Optional[str] = None
    custom_msgs: list[CustomMsgDef] = field(default_factory=list)
    resampling: ResamplingConfig = field(default_factory=ResamplingConfig)

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
    allowed = {f.name for f in FeatureMapping.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in raw.items() if k in allowed}
    if filtered.get("max_stamp_delay_ms") is not None:
        filtered["max_stamp_delay_ms"] = float(filtered["max_stamp_delay_ms"])
    return FeatureMapping(**filtered)


def _parse_custom_msg(raw: dict[str, Any]) -> CustomMsgDef:
    """Create a ``CustomMsgDef`` from a raw YAML dict."""
    return CustomMsgDef(
        msg_file=raw.get("msg_file", ""),
        package=raw.get("package", ""),
    )


def _parse_resampling(raw: dict[str, Any] | None) -> ResamplingConfig:
    """Create a ``ResamplingConfig`` from a raw YAML dict, using defaults for missing keys."""
    if raw is None:
        return ResamplingConfig()
    max_delay = raw.get("max_stamp_delay_ms")
    return ResamplingConfig(
        default_policy=raw.get("default_policy", "hold"),
        tolerance_ms=float(raw.get("tolerance_ms", 50.0)),
        trim_to_valid=bool(raw.get("trim_to_valid", True)),
        max_stamp_delay_ms=None if max_delay is None else float(max_delay),
        align_to_required=bool(raw.get("align_to_required", True)),
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

    return RobotConfig(
        robot_type=robot_type,
        fps=fps,
        task=task,
        repo_id=repo_id,
        observations=observations,
        actions=actions,
        custom_msgs=custom_msgs,
        resampling=resampling,
    )
