"""Shared fixtures.

``tiny_bag`` writes a minimal, self-contained rosbag2 (two JointState topics,
no images so no ffmpeg is involved) with full control over the relationship
between each message's ``header.stamp`` and its bag receive time. Tests that
need a bag with a *broken* clock — or just a small one that converts in
milliseconds — build it here instead of depending on the gitignored
``bagdata/`` tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

# ~2023 in UNIX nanoseconds; arbitrary but fixed so tests are deterministic.
BAG_START_NS = 1_700_000_000_000_000_000

STATE_TOPIC = "/joint_states"
ACTION_TOPIC = "/joint_commands"
JOINT_NAMES = ["j0", "j1", "j2"]


def write_tiny_bag(
    bag_path: Path,
    *,
    n_messages: int = 30,
    rate_hz: float = 30.0,
    header_offset_ns: int = 0,
    offset_from_index: int = 0,
    offset_topics: tuple[str, ...] = (STATE_TOPIC, ACTION_TOPIC),
    unset_header_stamp: bool = False,
) -> Path:
    """Write a two-topic JointState bag and return its path.

    Args:
        bag_path: Directory to create (parents are created as needed).
        n_messages: Messages written per topic.
        rate_hz: Publish rate used for both receive time and header stamp.
        header_offset_ns: Added to the header stamp only, so the header
            diverges from the receive time by exactly this much. ``0`` writes
            a bag whose clocks agree.
        offset_from_index: First message index the offset applies to; earlier
            messages keep a matching header stamp. Lets a test place the
            divergence in the middle of a bag.
        offset_topics: Topics the offset applies to.
        unset_header_stamp: Write ``sec=0, nanosec=0`` headers, which is how a
            publisher that never stamps its messages shows up in a bag.

    Returns:
        ``bag_path``, for chaining.
    """
    bag_path.parent.mkdir(parents=True, exist_ok=True)
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    JointState = typestore.types["sensor_msgs/msg/JointState"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]

    period_ns = int(1e9 / rate_hz)
    with Writer(bag_path, version=9) as writer:
        connections = {
            topic: writer.add_connection(
                topic, "sensor_msgs/msg/JointState", typestore=typestore
            )
            for topic in (STATE_TOPIC, ACTION_TOPIC)
        }
        for i in range(n_messages):
            recv_ns = BAG_START_NS + i * period_ns
            for topic, conn in connections.items():
                stamp_ns = recv_ns
                if topic in offset_topics and i >= offset_from_index:
                    stamp_ns += header_offset_ns
                if unset_header_stamp:
                    stamp_ns = 0
                msg = JointState(
                    header=Header(
                        stamp=Time(
                            sec=int(stamp_ns // 1_000_000_000),
                            nanosec=int(stamp_ns % 1_000_000_000),
                        ),
                        frame_id="base_link",
                    ),
                    name=np.array(JOINT_NAMES),
                    position=np.array([i * 0.01] * 3, dtype=np.float64),
                    velocity=np.zeros(3, dtype=np.float64),
                    effort=np.zeros(3, dtype=np.float64),
                )
                writer.write(
                    conn,
                    recv_ns,
                    typestore.serialize_cdr(msg, "sensor_msgs/msg/JointState"),
                )
    return bag_path


@pytest.fixture
def tiny_bag(tmp_path: Path) -> Callable[..., Path]:
    """Return a factory that writes tiny bags under ``tmp_path``."""

    def _factory(name: str = "bag", **kwargs: object) -> Path:
        return write_tiny_bag(tmp_path / name, **kwargs)  # type: ignore[arg-type]

    return _factory


def write_tf_bag(
    bag_path: Path,
    *,
    n_messages: int = 30,
    rate_hz: float = 30.0,
    tf_header_offset_ns: int = 0,
    static_header_offset_ns: int = 0,
    unset_tf_stamp: bool = False,
    extra_dynamic_frame: str | None = None,
    extra_dynamic_offset_ns: int = 0,
) -> Path:
    """Write a bag with joint states plus ``/tf`` and ``/tf_static``.

    The TF tree is ``base_link -> arm_link`` (static) and
    ``odom -> base_link`` (dynamic, translating along x), which is enough for
    a ``frame_from``/``frame_to`` feature to resolve. The offsets shift only
    the *header* stamps, leaving the bag receive times alone, which is what an
    unsynchronised publisher looks like on the wire.

    Args:
        bag_path: Directory to create (parents are created as needed).
        n_messages: Messages written per topic (``/tf_static`` gets one).
        rate_hz: Publish rate for receive times and header stamps.
        tf_header_offset_ns: Added to every dynamic transform's header stamp.
        static_header_offset_ns: Added to the static transform's header stamp.
        unset_tf_stamp: Write ``sec=0, nanosec=0`` on dynamic transforms.
        extra_dynamic_frame: When set, publish a second dynamic transform
            ``base_link -> <frame>`` on ``/tf``. Stands in for another sensor
            on the robot that no configured feature looks through.
        extra_dynamic_offset_ns: Header-stamp offset for that extra transform,
            so a test can skew it independently of the one in use.

    Returns:
        ``bag_path``, for chaining.
    """
    bag_path.parent.mkdir(parents=True, exist_ok=True)
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    JointState = typestore.types["sensor_msgs/msg/JointState"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]
    TFMessage = typestore.types["tf2_msgs/msg/TFMessage"]
    TransformStamped = typestore.types["geometry_msgs/msg/TransformStamped"]
    Transform = typestore.types["geometry_msgs/msg/Transform"]
    Vector3 = typestore.types["geometry_msgs/msg/Vector3"]
    Quaternion = typestore.types["geometry_msgs/msg/Quaternion"]

    def _stamp(ns: int) -> Any:
        return Time(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))

    def _transform(parent: str, child: str, stamp_ns: int, x: float) -> Any:
        return TransformStamped(
            header=Header(stamp=_stamp(stamp_ns), frame_id=parent),
            child_frame_id=child,
            transform=Transform(
                translation=Vector3(x=x, y=0.0, z=0.0),
                rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        )

    period_ns = int(1e9 / rate_hz)
    with Writer(bag_path, version=9) as writer:
        conn_state = writer.add_connection(
            STATE_TOPIC, "sensor_msgs/msg/JointState", typestore=typestore
        )
        conn_action = writer.add_connection(
            ACTION_TOPIC, "sensor_msgs/msg/JointState", typestore=typestore
        )
        conn_tf = writer.add_connection(
            "/tf", "tf2_msgs/msg/TFMessage", typestore=typestore
        )
        conn_tf_static = writer.add_connection(
            "/tf_static", "tf2_msgs/msg/TFMessage", typestore=typestore
        )

        static_msg = TFMessage(
            transforms=[
                _transform(
                    "base_link",
                    "arm_link",
                    BAG_START_NS + static_header_offset_ns,
                    0.5,
                )
            ]
        )
        writer.write(
            conn_tf_static,
            BAG_START_NS,
            typestore.serialize_cdr(static_msg, "tf2_msgs/msg/TFMessage"),
        )

        for i in range(n_messages):
            recv_ns = BAG_START_NS + i * period_ns
            joint = JointState(
                header=Header(stamp=_stamp(recv_ns), frame_id="base_link"),
                name=np.array(JOINT_NAMES),
                position=np.array([i * 0.01] * 3, dtype=np.float64),
                velocity=np.zeros(3, dtype=np.float64),
                effort=np.zeros(3, dtype=np.float64),
            )
            payload = typestore.serialize_cdr(joint, "sensor_msgs/msg/JointState")
            writer.write(conn_state, recv_ns, payload)
            writer.write(conn_action, recv_ns, payload)

            tf_stamp_ns = 0 if unset_tf_stamp else recv_ns + tf_header_offset_ns
            transforms = [_transform("odom", "base_link", tf_stamp_ns, i * 0.01)]
            if extra_dynamic_frame is not None:
                transforms.append(
                    _transform(
                        "base_link",
                        extra_dynamic_frame,
                        recv_ns + extra_dynamic_offset_ns,
                        i * 0.02,
                    )
                )
            tf_msg = TFMessage(transforms=transforms)
            writer.write(
                conn_tf,
                recv_ns,
                typestore.serialize_cdr(tf_msg, "tf2_msgs/msg/TFMessage"),
            )
    return bag_path


@pytest.fixture
def tf_bag(tmp_path: Path) -> Callable[..., Path]:
    """Return a factory that writes TF-carrying bags under ``tmp_path``."""

    def _factory(name: str = "bag", **kwargs: object) -> Path:
        return write_tf_bag(tmp_path / name, **kwargs)  # type: ignore[arg-type]

    return _factory


def tf_config_yaml(path: Path, *, fps: int = 10, extra: str = "") -> Path:
    """Write a config with a TF feature matching :func:`write_tf_bag`."""
    path.write_text(
        f"""robot_type: tiny_tf
fps: {fps}
task: tiny tf task

observations:
  - key: observation.state
    topic: {STATE_TOPIC}
    msg_type: sensor_msgs/msg/JointState
    selector: position
    dtype: float32
  - key: observation.ee_pose
    topic: /tf
    msg_type: tf2_msgs/msg/TFMessage
    frame_from: base_link
    frame_to: odom

actions:
  - key: action
    topic: {ACTION_TOPIC}
    msg_type: sensor_msgs/msg/JointState
    selector: position
    dtype: float32

resampling:
  default_policy: nearest
  tolerance_ms: 100.0
{extra}"""
    )
    return path


def tiny_config_yaml(
    path: Path,
    *,
    fps: int = 10,
    stamp_source: str = "header",
    extra: str = "",
) -> Path:
    """Write a robot_config.yaml matching :func:`write_tiny_bag` and return it.

    Args:
        path: File to write.
        fps: Target dataset fps.
        stamp_source: ``header`` or ``receive``, applied to both features.
        extra: Raw YAML appended verbatim (e.g. a ``timestamps:`` block).
    """
    path.write_text(
        f"""robot_type: tiny
fps: {fps}
task: tiny task

observations:
  - key: observation.state
    topic: {STATE_TOPIC}
    msg_type: sensor_msgs/msg/JointState
    selector: position
    dtype: float32
    stamp_source: {stamp_source}

actions:
  - key: action
    topic: {ACTION_TOPIC}
    msg_type: sensor_msgs/msg/JointState
    selector: position
    dtype: float32
    stamp_source: {stamp_source}

resampling:
  default_policy: nearest
  tolerance_ms: 100.0
{extra}"""
    )
    return path
