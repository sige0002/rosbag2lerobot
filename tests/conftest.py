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
from typing import Callable

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
