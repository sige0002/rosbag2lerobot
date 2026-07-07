"""Rosbag2 reader for rosbag2lerobot.

Reads ROS2 bag files using the ``rosbags`` library (no rclpy dependency).
Supports both SQLite3 (``.db3``) and MCAP storage backends.  Handles
custom message type registration from ``.msg`` files.

Main classes and functions:

- ``BagReader``       -- Context-managed reader for a single bag.
- ``discover_bags()`` -- Scan a directory tree for bag directories.
- ``TopicInfo``       -- Lightweight container for topic metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

from rosbag2lerobot.config import CustomMsgDef, RobotConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type-system helpers
# ---------------------------------------------------------------------------


def _register_custom_types(
    typestore: Any,
    custom_msgs: list[CustomMsgDef],
    base_dir: Path | None = None,
) -> None:
    """Register custom .msg definitions with the rosbags type store.

    Parameters
    ----------
    typestore:
        A rosbags ``Typestore`` instance (from ``get_typestore``).
    custom_msgs:
        List of ``CustomMsgDef`` entries from the robot config.
    base_dir:
        Base directory to resolve relative ``msg_file`` paths against.
        Defaults to the current working directory.
    """
    if base_dir is None:
        base_dir = Path.cwd()

    for cm in custom_msgs:
        msg_path = Path(cm.msg_file)
        if not msg_path.is_absolute():
            msg_path = base_dir / msg_path

        if not msg_path.exists():
            raise FileNotFoundError(
                f"Custom message file not found: {msg_path} (package={cm.package})"
            )

        msg_text = msg_path.read_text()
        # get_types_from_msg expects (msg_def, msg_type_name)
        # msg_type_name should be like "package/msg/TypeName"
        type_name = f"{cm.package}/msg/{msg_path.stem}"
        add_types = get_types_from_msg(msg_text, type_name)
        typestore.register(add_types)
        logger.info("Registered custom type: %s from %s", type_name, msg_path)


def _force_register(typestore: Any, types_dict: dict[str, Any]) -> None:
    """Register *types_dict* with *typestore*, overriding any conflicting type.

    ``typestore.register`` refuses to re-register a type whose fielddefs
    differ from the existing entry. Since the bag-embedded definition is
    authoritative for the data we are about to deserialize, we drop stale
    entries from the three internal dicts (``fielddefs``, ``types``,
    ``cache``) and then register fresh so the generated dataclass and CDR
    (de)serializer code reflect the new schema.
    """
    for tname in types_dict:
        typestore.fielddefs.pop(tname, None)
        typestore.types.pop(tname, None)
        typestore.cache.pop(tname, None)
    typestore.register(types_dict)


def _override_types_from_bag(typestore: Any, connections: list[Any]) -> None:
    """Override typestore definitions with bag-embedded msgdefs.

    rosbag2 stores the authoritative message definition for each connection
    in ``connection.msgdef.data``. When a locally-registered custom .msg file
    drifts from the definition the bag was recorded with, CDR deserialization
    fails with cryptic buffer/alignment errors. To make the reader robust, we
    re-register each connection's type from the bag's own embedded definition.

    Standard types provided by the default store (e.g. ``sensor_msgs/...``)
    are also overridden, which is safe because the bag's definition matches
    for standard types in practice.
    """
    processed: set[str] = set()
    for conn in connections:
        msgtype = conn.msgtype
        if msgtype in processed:
            continue
        processed.add(msgtype)

        msgdef = getattr(conn, "msgdef", None)
        if msgdef is None:
            continue
        data = (getattr(msgdef, "data", "") or "").strip()
        if not data:
            continue
        fmt = getattr(getattr(msgdef, "format", None), "name", None)
        if fmt != "MSG":
            # IDL-format msgdefs are rare in rosbag2 MCAP; skip for now.
            logger.debug("Skipping non-MSG msgdef for %s (format=%s)", msgtype, fmt)
            continue

        try:
            types_dict = get_types_from_msg(data, msgtype)
        except Exception as exc:
            logger.warning(
                "Could not parse embedded msgdef for %s (topic %s): %s",
                msgtype,
                conn.topic,
                exc,
            )
            continue
        if msgtype not in types_dict:
            continue

        try:
            _force_register(typestore, types_dict)
            logger.debug(
                "Overrode type %s from bag msgdef (topic %s)", msgtype, conn.topic
            )
        except Exception as exc:
            logger.warning(
                "Failed to register bag-embedded type %s (topic %s): %s",
                msgtype,
                conn.topic,
                exc,
            )


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def extract_header_stamp_ns(msg: Any) -> int | None:
    """Extract a ROS ``std_msgs/Header`` timestamp from *msg* in nanoseconds.

    Reads ``msg.header.stamp.sec`` and ``msg.header.stamp.nanosec`` and
    combines them into a single UNIX nanosecond timestamp. This is the
    timestamp the message producer assigned, as opposed to the bag receive
    time, and is used by callers to detect stale latched messages (e.g. a
    value left over from a TRANSIENT_LOCAL QoS queue whose header time lags
    far behind the receive time).

    Parameters
    ----------
    msg:
        A deserialized ROS2 message. May or may not carry a ``header``.

    Returns
    -------
    int | None
        ``sec * 1_000_000_000 + nanosec`` if a header stamp is present and
        non-zero. Returns ``None`` when the message has no ``header``/``stamp``,
        when the fields cannot be read, or when both ``sec`` and ``nanosec``
        are 0 (an unset stamp). Never raises.
    """
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is None or nanosec is None:
        return None
    try:
        sec_i = int(sec)
        nanosec_i = int(nanosec)
    except (TypeError, ValueError):
        return None
    if sec_i == 0 and nanosec_i == 0:
        return None
    return sec_i * 1_000_000_000 + nanosec_i


# ---------------------------------------------------------------------------
# Topic info
# ---------------------------------------------------------------------------


@dataclass
class TopicInfo:
    """Metadata about a single topic in a bag."""

    msg_type: str
    count: int


# ---------------------------------------------------------------------------
# BagReader
# ---------------------------------------------------------------------------


class BagReader:
    """High-level reader for a single ROS2 bag.

    Parameters
    ----------
    bag_path:
        Path to a bag directory (containing ``metadata.yaml``) or to the
        ``metadata.yaml`` file itself.
    config:
        Robot configuration – used to verify topics and register custom
        message types.
    typestore:
        Optional pre-built rosbags typestore.  If *None* a default
        ROS2 Humble store is created.
    """

    def __init__(
        self,
        bag_path: str | Path,
        config: RobotConfig,
        typestore: Any | None = None,
    ) -> None:
        self.bag_path = _resolve_bag_path(bag_path)
        self.config = config

        # Build typestore
        if typestore is None:
            self._typestore = get_typestore(Stores.ROS2_HUMBLE)
        else:
            self._typestore = typestore

        # Register custom types if any
        if config.custom_msgs:
            _register_custom_types(
                self._typestore,
                config.custom_msgs,
            )

        # Open reader (context-managed internally)
        self._reader = Reader(self.bag_path)
        self._reader.open()

        # Make the bag's embedded message definitions authoritative. This
        # protects against drift between local .msg files and what the bag
        # was actually recorded with (e.g. a firmware-specific message variant).
        _override_types_from_bag(self._typestore, self._reader.connections)

        # Verify expected topics
        self._topic_info = self._build_topic_info()
        self._verify_topics()

    # ----- public API -----

    @property
    def typestore(self) -> Any:
        return self._typestore

    def get_topics_info(self) -> dict[str, TopicInfo]:
        """Return mapping of topic name to ``TopicInfo``."""
        return dict(self._topic_info)

    def get_time_range(self) -> tuple[int, int]:
        """Return ``(start_ns, end_ns)`` for the bag.

        Returns nanosecond UNIX timestamps.  If the bag metadata reports
        zero duration (common with older / minimal bags), the range is
        computed by scanning all message timestamps.
        """
        duration = self._reader.duration
        start = self._reader.start_time
        end = start + duration

        # Validate: if start_time is 0 or duration looks implausible,
        # fall back to scanning actual message timestamps.
        if start > 0 and duration > 1_000_000:  # >1ms is plausible
            return (start, end)

        # Fallback: scan message timestamps to determine range
        min_ts = None
        max_ts = None
        for conn, timestamp, _rawdata in self._reader.messages():
            if min_ts is None or timestamp < min_ts:
                min_ts = timestamp
            if max_ts is None or timestamp > max_ts:
                max_ts = timestamp
        if min_ts is not None and max_ts is not None:
            return (min_ts, max_ts)
        return (start, end)

    def iter_messages(
        self,
        topics: list[str] | None = None,
    ) -> Generator[tuple[str, int, Any], None, None]:
        """Yield ``(topic, timestamp_ns, deserialized_msg)`` in time order.

        Parameters
        ----------
        topics:
            If given, only yield messages for these topics.  Otherwise yield
            all messages.
        """
        connections = self._reader.connections
        if topics is not None:
            topic_set = set(topics)
            connections = [c for c in connections if c.topic in topic_set]

        for conn, timestamp, rawdata in self._reader.messages(connections=connections):
            msg = self._typestore.deserialize_cdr(rawdata, conn.msgtype)
            yield conn.topic, timestamp, msg

    def close(self) -> None:
        """Close the underlying reader."""
        self._reader.close()

    def __enter__(self) -> "BagReader":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ----- internals -----

    def _build_topic_info(self) -> dict[str, TopicInfo]:
        """Aggregate per-topic metadata from all connections in the bag."""
        info: dict[str, TopicInfo] = {}
        for conn in self._reader.connections:
            topic = conn.topic
            msgtype = conn.msgtype
            # Count messages for this connection
            count = conn.msgcount
            if topic in info:
                info[topic] = TopicInfo(
                    msg_type=info[topic].msg_type,
                    count=info[topic].count + count,
                )
            else:
                info[topic] = TopicInfo(msg_type=msgtype, count=count)
        return info

    def _verify_topics(self) -> None:
        """Warn if any configured topics are missing from the bag.

        Topics marked as ``optional`` in the config only emit a debug-level
        message when absent.  Required (non-optional) topics emit a warning.
        """
        available = set(self._topic_info.keys())
        optional_topics = self.config.optional_topics
        for topic in self.config.all_topics:
            if topic not in available:
                if topic in optional_topics:
                    logger.debug(
                        "Optional topic '%s' not found in bag (skipped).",
                        topic,
                    )
                else:
                    logger.warning(
                        "Topic '%s' not found in bag. Available: %s",
                        topic,
                        sorted(available),
                    )


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------


def _resolve_bag_path(bag_path: str | Path) -> Path:
    """Resolve a bag path to the bag directory.

    Accepts:
    - A directory containing ``metadata.yaml``
    - A path to ``metadata.yaml`` directly
    - A directory containing ``.db3`` or ``.mcap`` files (rosbags auto-detects)
    """
    p = Path(bag_path)
    if not p.exists():
        raise FileNotFoundError(f"Bag path does not exist: {p}")

    if p.is_file():
        if p.name == "metadata.yaml":
            return p.parent
        raise ValueError(f"Expected a bag directory or metadata.yaml, got file: {p}")

    # It's a directory – check it looks like a bag
    if (p / "metadata.yaml").exists():
        return p

    # Look for storage files
    db3_files = list(p.glob("*.db3"))
    mcap_files = list(p.glob("*.mcap"))
    if db3_files or mcap_files:
        return p

    raise ValueError(
        f"Directory does not appear to be a rosbag2 bag: {p}. "
        "Expected metadata.yaml or .db3/.mcap files."
    )


def discover_bags(bags_dir: str | Path) -> list[Path]:
    """Discover all bag directories under *bags_dir*.

    Each discovered directory is expected to be a self-contained bag
    (containing ``metadata.yaml`` or storage files).

    The bags are returned sorted by name so that episode ordering is
    deterministic.
    """
    bags_dir = Path(bags_dir)
    if not bags_dir.is_dir():
        raise FileNotFoundError(f"Bags directory not found: {bags_dir}")

    # If the path itself is a bag, return just that
    try:
        _resolve_bag_path(bags_dir)
        return [bags_dir]
    except ValueError:
        pass

    # Otherwise scan subdirectories
    candidates: list[Path] = []
    for child in sorted(bags_dir.iterdir()):
        if not child.is_dir():
            continue
        try:
            resolved = _resolve_bag_path(child)
            candidates.append(resolved)
        except (ValueError, FileNotFoundError):
            continue

    if not candidates:
        raise FileNotFoundError(f"No bag directories found under: {bags_dir}")

    return candidates
