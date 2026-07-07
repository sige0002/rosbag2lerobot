"""Built-in decoders for common ROS2 message types.

All decoders output ``np.ndarray`` with dtype ``float32`` (except
``std_msgs/msg/String`` which returns ``str``).  Decoders accept an
optional ``selector`` list for sub-field extraction and a ``config`` dict
for unit conversion and padding.

Supported types include:

- ``sensor_msgs/msg/JointState`` -- joint positions / velocities / efforts
- ``geometry_msgs/msg/Twist``, ``TwistStamped`` -- linear + angular velocity
- ``geometry_msgs/msg/PoseStamped`` -- position + orientation
- ``nav_msgs/msg/Odometry`` -- pose + twist
- ``sensor_msgs/msg/Imu`` -- quaternion + gyro + accelerometer
- ``sensor_msgs/msg/Joy`` -- joystick axes + buttons
- ``std_msgs/msg/Float32``, ``Float64``, ``Float32MultiArray``, ``String``
- ``geometry_msgs/msg/Pose`` -- position + orientation (unstamped)
- ``geometry_msgs/msg/WrenchStamped`` -- force + torque (6D)
- ``trajectory_msgs/msg/JointTrajectory`` -- joint trajectory waypoints

Field aliases (e.g. ``"pos"`` -> ``"position"``) are supported for common
message types.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from rosbag2lerobot.decoders import register_decoder
from rosbag2lerobot.transforms import quat_xyzw_to_euler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Field aliases for flexible field name access
# ---------------------------------------------------------------------------
FIELD_ALIASES: dict[str, dict[str, str]] = {
    "sensor_msgs/msg/JointState": {
        "pos": "position",
        "vel": "velocity",
        "eff": "effort",
    },
    "geometry_msgs/msg/Twist": {
        "lin": "linear",
        "ang": "angular",
    },
    "geometry_msgs/msg/TwistStamped": {
        "lin": "linear",
        "ang": "angular",
    },
}


def _resolve_alias(msg_type: str, field: str) -> str:
    """Resolve a field alias to its canonical name."""
    aliases = FIELD_ALIASES.get(msg_type, {})
    return aliases.get(field, field)


def _apply_unit_conversion(values: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Apply unit conversion if specified in config."""
    conversion = config.get("unit_conversion")
    if conversion is None or conversion == 1.0 or conversion == "":
        return values
    if conversion == "rad2deg":
        return np.rad2deg(values).astype(np.float32)
    if conversion == "deg2rad":
        return np.deg2rad(values).astype(np.float32)
    if isinstance(conversion, (int, float)) and conversion != 1.0:
        return (values * conversion).astype(np.float32)
    logger.warning("Unknown unit_conversion: %s, skipping", conversion)
    return values


def _to_float_array(data: Any) -> np.ndarray:
    """Convert various data types to a float32 numpy array."""
    if isinstance(data, np.ndarray):
        return data.astype(np.float32)
    return np.array(data, dtype=np.float32)


def _finalize(values: Any, config: dict[str, Any]) -> np.ndarray:
    """Cast *values* to float32 and apply the config's unit conversion.

    *values* may be a list, tuple, or numpy array. This is the standard
    return path for all scalar / vector decoders; the flat structure
    (convert once, then apply the unit rule) keeps the per-decoder code
    short and makes unit handling consistent across the registry.
    """
    arr = (
        values
        if isinstance(values, np.ndarray)
        else np.asarray(values, dtype=np.float32)
    )
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return _apply_unit_conversion(arr, config)


def _get_nested_attr(obj: Any, dotted_path: str) -> Any:
    """Get a nested attribute using dot notation, e.g. 'linear.x'."""
    parts = dotted_path.split(".")
    current = obj
    for part in parts:
        current = getattr(current, part)
    return current


def _parse_array_selector(selector_str: str) -> tuple[str, int | None]:
    """Parse ``"field[idx]"`` into ``(field, idx)``; ``"field"`` returns ``(field, None)``.

    Negative indices are allowed (Python list semantics). Raises ``ValueError`` on
    malformed brackets so configs fail loudly at conversion time instead of
    silently producing wrong-shape output vectors.
    """
    if "[" not in selector_str:
        return selector_str, None
    if not selector_str.endswith("]"):
        raise ValueError(f"Invalid array selector (missing ']'): {selector_str!r}")
    field_name, rest = selector_str.split("[", 1)
    index_str = rest[:-1]
    if not field_name:
        raise ValueError(f"Invalid array selector (empty field name): {selector_str!r}")
    try:
        return field_name, int(index_str)
    except ValueError as e:
        raise ValueError(f"Invalid array index in selector: {selector_str!r}") from e


def _euler_convention(selector_str: str) -> tuple[str, str] | None:
    """Detect a trailing ``euler_<conv>`` token in a selector.

    Args:
        selector_str: A single selector entry, e.g. ``"orientation.euler_xyz"``,
            ``"pose.orientation.euler_zyx"``, or just ``"euler_xyz"``.

    Returns:
        ``(prefix, convention)`` when the last dotted segment is ``euler_<conv>``
        — ``prefix`` is the dotted path to the quaternion struct (``""`` when the
        token stands alone) and ``convention`` is ``"xyz"`` / ``"zyx"``.
        ``None`` for any non-euler selector.
    """
    last = selector_str.rsplit(".", 1)[-1]
    if not last.startswith("euler_"):
        return None
    convention = last[len("euler_") :]
    prefix = selector_str[: len(selector_str) - len(last)].rstrip(".")
    return prefix, convention


def _extract_by_selector(msg: Any, selector_str: str) -> list[float]:
    """Resolve one selector entry against *msg*, returning a flat list of floats.

    Handles, in order:
      * trailing ``euler_<conv>`` -- e.g. ``"orientation.euler_xyz"`` or
        ``"pose.orientation.euler_zyx"``: resolve the prefix to a quaternion
        struct (``.x/.y/.z/.w``) and convert to ``[roll, pitch, yaw]`` (3 floats).
      * ``"field[idx]"`` -- single element via :func:`_parse_array_selector`.
      * ``"a.b.c"``      -- nested attribute via :func:`_get_nested_attr`.
      * ``"field"``      -- full-array iteration, or single-value wrap for scalars.
    """
    euler = _euler_convention(selector_str)
    if euler is not None:
        prefix, convention = euler
        quat = _get_nested_attr(msg, prefix) if prefix else msg
        roll, pitch, yaw = quat_xyzw_to_euler(
            float(quat.x),
            float(quat.y),
            float(quat.z),
            float(quat.w),
            convention=convention,
        )
        return [roll, pitch, yaw]

    if "[" in selector_str:
        field_name, index = _parse_array_selector(selector_str)
        field_data = getattr(msg, field_name, None)
        if field_data is None:
            raise ValueError(f"Message has no field {field_name!r}")
        try:
            return [float(field_data[index])]
        except (IndexError, TypeError) as e:
            raise ValueError(
                f"Cannot access index {index} of field {field_name!r}: {e}"
            ) from e

    if "." in selector_str:
        return [float(_get_nested_attr(msg, selector_str))]

    field_data = getattr(msg, selector_str, None)
    if field_data is None:
        raise ValueError(f"Message has no field {selector_str!r}")
    if hasattr(field_data, "__iter__") and not isinstance(field_data, (str, bytes)):
        return [float(v) for v in field_data]
    return [float(field_data)]


# ---------------------------------------------------------------------------
# sensor_msgs/msg/JointState
# ---------------------------------------------------------------------------
@register_decoder("sensor_msgs/msg/JointState")
def decode_joint_state(
    msg: Any, selector: list[str] | None, config: dict[str, Any]
) -> np.ndarray:
    """Decode JointState messages.

    Selector grammar (checked in this order per entry):
      * ``"field[idx]"``       -- positional index (e.g. ``"position[0]"``).
      * ``"field.*"``           -- wildcard, all values of that field.
      * ``"field.joint_name"`` -- name-based indexing via ``msg.name``.
      * ``"field"``             -- full array for that field.

    For dual-arm robots, the config may contain:
        - "num_joints_per_arm": int (default 7)
        - "pad_to": int - pad each arm to this many joints

    If no selector is provided, returns all position values.
    """
    msg_type = "sensor_msgs/msg/JointState"
    joint_names = (
        list(msg.name) if hasattr(msg, "name") and msg.name is not None else []
    )

    if selector is None:
        values = _to_float_array(msg.position)
        return _apply_unit_conversion(values, config)

    results: list[float] = []
    for sel in selector:
        if "[" in sel:
            field_name, index = _parse_array_selector(sel)
            field_name = _resolve_alias(msg_type, field_name)
            field_data = getattr(msg, field_name, None)
            if field_data is None:
                raise ValueError(
                    f"JointState has no field '{field_name}' "
                    f"(available: position, velocity, effort)"
                )
            try:
                results.append(float(list(field_data)[index]))
            except (IndexError, TypeError) as e:
                raise ValueError(
                    f"Cannot access index {index} of field '{field_name}': {e}"
                ) from e
            continue
        parts = sel.split(".", 1)
        if len(parts) == 1:
            # Single field name like "position" -> return all values for that field
            field_name = _resolve_alias(msg_type, parts[0])
            field_data = getattr(msg, field_name, None)
            if field_data is None:
                raise ValueError(
                    f"JointState has no field '{field_name}' "
                    f"(available: position, velocity, effort)"
                )
            results.extend(float(v) for v in field_data)
            continue
        field_name, joint_name = parts
        field_name = _resolve_alias(msg_type, field_name)

        field_data = getattr(msg, field_name, None)
        if field_data is None:
            raise ValueError(
                f"JointState has no field '{field_name}' "
                f"(available: position, velocity, effort)"
            )

        field_array = list(field_data)

        if joint_name == "*":
            # Wildcard: return all values for this field
            results.extend(float(v) for v in field_array)
            continue

        if joint_name not in joint_names:
            raise ValueError(
                f"Joint '{joint_name}' not found in message. "
                f"Available joints: {joint_names}"
            )
        idx = joint_names.index(joint_name)
        results.append(float(field_array[idx]))

    values = np.array(results, dtype=np.float32)

    # Pad if configured
    pad_to = config.get("pad_to")
    if pad_to is not None and len(values) < pad_to:
        values = np.pad(
            values, (0, pad_to - len(values)), mode="constant", constant_values=0.0
        ).astype(np.float32)

    return _apply_unit_conversion(values, config)


# ---------------------------------------------------------------------------
# geometry_msgs/msg/Twist
# ---------------------------------------------------------------------------
@register_decoder("geometry_msgs/msg/Twist")
def decode_twist(
    msg: Any, selector: list[str] | None, config: dict[str, Any]
) -> np.ndarray:
    """Decode Twist messages.

    Selector format: ["linear.x", "angular.z", ...]
    If no selector, returns [linear.x, linear.y, linear.z, angular.x, angular.y, angular.z].
    """
    if selector is None:
        return _finalize(
            [
                msg.linear.x,
                msg.linear.y,
                msg.linear.z,
                msg.angular.x,
                msg.angular.y,
                msg.angular.z,
            ],
            config,
        )

    results: list[float] = []
    for sel in selector:
        resolved_parts = sel.split(".", 1)
        if len(resolved_parts) == 2:
            resolved_parts[0] = _resolve_alias(
                "geometry_msgs/msg/Twist", resolved_parts[0]
            )
            sel = ".".join(resolved_parts)
        results.append(float(_get_nested_attr(msg, sel)))
    return _finalize(results, config)


# ---------------------------------------------------------------------------
# geometry_msgs/msg/WrenchStamped
# ---------------------------------------------------------------------------
@register_decoder("geometry_msgs/msg/WrenchStamped")
def decode_wrench_stamped(
    msg: Any, selector: list[str] | None, config: dict[str, Any]
) -> np.ndarray:
    """Decode WrenchStamped messages.

    Selector format: ["wrench.force.x", "wrench.torque.z", ...]
    If no selector, returns [force.x, force.y, force.z, torque.x, torque.y, torque.z].
    """
    if selector is None:
        w = msg.wrench
        return _finalize(
            [w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z],
            config,
        )
    return _finalize(
        [float(_get_nested_attr(msg, sel)) for sel in selector],
        config,
    )


# ---------------------------------------------------------------------------
# trajectory_msgs/msg/JointTrajectory
# ---------------------------------------------------------------------------
@register_decoder("trajectory_msgs/msg/JointTrajectory")
def decode_joint_trajectory(
    msg: Any, selector: list[str] | None, config: dict[str, Any]
) -> np.ndarray:
    """Decode JointTrajectory messages.

    Extracts the last trajectory point's positions by default.
    Selector format: ["positions", "velocities", "accelerations", "effort"]
    If selector is a single field like "positions", returns all values for that field.
    """
    if not msg.points:
        # Empty trajectory -- return zeros matching joint_names length
        n_joints = (
            len(msg.joint_names)
            if hasattr(msg, "joint_names") and msg.joint_names
            else 0
        )
        return np.zeros(n_joints, dtype=np.float32)

    # Use the last point (the target) by default
    point = msg.points[-1]

    if selector is None:
        values = (
            _to_float_array(point.positions)
            if point.positions is not None and len(point.positions) > 0
            else np.array([], dtype=np.float32)
        )
        return _apply_unit_conversion(values, config)

    results: list[float] = []
    for sel in selector:
        field_data = getattr(point, sel, None)
        if field_data is not None and hasattr(field_data, "__iter__"):
            results.extend(float(v) for v in field_data)
        elif field_data is not None:
            results.append(float(field_data))

    values = np.array(results, dtype=np.float32)
    return _apply_unit_conversion(values, config)


# ---------------------------------------------------------------------------
# geometry_msgs/msg/TwistStamped
# ---------------------------------------------------------------------------
@register_decoder("geometry_msgs/msg/TwistStamped")
def decode_twist_stamped(
    msg: Any, selector: list[str] | None, config: dict[str, Any]
) -> np.ndarray:
    """Decode TwistStamped messages. Delegates to Twist decoder using msg.twist."""
    return decode_twist(msg.twist, selector, config)


# ---------------------------------------------------------------------------
# nav_msgs/msg/Odometry
# ---------------------------------------------------------------------------
@register_decoder("nav_msgs/msg/Odometry")
def decode_odometry(
    msg: Any, selector: list[str] | None, config: dict[str, Any]
) -> np.ndarray:
    """Decode Odometry messages.

    Selector format: ["pose.position.x", "twist.linear.x", ...]
    If no selector, returns pose (position xyz + orientation xyzw) + twist (linear + angular).
    """
    if selector is None:
        pose = msg.pose.pose
        twist = msg.twist.twist
        return _finalize(
            [
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
                twist.linear.x,
                twist.linear.y,
                twist.linear.z,
                twist.angular.x,
                twist.angular.y,
                twist.angular.z,
            ],
            config,
        )
    results: list[float] = []
    for sel in selector:
        results.extend(_extract_by_selector(msg, sel))
    return _finalize(results, config)


# ---------------------------------------------------------------------------
# sensor_msgs/msg/Imu
# ---------------------------------------------------------------------------
@register_decoder("sensor_msgs/msg/Imu")
def decode_imu(
    msg: Any, selector: list[str] | None, config: dict[str, Any]
) -> np.ndarray:
    """Decode IMU messages.

    Default output: float[10] = [qx, qy, qz, qw, gx, gy, gz, ax, ay, az]
    (quaternion + angular_velocity + linear_acceleration)
    """
    if selector is None:
        return _finalize(
            [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ],
            config,
        )
    results: list[float] = []
    for sel in selector:
        results.extend(_extract_by_selector(msg, sel))
    return _finalize(results, config)


# ---------------------------------------------------------------------------
# sensor_msgs/msg/Joy
# ---------------------------------------------------------------------------
@register_decoder("sensor_msgs/msg/Joy")
def decode_joy(
    msg: Any, selector: list[str] | None, config: dict[str, Any]
) -> np.ndarray:
    """Decode Joy (joystick) messages.

    Default output: concatenation of axes + buttons.
    Selector: ["axes.0", "axes.1", "buttons.2", ...] (field.index)
    """
    if selector is None:
        axes = (
            _to_float_array(msg.axes)
            if msg.axes is not None
            else np.array([], dtype=np.float32)
        )
        buttons = (
            _to_float_array(msg.buttons)
            if msg.buttons is not None
            else np.array([], dtype=np.float32)
        )
        return np.concatenate([axes, buttons]).astype(np.float32)

    results: list[float] = []
    for sel in selector:
        parts = sel.split(".", 1)
        field_name = parts[0]
        field_data = getattr(msg, field_name)
        if len(parts) == 2:
            idx = int(parts[1])
            results.append(float(field_data[idx]))
        else:
            results.extend(float(v) for v in field_data)

    return np.array(results, dtype=np.float32)


# ---------------------------------------------------------------------------
# std_msgs/msg/Float32
# ---------------------------------------------------------------------------
@register_decoder("std_msgs/msg/Float32")
def decode_float32(
    msg: Any, selector: list[str] | None, config: dict[str, Any]
) -> np.ndarray:
    """Decode Float32 scalar message."""
    return np.array([msg.data], dtype=np.float32)


# ---------------------------------------------------------------------------
# std_msgs/msg/Float64
# ---------------------------------------------------------------------------
@register_decoder("std_msgs/msg/Float64")
def decode_float64(
    msg: Any, selector: list[str] | None, config: dict[str, Any]
) -> np.ndarray:
    """Decode Float64 scalar message."""
    return np.array([msg.data], dtype=np.float32)


# ---------------------------------------------------------------------------
# std_msgs/msg/Float32MultiArray
# ---------------------------------------------------------------------------
@register_decoder("std_msgs/msg/Float32MultiArray")
def decode_float32_multi_array(
    msg: Any, selector: list[str] | None, config: dict[str, Any]
) -> np.ndarray:
    """Decode Float32MultiArray message.

    Selector: list of indices as strings, e.g. ["0", "2", "5"]
    """
    data = _to_float_array(msg.data)
    if selector is not None:
        indices = [int(s) for s in selector]
        data = data[indices]
    return data


# ---------------------------------------------------------------------------
# std_msgs/msg/String
# ---------------------------------------------------------------------------
@register_decoder("std_msgs/msg/String")
def decode_string(msg: Any, selector: list[str] | None, config: dict[str, Any]) -> str:
    """Decode String message. Returns the string data directly."""
    return str(msg.data)


# ---------------------------------------------------------------------------
# geometry_msgs/msg/PoseStamped
# ---------------------------------------------------------------------------
@register_decoder("geometry_msgs/msg/PoseStamped")
def decode_pose_stamped(
    msg: Any, selector: list[str] | None, config: dict[str, Any]
) -> np.ndarray:
    """Decode PoseStamped messages.

    Selector format: ["pose.position.x", "pose.orientation.w", ...]
    If no selector, returns [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w].
    """
    if selector is None:
        p = msg.pose.position
        o = msg.pose.orientation
        return _finalize([p.x, p.y, p.z, o.x, o.y, o.z, o.w], config)
    results: list[float] = []
    for sel in selector:
        results.extend(_extract_by_selector(msg, sel))
    return _finalize(results, config)


# ---------------------------------------------------------------------------
# geometry_msgs/msg/Pose
# ---------------------------------------------------------------------------
@register_decoder("geometry_msgs/msg/Pose")
def decode_pose(
    msg: Any, selector: list[str] | None, config: dict[str, Any]
) -> np.ndarray:
    """Decode Pose messages (unstamped).

    Selector format: ["position.x", "orientation.w", ...]
    If no selector, returns [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w].
    """
    if selector is None:
        p = msg.position
        o = msg.orientation
        return _finalize([p.x, p.y, p.z, o.x, o.y, o.z, o.w], config)
    results: list[float] = []
    for sel in selector:
        results.extend(_extract_by_selector(msg, sel))
    return _finalize(results, config)
