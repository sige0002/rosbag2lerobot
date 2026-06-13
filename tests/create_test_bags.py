"""Create synthetic rosbag2 files for E2E testing.

Generates realistic manipulator rosbag data:
1. manipulator_bag: 6-axis arm with JointState + CompressedImage + Twist action
2. dual_arm_bag: Dual 7-axis arm with two JointState topics + two cameras
"""

import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

TEST_BAGS_DIR = Path(__file__).parent.parent / "test_bags"


def create_manipulator_bag(output_dir: Path | None = None) -> Path:
    """Create a single-arm 6-axis manipulator rosbag2.

    Contains:
    - /joint_states (sensor_msgs/msg/JointState) at ~50Hz, 3 seconds
    - /camera/right_wrist/image_raw/compressed (sensor_msgs/msg/CompressedImage) at ~10Hz
    - /target_joint_positions (sensor_msgs/msg/JointState) at ~50Hz (action)

    Simulates a simple pick motion: joints move sinusoidally.
    """
    bag_path = (output_dir or TEST_BAGS_DIR) / "manipulator_bag"
    if bag_path.exists():
        shutil.rmtree(bag_path)

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    JointState = typestore.types["sensor_msgs/msg/JointState"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]
    CompressedImage = typestore.types["sensor_msgs/msg/CompressedImage"]

    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]
    duration_s = 3.0
    joint_hz = 50
    image_hz = 10
    start_ns = 1_700_000_000_000_000_000  # ~2023 timestamp

    with Writer(bag_path, version=9) as writer:
        conn_js = writer.add_connection(
            "/joint_states",
            "sensor_msgs/msg/JointState",
            typestore=typestore,
        )
        conn_img = writer.add_connection(
            "/camera/right_wrist/image_raw/compressed",
            "sensor_msgs/msg/CompressedImage",
            typestore=typestore,
        )
        conn_action = writer.add_connection(
            "/target_joint_positions",
            "sensor_msgs/msg/JointState",
            typestore=typestore,
        )

        # Write joint states at 50Hz
        n_joint_msgs = int(duration_s * joint_hz)
        for i in range(n_joint_msgs):
            t_s = i / joint_hz
            t_ns = start_ns + int(t_s * 1e9)
            sec = int(t_ns // 1_000_000_000)
            nanosec = int(t_ns % 1_000_000_000)

            positions = np.array(
                [math.sin(2 * math.pi * 0.5 * t_s + j * 0.3) for j in range(6)],
                dtype=np.float64,
            )
            velocities = np.array(
                [
                    2 * math.pi * 0.5 * math.cos(2 * math.pi * 0.5 * t_s + j * 0.3)
                    for j in range(6)
                ],
                dtype=np.float64,
            )
            effort = np.zeros(6, dtype=np.float64)

            header = Header(
                stamp=Time(sec=sec, nanosec=nanosec),
                frame_id="base_link",
            )
            msg = JointState(
                header=header,
                name=np.array(joint_names),
                position=positions,
                velocity=velocities,
                effort=effort,
            )
            data = typestore.serialize_cdr(msg, "sensor_msgs/msg/JointState")
            writer.write(conn_js, t_ns, data)

            # Action: target = current + small offset
            target_positions = positions + 0.1 * np.ones(6)
            action_msg = JointState(
                header=header,
                name=np.array(joint_names),
                position=target_positions,
                velocity=np.zeros(6, dtype=np.float64),
                effort=np.zeros(6, dtype=np.float64),
            )
            action_data = typestore.serialize_cdr(
                action_msg, "sensor_msgs/msg/JointState"
            )
            writer.write(conn_action, t_ns, action_data)

        # Write compressed images at 10Hz
        n_image_msgs = int(duration_s * image_hz)
        for i in range(n_image_msgs):
            t_s = i / image_hz
            t_ns = start_ns + int(t_s * 1e9)
            sec = int(t_ns // 1_000_000_000)
            nanosec = int(t_ns % 1_000_000_000)

            # Create a colored image that changes over time
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            color_val = int(128 + 127 * math.sin(2 * math.pi * 0.5 * t_s))
            img[:, :, 0] = color_val  # Blue channel varies
            img[:, :, 1] = 128  # Green constant
            img[:, :, 2] = 255 - color_val  # Red inverse

            # Draw frame number
            cv2.putText(
                img,
                f"Frame {i}",
                (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                3,
            )

            _, jpeg_data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])

            header = Header(
                stamp=Time(sec=sec, nanosec=nanosec),
                frame_id="right_wrist_camera",
            )
            img_msg = CompressedImage(
                header=header,
                format="jpeg",
                data=np.frombuffer(jpeg_data.tobytes(), dtype=np.uint8),
            )
            img_data = typestore.serialize_cdr(
                img_msg, "sensor_msgs/msg/CompressedImage"
            )
            writer.write(conn_img, t_ns, img_data)

    print(f"Created manipulator bag: {bag_path}")
    print(f"  JointState messages: {n_joint_msgs} at {joint_hz}Hz")
    print(f"  CompressedImage messages: {n_image_msgs} at {image_hz}Hz")
    print(f"  Action messages: {n_joint_msgs} at {joint_hz}Hz")
    print(f"  Duration: {duration_s}s")
    return bag_path


def create_dual_arm_bag(output_dir: Path | None = None) -> Path:
    """Create a dual 7-axis arm rosbag2.

    Contains:
    - /right_arm/joint_states (sensor_msgs/msg/JointState) at ~50Hz
    - /left_arm/joint_states (sensor_msgs/msg/JointState) at ~50Hz
    - /camera/right_wrist/image_raw/compressed (sensor_msgs/msg/CompressedImage) at ~10Hz
    - /camera/left_wrist/image_raw/compressed (sensor_msgs/msg/CompressedImage) at ~10Hz
    - /right_arm/target_joints (sensor_msgs/msg/JointState) at ~50Hz
    - /left_arm/target_joints (sensor_msgs/msg/JointState) at ~50Hz
    """
    bag_path = (output_dir or TEST_BAGS_DIR) / "dual_arm_bag"
    if bag_path.exists():
        shutil.rmtree(bag_path)

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    JointState = typestore.types["sensor_msgs/msg/JointState"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]
    CompressedImage = typestore.types["sensor_msgs/msg/CompressedImage"]

    right_joints = [f"right_j{i + 1}" for i in range(7)]
    left_joints = [f"left_j{i + 1}" for i in range(7)]
    duration_s = 2.0
    joint_hz = 50
    image_hz = 10
    start_ns = 1_700_000_000_000_000_000

    with Writer(bag_path, version=9) as writer:
        conn_rjs = writer.add_connection(
            "/right_arm/joint_states",
            "sensor_msgs/msg/JointState",
            typestore=typestore,
        )
        conn_ljs = writer.add_connection(
            "/left_arm/joint_states",
            "sensor_msgs/msg/JointState",
            typestore=typestore,
        )
        conn_rimg = writer.add_connection(
            "/camera/right_wrist/image_raw/compressed",
            "sensor_msgs/msg/CompressedImage",
            typestore=typestore,
        )
        conn_limg = writer.add_connection(
            "/camera/left_wrist/image_raw/compressed",
            "sensor_msgs/msg/CompressedImage",
            typestore=typestore,
        )
        conn_raction = writer.add_connection(
            "/right_arm/target_joints",
            "sensor_msgs/msg/JointState",
            typestore=typestore,
        )
        conn_laction = writer.add_connection(
            "/left_arm/target_joints",
            "sensor_msgs/msg/JointState",
            typestore=typestore,
        )

        n_joint_msgs = int(duration_s * joint_hz)
        for i in range(n_joint_msgs):
            t_s = i / joint_hz
            t_ns = start_ns + int(t_s * 1e9)
            sec = int(t_ns // 1_000_000_000)
            nanosec = int(t_ns % 1_000_000_000)
            header = Header(stamp=Time(sec=sec, nanosec=nanosec), frame_id="base_link")

            # Right arm
            r_pos = np.array(
                [math.sin(2 * math.pi * 0.3 * t_s + j * 0.4) for j in range(7)],
                dtype=np.float64,
            )
            r_msg = JointState(
                header=header,
                name=np.array(right_joints),
                position=r_pos,
                velocity=np.zeros(7, dtype=np.float64),
                effort=np.zeros(7, dtype=np.float64),
            )
            writer.write(
                conn_rjs,
                t_ns,
                typestore.serialize_cdr(r_msg, "sensor_msgs/msg/JointState"),
            )

            # Left arm
            l_pos = np.array(
                [
                    math.sin(2 * math.pi * 0.3 * t_s + j * 0.4 + math.pi)
                    for j in range(7)
                ],
                dtype=np.float64,
            )
            l_msg = JointState(
                header=header,
                name=np.array(left_joints),
                position=l_pos,
                velocity=np.zeros(7, dtype=np.float64),
                effort=np.zeros(7, dtype=np.float64),
            )
            writer.write(
                conn_ljs,
                t_ns,
                typestore.serialize_cdr(l_msg, "sensor_msgs/msg/JointState"),
            )

            # Actions (targets)
            r_action = JointState(
                header=header,
                name=np.array(right_joints),
                position=r_pos + 0.05,
                velocity=np.zeros(7, dtype=np.float64),
                effort=np.zeros(7, dtype=np.float64),
            )
            writer.write(
                conn_raction,
                t_ns,
                typestore.serialize_cdr(r_action, "sensor_msgs/msg/JointState"),
            )
            l_action = JointState(
                header=header,
                name=np.array(left_joints),
                position=l_pos + 0.05,
                velocity=np.zeros(7, dtype=np.float64),
                effort=np.zeros(7, dtype=np.float64),
            )
            writer.write(
                conn_laction,
                t_ns,
                typestore.serialize_cdr(l_action, "sensor_msgs/msg/JointState"),
            )

        # Images
        n_image_msgs = int(duration_s * image_hz)
        for i in range(n_image_msgs):
            t_s = i / image_hz
            t_ns = start_ns + int(t_s * 1e9)
            sec = int(t_ns // 1_000_000_000)
            nanosec = int(t_ns % 1_000_000_000)
            header = Header(stamp=Time(sec=sec, nanosec=nanosec), frame_id="camera")

            for label, conn in [
                ("RIGHT", conn_rimg),
                ("LEFT", conn_limg),
            ]:
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                if label == "RIGHT":
                    img[:, :, 2] = 200  # Red tint
                else:
                    img[:, :, 0] = 200  # Blue tint
                cv2.putText(
                    img,
                    f"{label} F{i}",
                    (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 255, 255),
                    3,
                )
                _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                img_msg = CompressedImage(
                    header=header,
                    format="jpeg",
                    data=np.frombuffer(jpeg.tobytes(), dtype=np.uint8),
                )
                writer.write(
                    conn,
                    t_ns,
                    typestore.serialize_cdr(img_msg, "sensor_msgs/msg/CompressedImage"),
                )

    print(f"Created dual-arm bag: {bag_path}")
    print(f"  Per-arm JointState: {n_joint_msgs} at {joint_hz}Hz")
    print(f"  Per-camera images: {n_image_msgs} at {image_hz}Hz")
    print(f"  Duration: {duration_s}s")
    return bag_path


if __name__ == "__main__":
    create_manipulator_bag()
    create_dual_arm_bag()
