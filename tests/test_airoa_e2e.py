"""End-to-end integration test for AIRoA HSR rosbag -> LeRobot v3.0 conversion.

Requires the real ROS2 bag at /workspace/airoa-moma-raw/235210_ros2/.
Tests the full pipeline: read -> decode -> resample -> write, then validates
the output dataset structure against the LeRobot v3.0 specification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

from rosbag2lerobot.config import (
    RobotConfig,
    load_config,
)
from rosbag2lerobot.decoders import decode
from rosbag2lerobot.reader import BagReader
from rosbag2lerobot.resampler import Resampler
from rosbag2lerobot.writer import DatasetWriter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
HSR_CONFIG_PATH = CONFIGS_DIR / "hsr.yaml"
BAG_PATH = Path("/workspace/airoa-moma-raw/235210_ros2")

BAG_EXISTS = BAG_PATH.exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_hsr_config() -> RobotConfig:
    """Load HSR config from YAML file."""
    return load_config(HSR_CONFIG_PATH)


def _run_full_pipeline(
    bag_path: Path,
    config: RobotConfig,
    output_dir: Path,
) -> Path:
    """Run the full conversion pipeline and return the output directory."""
    # 1. Read bag
    with BagReader(bag_path, config) as reader:
        start_ns, end_ns = reader.get_time_range()
        topics_info = reader.get_topics_info()

        raw_messages: list[tuple[str, int, object]] = []
        for topic, ts_ns, msg in reader.iter_messages(topics=config.all_topics):
            raw_messages.append((topic, ts_ns, msg))

    print(f"\n  Bag: {bag_path.name}")
    print(f"  Topics: {list(topics_info.keys())}")
    print(f"  Messages: {len(raw_messages)}")
    print(f"  Duration: {(end_ns - start_ns) / 1e9:.2f}s")

    # 2. Decode messages
    decoded_messages: list[tuple[str, int, object]] = []
    topic_to_features = config.topic_to_features

    for topic, ts_ns, msg in raw_messages:
        for fm in topic_to_features.get(topic, []):
            selector = fm.selector.split(",") if fm.selector else None
            decoder_config: dict[str, Any] = {}
            if fm.image_size is not None:
                decoder_config["image_size"] = fm.image_size
            if fm.unit_conversion != 1.0:
                decoder_config["unit_conversion"] = fm.unit_conversion
            try:
                decoded_value = decode(
                    fm.msg_type,
                    msg,
                    selector=selector,
                    config=decoder_config,
                )
                decoded_messages.append((fm.key, ts_ns, decoded_value))
            except Exception as e:
                print(f"  WARN: decode failed for {fm.key}: {e}")

    print(f"  Decoded: {len(decoded_messages)} values")

    # 3. Resample
    resampler = Resampler(
        fps=config.fps,
        policy=config.resampling.default_policy,
        tolerance_ms=config.resampling.tolerance_ms,
    )
    feature_keys = config.observation_keys + config.action_keys
    frames = resampler.resample(
        messages=decoded_messages,
        feature_keys=feature_keys,
        start_ns=start_ns,
        end_ns=end_ns,
    )
    print(f"  Resampled: {len(frames)} frames at {config.fps} fps")

    # 4. Build features spec
    features: dict[str, dict[str, Any]] = {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for fm in config.observations + config.actions:
        if fm.is_image:
            h, w = fm.image_size[0], fm.image_size[1]
            features[fm.key] = {
                "dtype": "video",
                "shape": [h, w, 3],
                "names": ["height", "width", "channels"],
            }
        else:
            shape = [1]
            for key, _, val in decoded_messages:
                if key == fm.key and isinstance(val, np.ndarray):
                    shape = list(val.shape)
                    break
            features[fm.key] = {
                "dtype": "float32",
                "shape": shape,
                "names": None,
            }

    # 5. Write dataset
    writer = DatasetWriter(
        output_dir=output_dir,
        config={"robot_type": config.robot_type},
        features=features,
        fps=config.fps,
        repo_id=config.repo_id,
        video_codec="libx264",
    )

    for frame in frames:
        writer_frame: dict[str, Any] = {"task": config.task}
        for key in feature_keys:
            val = frame.get(key)
            if val is not None:
                writer_frame[key] = val
        writer.add_frame(writer_frame)

    writer.save_episode()
    writer.finalize()

    print(f"  Output: {output_dir}")
    return output_dir


# ============================================================================
# Test: HSR config validation against real bag topics
# ============================================================================


@pytest.mark.skipif(
    not BAG_EXISTS,
    reason="AIRoA bag not found at /workspace/airoa-moma-raw/235210_ros2/",
)
class TestHSRConfigAgainstBag:
    """Validate that hsr.yaml topics match the actual ROS2 bag."""

    @pytest.fixture()
    def hsr_config(self) -> RobotConfig:
        return _load_hsr_config()

    def test_config_loads_successfully(self, hsr_config: RobotConfig) -> None:
        """Verify HSR config loads without errors."""
        assert hsr_config.robot_type in ("hsr", "toyota_hsr")
        assert hsr_config.fps == 10
        assert len(hsr_config.observations) >= 3
        assert len(hsr_config.actions) >= 2

    def test_all_config_topics_exist_in_bag(self, hsr_config: RobotConfig) -> None:
        """Verify every topic in the HSR config is present in the actual bag."""
        with BagReader(BAG_PATH, hsr_config) as reader:
            bag_topics = reader.get_topics_info()

        for topic in hsr_config.all_topics:
            assert topic in bag_topics, (
                f"Config topic {topic} not found in bag. "
                f"Available: {sorted(bag_topics.keys())}"
            )

    def test_message_types_match(self, hsr_config: RobotConfig) -> None:
        """Verify message types in config match those in the bag."""
        with BagReader(BAG_PATH, hsr_config) as reader:
            bag_topics = reader.get_topics_info()

        for fm in hsr_config.observations + hsr_config.actions:
            if fm.topic in bag_topics:
                # rosbags uses '/' separator in msgtype
                bag_type = bag_topics[fm.topic].msg_type
                # Normalize: config uses 'pkg/msg/Type', bag may use 'pkg/msg/Type'
                assert bag_type == fm.msg_type, (
                    f"Type mismatch for {fm.topic}: "
                    f"config={fm.msg_type}, bag={bag_type}"
                )

    def test_bag_has_sufficient_messages(self, hsr_config: RobotConfig) -> None:
        """Verify the bag has a reasonable number of messages per topic."""
        with BagReader(BAG_PATH, hsr_config) as reader:
            bag_topics = reader.get_topics_info()

        for fm in hsr_config.observations + hsr_config.actions:
            if fm.topic in bag_topics:
                count = bag_topics[fm.topic].count
                assert count > 10, (
                    f"Topic {fm.topic} has only {count} messages, expected >10"
                )
                print(f"  {fm.topic}: {count} msgs")

    def test_config_feature_keys_are_unique(self, hsr_config: RobotConfig) -> None:
        """Verify no duplicate feature keys."""
        all_keys = hsr_config.observation_keys + hsr_config.action_keys
        assert len(all_keys) == len(set(all_keys))

    def test_image_features_have_image_size(self, hsr_config: RobotConfig) -> None:
        """Verify all image features have image_size set."""
        for fm in hsr_config.observations + hsr_config.actions:
            if fm.dtype == "image":
                assert fm.image_size is not None
                assert len(fm.image_size) >= 2

    def test_config_message_types_are_valid(self, hsr_config: RobotConfig) -> None:
        """Verify all message types in config are recognized ROS2 types."""
        known_packages = {
            "sensor_msgs",
            "geometry_msgs",
            "std_msgs",
            "nav_msgs",
            "trajectory_msgs",
        }
        for fm in hsr_config.observations + hsr_config.actions:
            parts = fm.msg_type.split("/")
            assert len(parts) == 3, f"Invalid msg_type format: {fm.msg_type}"
            assert parts[1] == "msg"
            assert parts[0] in known_packages, (
                f"Unknown package: {parts[0]} in {fm.msg_type}"
            )


# ============================================================================
# Test: Full E2E conversion pipeline
# ============================================================================


@pytest.mark.skipif(
    not BAG_EXISTS,
    reason="AIRoA bag not found at /workspace/airoa-moma-raw/235210_ros2/",
)
class TestHSRE2EConversion:
    """Full end-to-end test: convert real HSR bag -> LeRobot v3.0 dataset."""

    @pytest.fixture
    def output_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "hsr_dataset"

    @pytest.fixture
    def hsr_config(self) -> RobotConfig:
        return _load_hsr_config()

    @pytest.fixture
    def converted_dataset(self, hsr_config: RobotConfig, output_dir: Path) -> Path:
        """Run conversion and return output dir (cached per test class)."""
        return _run_full_pipeline(BAG_PATH, hsr_config, output_dir)

    # --- Structure validation ---

    def test_info_json_exists_and_valid(
        self, converted_dataset: Path, hsr_config: RobotConfig
    ) -> None:
        """Validate meta/info.json has correct v3.0 fields."""
        info_path = converted_dataset / "meta" / "info.json"
        assert info_path.exists(), "meta/info.json missing"

        with open(info_path) as f:
            info = json.load(f)

        # Required v3.0 top-level fields
        assert info["codebase_version"] == "v3.0"
        assert info["robot_type"] == hsr_config.robot_type
        assert info["fps"] == hsr_config.fps
        assert info["total_episodes"] == 1
        assert "chunks_size" in info
        assert "data_files_size_in_mb" in info
        assert "video_files_size_in_mb" in info
        assert "splits" in info
        assert "features" in info

        # Verify splits format
        assert "train" in info["splits"]

        # Verify total_frames is reasonable (~10s * 10fps = ~100)
        assert 50 <= info["total_frames"] <= 200, (
            f"Unexpected frame count: {info['total_frames']}"
        )

        print(
            f"  info.json: {info['total_episodes']} eps, "
            f"{info['total_frames']} frames, v{info['codebase_version']}"
        )

    def test_info_json_has_all_features(
        self, converted_dataset: Path, hsr_config: RobotConfig
    ) -> None:
        """Verify info.json features include all observation and action keys."""
        with open(converted_dataset / "meta" / "info.json") as f:
            info = json.load(f)

        features = info["features"]

        # Meta features
        for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            assert key in features, f"Missing meta feature: {key}"

        # All observation and action features
        for fm in hsr_config.observations + hsr_config.actions:
            assert fm.key in features, f"Missing feature: {fm.key}"

    def test_data_parquet_exists_and_readable(
        self, converted_dataset: Path, hsr_config: RobotConfig
    ) -> None:
        """Validate data/chunk-000/file-000.parquet."""
        parquet_path = converted_dataset / "data" / "chunk-000" / "file-000.parquet"
        assert parquet_path.exists(), "data/chunk-000/file-000.parquet missing"

        table = pq.read_table(parquet_path)

        # Required columns
        for col in ["index", "timestamp", "frame_index", "episode_index", "task_index"]:
            assert col in table.column_names, f"Missing column: {col}"

        # Feature columns
        for fm in hsr_config.observations + hsr_config.actions:
            assert fm.key in table.column_names, f"Missing column: {fm.key}"

        # Reasonable row count
        n_rows = len(table)
        assert 50 <= n_rows <= 200, f"Unexpected row count: {n_rows}"

        # Timestamps are non-decreasing
        timestamps = table.column("timestamp").to_pylist()
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], "Timestamps not monotonic"

        # Frame indices are sequential
        frame_indices = table.column("frame_index").to_pylist()
        for i, fi in enumerate(frame_indices):
            assert fi == i, f"frame_index {fi} != expected {i}"

        print(f"  data parquet: {n_rows} rows, columns={table.column_names}")

    def test_videos_exist(
        self, converted_dataset: Path, hsr_config: RobotConfig
    ) -> None:
        """Validate MP4 video files exist for each camera stream."""
        video_features = [fm for fm in hsr_config.observations if fm.is_image]
        assert len(video_features) >= 2, "Expected at least 2 camera streams"

        for fm in video_features:
            video_dir = converted_dataset / "videos" / fm.key
            assert video_dir.exists(), f"Video dir missing: {video_dir}"
            mp4_files = list(video_dir.rglob("*.mp4"))
            assert len(mp4_files) >= 1, f"No MP4 files for {fm.key}"
            for vf in mp4_files:
                assert vf.stat().st_size > 100, f"Video file too small: {vf}"
            print(
                f"  video {fm.key}: {len(mp4_files)} file(s), "
                f"size={mp4_files[0].stat().st_size / 1024:.1f}KB"
            )

    def test_stats_json_exists_and_valid(
        self, converted_dataset: Path, hsr_config: RobotConfig
    ) -> None:
        """Validate meta/stats.json has min/max/mean/std for each feature."""
        stats_path = converted_dataset / "meta" / "stats.json"
        assert stats_path.exists(), "meta/stats.json missing"

        with open(stats_path) as f:
            stats = json.load(f)

        for fm in hsr_config.observations + hsr_config.actions:
            assert fm.key in stats, f"Missing stats for: {fm.key}"
            entry = stats[fm.key]
            for stat_name in ["min", "max", "mean", "std"]:
                assert stat_name in entry, f"Missing stat '{stat_name}' for {fm.key}"
                assert isinstance(entry[stat_name], list), (
                    f"Stat {stat_name} for {fm.key} should be a list"
                )

        print(f"  stats.json: {len(stats)} features")

    def test_tasks_parquet_exists(
        self, converted_dataset: Path, hsr_config: RobotConfig
    ) -> None:
        """Validate meta/tasks.parquet (LeRobot v3 layout)."""
        import pandas as pd

        tasks_path = converted_dataset / "meta" / "tasks.parquet"
        assert tasks_path.exists(), "meta/tasks.parquet missing"

        df = pd.read_parquet(tasks_path)
        # v3: task strings are the (unnamed) Index; task_index is the only column.
        assert list(df.columns) == ["task_index"]
        assert df["task_index"].dtype.name == "int64"
        assert df.index.name is None
        assert len(df) >= 1

        assert hsr_config.task in df.index.tolist()

    def test_episodes_parquet_exists(self, converted_dataset: Path) -> None:
        """Validate meta/episodes/chunk-000/file-000.parquet."""
        ep_path = (
            converted_dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        )
        assert ep_path.exists(), "meta/episodes/chunk-000/file-000.parquet missing"

        table = pq.read_table(ep_path)
        assert "episode_index" in table.column_names
        assert "length" in table.column_names
        assert len(table) == 1  # 1 episode

        length = table.column("length").to_pylist()[0]
        assert 50 <= length <= 200, f"Unexpected episode length: {length}"

    def test_frame_count_matches_duration(
        self, converted_dataset: Path, hsr_config: RobotConfig
    ) -> None:
        """Verify frame count is consistent: ~10s * 10fps = ~100 frames."""
        with open(converted_dataset / "meta" / "info.json") as f:
            info = json.load(f)

        total_frames = info["total_frames"]
        expected_frames = 10.1 * hsr_config.fps  # ~101
        # Allow +/- 30% tolerance
        assert abs(total_frames - expected_frames) / expected_frames < 0.3, (
            f"Frame count {total_frames} too far from expected ~{expected_frames:.0f}"
        )

    # --- Data quality validation ---

    def test_numeric_feature_values_are_finite(
        self, converted_dataset: Path, hsr_config: RobotConfig
    ) -> None:
        """Verify all numeric feature values are finite (no NaN/Inf)."""
        data_files = list((converted_dataset / "data").rglob("*.parquet"))
        table = pq.read_table(data_files[0])

        numeric_features = [
            fm for fm in hsr_config.observations + hsr_config.actions if not fm.is_image
        ]

        for fm in numeric_features:
            col = table.column(fm.key).to_pylist()
            non_none_count = sum(1 for row in col if row is not None)
            assert non_none_count > 0, f"No data for {fm.key}"

            for row in col:
                if row is not None:
                    vals = np.array(row)
                    assert np.all(np.isfinite(vals)), (
                        f"Non-finite values in {fm.key}: {vals}"
                    )

            print(
                f"  {fm.key}: {non_none_count}/{len(col)} non-null, "
                f"dim={len(np.array(col[0]))}"
            )

    def test_joint_state_feature_dimensions(
        self, converted_dataset: Path, hsr_config: RobotConfig
    ) -> None:
        """Verify JointState-based features have expected dimensions."""
        data_files = list((converted_dataset / "data").rglob("*.parquet"))
        table = pq.read_table(data_files[0])

        # Find all JointState features and check they have > 0 dimensions
        for fm in hsr_config.observations + hsr_config.actions:
            if (
                fm.msg_type == "sensor_msgs/msg/JointState"
                and fm.key in table.column_names
            ):
                col = table.column(fm.key).to_pylist()
                for row in col:
                    if row is not None:
                        vals = np.array(row)
                        assert len(vals) > 0, f"{fm.key} has 0 dimensions"
                        assert np.all(np.isfinite(vals)), f"Non-finite in {fm.key}"
                        print(f"  {fm.key}: dim={len(vals)}")
                        break

    def test_wrench_feature_has_6_values(
        self, converted_dataset: Path, hsr_config: RobotConfig
    ) -> None:
        """Verify WrenchStamped feature has 6 values (force xyz + torque xyz)."""
        data_files = list((converted_dataset / "data").rglob("*.parquet"))
        table = pq.read_table(data_files[0])

        for fm in hsr_config.observations:
            if (
                fm.msg_type == "geometry_msgs/msg/WrenchStamped"
                and fm.key in table.column_names
            ):
                col = table.column(fm.key).to_pylist()
                for row in col:
                    if row is not None:
                        vals = np.array(row)
                        assert vals.shape[0] == 6, (
                            f"Expected 6 wrench values for {fm.key}, got {vals.shape[0]}"
                        )
                        break

    def test_twist_feature_has_6_values(
        self, converted_dataset: Path, hsr_config: RobotConfig
    ) -> None:
        """Verify Twist action has 6 values (linear xyz + angular xyz)."""
        data_files = list((converted_dataset / "data").rglob("*.parquet"))
        table = pq.read_table(data_files[0])

        for fm in hsr_config.actions:
            if (
                fm.msg_type == "geometry_msgs/msg/Twist"
                and fm.key in table.column_names
            ):
                col = table.column(fm.key).to_pylist()
                for row in col:
                    if row is not None:
                        vals = np.array(row)
                        assert vals.shape[0] == 6, (
                            f"Expected 6 twist values for {fm.key}, got {vals.shape[0]}"
                        )
                        break


# ============================================================================
# Test: Individual decoder tests against real bag messages
# ============================================================================


@pytest.mark.skipif(not BAG_EXISTS, reason="AIRoA bag not found")
class TestHSRDecoders:
    """Test individual decoders against real messages from the HSR bag."""

    @pytest.fixture
    def hsr_config(self) -> RobotConfig:
        return _load_hsr_config()

    def test_decode_joint_state(self, hsr_config: RobotConfig) -> None:
        """Decode JointState from the real bag."""
        with BagReader(BAG_PATH, hsr_config) as reader:
            msgs = list(reader.iter_messages(topics=["/hsrb/joint_states"]))

        assert len(msgs) > 100, f"Expected >100 JointState msgs, got {len(msgs)}"
        _, _, msg = msgs[0]
        decoded = decode(
            "sensor_msgs/msg/JointState",
            msg,
            selector=["position"],
            config={},
        )
        assert isinstance(decoded, np.ndarray)
        assert decoded.dtype == np.float32
        assert len(decoded) == 13  # HSR has 13 joints
        print(f"  JointState: {len(decoded)} joints, values={decoded[:5]}...")

    def test_decode_wrench_stamped(self, hsr_config: RobotConfig) -> None:
        """Decode WrenchStamped from the real bag."""
        with BagReader(BAG_PATH, hsr_config) as reader:
            msgs = list(reader.iter_messages(topics=["/hsrb/wrist_wrench/raw"]))

        assert len(msgs) > 100
        _, _, msg = msgs[0]
        decoded = decode(
            "geometry_msgs/msg/WrenchStamped",
            msg,
            selector=None,
            config={},
        )
        assert isinstance(decoded, np.ndarray)
        assert decoded.dtype == np.float32
        assert len(decoded) == 6
        print(f"  WrenchStamped: {decoded}")

    def test_decode_compressed_image(self, hsr_config: RobotConfig) -> None:
        """Decode CompressedImage from the real bag."""
        with BagReader(BAG_PATH, hsr_config) as reader:
            msgs = list(
                reader.iter_messages(
                    topics=["/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed"]
                )
            )

        assert len(msgs) > 50
        _, _, msg = msgs[0]
        decoded = decode(
            "sensor_msgs/msg/CompressedImage",
            msg,
            selector=None,
            config={"image_size": [480, 640]},
        )
        assert isinstance(decoded, Image.Image)
        assert decoded.size == (640, 480)
        print(f"  CompressedImage: {decoded.size} ({decoded.mode})")

    def test_decode_twist(self, hsr_config: RobotConfig) -> None:
        """Decode Twist from the real bag."""
        with BagReader(BAG_PATH, hsr_config) as reader:
            msgs = list(reader.iter_messages(topics=["/hsrb/command_velocity"]))

        assert len(msgs) > 50
        _, _, msg = msgs[0]
        decoded = decode(
            "geometry_msgs/msg/Twist",
            msg,
            selector=None,
            config={},
        )
        assert isinstance(decoded, np.ndarray)
        assert decoded.dtype == np.float32
        assert len(decoded) == 6
        print(f"  Twist: {decoded}")

    def test_decode_joint_trajectory(self, hsr_config: RobotConfig) -> None:
        """Decode JointTrajectory from the real bag."""
        with BagReader(BAG_PATH, hsr_config) as reader:
            msgs = list(
                reader.iter_messages(topics=["/hsrb/arm_trajectory_controller/command"])
            )

        assert len(msgs) > 50
        _, _, msg = msgs[0]
        decoded = decode(
            "trajectory_msgs/msg/JointTrajectory",
            msg,
            selector=["positions"],
            config={},
        )
        assert isinstance(decoded, np.ndarray)
        assert decoded.dtype == np.float32
        assert len(decoded) > 0
        print(f"  JointTrajectory: {len(decoded)} positions, values={decoded}")

    def test_decode_odometry(self, hsr_config: RobotConfig) -> None:
        """Decode Odometry from the real bag."""
        with BagReader(BAG_PATH, hsr_config) as reader:
            msgs = list(reader.iter_messages(topics=["/hsrb/odom"]))

        assert len(msgs) > 50
        _, _, msg = msgs[0]
        decoded = decode(
            "nav_msgs/msg/Odometry",
            msg,
            selector=None,
            config={},
        )
        assert isinstance(decoded, np.ndarray)
        assert decoded.dtype == np.float32
        assert len(decoded) == 13  # 7 pose + 6 twist
        print(f"  Odometry: {len(decoded)} values")
