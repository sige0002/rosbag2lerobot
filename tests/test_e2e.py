"""End-to-end integration tests using real and synthetic rosbag2 data.

Tests the full pipeline: bag reading → decoding → resampling → dataset writing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from PIL import Image

from rosbag2lerobot.config import FeatureMapping, RobotConfig, ResamplingConfig
from rosbag2lerobot.decoders import decode
from rosbag2lerobot.reader import BagReader
from rosbag2lerobot.resampler import Resampler
from rosbag2lerobot.writer import DatasetWriter

TEST_BAGS_DIR = Path(__file__).parent.parent / "test_bags"
MANIPULATOR_BAG = TEST_BAGS_DIR / "manipulator_bag"
DUAL_ARM_BAG = TEST_BAGS_DIR / "dual_arm_bag"
TWIST_BAG = TEST_BAGS_DIR / "twist_bag"


def _make_single_arm_config() -> RobotConfig:
    """Config matching the synthetic manipulator_bag."""
    return RobotConfig(
        robot_type="test_manipulator",
        fps=10,
        task="pick and place",
        repo_id="test/manipulator_e2e",
        observations=[
            FeatureMapping(
                key="observation.state",
                topic="/joint_states",
                msg_type="sensor_msgs/msg/JointState",
                selector="position",
                dtype="float32",
            ),
            FeatureMapping(
                key="observation.images.right_wrist",
                topic="/camera/right_wrist/image_raw/compressed",
                msg_type="sensor_msgs/msg/CompressedImage",
                dtype="image",
                image_size=[480, 640],
            ),
        ],
        actions=[
            FeatureMapping(
                key="action",
                topic="/target_joint_positions",
                msg_type="sensor_msgs/msg/JointState",
                selector="position",
                dtype="float32",
            ),
        ],
        resampling=ResamplingConfig(default_policy="nearest", tolerance_ms=100),
    )


def _make_dual_arm_config() -> RobotConfig:
    """Config matching the synthetic dual_arm_bag."""
    return RobotConfig(
        robot_type="test_dual_arm",
        fps=10,
        task="bimanual manipulation",
        repo_id="test/dual_arm_e2e",
        observations=[
            FeatureMapping(
                key="observation.state.right",
                topic="/right_arm/joint_states",
                msg_type="sensor_msgs/msg/JointState",
                selector="position",
                dtype="float32",
            ),
            FeatureMapping(
                key="observation.state.left",
                topic="/left_arm/joint_states",
                msg_type="sensor_msgs/msg/JointState",
                selector="position",
                dtype="float32",
            ),
            FeatureMapping(
                key="observation.images.right_wrist",
                topic="/camera/right_wrist/image_raw/compressed",
                msg_type="sensor_msgs/msg/CompressedImage",
                dtype="image",
                image_size=[480, 640],
            ),
            FeatureMapping(
                key="observation.images.left_wrist",
                topic="/camera/left_wrist/image_raw/compressed",
                msg_type="sensor_msgs/msg/CompressedImage",
                dtype="image",
                image_size=[480, 640],
            ),
        ],
        actions=[
            FeatureMapping(
                key="action.right",
                topic="/right_arm/target_joints",
                msg_type="sensor_msgs/msg/JointState",
                selector="position",
                dtype="float32",
            ),
            FeatureMapping(
                key="action.left",
                topic="/left_arm/target_joints",
                msg_type="sensor_msgs/msg/JointState",
                selector="position",
                dtype="float32",
            ),
        ],
        resampling=ResamplingConfig(default_policy="nearest", tolerance_ms=100),
    )


def _make_twist_config() -> RobotConfig:
    """Config matching the downloaded eloquent-twist bag."""
    return RobotConfig(
        robot_type="turtlebot",
        fps=2,
        task="navigation",
        observations=[],
        actions=[
            FeatureMapping(
                key="action",
                topic="/turtle1/cmd_vel",
                msg_type="geometry_msgs/msg/Twist",
                selector="linear.x,angular.z",
                dtype="float32",
            ),
        ],
        resampling=ResamplingConfig(default_policy="nearest", tolerance_ms=500),
    )


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

        # Collect all raw messages
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
            decoder_config = {
                "image_size": fm.image_size,
                "unit_conversion": fm.unit_conversion,
            }
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

    # 4. Build features spec for writer
    features: dict[str, dict] = {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    # Determine feature shapes from decoded data
    for fm in config.observations + config.actions:
        if fm.is_image:
            h, w = fm.image_size[0], fm.image_size[1]
            features[fm.key] = {
                "dtype": "video",
                "shape": [h, w, 3],
                "names": ["height", "width", "channels"],
            }
        else:
            # Find first decoded value to get shape
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
    )

    for frame in frames:
        writer_frame = {"task": config.task}
        for key in feature_keys:
            val = frame.get(key)
            if val is not None:
                writer_frame[key] = val
        writer.add_frame(writer_frame)

    writer.save_episode()
    writer.finalize()

    print(f"  Output: {output_dir}")
    return output_dir


def _validate_dataset(
    output_dir: Path, config: RobotConfig, expected_min_frames: int = 1
) -> None:
    """Validate the output dataset structure and content."""
    # Check directory structure
    assert (output_dir / "meta" / "info.json").exists(), "info.json missing"
    assert (output_dir / "meta" / "stats.json").exists(), "stats.json missing"
    assert (output_dir / "meta" / "tasks.parquet").exists(), "tasks.parquet missing"

    # Validate info.json
    with open(output_dir / "meta" / "info.json") as f:
        info = json.load(f)

    assert info["codebase_version"] == "v3.0"
    assert info["robot_type"] == config.robot_type
    assert info["fps"] == config.fps
    assert info["total_episodes"] >= 1
    assert info["total_frames"] >= expected_min_frames
    assert "features" in info

    # Required meta features
    for meta_key in [
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]:
        assert meta_key in info["features"], f"Missing meta feature: {meta_key}"

    # User-defined features
    for fm in config.observations + config.actions:
        assert fm.key in info["features"], f"Missing feature: {fm.key}"

    print(
        f"  info.json: OK (v3.0, {info['total_episodes']} eps, {info['total_frames']} frames)"
    )

    # Validate stats.json
    with open(output_dir / "meta" / "stats.json") as f:
        stats = json.load(f)

    for fm in config.observations + config.actions:
        assert fm.key in stats, f"Missing stats for: {fm.key}"
        stat_entry = stats[fm.key]
        for stat_name in ["min", "max", "mean", "std", "count"]:
            assert stat_name in stat_entry, f"Missing stat {stat_name} for {fm.key}"
    print(f"  stats.json: OK ({len(stats)} features)")

    # Validate tasks.parquet (LeRobot v3: task strings are the pandas Index)
    tasks_df = pd.read_parquet(output_dir / "meta" / "tasks.parquet")
    assert list(tasks_df.columns) == ["task_index"]
    assert tasks_df["task_index"].dtype.name == "int64"
    assert tasks_df.index.name is None
    assert len(tasks_df) >= 1
    assert config.task in tasks_df.index.tolist()
    print(f"  tasks.parquet: OK ({len(tasks_df)} tasks)")

    # Validate data parquet
    data_files = list((output_dir / "data").rglob("*.parquet"))
    assert len(data_files) >= 1, "No data parquet files found"
    data_table = pq.read_table(data_files[0])
    assert len(data_table) >= expected_min_frames
    for col in ["index", "timestamp", "frame_index", "episode_index", "task_index"]:
        assert col in data_table.column_names, f"Missing column: {col}"
    print(
        f"  data parquet: OK ({len(data_table)} rows, cols={data_table.column_names})"
    )

    # Validate episodes parquet
    ep_files = list((output_dir / "meta" / "episodes").rglob("*.parquet"))
    assert len(ep_files) >= 1, "No episode parquet files found"
    ep_table = pq.read_table(ep_files[0])
    assert "episode_index" in ep_table.column_names
    assert "length" in ep_table.column_names
    print(f"  episodes parquet: OK ({len(ep_table)} episodes)")

    # Validate video files if images are in config
    video_features = [fm for fm in config.observations if fm.is_image]
    for fm in video_features:
        video_dir = output_dir / "videos" / fm.key
        if video_dir.exists():
            video_files = list(video_dir.rglob("*.mp4"))
            assert len(video_files) >= 1, f"No video files for {fm.key}"
            for vf in video_files:
                assert vf.stat().st_size > 0, f"Empty video file: {vf}"
            print(f"  video {fm.key}: OK ({len(video_files)} files)")
        else:
            print(f"  video {fm.key}: SKIPPED (no video dir)")


# ============================================================================
# Test: Real rosbag2 data (downloaded twist bag)
# ============================================================================


class TestRealTwistBag:
    """Test with the real downloaded eloquent-twist.db3 bag."""

    @pytest.fixture
    def output_dir(self, tmp_path):
        return tmp_path / "twist_dataset"

    def test_read_twist_bag(self):
        """Verify we can read the real twist bag."""
        if not TWIST_BAG.exists():
            pytest.skip("twist_bag not available")

        config = _make_twist_config()
        with BagReader(TWIST_BAG, config) as reader:
            topics = reader.get_topics_info()
            assert "/turtle1/cmd_vel" in topics
            assert topics["/turtle1/cmd_vel"].msg_type == "geometry_msgs/msg/Twist"
            assert topics["/turtle1/cmd_vel"].count == 4

            start_ns, end_ns = reader.get_time_range()
            assert end_ns > start_ns

            msgs = list(reader.iter_messages())
            assert len(msgs) == 4

            # Decode first message
            topic, ts, msg = msgs[0]
            decoded = decode(
                "geometry_msgs/msg/Twist",
                msg,
                selector=["linear.x", "angular.z"],
                config={},
            )
            assert isinstance(decoded, np.ndarray)
            assert decoded.dtype == np.float32
            assert len(decoded) == 2
            assert decoded[0] == pytest.approx(2.0, abs=0.01)
            assert decoded[1] == pytest.approx(0.0, abs=0.01)

    def test_full_pipeline_twist(self, output_dir):
        """E2E test with real twist bag data."""
        if not TWIST_BAG.exists():
            pytest.skip("twist_bag not available")

        config = _make_twist_config()
        _run_full_pipeline(TWIST_BAG, config, output_dir)
        _validate_dataset(output_dir, config, expected_min_frames=2)


# ============================================================================
# Test: Synthetic single-arm manipulator bag
# ============================================================================


class TestManipulatorBag:
    """Test with synthetic 6-axis manipulator bag (JointState + CompressedImage)."""

    @pytest.fixture
    def output_dir(self, tmp_path):
        return tmp_path / "manipulator_dataset"

    def test_read_manipulator_bag(self):
        """Verify reading the synthetic manipulator bag."""
        if not MANIPULATOR_BAG.exists():
            pytest.skip("manipulator_bag not generated")

        config = _make_single_arm_config()
        with BagReader(MANIPULATOR_BAG, config) as reader:
            topics = reader.get_topics_info()
            assert "/joint_states" in topics
            assert "/camera/right_wrist/image_raw/compressed" in topics
            assert "/target_joint_positions" in topics

            assert topics["/joint_states"].count == 150
            assert topics["/camera/right_wrist/image_raw/compressed"].count == 30
            assert topics["/target_joint_positions"].count == 150

    def test_decode_joint_state(self):
        """Verify JointState decoding with selector."""
        if not MANIPULATOR_BAG.exists():
            pytest.skip("manipulator_bag not generated")

        config = _make_single_arm_config()
        with BagReader(MANIPULATOR_BAG, config) as reader:
            msgs = list(reader.iter_messages(topics=["/joint_states"]))
            assert len(msgs) == 150

            # Decode with "position" selector (all positions)
            _, _, msg = msgs[0]
            decoded = decode(
                "sensor_msgs/msg/JointState",
                msg,
                selector=["position"],
                config={},
            )
            assert isinstance(decoded, np.ndarray)
            assert decoded.dtype == np.float32
            assert len(decoded) == 6  # 6-axis arm

    def test_decode_compressed_image(self):
        """Verify CompressedImage decoding."""
        if not MANIPULATOR_BAG.exists():
            pytest.skip("manipulator_bag not generated")

        config = _make_single_arm_config()
        with BagReader(MANIPULATOR_BAG, config) as reader:
            msgs = list(
                reader.iter_messages(
                    topics=["/camera/right_wrist/image_raw/compressed"]
                )
            )
            assert len(msgs) == 30

            _, _, msg = msgs[0]
            decoded = decode(
                "sensor_msgs/msg/CompressedImage",
                msg,
                selector=None,
                config={"image_size": [480, 640]},
            )
            assert isinstance(decoded, Image.Image)
            assert decoded.size == (640, 480)  # PIL is (W, H)

    def test_resampling(self):
        """Verify resampling with multi-rate topics."""
        if not MANIPULATOR_BAG.exists():
            pytest.skip("manipulator_bag not generated")

        config = _make_single_arm_config()
        with BagReader(MANIPULATOR_BAG, config) as reader:
            start_ns, end_ns = reader.get_time_range()
            topic_to_features = config.topic_to_features

            decoded_msgs: list[tuple[str, int, object]] = []
            for topic, ts_ns, msg in reader.iter_messages(topics=config.all_topics):
                for fm in topic_to_features.get(topic, []):
                    selector = fm.selector.split(",") if fm.selector else None
                    decoder_config = {"image_size": fm.image_size}
                    decoded = decode(
                        fm.msg_type, msg, selector=selector, config=decoder_config
                    )
                    decoded_msgs.append((fm.key, ts_ns, decoded))

        resampler = Resampler(fps=10, policy="nearest", tolerance_ms=100)
        feature_keys = config.observation_keys + config.action_keys
        frames = resampler.resample(decoded_msgs, feature_keys, start_ns, end_ns)

        assert len(frames) >= 20  # 3s at 10fps ≈ 30 frames
        for frame in frames:
            assert "frame_index" in frame
            assert "timestamp" in frame
            # Check all features have values (nearest with 100ms tolerance)
            for key in feature_keys:
                assert frame[key] is not None, (
                    f"Frame {frame['frame_index']}: {key} is None"
                )

    def test_full_pipeline_manipulator(self, output_dir):
        """Full E2E test: bag → decode → resample → write LeRobot v3.0 dataset."""
        if not MANIPULATOR_BAG.exists():
            pytest.skip("manipulator_bag not generated")

        config = _make_single_arm_config()
        _run_full_pipeline(MANIPULATOR_BAG, config, output_dir)
        _validate_dataset(output_dir, config, expected_min_frames=20)

        # Additional validation: check data values are reasonable
        data_files = list((output_dir / "data").rglob("*.parquet"))
        table = pq.read_table(data_files[0])

        # Check observation.state values (sin wave, should be in [-1, 1])
        state_col = table.column("observation.state").to_pylist()
        for row in state_col:
            if row is not None:
                vals = np.array(row)
                assert np.all(np.abs(vals) <= 1.5), f"Joint values out of range: {vals}"

        # Check timestamps are monotonically increasing
        timestamps = table.column("timestamp").to_pylist()
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], "Timestamps not monotonic"


# ============================================================================
# Test: Synthetic dual-arm bag
# ============================================================================


class TestDualArmBag:
    """Test with synthetic dual 7-axis arm bag."""

    @pytest.fixture
    def output_dir(self, tmp_path):
        return tmp_path / "dual_arm_dataset"

    def test_read_dual_arm_bag(self):
        """Verify reading the dual-arm bag."""
        if not DUAL_ARM_BAG.exists():
            pytest.skip("dual_arm_bag not generated")

        config = _make_dual_arm_config()
        with BagReader(DUAL_ARM_BAG, config) as reader:
            topics = reader.get_topics_info()
            assert "/right_arm/joint_states" in topics
            assert "/left_arm/joint_states" in topics
            assert "/camera/right_wrist/image_raw/compressed" in topics
            assert "/camera/left_wrist/image_raw/compressed" in topics

    def test_decode_dual_arm_joints(self):
        """Verify decoding both arms' joint states."""
        if not DUAL_ARM_BAG.exists():
            pytest.skip("dual_arm_bag not generated")

        config = _make_dual_arm_config()
        with BagReader(DUAL_ARM_BAG, config) as reader:
            # Right arm
            right_msgs = list(reader.iter_messages(topics=["/right_arm/joint_states"]))
            assert len(right_msgs) == 100

            _, _, msg = right_msgs[0]
            decoded = decode(
                "sensor_msgs/msg/JointState",
                msg,
                selector=["position"],
                config={},
            )
            assert len(decoded) == 7  # 7-axis

            # Left arm
        with BagReader(DUAL_ARM_BAG, config) as reader:
            left_msgs = list(reader.iter_messages(topics=["/left_arm/joint_states"]))
            assert len(left_msgs) == 100

            _, _, msg = left_msgs[0]
            decoded = decode(
                "sensor_msgs/msg/JointState",
                msg,
                selector=["position"],
                config={},
            )
            assert len(decoded) == 7

    def test_full_pipeline_dual_arm(self, output_dir):
        """Full E2E test for dual-arm robot."""
        if not DUAL_ARM_BAG.exists():
            pytest.skip("dual_arm_bag not generated")

        config = _make_dual_arm_config()
        _run_full_pipeline(DUAL_ARM_BAG, config, output_dir)
        _validate_dataset(output_dir, config, expected_min_frames=10)

        # Verify both arms' data exists in parquet
        data_files = list((output_dir / "data").rglob("*.parquet"))
        table = pq.read_table(data_files[0])
        assert "observation.state.right" in table.column_names
        assert "observation.state.left" in table.column_names
        assert "action.right" in table.column_names
        assert "action.left" in table.column_names

        # Verify both cameras have videos
        info_path = output_dir / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)
        assert "observation.images.right_wrist" in info["features"]
        assert "observation.images.left_wrist" in info["features"]


# ============================================================================
# Test: Multi-episode conversion
# ============================================================================


class TestMultiEpisode:
    """Test converting multiple bags as separate episodes."""

    @pytest.fixture
    def output_dir(self, tmp_path):
        return tmp_path / "multi_episode_dataset"

    def test_two_episodes(self, output_dir):
        """Convert manipulator_bag twice as two episodes."""
        if not MANIPULATOR_BAG.exists():
            pytest.skip("manipulator_bag not generated")

        config = _make_single_arm_config()

        # Build features spec
        features = {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "observation.state": {"dtype": "float32", "shape": [6], "names": None},
            "observation.images.right_wrist": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
            },
            "action": {"dtype": "float32", "shape": [6], "names": None},
        }

        writer = DatasetWriter(
            output_dir=output_dir,
            config={"robot_type": config.robot_type},
            features=features,
            fps=config.fps,
            repo_id=config.repo_id,
        )

        # Process the same bag twice as two episodes
        for ep_idx in range(2):
            with BagReader(MANIPULATOR_BAG, config) as reader:
                start_ns, end_ns = reader.get_time_range()
                topic_to_features = config.topic_to_features

                decoded_msgs = []
                for topic, ts_ns, msg in reader.iter_messages(topics=config.all_topics):
                    for fm in topic_to_features.get(topic, []):
                        selector = fm.selector.split(",") if fm.selector else None
                        decoder_config = {"image_size": fm.image_size}
                        decoded = decode(
                            fm.msg_type, msg, selector=selector, config=decoder_config
                        )
                        decoded_msgs.append((fm.key, ts_ns, decoded))

            resampler = Resampler(fps=config.fps, policy="nearest", tolerance_ms=100)
            feature_keys = config.observation_keys + config.action_keys
            frames = resampler.resample(decoded_msgs, feature_keys, start_ns, end_ns)

            for frame in frames:
                writer_frame = {"task": config.task}
                for key in feature_keys:
                    val = frame.get(key)
                    if val is not None:
                        writer_frame[key] = val
                writer.add_frame(writer_frame)
            writer.save_episode()

        writer.finalize()

        # Validate
        with open(output_dir / "meta" / "info.json") as f:
            info = json.load(f)
        assert info["total_episodes"] == 2
        assert info["total_frames"] >= 40  # At least 20 frames per episode

        # Check episode parquet files
        ep_files = list((output_dir / "meta" / "episodes").rglob("*.parquet"))
        assert len(ep_files) == 2
