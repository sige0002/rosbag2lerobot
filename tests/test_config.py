"""Tests for robot configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rosbag2lerobot.config import (
    CustomMsgDef,
    FeatureMapping,
    ResamplingConfig,
    SplitConfig,
    build_default_config,
    compute_splits,
    load_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _minimal_config_dict() -> dict:
    """Return a minimal valid config as a Python dict."""
    return {
        "robot_type": "test_robot",
        "fps": 30,
        "task": "test_task",
        "observations": [
            {
                "key": "observation.state",
                "topic": "/joint_states",
                "msg_type": "sensor_msgs/msg/JointState",
                "selector": "position",
                "dtype": "float32",
            }
        ],
        "actions": [
            {
                "key": "action",
                "topic": "/joint_commands",
                "msg_type": "sensor_msgs/msg/JointState",
                "selector": "position",
                "dtype": "float32",
            }
        ],
    }


@pytest.fixture()
def minimal_config_path(tmp_path: Path) -> Path:
    """Write a minimal valid config YAML and return its path."""
    cfg = _minimal_config_dict()
    path = tmp_path / "robot_config.yaml"
    path.write_text(yaml.dump(cfg))
    return path


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestLoadValidConfig:
    """Test loading valid configuration files."""

    def test_load_minimal(self, minimal_config_path: Path) -> None:
        cfg = load_config(minimal_config_path)
        assert cfg.robot_type == "test_robot"
        assert cfg.fps == 30
        assert cfg.task == "test_task"
        assert len(cfg.observations) == 1
        assert len(cfg.actions) == 1

    def test_observation_keys(self, minimal_config_path: Path) -> None:
        cfg = load_config(minimal_config_path)
        assert cfg.observation_keys == ["observation.state"]

    def test_action_keys(self, minimal_config_path: Path) -> None:
        cfg = load_config(minimal_config_path)
        assert cfg.action_keys == ["action"]

    def test_all_topics(self, minimal_config_path: Path) -> None:
        cfg = load_config(minimal_config_path)
        assert "/joint_states" in cfg.all_topics
        assert "/joint_commands" in cfg.all_topics

    def test_resampling_defaults(self, minimal_config_path: Path) -> None:
        cfg = load_config(minimal_config_path)
        assert cfg.resampling.default_policy == "hold"
        assert cfg.resampling.tolerance_ms == 50.0

    def test_stamp_delay_defaults(self, minimal_config_path: Path) -> None:
        """New stale-stamp fields default to disabled / align-to-required."""
        cfg = load_config(minimal_config_path)
        assert cfg.resampling.align_to_required is True
        assert cfg.resampling.max_stamp_delay_ms is None
        # Per-feature default is also None (no individual threshold).
        assert cfg.observations[0].max_stamp_delay_ms is None
        assert cfg.actions[0].max_stamp_delay_ms is None

    def test_load_with_stamp_delay(self, tmp_path: Path) -> None:
        """Global stale-stamp settings are read from the resampling block."""
        cfg_dict = _minimal_config_dict()
        cfg_dict["resampling"] = {
            "max_stamp_delay_ms": 120,
            "align_to_required": False,
        }
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(cfg_dict))
        cfg = load_config(path)
        assert cfg.resampling.max_stamp_delay_ms == 120.0
        assert isinstance(cfg.resampling.max_stamp_delay_ms, float)
        assert cfg.resampling.align_to_required is False

    def test_per_feature_stamp_delay_override(self, tmp_path: Path) -> None:
        """A per-feature max_stamp_delay_ms is parsed (and coerced to float)."""
        cfg_dict = _minimal_config_dict()
        cfg_dict["observations"][0]["max_stamp_delay_ms"] = 0
        cfg_dict["actions"][0]["max_stamp_delay_ms"] = 250
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(cfg_dict))
        cfg = load_config(path)
        # 0 is a valid threshold (drop any lag > 0ms), not falsy-None.
        assert cfg.observations[0].max_stamp_delay_ms == 0.0
        assert isinstance(cfg.observations[0].max_stamp_delay_ms, float)
        assert cfg.actions[0].max_stamp_delay_ms == 250.0

    def test_load_with_images(self, tmp_path: Path) -> None:
        cfg_dict = _minimal_config_dict()
        cfg_dict["observations"].append(
            {
                "key": "observation.images.top",
                "topic": "/camera/top/compressed",
                "msg_type": "sensor_msgs/msg/CompressedImage",
                "dtype": "image",
                "image_size": [480, 640, 3],
            }
        )
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(cfg_dict))
        cfg = load_config(path)
        assert len(cfg.image_features) == 1
        assert cfg.image_features[0].key == "observation.images.top"

    def test_load_with_custom_msgs(self, tmp_path: Path) -> None:
        cfg_dict = _minimal_config_dict()
        cfg_dict["custom_msgs"] = [
            {"msg_file": "msgs/my/Custom.msg", "package": "my_msgs"}
        ]
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(cfg_dict))
        cfg = load_config(path)
        assert len(cfg.custom_msgs) == 1
        assert cfg.custom_msgs[0].package == "my_msgs"

    def test_load_with_resampling(self, tmp_path: Path) -> None:
        cfg_dict = _minimal_config_dict()
        cfg_dict["resampling"] = {"default_policy": "nearest", "tolerance_ms": 100.0}
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(cfg_dict))
        cfg = load_config(path)
        assert cfg.resampling.default_policy == "nearest"
        assert cfg.resampling.tolerance_ms == 100.0

    def test_topic_to_features_map(self, minimal_config_path: Path) -> None:
        cfg = load_config(minimal_config_path)
        t2f = cfg.topic_to_features
        assert "/joint_states" in t2f
        assert len(t2f["/joint_states"]) == 1

    def test_load_bundled_so101(self) -> None:
        """Verify the bundled so101.yaml loads without errors."""
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "so101.yaml"
        cfg = load_config(cfg_path)
        assert cfg.robot_type == "so101"
        assert cfg.fps == 30

    def test_load_bundled_hsr(self) -> None:
        """Verify the bundled hsr.yaml loads without errors."""
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "hsr.yaml"
        cfg = load_config(cfg_path)
        assert cfg.robot_type == "hsr"
        assert cfg.fps == 10


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


class TestValidationErrors:
    """Test that invalid configs raise appropriate errors."""

    def test_missing_robot_type(self, tmp_path: Path) -> None:
        cfg_dict = _minimal_config_dict()
        del cfg_dict["robot_type"]
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(cfg_dict))
        with pytest.raises(ValueError, match="robot_type"):
            load_config(path)

    def test_missing_fps(self, tmp_path: Path) -> None:
        cfg_dict = _minimal_config_dict()
        del cfg_dict["fps"]
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(cfg_dict))
        with pytest.raises(ValueError, match="fps"):
            load_config(path)

    def test_missing_task(self, tmp_path: Path) -> None:
        cfg_dict = _minimal_config_dict()
        del cfg_dict["task"]
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(cfg_dict))
        with pytest.raises(ValueError, match="task"):
            load_config(path)

    def test_missing_observations(self, tmp_path: Path) -> None:
        cfg_dict = _minimal_config_dict()
        del cfg_dict["observations"]
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(cfg_dict))
        with pytest.raises(ValueError, match="observations"):
            load_config(path)

    def test_missing_actions(self, tmp_path: Path) -> None:
        cfg_dict = _minimal_config_dict()
        del cfg_dict["actions"]
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(cfg_dict))
        with pytest.raises(ValueError, match="actions"):
            load_config(path)

    def test_negative_fps(self, tmp_path: Path) -> None:
        cfg_dict = _minimal_config_dict()
        cfg_dict["fps"] = -10
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(cfg_dict))
        with pytest.raises(ValueError, match="fps must be positive"):
            load_config(path)

    def test_invalid_dtype(self, tmp_path: Path) -> None:
        cfg_dict = _minimal_config_dict()
        cfg_dict["observations"][0]["dtype"] = "complex128"
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(cfg_dict))
        with pytest.raises(ValueError, match="Unsupported dtype"):
            load_config(path)

    def test_invalid_resampling_policy(self) -> None:
        with pytest.raises(ValueError, match="Invalid resampling policy"):
            ResamplingConfig(default_policy="interpolate")

    def test_negative_tolerance(self) -> None:
        with pytest.raises(ValueError, match="tolerance_ms must be non-negative"):
            ResamplingConfig(tolerance_ms=-1.0)

    def test_negative_resampling_stamp_delay(self) -> None:
        with pytest.raises(ValueError, match="max_stamp_delay_ms must be non-negative"):
            ResamplingConfig(max_stamp_delay_ms=-1.0)

    def test_resampling_stamp_delay_zero_allowed(self) -> None:
        """0 is a valid threshold and must not raise."""
        cfg = ResamplingConfig(max_stamp_delay_ms=0.0)
        assert cfg.max_stamp_delay_ms == 0.0

    def test_negative_feature_stamp_delay(self) -> None:
        with pytest.raises(ValueError, match="max_stamp_delay_ms must be non-negative"):
            FeatureMapping(
                key="k",
                topic="/t",
                msg_type="std_msgs/msg/String",
                max_stamp_delay_ms=-5.0,
            )

    def test_duplicate_feature_key(self, tmp_path: Path) -> None:
        cfg_dict = _minimal_config_dict()
        # Give the action the same key as the observation
        cfg_dict["actions"][0]["key"] = "observation.state"
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(cfg_dict))
        with pytest.raises(ValueError, match="Duplicate feature key"):
            load_config(path)

    def test_empty_feature_key(self) -> None:
        with pytest.raises(ValueError, match="key.*must not be empty"):
            FeatureMapping(key="", topic="/t", msg_type="std_msgs/msg/String")

    def test_empty_topic(self) -> None:
        with pytest.raises(ValueError, match="topic.*must not be empty"):
            FeatureMapping(key="k", topic="", msg_type="std_msgs/msg/String")

    def test_empty_msg_type(self) -> None:
        with pytest.raises(ValueError, match="msg_type.*must not be empty"):
            FeatureMapping(key="k", topic="/t", msg_type="")

    def test_invalid_stamp_source(self) -> None:
        with pytest.raises(ValueError, match="Invalid stamp_source"):
            FeatureMapping(
                key="k",
                topic="/t",
                msg_type="std_msgs/msg/String",
                stamp_source="invalid",
            )

    def test_invalid_image_size(self) -> None:
        with pytest.raises(ValueError, match="image_size must have 2"):
            FeatureMapping(
                key="k",
                topic="/t",
                msg_type="sensor_msgs/msg/Image",
                dtype="image",
                image_size=[480],
            )

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_non_mapping_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.yaml"
        path.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="root must be a mapping"):
            load_config(path)


# ---------------------------------------------------------------------------
# Default config tests
# ---------------------------------------------------------------------------


class TestDefaultArmConfig:
    """Test default arm configurations."""

    def test_single_arm_drops_left_wrist(self) -> None:
        cfg = build_default_config(robot_type="single_arm")
        image_keys = [fm.key for fm in cfg.image_features]
        assert "observation.images.right_wrist" in image_keys
        assert "observation.images.left_wrist" not in image_keys

    def test_dual_arm_has_both_wrists(self) -> None:
        cfg = build_default_config(robot_type="dual_arm")
        image_keys = [fm.key for fm in cfg.image_features]
        assert "observation.images.right_wrist" in image_keys
        assert "observation.images.left_wrist" in image_keys

    def test_default_fps(self) -> None:
        cfg = build_default_config()
        assert cfg.fps == 30

    def test_custom_fps(self) -> None:
        cfg = build_default_config(fps=60)
        assert cfg.fps == 60


# ---------------------------------------------------------------------------
# FeatureMapping property tests
# ---------------------------------------------------------------------------


class TestFeatureMapping:
    """Test FeatureMapping helper properties."""

    def test_is_image_by_dtype(self) -> None:
        fm = FeatureMapping(
            key="obs.img",
            topic="/cam",
            msg_type="sensor_msgs/msg/Image",
            dtype="image",
        )
        assert fm.is_image is True

    def test_is_image_by_image_size(self) -> None:
        fm = FeatureMapping(
            key="obs.img",
            topic="/cam",
            msg_type="sensor_msgs/msg/Image",
            dtype="uint8",
            image_size=[480, 640, 3],
        )
        assert fm.is_image is True

    def test_not_image(self) -> None:
        fm = FeatureMapping(
            key="obs.state",
            topic="/js",
            msg_type="sensor_msgs/msg/JointState",
        )
        assert fm.is_image is False

    def test_lerobot_key_passthrough(self) -> None:
        fm = FeatureMapping(
            key="observation.state",
            topic="/js",
            msg_type="sensor_msgs/msg/JointState",
        )
        assert fm.lerobot_key == "observation.state"

    def test_unit_conversion_default(self) -> None:
        fm = FeatureMapping(
            key="k",
            topic="/t",
            msg_type="sensor_msgs/msg/JointState",
        )
        assert fm.unit_conversion == 1.0


# ---------------------------------------------------------------------------
# CustomMsgDef validation
# ---------------------------------------------------------------------------


class TestCustomMsgDef:
    def test_empty_msg_file(self) -> None:
        with pytest.raises(ValueError, match="msg_file.*must not be empty"):
            CustomMsgDef(msg_file="", package="pkg")

    def test_empty_package(self) -> None:
        with pytest.raises(ValueError, match="package.*must not be empty"):
            CustomMsgDef(msg_file="file.msg", package="")


# ---------------------------------------------------------------------------
# Unknown-key (typo) detection (⑪)
# ---------------------------------------------------------------------------


class TestUnknownKeyDetection:
    """Unknown config keys raise ValueError with a difflib suggestion."""

    def test_resampling_typo(self, tmp_path: Path) -> None:
        d = _minimal_config_dict()
        d["resampling"] = {"align_to_require": False}
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump(d))
        with pytest.raises(ValueError, match="did you mean: align_to_required"):
            load_config(path)

    def test_feature_image_size_typo(self, tmp_path: Path) -> None:
        d = _minimal_config_dict()
        d["observations"][0]["image_sizes"] = [480, 640, 3]
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump(d))
        with pytest.raises(ValueError, match="did you mean: image_size"):
            load_config(path)

    def test_top_level_unknown_key(self, tmp_path: Path) -> None:
        # Singular 'observation' is a typo for 'observations'.
        d = _minimal_config_dict()
        d["observation"] = "oops"
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump(d))
        with pytest.raises(ValueError, match="Unknown top-level key"):
            load_config(path)

    def test_split_typo(self, tmp_path: Path) -> None:
        d = _minimal_config_dict()
        d["split"] = {"trian": 1.0}
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump(d))
        with pytest.raises(ValueError, match="did you mean: train"):
            load_config(path)

    def test_custom_msg_typo(self, tmp_path: Path) -> None:
        d = _minimal_config_dict()
        d["custom_msgs"] = [{"msg_file": "x.msg", "packge": "pkg"}]
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump(d))
        with pytest.raises(ValueError, match="did you mean: package"):
            load_config(path)


def test_all_shipped_configs_load() -> None:
    """Regression: every parseable configs/*.yaml loads without error.

    Guards against the unknown-key check rejecting a key actually used by a
    shipped config. ``robot_template.yaml`` is fully commented out (YAML root
    is ``None``) and is not a loadable config, so it is excluded here.
    """
    configs_dir = Path(__file__).resolve().parent.parent / "configs"
    loadable = [
        p
        for p in sorted(configs_dir.glob("*.yaml"))
        if yaml.safe_load(p.read_text()) is not None
    ]
    assert loadable, "expected at least one loadable shipped config"
    for path in loadable:
        load_config(path)  # must not raise


# ---------------------------------------------------------------------------
# SplitConfig + compute_splits (⑨)
# ---------------------------------------------------------------------------


class TestSplitConfig:
    def test_default_is_train_only(self) -> None:
        sc = SplitConfig()
        assert sc.train == 1.0
        assert sc.val == 0.0
        assert sc.test == 0.0
        assert sc.min_length == 0
        assert sc.ratios == {"train": 1.0, "val": 0.0, "test": 0.0}

    def test_three_way_valid(self) -> None:
        sc = SplitConfig(train=0.7, val=0.15, test=0.15)
        assert abs(sc.train + sc.val + sc.test - 1.0) < 1e-9

    def test_ratios_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            SplitConfig(train=0.7, val=0.1, test=0.1)

    def test_ratio_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="in .0, 1."):
            SplitConfig(train=1.5, val=0.0, test=0.0)

    def test_negative_min_length(self) -> None:
        with pytest.raises(ValueError, match="min_length"):
            SplitConfig(train=1.0, min_length=-1)

    def test_split_parsed_from_yaml(self, tmp_path: Path) -> None:
        d = _minimal_config_dict()
        d["split"] = {"train": 0.8, "val": 0.1, "test": 0.1, "min_length": 5}
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump(d))
        cfg = load_config(path)
        assert cfg.split.train == 0.8
        assert cfg.split.min_length == 5

    def test_split_defaults_when_absent(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump(_minimal_config_dict()))
        cfg = load_config(path)
        assert cfg.split == SplitConfig()


class TestComputeSplits:
    def test_default_is_legacy_single_split(self) -> None:
        # Byte-identical to the legacy {"train": "0:N"}.
        assert compute_splits(5, {"train": 1.0, "val": 0.0, "test": 0.0}) == {
            "train": "0:5"
        }

    def test_three_way_contiguous_partition(self) -> None:
        splits = compute_splits(7, {"train": 0.7, "val": 0.15, "test": 0.15})
        # round(0.7*7)=5, round(0.15*7)=1, test absorbs 7-5-1=1.
        assert splits == {"train": "0:5", "val": "5:6", "test": "6:7"}
        _assert_partition(splits, 7)

    def test_zero_width_splits_omitted(self) -> None:
        splits = compute_splits(4, {"train": 0.5, "val": 0.0, "test": 0.5})
        assert "val" not in splits
        _assert_partition(splits, 4)

    def test_test_absorbs_remainder(self) -> None:
        splits = compute_splits(10, {"train": 0.33, "val": 0.33, "test": 0.34})
        _assert_partition(splits, 10)

    def test_ratio_overrun_clamped(self) -> None:
        # round(0.5*3)=2 for both train and val -> naive sum 4 > 3 would force
        # n_test=-1. The clamp keeps it a valid partition of [0, 3).
        splits = compute_splits(3, {"train": 0.5, "val": 0.5, "test": 0.0})
        assert splits == {"train": "0:2", "val": "2:3"}
        _assert_partition(splits, 3)
        _assert_passes_validation(splits, 3)

    def test_small_n_partitions(self) -> None:
        for n in (1, 2):
            for ratios in (
                {"train": 0.5, "val": 0.5, "test": 0.0},
                {"train": 0.34, "val": 0.33, "test": 0.33},
                {"train": 1.0, "val": 0.0, "test": 0.0},
            ):
                splits = compute_splits(n, ratios)
                _assert_partition(splits, n)
                _assert_passes_validation(splits, n)

    def test_default_passes_validation_byte_identical(self) -> None:
        # The default single-split is byte-identical to {"train": "0:N"} and
        # passes the dataset validator's partition check.
        splits = compute_splits(4, {"train": 1.0, "val": 0.0, "test": 0.0})
        assert splits == {"train": "0:4"}
        _assert_passes_validation(splits, 4)


def _assert_passes_validation(splits: dict[str, str], total: int) -> None:
    """Assert ``splits`` is accepted by validation._check_splits_partition."""
    from rosbag2lerobot.validation import (
        DatasetValidationReport,
        _check_splits_partition,
    )

    report = DatasetValidationReport(dataset="x")
    _check_splits_partition(splits, total, report)
    assert report.errors() == [], [i.to_dict() for i in report.errors()]


def _assert_partition(splits: dict[str, str], total: int) -> None:
    """Assert ``splits`` is a contiguous gap-free partition of ``[0, total)``."""
    intervals = sorted(
        (int(v.split(":")[0]), int(v.split(":")[1])) for v in splits.values()
    )
    prev = 0
    for a, b in intervals:
        assert a == prev, (a, prev, splits)
        assert a < b
        prev = b
    assert prev == total, (prev, total, splits)
