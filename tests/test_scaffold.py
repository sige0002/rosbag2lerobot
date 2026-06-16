"""Unit tests for ``bagel scaffold`` and :func:`bagel.config.config_to_yaml`.

Covers the pure mapping heuristics (slug derivation, collision dedup, fps
pick, decoder-availability annotation) and the YAML round-trip guarantee on
synthetic data. Real-bag E2E coverage lives in ``test_scaffold_e2e.py``.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
from click.testing import CliRunner
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from bagel.cli import (
    _dedupe_key,
    _pick_target_fps,
    _scaffold_from_topics,
    _slug_from_topic,
    main,
)
from bagel.config import (
    FeatureMapping,
    RobotConfig,
    config_to_yaml,
    load_config,
)
from bagel.reader import TopicInfo


# ---------------------------------------------------------------------------
# Fixture helper — synthetic bag with images + dual-arm JointState + no-decoder
# ---------------------------------------------------------------------------


def _write_bag(
    bag_path: Path,
    joint_hz: int = 50,
    image_hz: int = 10,
    duration_s: float = 2.0,
    start_ns: int = 1_700_000_000_000_000_000,
) -> Path:
    """Write a synthetic bag resembling a dual-arm RealMan recording.

    Topics:
      * /camera/color/image_raw0  (sensor_msgs/msg/Image, 480x640)
      * /camera/depth/image_raw0  (sensor_msgs/msg/Image, 480x640)
      * /left_arm_controller/joint_states  (JointState)
      * /right_arm_controller/joint_states (JointState)
      * /arm/rm_driver/udp_six_force       (rm_ros_interfaces/* -> no decoder)
      * /arm/rm_driver/movej_p_cmd         (command candidate, no decoder)
      * /rosout                            (infra, must be dropped)
    """
    if bag_path.exists():
        shutil.rmtree(bag_path)

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    # Register placeholder no-decoder types so the writer can declare
    # connections with these msg_types (scaffold reads only the type string,
    # never deserializes their payloads).
    typestore.register(
        get_types_from_msg("float64 x\n", "rm_ros_interfaces/msg/Sixforce")
    )
    typestore.register(
        get_types_from_msg("float64 x\n", "rm_ros_interfaces/msg/Movejp")
    )
    JointState = typestore.types["sensor_msgs/msg/JointState"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]
    Image = typestore.types["sensor_msgs/msg/Image"]
    Log = typestore.types["rcl_interfaces/msg/Log"]

    joint_names = [f"j{i}" for i in range(6)]

    with Writer(bag_path, version=9) as writer:
        conn_left = writer.add_connection(
            "/left_arm_controller/joint_states",
            "sensor_msgs/msg/JointState",
            typestore=typestore,
        )
        conn_right = writer.add_connection(
            "/right_arm_controller/joint_states",
            "sensor_msgs/msg/JointState",
            typestore=typestore,
        )
        conn_color = writer.add_connection(
            "/camera/color/image_raw0",
            "sensor_msgs/msg/Image",
            typestore=typestore,
        )
        conn_depth = writer.add_connection(
            "/camera/depth/image_raw0",
            "sensor_msgs/msg/Image",
            typestore=typestore,
        )
        # No-decoder numeric topic: declare an rm_ros_interfaces type but write
        # a JointState payload — the bag stays openable; we only need the type
        # string in topic metadata, which scaffold reads without deserializing.
        conn_force = writer.add_connection(
            "/right_arm_controller/rm_driver/udp_six_force",
            "rm_ros_interfaces/msg/Sixforce",
            typestore=typestore,
            offered_qos_profiles="",
        )
        conn_cmd = writer.add_connection(
            "/right_arm_controller/rm_driver/movej_p_cmd",
            "rm_ros_interfaces/msg/Movejp",
            typestore=typestore,
        )
        conn_log = writer.add_connection(
            "/rosout",
            "rcl_interfaces/msg/Log",
            typestore=typestore,
        )

        n_joint = int(duration_s * joint_hz)
        for i in range(n_joint):
            t_s = i / joint_hz
            t_ns = start_ns + int(t_s * 1e9)
            sec = int(t_ns // 1_000_000_000)
            nsec = int(t_ns % 1_000_000_000)
            header = Header(stamp=Time(sec=sec, nanosec=nsec), frame_id="base")
            positions = np.array(
                [math.sin(0.3 * t_s + j * 0.2) for j in range(6)],
                dtype=np.float64,
            )
            msg = JointState(
                header=header,
                name=np.array(joint_names),
                position=positions,
                velocity=np.zeros(6, dtype=np.float64),
                effort=np.zeros(6, dtype=np.float64),
            )
            serialized = typestore.serialize_cdr(msg, "sensor_msgs/msg/JointState")
            writer.write(conn_left, t_ns, serialized)
            writer.write(conn_right, t_ns, serialized)

        n_img = int(duration_s * image_hz)
        for i in range(n_img):
            t_s = i / image_hz
            t_ns = start_ns + int(t_s * 1e9)
            sec = int(t_ns // 1_000_000_000)
            nsec = int(t_ns % 1_000_000_000)
            header = Header(stamp=Time(sec=sec, nanosec=nsec), frame_id="cam")
            arr = np.full((480, 640, 3), 100, dtype=np.uint8)
            img_msg = Image(
                header=header,
                height=480,
                width=640,
                encoding="rgb8",
                is_bigendian=0,
                step=640 * 3,
                data=arr.reshape(-1),
            )
            serialized = typestore.serialize_cdr(img_msg, "sensor_msgs/msg/Image")
            writer.write(conn_color, t_ns, serialized)
            writer.write(conn_depth, t_ns, serialized)

        # A couple of force / command / log messages (low rate). We only need
        # these connections to appear in the topic metadata with the right
        # msg_type; the payload itself is never deserialized by scaffold, so a
        # placeholder empty JointState payload is fine for the non-Log topics.
        placeholder = typestore.serialize_cdr(
            JointState(
                header=Header(stamp=Time(sec=0, nanosec=0), frame_id=""),
                name=np.array([]),
                position=np.zeros(0),
                velocity=np.zeros(0),
                effort=np.zeros(0),
            ),
            "sensor_msgs/msg/JointState",
        )
        log_msg = Log(
            stamp=Time(sec=0, nanosec=0),
            level=20,
            name="n",
            msg="m",
            file="f",
            function="fn",
            line=1,
        )
        log_payload = typestore.serialize_cdr(log_msg, "rcl_interfaces/msg/Log")
        for i in range(3):
            t_ns = start_ns + int(i * 1e9)
            writer.write(conn_force, t_ns, placeholder)
            writer.write(conn_cmd, t_ns, placeholder)
            writer.write(conn_log, t_ns, log_payload)

    return bag_path


# ---------------------------------------------------------------------------
# Slug / collision / dedup
# ---------------------------------------------------------------------------


class TestSlug:
    def test_color_depth_mono_distinct(self) -> None:
        assert _slug_from_topic("/camera/color/image_raw0") == "camera_color_0"
        assert _slug_from_topic("/camera/depth/image_raw0") == "camera_depth_0"
        assert _slug_from_topic("/camera/image_raw0") == "camera_0"

    def test_numeric_suffix_preserved(self) -> None:
        assert _slug_from_topic("/camera/color/image_raw2") == "camera_color_2"

    def test_sanitization(self) -> None:
        # Non-alnum chars collapse to underscores; result is lower-case [a-z0-9_].
        slug = _slug_from_topic("/My-Cam/Foo.Bar")
        assert all(c.isalnum() or c == "_" for c in slug)
        assert slug == slug.lower()

    def test_dedupe_key(self) -> None:
        used: set[str] = set()
        assert _dedupe_key("observation.state", used) == "observation.state"
        assert _dedupe_key("observation.state", used) == "observation.state_2"
        assert _dedupe_key("observation.state", used) == "observation.state_3"


# ---------------------------------------------------------------------------
# FPS pick
# ---------------------------------------------------------------------------


class TestPickFps:
    def test_min_over_images(self) -> None:
        fps = _pick_target_fps(
            image_topics=["/a", "/b"],
            state_topics=["/c"],
            fps_by_topic={"/a": 30.4, "/b": 14.6, "/c": 200.0},
        )
        # min image fps = 14.6 -> round = 15 (state fps ignored when images exist)
        assert fps == 15

    def test_median_state_when_no_images(self) -> None:
        fps = _pick_target_fps(
            image_topics=[],
            state_topics=["/a", "/b", "/c"],
            fps_by_topic={"/a": 10.0, "/b": 50.0, "/c": 90.0},
        )
        assert fps == 50

    def test_fallback_30(self) -> None:
        assert _pick_target_fps([], [], {}) == 30


# ---------------------------------------------------------------------------
# Mapping heuristics via _scaffold_from_topics (pure)
# ---------------------------------------------------------------------------


def _topics_info() -> dict[str, TopicInfo]:
    return {
        "/camera/color/image_raw0": TopicInfo("sensor_msgs/msg/Image", 100),
        "/camera/depth/image_raw0": TopicInfo("sensor_msgs/msg/Image", 100),
        "/camera/image_raw0": TopicInfo("sensor_msgs/msg/Image", 150),
        "/left_arm_controller/joint_states": TopicInfo(
            "sensor_msgs/msg/JointState", 5000
        ),
        "/right_arm_controller/joint_states": TopicInfo(
            "sensor_msgs/msg/JointState", 5000
        ),
        "/right_arm_controller/rm_driver/udp_six_force": TopicInfo(
            "rm_ros_interfaces/msg/Sixforce", 5000
        ),
        "/right_arm_controller/rm_driver/movej_p_cmd": TopicInfo(
            "rm_ros_interfaces/msg/Movejp", 5
        ),
        "/rosout": TopicInfo("rcl_interfaces/msg/Log", 200),
        "/rosbag/status": TopicInfo("std_msgs/msg/String", 50),
        "/zero_count": TopicInfo("std_msgs/msg/Bool", 0),
    }


class TestScaffoldHeuristics:
    def _build(self) -> tuple:
        return _scaffold_from_topics(
            topics_info=_topics_info(),
            fps_by_topic={
                "/camera/color/image_raw0": 15.0,
                "/camera/depth/image_raw0": 15.0,
                "/camera/image_raw0": 20.0,
                "/left_arm_controller/joint_states": 200.0,
                "/right_arm_controller/joint_states": 200.0,
            },
            image_shapes={
                "/camera/color/image_raw0": (480, 640, 3),
                "/camera/depth/image_raw0": (480, 640, 3),
                "/camera/image_raw0": None,  # detection failed
            },
            registered={"sensor_msgs/msg/JointState", "sensor_msgs/msg/Image"},
            robot_type="rig",
            task="t",
            fps_override=None,
            min_count=1,
            bag_name="bag0",
        )

    def test_image_collision_disambiguation(self) -> None:
        cfg, _ann, _obsc, _actc = self._build()
        keys = {fm.key for fm in cfg.observations}
        assert "observation.images.camera_color_0" in keys
        assert "observation.images.camera_depth_0" in keys
        assert "observation.images.camera_0" in keys

    def test_dual_arm_both_states(self) -> None:
        cfg, _ann, _obsc, _actc = self._build()
        keys = {fm.key for fm in cfg.observations}
        assert "observation.state_left" in keys
        assert "observation.state_right" in keys

    def test_infra_and_zero_count_dropped(self) -> None:
        cfg, _ann, obs_cand, _actc = self._build()
        topics = {fm.topic for fm in cfg.observations}
        assert "/rosout" not in topics
        assert "/rosbag/status" not in topics
        assert "/zero_count" not in topics
        # And they are not surfaced as candidates either.
        cand_text = "\n".join(obs_cand)
        assert "/rosout" not in cand_text
        assert "/zero_count" not in cand_text

    def test_target_fps_min_image(self) -> None:
        cfg, _ann, _obsc, _actc = self._build()
        assert cfg.fps == 15

    def test_no_decoder_is_commented_candidate(self) -> None:
        _cfg, _ann, obs_cand, _actc = self._build()
        cand_text = "\n".join(obs_cand)
        assert "/right_arm_controller/rm_driver/udp_six_force" in cand_text
        assert "decoder: NONE" in cand_text

    def test_command_is_action_candidate(self) -> None:
        _cfg, _ann, _obsc, act_cand = self._build()
        act_text = "\n".join(act_cand)
        assert "/right_arm_controller/rm_driver/movej_p_cmd" in act_text

    def test_decoder_availability_annotation(self) -> None:
        _cfg, ann, _obsc, _actc = self._build()
        # JointState (registered) -> "decoder: builtin"
        left = ann["observation.state_left"]
        assert any("decoder: builtin" in n for n in left)
        # Measured fps surfaced.
        assert any("measured fps" in n for n in left)

    def test_image_size_todo_on_failed_detection(self) -> None:
        _cfg, ann, _obsc, _actc = self._build()
        mono = ann["observation.images.camera_0"]
        assert any("TODO image_size" in n for n in mono)

    def test_fps_override(self) -> None:
        cfg, _ann, _obsc, _actc = _scaffold_from_topics(
            topics_info=_topics_info(),
            fps_by_topic={},
            image_shapes={},
            registered={"sensor_msgs/msg/JointState", "sensor_msgs/msg/Image"},
            robot_type="rig",
            task="t",
            fps_override=12,
            min_count=1,
            bag_name="bag0",
        )
        assert cfg.fps == 12

    def test_min_count_filter(self) -> None:
        # Raising min_count above the cmd count drops the cmd candidate.
        _cfg, _ann, _obsc, act_cand = _scaffold_from_topics(
            topics_info=_topics_info(),
            fps_by_topic={},
            image_shapes={},
            registered={"sensor_msgs/msg/JointState", "sensor_msgs/msg/Image"},
            robot_type="rig",
            task="t",
            fps_override=10,
            min_count=10,
            bag_name="bag0",
        )
        act_text = "\n".join(act_cand)
        assert "movej_p_cmd" not in act_text


# ---------------------------------------------------------------------------
# config_to_yaml round-trip
# ---------------------------------------------------------------------------


class TestConfigToYaml:
    def _sample_cfg(self) -> RobotConfig:
        return RobotConfig(
            robot_type="rig",
            fps=15,
            task="pick and place",
            repo_id="org/ds",
            observations=[
                FeatureMapping(
                    key="observation.images.cam",
                    topic="/cam",
                    msg_type="sensor_msgs/msg/Image",
                    dtype="image",
                    image_size=[480, 640, 3],
                ),
                FeatureMapping(
                    key="observation.state",
                    topic="/js",
                    msg_type="sensor_msgs/msg/JointState",
                    selector="position",
                ),
            ],
            actions=[
                FeatureMapping(
                    key="action",
                    topic="/cmd",
                    msg_type="sensor_msgs/msg/JointState",
                    selector="position",
                ),
            ],
        )

    def test_round_trip_equal(self, tmp_path: Path) -> None:
        cfg = self._sample_cfg()
        text = config_to_yaml(
            cfg,
            header_lines=["header note"],
            obs_annotations={"observation.state": ["measured fps: 50.00"]},
            obs_candidates=["  # candidate"],
            act_candidates=["  # action candidate"],
        )
        path = tmp_path / "out.yaml"
        path.write_text(text)
        loaded = load_config(path)

        assert loaded.robot_type == cfg.robot_type
        assert loaded.fps == cfg.fps
        assert loaded.task == cfg.task
        assert loaded.repo_id == cfg.repo_id
        assert [f.key for f in loaded.observations] == [f.key for f in cfg.observations]
        assert [f.topic for f in loaded.observations] == [
            f.topic for f in cfg.observations
        ]
        assert loaded.observations[0].image_size == [480, 640, 3]
        assert loaded.observations[1].selector == "position"
        assert [f.key for f in loaded.actions] == ["action"]

    def test_empty_actions_round_trip(self, tmp_path: Path) -> None:
        cfg = RobotConfig(
            robot_type="rig",
            fps=10,
            task="t",
            observations=[
                FeatureMapping(
                    key="observation.state",
                    topic="/js",
                    msg_type="sensor_msgs/msg/JointState",
                    selector="position",
                ),
            ],
            actions=[],
        )
        text = config_to_yaml(cfg)
        assert "actions: []" in text
        path = tmp_path / "out.yaml"
        path.write_text(text)
        loaded = load_config(path)
        assert loaded.actions == []

    def test_non_default_resampling_emitted(self, tmp_path: Path) -> None:
        from bagel.config import ResamplingConfig

        cfg = RobotConfig(
            robot_type="rig",
            fps=10,
            task="t",
            observations=[
                FeatureMapping(
                    key="observation.state",
                    topic="/js",
                    msg_type="sensor_msgs/msg/JointState",
                    selector="position",
                ),
            ],
            actions=[],
            resampling=ResamplingConfig(default_policy="nearest", tolerance_ms=100.0),
        )
        text = config_to_yaml(cfg)
        assert "resampling:" in text
        path = tmp_path / "out.yaml"
        path.write_text(text)
        loaded = load_config(path)
        assert loaded.resampling.default_policy == "nearest"
        assert loaded.resampling.tolerance_ms == 100.0


# ---------------------------------------------------------------------------
# CLI on a synthetic bag (fast)
# ---------------------------------------------------------------------------


class TestScaffoldCli:
    def test_scaffold_to_file_round_trips(self, tmp_path: Path) -> None:
        bag = _write_bag(tmp_path / "bag")
        out = tmp_path / "config.yaml"
        result = CliRunner().invoke(
            main,
            [
                "scaffold",
                "--bags",
                str(bag),
                "-o",
                str(out),
                "--robot-type",
                "rig",
                "--task",
                "demo",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

        cfg = load_config(out)
        keys = {fm.key for fm in cfg.observations}
        assert "observation.images.camera_color_0" in keys
        assert "observation.images.camera_depth_0" in keys
        assert "observation.state_left" in keys
        assert "observation.state_right" in keys

        raw = out.read_text()
        # No-decoder rm_ros_interfaces topic appears only as a commented candidate.
        assert "/right_arm_controller/rm_driver/udp_six_force" in raw
        assert "decoder: NONE" in raw
        # Command topic is a commented action candidate.
        assert "movej_p_cmd" in raw

    def test_scaffold_stdout(self, tmp_path: Path) -> None:
        bag = _write_bag(tmp_path / "bag2")
        result = CliRunner().invoke(main, ["scaffold", "--bags", str(bag)])
        assert result.exit_code == 0, result.output
        assert "observations:" in result.output
        assert "actions: []" in result.output

    def test_no_validate_flag(self, tmp_path: Path) -> None:
        bag = _write_bag(tmp_path / "bag3")
        out = tmp_path / "c.yaml"
        result = CliRunner().invoke(
            main,
            ["scaffold", "--bags", str(bag), "-o", str(out), "--no-validate"],
        )
        assert result.exit_code == 0, result.output
        # Validation summary should not be printed.
        assert "Config validation:" not in result.output
        # Sanity: still a valid config.
        load_config(out)

    def test_cli_and_ui_scaffold_yaml_equivalent(self, tmp_path: Path) -> None:
        """The UI ``Api.scaffold`` path produces the same YAML as the CLI.

        Both routes go through :func:`bagel.cli.scaffold_config_yaml` with the
        same defaults, so the YAML must be byte-identical (modulo the trailing
        newline ``click.echo`` adds to the CLI stdout form).
        """
        from bagel.cli import scaffold_config_yaml
        from bagel.ui.api import Api
        from bagel.ui.jobs import JobRegistry
        from bagel.ui.security import Root

        bag = _write_bag(tmp_path / "bag4")

        # CLI form: the shared helper with the CLI's own defaults.
        cli_yaml = scaffold_config_yaml(
            bag,
            robot_type="rig",
            task="demo",
            fps=None,
            min_count=1,
            samples=3,
        )

        # UI form: drive Api.scaffold over a bags root containing the bag.
        api = Api(
            roots=[
                Root(id=0, label="bags", path=tmp_path),
                Root(id=1, label="output", path=tmp_path / "out"),
            ],
            token="t" * 40,
            registry=JobRegistry(),
        )
        resp = api.scaffold({"bags": [str(bag)], "robot_type": "rig", "task": "demo"})
        assert resp["yaml"] == cli_yaml
        assert resp["command"].startswith("bagel scaffold")
