"""Unit tests for :mod:`rosbag2lerobot.diagnostics`.

Covers :func:`compute_topic_fps_report`, :func:`detect_image_shape`,
and :func:`validate_config_against_bag`, plus CLI integration for
``inspect --fps-stats`` and ``validate-config``.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from rosbag2lerobot.cli import main
from rosbag2lerobot.config import FeatureMapping, RobotConfig
from rosbag2lerobot.diagnostics import (
    ValidationReport,
    compute_topic_fps_report,
    detect_image_shape,
    validate_config_against_bag,
)
from rosbag2lerobot.reader import BagReader


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_bag(
    bag_path: Path,
    joint_hz: int = 50,
    image_hz: int = 10,
    duration_s: float = 2.0,
    start_ns: int = 1_700_000_000_000_000_000,
    image_shape: tuple[int, int] = (480, 640),
    drop_action: bool = False,
    drop_images: bool = False,
    action_msg_type: str = "sensor_msgs/msg/JointState",
    extra_topic: str | None = None,
    vary_image_shapes: bool = False,
) -> Path:
    """Write a synthetic bag.

    - ``drop_action``: skip writing the action topic entirely.
    - ``drop_images``: skip writing the image topic entirely.
    - ``action_msg_type``: override the action connection's msgtype to
      test mismatch detection. For non-JointState types we still write a
      JointState payload on a connection declaring the wrong type so the
      bag has the mismatch but stays syntactically valid.
    - ``extra_topic``: additional topic name to simulate an unused bag
      topic not referenced by any config.
    - ``vary_image_shapes``: alternate between 480x640 and 240x320
      frames to exercise the "inconsistent shapes" branch of
      :func:`detect_image_shape`.
    """
    if bag_path.exists():
        shutil.rmtree(bag_path)

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    JointState = typestore.types["sensor_msgs/msg/JointState"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]
    CompressedImage = typestore.types["sensor_msgs/msg/CompressedImage"]

    joint_names = [f"j{i}" for i in range(6)]

    with Writer(bag_path, version=9) as writer:
        conn_js = writer.add_connection(
            "/joint_states",
            "sensor_msgs/msg/JointState",
            typestore=typestore,
        )
        conn_action = None
        if not drop_action:
            conn_action = writer.add_connection(
                "/target_joint_positions",
                action_msg_type,
                typestore=typestore,
            )
        conn_img = None
        if not drop_images:
            conn_img = writer.add_connection(
                "/camera/front/image_raw/compressed",
                "sensor_msgs/msg/CompressedImage",
                typestore=typestore,
            )
        conn_extra = None
        if extra_topic is not None:
            conn_extra = writer.add_connection(
                extra_topic,
                "sensor_msgs/msg/JointState",
                typestore=typestore,
            )

        n_joint = int(duration_s * joint_hz)
        for i in range(n_joint):
            t_s = i / joint_hz
            t_ns = start_ns + int(t_s * 1e9)
            sec = int(t_ns // 1_000_000_000)
            nsec = int(t_ns % 1_000_000_000)
            header = Header(
                stamp=Time(sec=sec, nanosec=nsec),
                frame_id="base_link",
            )
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
            writer.write(conn_js, t_ns, serialized)
            if conn_action is not None:
                writer.write(conn_action, t_ns, serialized)
            if conn_extra is not None:
                writer.write(conn_extra, t_ns, serialized)

        if conn_img is not None:
            n_img = int(duration_s * image_hz)
            for i in range(n_img):
                t_s = i / image_hz
                t_ns = start_ns + int(t_s * 1e9)
                sec = int(t_ns // 1_000_000_000)
                nsec = int(t_ns % 1_000_000_000)
                header = Header(
                    stamp=Time(sec=sec, nanosec=nsec),
                    frame_id="cam",
                )
                if vary_image_shapes and i % 2 == 1:
                    h, w = 240, 320
                else:
                    h, w = image_shape
                img = np.full((h, w, 3), 128, dtype=np.uint8)
                _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                img_msg = CompressedImage(
                    header=header,
                    format="jpeg",
                    data=np.frombuffer(jpeg.tobytes(), dtype=np.uint8),
                )
                writer.write(
                    conn_img,
                    t_ns,
                    typestore.serialize_cdr(
                        img_msg,
                        "sensor_msgs/msg/CompressedImage",
                    ),
                )

    return bag_path


def _make_config(image_size: list[int] | None = None) -> RobotConfig:
    """Config matching the fixture bag."""
    return RobotConfig(
        robot_type="test_rig",
        fps=10,
        task="test",
        observations=[
            FeatureMapping(
                key="observation.state",
                topic="/joint_states",
                msg_type="sensor_msgs/msg/JointState",
                selector="position",
                dtype="float32",
            ),
            FeatureMapping(
                key="observation.images.front",
                topic="/camera/front/image_raw/compressed",
                msg_type="sensor_msgs/msg/CompressedImage",
                dtype="image",
                image_size=image_size or [480, 640],
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
    )


# ---------------------------------------------------------------------------
# F1 — compute_topic_fps_report
# ---------------------------------------------------------------------------


class TestComputeFpsReport:
    """Cover the four edge cases called out in plan.md."""

    def test_uniform_30hz(self) -> None:
        bag_start = 1_000_000_000_000
        n = 90
        ts = bag_start + np.arange(n, dtype=np.int64) * int(1e9 / 30)
        bag_end = int(ts[-1])

        r = compute_topic_fps_report(
            ts,
            bag_start,
            bag_end,
            msg_type="sensor_msgs/msg/JointState",
            msg_count=n,
            gap_threshold_ms=200.0,
            head_n=5,
        )

        assert r["msg_count"] == n
        assert r["fps"]["mean"] == pytest.approx(30.0, abs=0.1)
        assert r["fps"]["std"] == pytest.approx(0.0, abs=0.01)
        assert r["gaps"] == []
        assert r["head_lag_ms"] == pytest.approx(0.0, abs=0.001)
        assert len(r["head_intervals_ms"]) == 5

    def test_central_gap(self) -> None:
        bag_start = 0
        base = np.arange(20, dtype=np.int64) * int(1e9 / 30)
        # Insert a 500ms gap by removing the middle entry.
        ts = np.concatenate([base[:10], base[10:] + 500_000_000])
        bag_end = int(ts[-1])

        r = compute_topic_fps_report(
            ts,
            bag_start,
            bag_end,
            msg_type="sensor_msgs/msg/JointState",
            msg_count=int(ts.size),
            gap_threshold_ms=200.0,
            head_n=3,
        )

        assert len(r["gaps"]) == 1
        assert r["gaps"][0]["duration_ms"] == pytest.approx(533.33, abs=1.0)

    def test_head_lag(self) -> None:
        bag_start = 0
        # First message arrives 250ms after the bag starts.
        ts = 250_000_000 + np.arange(30, dtype=np.int64) * int(1e9 / 30)
        bag_end = int(ts[-1])

        r = compute_topic_fps_report(
            ts,
            bag_start,
            bag_end,
            msg_type="sensor_msgs/msg/JointState",
            msg_count=30,
            gap_threshold_ms=200.0,
            head_n=3,
        )
        assert r["head_lag_ms"] == pytest.approx(250.0, abs=0.001)

    def test_tail_lag(self) -> None:
        bag_start = 0
        ts = np.arange(30, dtype=np.int64) * int(1e9 / 30)
        # Bag runs for 300ms longer than the last message.
        bag_end = int(ts[-1]) + 300_000_000

        r = compute_topic_fps_report(
            ts,
            bag_start,
            bag_end,
            msg_type="sensor_msgs/msg/JointState",
            msg_count=30,
            gap_threshold_ms=200.0,
            head_n=3,
        )
        assert r["tail_lag_ms"] == pytest.approx(300.0, abs=0.001)

    def test_near_duplicate_timestamps_mean_is_realistic(self) -> None:
        # Regression: a publisher that emits messages in bursts produces
        # pairs of near-duplicate timestamps (dt of a few microseconds).
        # Averaging the per-interval reciprocals (mean(1/dt)) blows MEAN_FPS
        # up to tens of thousands of fps; the span-based mean must stay at
        # the true effective rate (~50 Hz here).
        bag_start = 1_000_000_000_000
        nominal_dt = int(1e9 / 50)  # 50 Hz nominal spacing (20 ms).
        ts_list: list[int] = []
        t = bag_start
        for _ in range(100):
            ts_list.append(t)
            # A duplicate sample arriving 5 us later (dt -> 1/5us = 200000 fps).
            ts_list.append(t + 5_000)
            t += nominal_dt
        ts = np.asarray(ts_list, dtype=np.int64)
        bag_end = int(ts[-1])

        r = compute_topic_fps_report(
            ts,
            bag_start,
            bag_end,
            msg_type="sensor_msgs/msg/JointState",
            msg_count=int(ts.size),
            gap_threshold_ms=200.0,
            head_n=5,
        )

        mean = r["fps"]["mean"]
        # Effective rate = (intervals) / span. With 200 samples bunched into
        # 100 ~20ms cells the rate is ~100 Hz, and must be nowhere near the
        # ~200000 fps that mean(1/dt) would yield.
        assert mean is not None
        assert 50.0 < mean < 500.0
        # The instantaneous max still reflects the tiny-dt burst (unchanged
        # meaning); only the mean is robust.
        assert r["fps"]["max"] > 10_000.0

    def test_empty_timestamps(self) -> None:
        r = compute_topic_fps_report(
            np.array([], dtype=np.int64),
            0,
            1_000_000_000,
            msg_type="sensor_msgs/msg/JointState",
            msg_count=0,
            gap_threshold_ms=200.0,
            head_n=5,
        )
        assert r["fps"]["mean"] is None
        assert r["head_lag_ms"] is None


# ---------------------------------------------------------------------------
# F3 — detect_image_shape
# ---------------------------------------------------------------------------


class TestDetectImageShape:
    def test_shape_match(self, tmp_path: Path) -> None:
        bag = _write_bag(tmp_path / "bag_match", image_shape=(480, 640))
        cfg = _make_config(image_size=[480, 640])
        img_fm = cfg.image_features[0]

        with BagReader(bag, cfg) as reader:
            shape = detect_image_shape(reader, img_fm, n_samples=3)

        assert shape == (480, 640, 3)

    def test_shape_mismatch(self, tmp_path: Path) -> None:
        bag = _write_bag(tmp_path / "bag_small", image_shape=(240, 320))
        cfg = _make_config(image_size=[480, 640])
        img_fm = cfg.image_features[0]

        with BagReader(bag, cfg) as reader:
            shape = detect_image_shape(reader, img_fm, n_samples=3)

        assert shape == (240, 320, 3)

    def test_inconsistent_samples_returns_none(self, tmp_path: Path) -> None:
        bag = _write_bag(tmp_path / "bag_vary", vary_image_shapes=True)
        cfg = _make_config(image_size=[480, 640])
        img_fm = cfg.image_features[0]

        with BagReader(bag, cfg) as reader:
            shape = detect_image_shape(reader, img_fm, n_samples=5)

        assert shape is None


# ---------------------------------------------------------------------------
# F4 — validate_config_against_bag
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_ok(self, tmp_path: Path) -> None:
        bag = _write_bag(tmp_path / "bag_ok")
        cfg = _make_config()
        with BagReader(bag, cfg) as reader:
            report = validate_config_against_bag(cfg, reader, samples=3)
        report.apply_verdict(strict=False)
        assert report.verdict == "OK"
        assert report.exit_code == 0
        assert not report.missing_required_topics
        assert not report.msg_type_mismatches
        assert not report.image_shape_mismatches

    def test_missing_required(self, tmp_path: Path) -> None:
        bag = _write_bag(tmp_path / "bag_no_action", drop_action=True)
        cfg = _make_config()
        with BagReader(bag, cfg) as reader:
            report = validate_config_against_bag(cfg, reader, samples=3)
        report.apply_verdict(strict=False)
        assert "/target_joint_positions" in report.missing_required_topics
        assert report.exit_code == 1

    def test_missing_optional(self, tmp_path: Path) -> None:
        bag = _write_bag(tmp_path / "bag_no_img", drop_images=True)
        cfg = _make_config()
        # Flag image feature optional to exercise the optional branch.
        cfg.observations[1].optional = True

        with BagReader(bag, cfg) as reader:
            report = validate_config_against_bag(cfg, reader, samples=3)
        report.apply_verdict(strict=False)

        assert "/camera/front/image_raw/compressed" in report.missing_optional_topics
        assert report.exit_code == 0

    def test_msg_type_mismatch(self, tmp_path: Path) -> None:
        bag = _write_bag(
            tmp_path / "bag_badtype",
            action_msg_type="geometry_msgs/msg/Twist",
        )
        cfg = _make_config()
        with BagReader(bag, cfg) as reader:
            report = validate_config_against_bag(cfg, reader, samples=3)
        report.apply_verdict(strict=False)

        topics = [m.topic for m in report.msg_type_mismatches]
        assert "/target_joint_positions" in topics
        assert report.exit_code == 1

    def test_image_shape_mismatch(self, tmp_path: Path) -> None:
        bag = _write_bag(tmp_path / "bag_smallimg", image_shape=(240, 320))
        cfg = _make_config(image_size=[480, 640])
        with BagReader(bag, cfg) as reader:
            report = validate_config_against_bag(cfg, reader, samples=3)
        report.apply_verdict(strict=False)

        assert len(report.image_shape_mismatches) == 1
        assert report.image_shape_mismatches[0].decoded == [240, 320, 3]
        assert report.exit_code == 0  # shape mismatches are warnings by default

        # --strict escalates the image shape mismatch to a failure.
        report2 = ValidationReport(
            image_shape_mismatches=list(report.image_shape_mismatches),
        )
        report2.apply_verdict(strict=True)
        assert report2.exit_code == 1

    def test_unused_bag_topics(self, tmp_path: Path) -> None:
        bag = _write_bag(tmp_path / "bag_extra", extra_topic="/diagnostics")
        cfg = _make_config()
        with BagReader(bag, cfg) as reader:
            report = validate_config_against_bag(cfg, reader, samples=3)

        assert "/diagnostics" in report.unused_bag_topics

        report.apply_verdict(strict=False)
        assert report.exit_code == 0

        report_strict = ValidationReport(
            unused_bag_topics=list(report.unused_bag_topics),
        )
        report_strict.apply_verdict(strict=True)
        assert report_strict.exit_code == 1


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCliIntegration:
    def _run(self, tmp_path: Path, bag: Path, cfg_path: Path, extra: list[str]):
        runner = CliRunner()
        return runner.invoke(main, extra, catch_exceptions=False)

    def test_inspect_fps_stats_json(self, tmp_path: Path) -> None:
        bag = _write_bag(tmp_path / "bag_cli")
        json_path = tmp_path / "fps.json"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "inspect",
                "--bags",
                str(bag),
                "--fps-stats",
                "--json-out",
                str(json_path),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        data = json.loads(json_path.read_text())
        assert "bags" in data
        assert len(data["bags"]) == 1
        topics = {t["name"]: t for t in data["bags"][0]["topics"]}
        assert "/joint_states" in topics

    def test_validate_config_json(self, tmp_path: Path) -> None:
        bag = _write_bag(tmp_path / "bag_cli_vc")
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(_config_yaml_text())
        json_path = tmp_path / "vc.json"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "validate-config",
                "--config",
                str(cfg_path),
                "--bags",
                str(bag),
                "--json-out",
                str(json_path),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(json_path.read_text())
        assert payload["results"]["verdict"] == "OK"
        assert payload["results"]["exit_code"] == 0

    def test_validate_config_missing_required_exits_1(
        self,
        tmp_path: Path,
    ) -> None:
        bag = _write_bag(tmp_path / "bag_cli_fail", drop_action=True)
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(_config_yaml_text())

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "validate-config",
                "--config",
                str(cfg_path),
                "--bags",
                str(bag),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 1


def _config_yaml_text() -> str:
    return """
robot_type: test_rig
fps: 10
task: test
observations:
  - key: observation.state
    topic: /joint_states
    msg_type: sensor_msgs/msg/JointState
    selector: position
    dtype: float32
  - key: observation.images.front
    topic: /camera/front/image_raw/compressed
    msg_type: sensor_msgs/msg/CompressedImage
    dtype: image
    image_size: [480, 640]
actions:
  - key: action
    topic: /target_joint_positions
    msg_type: sensor_msgs/msg/JointState
    selector: position
    dtype: float32
"""
