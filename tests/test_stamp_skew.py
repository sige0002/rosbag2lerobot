"""Tests for the header/receive timestamp skew guard.

A message carries a publisher-written ``header.stamp`` and a recorder-written
receive time. When those disagree by more than transport latency explains —
an unsynchronised clock — the converter used to adopt the header stamp anyway
and produce a dataset whose timing was silently wrong. These tests pin down
the loud failure that replaced that:

  * ``timestamps.max_header_receive_skew_ms`` parsing (enabled by default, an
    explicit ``null`` disables it, unknown keys are rejected);
  * the guard fires per message, naming the bag/topic/feature;
  * exactly where it does *not* apply: ``stamp_source: receive`` features,
    messages without a header stamp, and messages already dropped by
    ``resampling.max_stamp_delay_ms``;
  * the same guard on TF inputs, which reach the pipeline by a different
    route (a ``TFMessage`` has no header of its own) and corrupt in a
    different way (a frozen pose rather than shifted sample times);
  * the CLI contract — abort by default, record-and-continue with
    ``--skip-failed``.

Bags are synthesized by the ``tiny_bag`` / ``tf_bag`` fixtures, so nothing
here depends on the gitignored ``bagdata/`` tree.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner, Result

from rosbag2lerobot.cli import main
from rosbag2lerobot.cli.convert import _process_episode
from rosbag2lerobot.config import (
    RobotConfig,
    TimestampsConfig,
    config_to_yaml,
    load_config,
)
from rosbag2lerobot.resampler import Resampler
from rosbag2lerobot.timestamps import (
    StampSkewError,
    format_skew_error,
    format_tf_skew_error,
    tf_skews,
)

from .conftest import ACTION_TOPIC, STATE_TOPIC, tf_config_yaml, tiny_config_yaml

ONE_HOUR_NS = 3_600 * 1_000_000_000


def _resampler(cfg: RobotConfig) -> Resampler:
    return Resampler(
        fps=cfg.fps,
        policy=cfg.resampling.default_policy,
        tolerance_ms=cfg.resampling.tolerance_ms,
    )


def _convert(tmp_path: Path, bags: Path, cfg_path: Path, *extra: str) -> Result:
    return CliRunner().invoke(
        main,
        [
            "convert",
            "--config",
            str(cfg_path),
            "--bags",
            str(bags),
            "--output",
            str(tmp_path / "out"),
            *extra,
        ],
    )


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


class TestTimestampsConfig:
    def test_enabled_by_default(self) -> None:
        """Silent corruption is worse than a new error: the guard ships on."""
        assert TimestampsConfig().max_header_receive_skew_ms == 60_000.0

    def test_negative_threshold_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            TimestampsConfig(max_header_receive_skew_ms=-1.0)

    def test_absent_section_keeps_default(self, tmp_path: Path) -> None:
        cfg = load_config(tiny_config_yaml(tmp_path / "c.yaml"))
        assert cfg.timestamps.max_header_receive_skew_ms == 60_000.0

    def test_explicit_null_disables(self, tmp_path: Path) -> None:
        """An explicit null must not collapse into 'key absent' (= default)."""
        cfg = load_config(
            tiny_config_yaml(
                tmp_path / "c.yaml",
                extra="\ntimestamps:\n  max_header_receive_skew_ms: null\n",
            )
        )
        assert cfg.timestamps.max_header_receive_skew_ms is None

    def test_value_is_parsed_as_float(self, tmp_path: Path) -> None:
        cfg = load_config(
            tiny_config_yaml(
                tmp_path / "c.yaml",
                extra="\ntimestamps:\n  max_header_receive_skew_ms: 250\n",
            )
        )
        assert cfg.timestamps.max_header_receive_skew_ms == 250.0

    def test_unknown_key_rejected_with_suggestion(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="max_header_receive_skew_ms"):
            load_config(
                tiny_config_yaml(
                    tmp_path / "c.yaml",
                    extra="\ntimestamps:\n  max_header_receive_skew_m: 250\n",
                )
            )

    def test_yaml_round_trip(self, tmp_path: Path) -> None:
        """A non-default threshold survives config_to_yaml -> load_config."""
        cfg = load_config(
            tiny_config_yaml(
                tmp_path / "c.yaml",
                extra="\ntimestamps:\n  max_header_receive_skew_ms: 1500\n",
            )
        )
        rewritten = tmp_path / "round.yaml"
        rewritten.write_text(config_to_yaml(cfg))
        assert load_config(rewritten).timestamps == cfg.timestamps

        cfg.timestamps = TimestampsConfig(max_header_receive_skew_ms=None)
        rewritten.write_text(config_to_yaml(cfg))
        assert load_config(rewritten).timestamps.max_header_receive_skew_ms is None


def test_format_skew_error_is_actionable() -> None:
    msg = format_skew_error(
        bag_path="/bags/ep0",
        topic="/joint_states",
        feature_key="observation.state",
        header_ns=1_700_000_000_000_000_000,
        receive_ns=1_700_000_000_000_000_000 + ONE_HOUR_NS,
        threshold_ms=60_000.0,
    )
    # Names what tripped, by how much, against which limit.
    assert "/bags/ep0" in msg
    assert "/joint_states" in msg
    assert "observation.state" in msg
    assert "3600000 ms" in msg
    assert "60000 ms" in msg
    # And how to get out of it.
    assert "stamp_source: receive" in msg
    assert "max_header_receive_skew_ms" in msg


# ---------------------------------------------------------------------------
# Pipeline behaviour
# ---------------------------------------------------------------------------


class TestGuardFires:
    def test_skew_beyond_threshold_fails_the_episode(self, tmp_path, tiny_bag) -> None:
        bag = tiny_bag(header_offset_ns=ONE_HOUR_NS)
        cfg = load_config(tiny_config_yaml(tmp_path / "c.yaml"))

        with pytest.raises(StampSkewError) as excinfo:
            _process_episode(bag, cfg, _resampler(cfg))

        message = str(excinfo.value)
        assert STATE_TOPIC in message or ACTION_TOPIC in message
        assert "3600000 ms" in message

    def test_skew_within_threshold_converts(self, tmp_path, tiny_bag) -> None:
        """A 100 ms divergence is ordinary latency, not a broken clock."""
        bag = tiny_bag(header_offset_ns=100_000_000)
        cfg = load_config(tiny_config_yaml(tmp_path / "c.yaml"))
        assert _process_episode(bag, cfg, _resampler(cfg))

    def test_threshold_is_the_boundary(self, tmp_path, tiny_bag) -> None:
        """Exactly at the threshold passes; just over it fails."""
        bag = tiny_bag(header_offset_ns=1_000_000_000)  # 1000 ms
        at_limit = load_config(
            tiny_config_yaml(
                tmp_path / "at.yaml",
                extra="\ntimestamps:\n  max_header_receive_skew_ms: 1000\n",
            )
        )
        assert _process_episode(bag, at_limit, _resampler(at_limit))

        just_under = load_config(
            tiny_config_yaml(
                tmp_path / "under.yaml",
                extra="\ntimestamps:\n  max_header_receive_skew_ms: 999.9\n",
            )
        )
        with pytest.raises(StampSkewError):
            _process_episode(bag, just_under, _resampler(just_under))

    def test_mid_bag_divergence_is_caught(self, tmp_path, tiny_bag) -> None:
        """A clock that jumps part-way through a recording still fails."""
        bag = tiny_bag(header_offset_ns=ONE_HOUR_NS, offset_from_index=15)
        cfg = load_config(tiny_config_yaml(tmp_path / "c.yaml"))
        with pytest.raises(StampSkewError):
            _process_episode(bag, cfg, _resampler(cfg))


class TestGuardDoesNotFire:
    def test_disabled_by_null(self, tmp_path, tiny_bag) -> None:
        bag = tiny_bag(header_offset_ns=ONE_HOUR_NS)
        cfg = load_config(
            tiny_config_yaml(
                tmp_path / "c.yaml",
                extra="\ntimestamps:\n  max_header_receive_skew_ms: null\n",
            )
        )
        assert _process_episode(bag, cfg, _resampler(cfg))

    def test_receive_stamp_source_is_unaffected(self, tmp_path, tiny_bag) -> None:
        """With stamp_source: receive the header stamp is never adopted, so a
        divergent one cannot corrupt the output and must not fail the run."""
        bag = tiny_bag(header_offset_ns=ONE_HOUR_NS)
        cfg = load_config(tiny_config_yaml(tmp_path / "c.yaml", stamp_source="receive"))
        assert _process_episode(bag, cfg, _resampler(cfg))

    def test_stale_dropped_messages_are_exempt(self, tmp_path, tiny_bag) -> None:
        """max_stamp_delay_ms is an explicit 'discard these' policy: messages it
        drops are handled, so the guard must not also abort on them."""
        bag = tiny_bag(header_offset_ns=ONE_HOUR_NS)
        cfg = load_config(
            tiny_config_yaml(
                tmp_path / "c.yaml",
                extra="  max_stamp_delay_ms: 500.0\n",
            )
        )
        assert cfg.resampling.max_stamp_delay_ms == 500.0
        # Every message is dropped as stale -> empty episode, but no exception.
        assert _process_episode(bag, cfg, _resampler(cfg)) == []

    def test_messages_without_header_stamps_are_unaffected(
        self, tmp_path, tiny_bag
    ) -> None:
        """Nothing to compare against: an unset header stamp falls back to the
        receive time, which no threshold can invalidate."""
        bag = tiny_bag(unset_header_stamp=True)
        cfg = load_config(tiny_config_yaml(tmp_path / "c.yaml"))
        assert _process_episode(bag, cfg, _resampler(cfg))


# ---------------------------------------------------------------------------
# TF inputs
# ---------------------------------------------------------------------------


def _tf_transform(parent: str, child: str, sec: int, nanosec: int = 0):
    """A stand-in for one geometry_msgs/TransformStamped."""
    return SimpleNamespace(
        header=SimpleNamespace(
            frame_id=parent, stamp=SimpleNamespace(sec=sec, nanosec=nanosec)
        ),
        child_frame_id=child,
    )


class TestTfSkews:
    """Unit-level: which transforms trip the limit."""

    def test_empty_when_all_within_limit(self) -> None:
        msg = SimpleNamespace(
            transforms=[_tf_transform("odom", "base_link", 100, 500_000_000)]
        )
        assert tf_skews(msg, 100_000_000_000, 1_000_000_000) == []

    def test_reports_every_offender_with_its_frames(self) -> None:
        """All of them, not just the first: which ones matter is decided later,
        from the frame paths the configured features actually read through."""
        msg = SimpleNamespace(
            transforms=[
                _tf_transform("odom", "base_link", 100),
                _tf_transform("base_link", "arm_link", 4_000),
                _tf_transform("arm_link", "tool", 9_000),
            ]
        )
        offenders = tf_skews(msg, 100_000_000_000, 1_000_000_000)
        assert [(p, c) for p, c, _ in offenders] == [
            ("base_link", "arm_link"),
            ("arm_link", "tool"),
        ]
        assert offenders[0][2] == 4_000 * 1_000_000_000

    def test_unset_stamps_are_not_offenders(self) -> None:
        """sec and nanosec both 0 means "never stamped", not "stamped at epoch";
        the receive-time fallback in add_dynamic handles those instead."""
        msg = SimpleNamespace(transforms=[_tf_transform("odom", "base_link", 0, 0)])
        assert tf_skews(msg, 100_000_000_000, 1_000_000_000) == []

    def test_empty_message_is_fine(self) -> None:
        assert tf_skews(SimpleNamespace(transforms=[]), 1, 1) == []


def test_format_tf_skew_error_names_the_transform() -> None:
    msg = format_tf_skew_error(
        bag_path="/bags/ep3",
        topic="/tf",
        parent_frame="odom",
        child_frame="base_link",
        header_ns=1_700_000_000_000_000_000 + ONE_HOUR_NS,
        receive_ns=1_700_000_000_000_000_000,
        threshold_ms=60_000.0,
        feature_key="observation.ee_pose",
    )
    assert "observation.ee_pose" in msg
    assert "/bags/ep3" in msg
    assert "'odom' -> 'base_link'" in msg
    assert "3600000 ms" in msg
    assert "ahead of" in msg
    # The TF-specific damage and the TF-specific way out.
    assert "never moves" in msg
    assert "max_header_receive_skew_ms" in msg
    # stamp_source is not a lever for TF features; it must not be suggested.
    assert "stamp_source" not in msg


class TestTfGuard:
    """A TFMessage reaches TransformLookup by its own route, and is guarded there."""

    def test_skewed_dynamic_tf_fails_the_episode(self, tmp_path, tf_bag) -> None:
        bag = tf_bag(tf_header_offset_ns=ONE_HOUR_NS)
        cfg = load_config(tf_config_yaml(tmp_path / "c.yaml"))

        with pytest.raises(StampSkewError) as excinfo:
            _process_episode(bag, cfg, _resampler(cfg))

        message = str(excinfo.value)
        assert "/tf" in message
        assert "'odom' -> 'base_link'" in message
        assert "3600000 ms" in message

    def test_this_is_what_the_guard_prevents(self, tmp_path, tf_bag) -> None:
        """Turn the check off and the corruption is right there: every frame
        gets the same pose, because the lookup timeline sits an hour away from
        the frame grid and nearest-in-time clamps to its edge — a TF feature
        that looks valid and never moves, with the run reporting success."""
        bag = tf_bag(tf_header_offset_ns=ONE_HOUR_NS)
        cfg = load_config(
            tf_config_yaml(
                tmp_path / "off.yaml",
                extra="\ntimestamps:\n  max_header_receive_skew_ms: null\n",
            )
        )

        frames = _process_episode(bag, cfg, _resampler(cfg))

        poses = {tuple(f["observation.ee_pose"]) for f in frames}
        assert len(frames) > 1
        assert len(poses) == 1  # frozen

    def test_healthy_tf_still_converts_and_moves(self, tmp_path, tf_bag) -> None:
        bag = tf_bag()
        cfg = load_config(tf_config_yaml(tmp_path / "c.yaml"))

        frames = _process_episode(bag, cfg, _resampler(cfg))

        poses = {tuple(f["observation.ee_pose"]) for f in frames}
        assert len(poses) == len(frames)  # a distinct pose per frame

    def test_tf_within_threshold_converts(self, tmp_path, tf_bag) -> None:
        """100 ms of publisher latency is not a broken clock."""
        bag = tf_bag(tf_header_offset_ns=100_000_000)
        cfg = load_config(tf_config_yaml(tmp_path / "c.yaml"))
        assert _process_episode(bag, cfg, _resampler(cfg))

    def test_static_tf_skew_is_exempt(self, tmp_path, tf_bag) -> None:
        """/tf_static is latched, so its stamps are legitimately old — and
        TransformLookup.add_static discards them anyway, so they cannot move
        a pose. Failing on them would break every real robot bag."""
        bag = tf_bag(static_header_offset_ns=-100 * ONE_HOUR_NS)
        cfg = load_config(tf_config_yaml(tmp_path / "c.yaml"))
        assert _process_episode(bag, cfg, _resampler(cfg))

    def test_unset_tf_stamps_fall_back_to_the_receive_time(
        self, tmp_path, tf_bag
    ) -> None:
        """A publisher that never stamps its transforms must not freeze the
        pose. Exempting an unset stamp from the *error* is not enough: taken
        literally it is 1970, so the transform would enter the timeline there
        and every lookup would clamp to it — the same corruption the guard
        exists to prevent, reached by a different door. The receive time is the
        fallback the non-TF path already uses for an unstamped message."""
        bag = tf_bag(unset_tf_stamp=True)
        cfg = load_config(tf_config_yaml(tmp_path / "c.yaml"))

        frames = _process_episode(bag, cfg, _resampler(cfg))

        poses = {tuple(f["observation.ee_pose"]) for f in frames}
        assert len(frames) > 1
        assert len(poses) == len(frames)  # moving, not frozen

    def test_skew_on_an_unused_edge_does_not_abort(self, tmp_path, tf_bag) -> None:
        """One sensor on a bad clock must not fail a conversion that never
        looks through its frames — the output would have been correct, and the
        only escape hatch also disables the check where it matters."""
        bag = tf_bag(
            extra_dynamic_frame="imu_link", extra_dynamic_offset_ns=ONE_HOUR_NS
        )
        cfg = load_config(tf_config_yaml(tmp_path / "c.yaml"))

        frames = _process_episode(bag, cfg, _resampler(cfg))

        poses = {tuple(f["observation.ee_pose"]) for f in frames}
        assert len(poses) == len(frames)

    def test_skew_on_a_used_edge_still_aborts_with_others_present(
        self, tmp_path, tf_bag
    ) -> None:
        """The scoping must not swallow the case that matters: the configured
        path is skewed here, and an unrelated edge is fine."""
        bag = tf_bag(tf_header_offset_ns=ONE_HOUR_NS, extra_dynamic_frame="imu_link")
        cfg = load_config(tf_config_yaml(tmp_path / "c.yaml"))

        with pytest.raises(StampSkewError) as excinfo:
            _process_episode(bag, cfg, _resampler(cfg))

        message = str(excinfo.value)
        assert "'odom' -> 'base_link'" in message
        assert "observation.ee_pose" in message
        assert "imu_link" not in message

    def test_disabled_guard_skips_the_tf_check(self, tmp_path, tf_bag) -> None:
        bag = tf_bag(tf_header_offset_ns=ONE_HOUR_NS)
        cfg = load_config(
            tf_config_yaml(
                tmp_path / "c.yaml",
                extra="\ntimestamps:\n  max_header_receive_skew_ms: null\n",
            )
        )
        assert _process_episode(bag, cfg, _resampler(cfg))

    def test_cli_aborts_and_skip_failed_continues(self, tmp_path, tf_bag) -> None:
        """Same two semantics as the per-feature guard, over the TF path."""
        tf_bag(name="bags/ep0")
        tf_bag(name="bags/ep1", tf_header_offset_ns=ONE_HOUR_NS)
        cfg_path = tf_config_yaml(tmp_path / "c.yaml")

        aborted = _convert(tmp_path, tmp_path / "bags", cfg_path)
        assert aborted.exit_code != 0
        assert "timestamp skew" in aborted.output
        assert not (tmp_path / "out" / "meta" / "info.json").exists()

        skipped = _convert(
            tmp_path, tmp_path / "bags", cfg_path, "--skip-failed", "--resume"
        )
        assert skipped.exit_code == 0, skipped.output
        info = json.loads((tmp_path / "out" / "meta" / "info.json").read_text())
        assert info["total_episodes"] == 1
        summary = json.loads(
            (tmp_path / "out" / "meta" / "job_summary.json").read_text()
        )
        failed = [ep for ep in summary["episodes"] if not ep["success"]]
        assert len(failed) == 1
        assert "'odom' -> 'base_link'" in failed[0]["error"]


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


class TestCliInterplay:
    def test_run_aborts_without_skip_failed(self, tmp_path, tiny_bag) -> None:
        tiny_bag(name="bags/ep0")
        tiny_bag(name="bags/ep1", header_offset_ns=ONE_HOUR_NS)
        cfg_path = tiny_config_yaml(tmp_path / "c.yaml")

        result = _convert(tmp_path, tmp_path / "bags", cfg_path)

        assert result.exit_code != 0
        assert "timestamp skew" in result.output
        assert "max_header_receive_skew_ms" in result.output
        # Aborted before finalize -> no dataset claims to be complete.
        assert not (tmp_path / "out" / "meta" / "info.json").exists()

    def test_skip_failed_records_and_continues(self, tmp_path, tiny_bag) -> None:
        tiny_bag(name="bags/ep0")
        tiny_bag(name="bags/ep1", header_offset_ns=ONE_HOUR_NS)
        cfg_path = tiny_config_yaml(tmp_path / "c.yaml")

        result = _convert(tmp_path, tmp_path / "bags", cfg_path, "--skip-failed")

        assert result.exit_code == 0, result.output
        info = json.loads((tmp_path / "out" / "meta" / "info.json").read_text())
        assert info["total_episodes"] == 1

        summary = json.loads(
            (tmp_path / "out" / "meta" / "job_summary.json").read_text()
        )
        assert summary["n_success"] == 1
        assert summary["n_failed"] == 1
        failed = [ep for ep in summary["episodes"] if not ep["success"]]
        assert "timestamp skew" in failed[0]["error"]

    def test_skew_survives_the_worker_boundary(self, tmp_path, tiny_bag) -> None:
        """--workers > 1 pickles the exception back from a worker process; the
        actionable message has to survive that round trip."""
        tiny_bag(name="bags/ep0")
        tiny_bag(name="bags/ep1", header_offset_ns=ONE_HOUR_NS)
        cfg_path = tiny_config_yaml(tmp_path / "c.yaml")

        result = _convert(
            tmp_path, tmp_path / "bags", cfg_path, "--workers", "2", "--skip-failed"
        )

        assert result.exit_code == 0, result.output
        summary = json.loads(
            (tmp_path / "out" / "meta" / "job_summary.json").read_text()
        )
        failed = [ep for ep in summary["episodes"] if not ep["success"]]
        assert len(failed) == 1
        assert "max_header_receive_skew_ms" in failed[0]["error"]
