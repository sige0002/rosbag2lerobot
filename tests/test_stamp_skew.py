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
  * the CLI contract — abort by default, record-and-continue with
    ``--skip-failed``.

Bags are synthesized by the ``tiny_bag`` fixture, so nothing here depends on
the gitignored ``bagdata/`` tree.
"""

from __future__ import annotations

import json
from pathlib import Path

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
from rosbag2lerobot.timestamps import StampSkewError, format_skew_error

from .conftest import ACTION_TOPIC, STATE_TOPIC, tiny_config_yaml

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
