"""Tests for the ``meta/progress.json`` heartbeat and non-TTY progress logging.

A tqdm bar tells a human at a terminal how far along a conversion is. These
are the answers for everything else — a supervisor polling a file, and a log
stream that must stay free of redrawn bars. Covered here:

  * :func:`bag_message_count` reads ``metadata.yaml`` (whole bag or a topic
    subset) and reports "unknown" rather than guessing;
  * :class:`ProgressReporter` cadence, atomic writes, and the exact JSON
    schema consumers depend on;
  * the wiring: the serial path advances per message, the parallel path per
    episode, and a successful run leaves no heartbeat behind;
  * no tqdm bar when stdout is not a TTY.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from rosbag2lerobot.cli import _iter_episodes_serial, main
from rosbag2lerobot.cli._common import _make_progress
from rosbag2lerobot.config import load_config
from rosbag2lerobot.progress import (
    PROGRESS_FILENAME,
    ProgressReporter,
    bag_message_count,
    format_progress_line,
)
from rosbag2lerobot.resampler import Resampler

from .conftest import ACTION_TOPIC, STATE_TOPIC, tiny_config_yaml


class _FakeClock:
    """Monotonic clock a test can step by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


# ---------------------------------------------------------------------------
# bag_message_count
# ---------------------------------------------------------------------------


class TestBagMessageCount:
    def test_counts_the_whole_bag(self, tiny_bag) -> None:
        bag = tiny_bag(n_messages=20)
        assert bag_message_count(bag) == 40  # two topics x 20

    def test_counts_a_topic_subset(self, tiny_bag) -> None:
        bag = tiny_bag(n_messages=20)
        assert bag_message_count(bag, [STATE_TOPIC]) == 20
        assert bag_message_count(bag, [STATE_TOPIC, ACTION_TOPIC]) == 40

    def test_unknown_topics_count_zero_reported_as_unknown(self, tiny_bag) -> None:
        bag = tiny_bag(n_messages=20)
        assert bag_message_count(bag, ["/nope"]) is None

    def test_missing_metadata_is_unknown(self, tmp_path: Path) -> None:
        bare = tmp_path / "bare"
        bare.mkdir()
        assert bag_message_count(bare) is None

    def test_unparseable_metadata_is_unknown(self, tmp_path: Path) -> None:
        bag = tmp_path / "bag"
        bag.mkdir()
        (bag / "metadata.yaml").write_text("{{{ not yaml")
        assert bag_message_count(bag) is None

    def test_flat_count_is_not_scoped_to_a_subset(self, tmp_path: Path) -> None:
        """Without per-topic counts, a subset total would overstate the work."""
        bag = tmp_path / "bag"
        bag.mkdir()
        (bag / "metadata.yaml").write_text(
            "rosbag2_bagfile_information:\n  message_count: 100\n"
        )
        assert bag_message_count(bag) == 100
        assert bag_message_count(bag, [STATE_TOPIC]) is None


# ---------------------------------------------------------------------------
# ProgressReporter
# ---------------------------------------------------------------------------


class TestProgressReporter:
    def _reporter(
        self, tmp_path: Path, **kwargs
    ) -> tuple[ProgressReporter, _FakeClock]:
        clock = _FakeClock()
        reporter = ProgressReporter(
            tmp_path / "meta" / PROGRESS_FILENAME,
            episode_total=kwargs.pop("episode_total", 4),
            clock=clock,
            **kwargs,
        )
        return reporter, clock

    def _read(self, tmp_path: Path) -> dict:
        return json.loads((tmp_path / "meta" / PROGRESS_FILENAME).read_text())

    def test_start_episode_writes_the_documented_schema(self, tmp_path: Path) -> None:
        reporter, _ = self._reporter(tmp_path)
        reporter.start_episode(2, 500)

        state = self._read(tmp_path)
        assert set(state) == {
            "episode_index",
            "episode_total",
            "messages_done",
            "messages_total",
            "updated_at",
        }
        assert state["episode_index"] == 2
        assert state["episode_total"] == 4
        assert state["messages_done"] == 0
        assert state["messages_total"] == 500
        assert state["updated_at"].endswith("+00:00")

    def test_unknown_total_is_null_not_zero(self, tmp_path: Path) -> None:
        reporter, _ = self._reporter(tmp_path)
        reporter.start_episode(0, None)
        assert self._read(tmp_path)["messages_total"] is None

    def test_writes_are_rate_limited(self, tmp_path: Path) -> None:
        reporter, clock = self._reporter(
            tmp_path, write_interval_s=2.0, message_stride=10
        )
        reporter.start_episode(0, 1000)

        # Under both thresholds: the file still shows the start-of-episode state.
        for _ in range(9):
            reporter.advance()
        assert self._read(tmp_path)["messages_done"] == 0

        # Stride reached but no time has passed -> still no write.
        reporter.advance()
        assert self._read(tmp_path)["messages_done"] == 0

        # Both thresholds crossed -> the file catches up.
        clock.tick(2.0)
        for _ in range(10):
            reporter.advance()
        assert self._read(tmp_path)["messages_done"] == 20

    def test_finish_episode_forces_a_write(self, tmp_path: Path) -> None:
        reporter, _ = self._reporter(tmp_path, message_stride=1000)
        reporter.start_episode(1, 30)
        for _ in range(30):
            reporter.advance()
        assert self._read(tmp_path)["messages_done"] == 0  # nothing was due

        reporter.finish_episode()
        assert self._read(tmp_path)["messages_done"] == 30

    def test_episode_completed_reports_a_finished_episode(self, tmp_path: Path) -> None:
        reporter, _ = self._reporter(tmp_path)
        reporter.episode_completed(3, 700)
        state = self._read(tmp_path)
        assert state["episode_index"] == 3
        assert state["messages_done"] == 700
        assert state["messages_total"] == 700

    def test_write_is_atomic(self, tmp_path: Path) -> None:
        """Readers must never see a partial file, so the write lands via
        rename and leaves no temp file behind."""
        reporter, _ = self._reporter(tmp_path)
        reporter.start_episode(0, 10)
        meta = tmp_path / "meta"
        assert [p.name for p in meta.iterdir()] == [PROGRESS_FILENAME]

    def test_remove_is_idempotent(self, tmp_path: Path) -> None:
        reporter, _ = self._reporter(tmp_path)
        reporter.start_episode(0, 10)
        reporter.remove()
        reporter.remove()
        assert not (tmp_path / "meta" / PROGRESS_FILENAME).exists()

    def test_logs_every_fraction_of_an_episode(self, tmp_path: Path) -> None:
        lines: list[str] = []
        clock = _FakeClock()
        reporter = ProgressReporter(
            tmp_path / "meta" / PROGRESS_FILENAME,
            episode_total=40,
            log_fn=lines.append,
            message_stride=100,
            log_fraction=0.10,
            log_interval_s=30.0,
            clock=clock,
        )
        reporter.start_episode(2, 1000)
        for _ in range(1000):
            reporter.advance()

        # 10% of 1000 messages -> ~10 lines, and they read like the spec.
        assert len(lines) == 10
        assert lines[0] == "episode 3/40: 10% (100/1000 messages)"
        assert lines[-1] == "episode 3/40: 100% (1000/1000 messages)"

    def test_logs_on_time_when_the_total_is_unknown(self, tmp_path: Path) -> None:
        lines: list[str] = []
        clock = _FakeClock()
        reporter = ProgressReporter(
            tmp_path / "meta" / PROGRESS_FILENAME,
            episode_total=2,
            log_fn=lines.append,
            message_stride=10,
            log_interval_s=30.0,
            clock=clock,
        )
        reporter.start_episode(0, None)
        for _ in range(100):
            reporter.advance()
        assert lines == []  # no elapsed time, no known total -> nothing to say

        clock.tick(30.0)
        for _ in range(10):
            reporter.advance()
        assert lines == ["episode 1/2: 110 messages"]

    def test_no_logging_without_a_log_fn(self, tmp_path: Path) -> None:
        reporter, clock = self._reporter(tmp_path, message_stride=1)
        reporter.start_episode(0, 10)
        clock.tick(1000.0)
        reporter.advance()  # must not raise


def test_format_progress_line_is_one_based() -> None:
    assert (
        format_progress_line(
            episode_index=2, episode_total=40, messages_done=12000, messages_total=19500
        )
        == "episode 3/40: 62% (12000/19500 messages)"
    )
    assert (
        format_progress_line(
            episode_index=0, episode_total=1, messages_done=5, messages_total=None
        )
        == "episode 1/1: 5 messages"
    )


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


class TestConvertWiring:
    def test_serial_path_advances_per_message(self, tmp_path: Path, tiny_bag) -> None:
        bag = tiny_bag(name="bags/ep0", n_messages=20)
        cfg = load_config(tiny_config_yaml(tmp_path / "c.yaml"))
        resampler = Resampler(fps=cfg.fps, policy="nearest", tolerance_ms=100.0)
        reporter = ProgressReporter(
            tmp_path / "meta" / PROGRESS_FILENAME,
            episode_total=1,
            message_stride=1,
            write_interval_s=0.0,
        )

        episodes = list(
            _iter_episodes_serial([bag], cfg, resampler, [("t", [])], progress=reporter)
        )

        assert len(episodes) == 1
        state = json.loads((tmp_path / "meta" / PROGRESS_FILENAME).read_text())
        # Both topics are read, and the episode finished, so done == total.
        assert state["messages_total"] == 40
        assert state["messages_done"] == 40
        assert state["episode_index"] == 0
        assert state["episode_total"] == 1

    def test_successful_run_leaves_no_heartbeat(self, tmp_path: Path, tiny_bag) -> None:
        tiny_bag(name="bags/ep0")
        cfg_path = tiny_config_yaml(tmp_path / "c.yaml")
        out = tmp_path / "out"

        result = CliRunner().invoke(
            main,
            [
                "convert",
                "--config",
                str(cfg_path),
                "--bags",
                str(tmp_path / "bags"),
                "--output",
                str(out),
            ],
        )

        assert result.exit_code == 0, result.output
        # Transient run state: gone once job_summary.json records the outcome.
        assert not (out / "meta" / PROGRESS_FILENAME).exists()
        assert (out / "meta" / "job_summary.json").exists()

    def test_heartbeat_survives_a_failed_run(self, tmp_path: Path, tiny_bag) -> None:
        """A run that dies leaves the file behind, saying how far it got."""
        tiny_bag(name="bags/ep0")
        # A bag whose metadata promises storage files that are not there: the
        # reader fails on open, which (without --skip-failed) aborts the run.
        broken = tiny_bag(name="bags/ep1")
        for storage in list(broken.glob("*.db3")) + list(broken.glob("*.mcap")):
            storage.unlink()
        cfg_path = tiny_config_yaml(tmp_path / "c.yaml")
        out = tmp_path / "out"

        result = CliRunner().invoke(
            main,
            [
                "convert",
                "--config",
                str(cfg_path),
                "--bags",
                str(tmp_path / "bags"),
                "--output",
                str(out),
            ],
        )

        assert result.exit_code != 0
        state = json.loads((out / "meta" / PROGRESS_FILENAME).read_text())
        assert state["episode_index"] == 1
        assert state["episode_total"] == 2

    @pytest.mark.parametrize("workers", ["1", "2"])
    def test_progress_json_is_written_during_a_run(
        self, tmp_path: Path, tiny_bag, workers: str
    ) -> None:
        """Both paths keep the heartbeat current; capture it before the final
        delete by snapshotting it as each episode is handed to the writer.

        Only the serial path is asserted to advance across the snapshots: with
        workers the episodes can all finish before the first one is drained to
        the writer, so every snapshot may legitimately show the same index.
        """
        for i in range(3):
            tiny_bag(name=f"bags/ep{i}")
        cfg_path = tiny_config_yaml(tmp_path / "c.yaml")
        out = tmp_path / "out"
        seen: list[dict] = []

        from rosbag2lerobot import writer as writer_module

        real_write_dataset = writer_module.write_dataset

        def spy(episodes, *args, **kwargs):
            def watched():
                for ep in episodes:
                    path = out / "meta" / PROGRESS_FILENAME
                    if path.exists():
                        seen.append(json.loads(path.read_text()))
                    yield ep

            return real_write_dataset(watched(), *args, **kwargs)

        with mock.patch.object(writer_module, "write_dataset", spy):
            result = CliRunner().invoke(
                main,
                [
                    "convert",
                    "--config",
                    str(cfg_path),
                    "--bags",
                    str(tmp_path / "bags"),
                    "--output",
                    str(out),
                    "--workers",
                    workers,
                ],
            )

        assert result.exit_code == 0, result.output
        assert seen, "progress.json was never written during the run"
        assert all(s["episode_total"] == 3 for s in seen)
        assert all(0 <= s["episode_index"] < 3 for s in seen)
        if workers == "1":
            assert [s["episode_index"] for s in seen] == [0, 1, 2]


class TestNonTtyBar:
    def test_no_bar_when_stdout_is_not_a_tty(self) -> None:
        with mock.patch("sys.stdout.isatty", return_value=False):
            assert _make_progress(10, disable=False) is None

    def test_bar_at_a_terminal(self) -> None:
        with mock.patch("sys.stdout.isatty", return_value=True):
            bar = _make_progress(10, disable=False)
            assert bar is not None
            bar.close()

    def test_disable_wins_at_a_terminal(self) -> None:
        with mock.patch("sys.stdout.isatty", return_value=True):
            assert _make_progress(10, disable=True) is None
