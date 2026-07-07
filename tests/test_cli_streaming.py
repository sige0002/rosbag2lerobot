"""Tests for the generator-based CLI episode pipeline (T11).

Covers the two helpers that replaced the old ``all_episodes`` accumulator:

* :func:`rosbag2lerobot.cli._iter_episodes_serial` -- yields one episode at a
  time in ``bag_paths`` order.
* :func:`rosbag2lerobot.cli._iter_episodes_parallel` -- uses a ProcessPool
  internally, buffers out-of-order completions, and yields episodes in
  original ``bag_paths`` order regardless of completion order.

Also pins down that :func:`rosbag2lerobot.writer.write_dataset` accepts an
arbitrary iterator (e.g. ``iter([...])``) and consumes it lazily via
:func:`itertools.chain` on top of ``next()`` on the first episode.
"""

from __future__ import annotations

import json
from concurrent.futures import Future
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

from rosbag2lerobot import cli as cli_module
from rosbag2lerobot.cli import _iter_episodes_parallel, _iter_episodes_serial
from rosbag2lerobot.config import SplitConfig


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeResampler:
    """Minimal stand-in for :class:`rosbag2lerobot.resampler.Resampler`.

    ``_iter_episodes_parallel`` extracts ``fps`` / ``policy`` / ``tolerance_ms``
    from the passed resampler to rebuild it inside each worker. We never
    actually submit to the real pool in these tests (the executor is mocked),
    so the attribute surface is all we need to cover.
    """

    def __init__(self) -> None:
        self.fps = 30
        self.policy = "hold"
        self.tolerance_ms = 50.0


class _FakeCfg:
    """Minimal RobotConfig stand-in with just the fields the CLI touches."""

    def __init__(
        self, task: str = "default-task", fps: int = 30, min_length: int = 0
    ) -> None:
        self.task = task
        self.fps = fps
        # The episode iterators read cfg.split.min_length for the producer-side
        # length filter; default 0 keeps every episode (legacy behavior).
        self.split = SplitConfig(train=1.0, min_length=min_length)


def _make_frames(n: int, tag: int) -> list[dict[str, Any]]:
    """Build a synthetic frame list. ``tag`` is stashed so tests can
    verify which fake bag the frames came from after they come out of
    the iterator."""
    return [{"frame_index": i, "_tag": tag} for i in range(n)]


# ---------------------------------------------------------------------------
# _iter_episodes_serial
# ---------------------------------------------------------------------------


class TestIterEpisodesSerial:
    """Serial generator: deterministic order, task resolution applied."""

    def test_yields_in_order(self, tmp_path: Path) -> None:
        """3 bags -> 3 episodes, yielded in ``bag_paths`` order, each
        frame is tagged with the bag-specific resolved task."""
        bag_paths = [tmp_path / f"bag_{i}" for i in range(3)]
        for bp in bag_paths:
            bp.mkdir()

        fake_episodes = {
            bag_paths[0]: _make_frames(2, tag=0),
            bag_paths[1]: _make_frames(3, tag=1),
            bag_paths[2]: _make_frames(4, tag=2),
        }

        def fake_process_episode(
            bag_path: Path,
            cfg: _FakeCfg,
            resampler: _FakeResampler,
        ) -> list[dict[str, Any]]:
            return list(fake_episodes[bag_path])

        cfg = _FakeCfg(task="yaml-default", fps=30)
        resampler = _FakeResampler()
        bag_specs = [(f"task-{bp.name}", []) for bp in bag_paths]

        with mock.patch.object(
            cli_module,
            "_process_episode",
            side_effect=fake_process_episode,
        ):
            result = list(
                _iter_episodes_serial(bag_paths, cfg, resampler, bag_specs),  # type: ignore[arg-type]
            )

        assert len(result) == 3
        assert [len(ep) for ep in result] == [2, 3, 4]
        # Tag matches the bag index -> order preserved.
        assert [ep[0]["_tag"] for ep in result] == [0, 1, 2]
        # Each frame has the per-bag resolved task injected.
        for i, ep in enumerate(result):
            for frame in ep:
                assert frame["task"] == f"task-bag_{i}"

    def test_empty_episode_still_yielded(self, tmp_path: Path) -> None:
        """An empty frame list is still yielded (len(ep) == 0 is a valid
        signal for downstream code to handle / warn about)."""
        bag = tmp_path / "empty_bag"
        bag.mkdir()

        with mock.patch.object(cli_module, "_process_episode", return_value=[]):
            result = list(
                _iter_episodes_serial(
                    [bag],
                    _FakeCfg(),
                    _FakeResampler(),
                    [("t", [])],
                ),  # type: ignore[arg-type]
            )

        assert result == [[]]

    def test_min_length_drops_short_episode_everywhere(self, tmp_path: Path) -> None:
        """Episodes shorter than min_length are never yielded and never
        reported via on_episode_done (so totals stay consistent)."""
        bag_paths = [tmp_path / f"bag_{i}" for i in range(3)]
        for bp in bag_paths:
            bp.mkdir()

        # bag_1 is 1 frame (below the threshold of 3) -> dropped.
        fake_episodes = {
            bag_paths[0]: _make_frames(5, tag=0),
            bag_paths[1]: _make_frames(1, tag=1),
            bag_paths[2]: _make_frames(4, tag=2),
        }

        def fake_process_episode(bag_path, cfg, resampler):  # type: ignore[no-untyped-def]
            return list(fake_episodes[bag_path])

        cfg = _FakeCfg(fps=30, min_length=3)
        bag_specs = [(f"t{i}", []) for i in range(3)]
        reported: list[tuple[int, bool, int]] = []

        def on_done(res) -> None:  # type: ignore[no-untyped-def]
            reported.append((res.index, res.success, res.n_frames))

        with mock.patch.object(
            cli_module, "_process_episode", side_effect=fake_process_episode
        ):
            result = list(
                _iter_episodes_serial(
                    bag_paths,
                    cfg,
                    _FakeResampler(),
                    bag_specs,
                    on_episode_done=on_done,
                ),  # type: ignore[arg-type]
            )

        # Only the two long episodes are yielded; the short one is absent.
        assert [ep[0]["_tag"] for ep in result] == [0, 2]
        # And the dropped episode never reaches on_episode_done.
        assert [r[0] for r in reported] == [0, 2]
        assert all(r[1] for r in reported)


# ---------------------------------------------------------------------------
# _iter_episodes_parallel
# ---------------------------------------------------------------------------


def _make_completed_future(
    result: tuple[int, list[dict[str, Any]], str, int, float],
) -> Future:
    """Build a pre-completed Future carrying ``result``. Used to simulate
    workers that have already finished when ``as_completed`` observes them."""
    fut: Future = Future()
    fut.set_result(result)
    return fut


class _FakeExecutor:
    """Minimal ProcessPoolExecutor stand-in.

    ``submit`` stashes the job tuple and returns a fresh Future. We don't
    actually run anything — the test drives completion order via a patched
    ``as_completed`` so we can assert the ordering buffer behavior in
    isolation from real multiprocessing.
    """

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self._futures: list[Future] = []
        self._jobs: list[tuple[Any, ...]] = []

    def __enter__(self) -> "_FakeExecutor":
        return self

    def __exit__(self, *_exc: Any) -> None:
        pass

    def submit(self, _fn: Any, job: tuple[Any, ...]) -> Future:
        fut: Future = Future()
        self._futures.append(fut)
        self._jobs.append(job)
        return fut

    # Convenience: resolve a given future in-place with a result.
    def complete(
        self, idx: int, result: tuple[int, list[dict[str, Any]], str, int, float]
    ) -> Future:
        self._futures[idx].set_result(result)
        return self._futures[idx]


class TestIterEpisodesParallel:
    """Parallel generator: out-of-order completion, in-order output."""

    def test_preserves_order_when_completion_is_reversed(
        self,
        tmp_path: Path,
    ) -> None:
        """Workers finish in reverse order (bag 2 first, then bag 1, then
        bag 0); the generator must still yield in bag_paths order (0,1,2).

        This is the load-bearing invariant of ``_iter_episodes_parallel``:
        completion order is non-deterministic but output order must match
        the serial path exactly.
        """
        bag_paths = [tmp_path / f"bag_{i}" for i in range(3)]
        for bp in bag_paths:
            bp.mkdir()

        # Build the per-bag worker results (tagged so we can verify order).
        worker_results = [
            (0, _make_frames(2, tag=0), "task-0", 1000, 0.1),
            (1, _make_frames(3, tag=1), "task-1", 1001, 0.1),
            (2, _make_frames(4, tag=2), "task-2", 1002, 0.1),
        ]

        fake_pool = _FakeExecutor()

        def fake_executor_ctor(*args: Any, **kwargs: Any) -> _FakeExecutor:
            return fake_pool

        # Force as_completed to return futures in REVERSE order (2, 1, 0)
        # while the original submission order is (0, 1, 2).
        def fake_as_completed(
            futs: dict[Future, int] | list[Future],
        ) -> list[Future]:
            # Resolve backward before returning — simulates slowest-bag-first.
            for ep_idx in (2, 1, 0):
                fake_pool.complete(ep_idx, worker_results[ep_idx])
            return [fake_pool._futures[i] for i in (2, 1, 0)]

        cfg = _FakeCfg(task="yaml-default", fps=30)
        resampler = _FakeResampler()

        bag_specs = [(f"t{i}", []) for i in range(3)]

        with (
            mock.patch.object(
                cli_module, "ProcessPoolExecutor", side_effect=fake_executor_ctor
            ),
            mock.patch.object(
                cli_module, "as_completed", side_effect=fake_as_completed
            ),
        ):
            yielded = list(
                _iter_episodes_parallel(
                    bag_paths,
                    cfg,
                    resampler,
                    workers=3,
                    bag_specs=bag_specs,
                ),  # type: ignore[arg-type]
            )

        # Output order must match bag_paths order, not completion order.
        assert [ep[0]["_tag"] for ep in yielded] == [0, 1, 2]
        assert [len(ep) for ep in yielded] == [2, 3, 4]

    def test_drains_contiguous_prefix_as_it_becomes_available(
        self,
        tmp_path: Path,
    ) -> None:
        """If bag 0 finishes before bag 1, bag 0 should be yielded
        immediately (i.e. the generator does not wait for all futures).

        Concretely: arrive in order (1, 0, 2). After bag 1 arrives the
        buffer holds {1: ...} and nothing is yielded. After bag 0 arrives
        the contiguous prefix 0,1 should drain. Then bag 2 completes and
        drains on its own.
        """
        bag_paths = [tmp_path / f"bag_{i}" for i in range(3)]
        for bp in bag_paths:
            bp.mkdir()

        worker_results = [
            (0, _make_frames(2, tag=0), "t0", 1000, 0.1),
            (1, _make_frames(3, tag=1), "t1", 1001, 0.1),
            (2, _make_frames(4, tag=2), "t2", 1002, 0.1),
        ]

        fake_pool = _FakeExecutor()

        def fake_executor_ctor(*args: Any, **kwargs: Any) -> _FakeExecutor:
            return fake_pool

        def fake_as_completed(
            futs: dict[Future, int] | list[Future],
        ) -> list[Future]:
            order = (1, 0, 2)
            for ep_idx in order:
                fake_pool.complete(ep_idx, worker_results[ep_idx])
            return [fake_pool._futures[i] for i in order]

        bag_specs = [(f"t{i}", []) for i in range(3)]

        with (
            mock.patch.object(
                cli_module, "ProcessPoolExecutor", side_effect=fake_executor_ctor
            ),
            mock.patch.object(
                cli_module, "as_completed", side_effect=fake_as_completed
            ),
        ):
            yielded = list(
                _iter_episodes_parallel(
                    bag_paths,
                    _FakeCfg(task="y", fps=30),
                    _FakeResampler(),
                    workers=2,
                    bag_specs=bag_specs,
                ),  # type: ignore[arg-type]
            )

        assert [ep[0]["_tag"] for ep in yielded] == [0, 1, 2]

    def test_min_length_drops_short_episode_keeps_order(
        self,
        tmp_path: Path,
    ) -> None:
        """A short middle episode is hopped over in the drain so the surviving
        episodes still stream out contiguously and are never reported."""
        bag_paths = [tmp_path / f"bag_{i}" for i in range(3)]
        for bp in bag_paths:
            bp.mkdir()

        # bag_1 is 1 frame (< min_length 3) -> dropped.
        worker_results = [
            (0, _make_frames(5, tag=0), "t0", 1000, 0.1),
            (1, _make_frames(1, tag=1), "t1", 1001, 0.1),
            (2, _make_frames(4, tag=2), "t2", 1002, 0.1),
        ]

        fake_pool = _FakeExecutor()

        def fake_executor_ctor(*args: Any, **kwargs: Any) -> _FakeExecutor:
            return fake_pool

        def fake_as_completed(
            futs: dict[Future, int] | list[Future],
        ) -> list[Future]:
            order = (0, 1, 2)
            for ep_idx in order:
                fake_pool.complete(ep_idx, worker_results[ep_idx])
            return [fake_pool._futures[i] for i in order]

        bag_specs = [(f"t{i}", []) for i in range(3)]
        reported: list[int] = []

        with (
            mock.patch.object(
                cli_module, "ProcessPoolExecutor", side_effect=fake_executor_ctor
            ),
            mock.patch.object(
                cli_module, "as_completed", side_effect=fake_as_completed
            ),
        ):
            yielded = list(
                _iter_episodes_parallel(
                    bag_paths,
                    _FakeCfg(task="y", fps=30, min_length=3),
                    _FakeResampler(),
                    workers=3,
                    bag_specs=bag_specs,
                    on_episode_done=lambda r: reported.append(r.index),
                ),  # type: ignore[arg-type]
            )

        # The short episode is absent from the yielded stream and the report.
        assert [ep[0]["_tag"] for ep in yielded] == [0, 2]
        assert reported == [0, 2]


# ---------------------------------------------------------------------------
# write_dataset accepts a plain iterator
# ---------------------------------------------------------------------------


class TestWriteDatasetAcceptsIterator:
    """``write_dataset`` must consume ``Iterable[list[dict]]`` lazily.

    Guards the T11 invariant that the CLI can hand the writer a generator
    and the writer never materializes the full ``list[list[dict]]``.
    """

    def test_accepts_iter_of_lists(self, tmp_path: Path) -> None:
        """Two synthetic episodes, passed as ``iter([...])``, must produce
        a valid dataset with no errors."""
        pytest.importorskip("rosbag2lerobot.writer")
        from rosbag2lerobot.writer import write_dataset

        # Build a minimal RobotConfig-compatible object. We only touch
        # the fields that write_dataset reads.
        class _Feat:
            def __init__(self, key: str, dim: int = 3) -> None:
                self.key = key
                self._dim = dim
                self.is_image = False
                self.image_size = None
                self.names = None

        class _Cfg:
            def __init__(self) -> None:
                self.robot_type = "test"
                self.task = "default"
                self.fps = 10
                self.repo_id = None
                self.observations = [_Feat("observation.state", dim=3)]
                self.actions = [_Feat("action", dim=3)]
                self.split = SplitConfig()

        cfg = _Cfg()

        ep1 = [
            {
                "observation.state": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "action": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "frame_index": i,
                "timestamp": np.float32(i / cfg.fps),
                "task": "ep-a",
            }
            for i in range(3)
        ]
        ep2 = [
            {
                "observation.state": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "action": np.array([0.5, 0.5, 0.5], dtype=np.float32),
                "frame_index": i,
                "timestamp": np.float32(i / cfg.fps),
                "task": "ep-b",
            }
            for i in range(4)
        ]

        # Pass as a bare iterator — this is the critical bit. If
        # write_dataset tried to e.g. ``len(episodes)`` or iterate twice,
        # this test would fail.
        write_dataset(
            episodes=iter([ep1, ep2]),
            config=cfg,  # type: ignore[arg-type]
            output_dir=tmp_path,
            video_codec="libx264",
        )

        info_path = tmp_path / "meta" / "info.json"
        assert info_path.exists()
        with open(info_path) as fh:
            info = json.load(fh)
        assert info["total_episodes"] == 2
        assert info["total_frames"] == 7

    def test_empty_iterator_returns_without_error(self, tmp_path: Path) -> None:
        """An empty episode stream must short-circuit cleanly (no writer
        is constructed, no metadata is written)."""
        pytest.importorskip("rosbag2lerobot.writer")
        from rosbag2lerobot.writer import write_dataset

        class _Cfg:
            def __init__(self) -> None:
                self.robot_type = "test"
                self.task = "default"
                self.fps = 10
                self.repo_id = None
                self.observations: list[Any] = []
                self.actions: list[Any] = []

        write_dataset(
            episodes=iter([]),
            config=_Cfg(),  # type: ignore[arg-type]
            output_dir=tmp_path,
            video_codec="libx264",
        )

        # Nothing should have been written.
        assert not (tmp_path / "meta" / "info.json").exists()
