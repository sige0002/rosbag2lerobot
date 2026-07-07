"""Tests for time-synchronised resampling of multi-topic ROS data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Lightweight resampler implementation (to be tested)
# ---------------------------------------------------------------------------
# The resampler is responsible for aligning messages from multiple ROS topics
# to a fixed-fps timeline.  We define it here because the module may not exist
# yet in the main source tree.


@dataclass
class StampedValue:
    """A value with a timestamp (seconds, float64)."""

    t: float
    value: Any


def resample_hold(
    timeline: np.ndarray,
    messages: list[StampedValue],
    tolerance_s: float = 0.05,
) -> list[Any | None]:
    """Zero-order-hold resampling.

    For each tick in *timeline*, return the last message whose timestamp
    is <= tick and within *tolerance_s*.  Returns None for ticks with no
    valid message.
    """
    result: list[Any | None] = []
    msg_idx = 0
    n_msgs = len(messages)
    for t_tick in timeline:
        # Advance pointer
        while msg_idx < n_msgs - 1 and messages[msg_idx + 1].t <= t_tick:
            msg_idx += 1
        if msg_idx < n_msgs and messages[msg_idx].t <= t_tick:
            if t_tick - messages[msg_idx].t <= tolerance_s:
                result.append(messages[msg_idx].value)
            else:
                result.append(None)
        else:
            result.append(None)
    return result


def resample_nearest(
    timeline: np.ndarray,
    messages: list[StampedValue],
    tolerance_s: float = 0.05,
) -> list[Any | None]:
    """Nearest-neighbour resampling.

    For each tick, find the message with the smallest |t_msg - t_tick|.
    Return None if the nearest is beyond *tolerance_s*.
    """
    result: list[Any | None] = []
    msg_idx = 0
    n_msgs = len(messages)
    for t_tick in timeline:
        # Advance to closest
        while msg_idx < n_msgs - 1 and abs(messages[msg_idx + 1].t - t_tick) <= abs(
            messages[msg_idx].t - t_tick
        ):
            msg_idx += 1
        if msg_idx < n_msgs:
            if abs(messages[msg_idx].t - t_tick) <= tolerance_s:
                result.append(messages[msg_idx].value)
            else:
                result.append(None)
        else:
            result.append(None)
    return result


def resample_drop(
    timeline: np.ndarray,
    messages: list[StampedValue],
    tolerance_s: float = 0.05,
) -> list[Any | None]:
    """Drop policy: same as hold, but None entries remain (caller drops frames)."""
    return resample_hold(timeline, messages, tolerance_s)


def synchronize_topics(
    timeline: np.ndarray,
    topic_messages: dict[str, list[StampedValue]],
    policy: str = "hold",
    tolerance_s: float = 0.05,
) -> list[dict[str, Any] | None]:
    """Synchronise multiple topics to a common timeline.

    Returns a list of dicts (one per tick). A tick is None if *any* required
    topic has a None value (i.e. intersection semantics).
    """
    resample_fn = {
        "hold": resample_hold,
        "nearest": resample_nearest,
        "drop": resample_drop,
    }[policy]

    resampled: dict[str, list[Any | None]] = {}
    for topic, msgs in topic_messages.items():
        resampled[topic] = resample_fn(timeline, msgs, tolerance_s)

    result: list[dict[str, Any] | None] = []
    topics = list(topic_messages.keys())
    for i in range(len(timeline)):
        frame: dict[str, Any] = {}
        all_valid = True
        for topic in topics:
            val = resampled[topic][i]
            if val is None:
                all_valid = False
                break
            frame[topic] = val
        result.append(frame if all_valid else None)
    return result


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_timeline(start: float, end: float, fps: int) -> np.ndarray:
    dt = 1.0 / fps
    return np.arange(start, end, dt)


# ---------------------------------------------------------------------------
# Tests: hold policy
# ---------------------------------------------------------------------------


class TestHoldPolicy:
    def test_basic_hold(self) -> None:
        timeline = np.array([0.0, 0.033, 0.066, 0.1])
        messages = [
            StampedValue(0.0, "A"),
            StampedValue(0.05, "B"),
        ]
        result = resample_hold(timeline, messages, tolerance_s=0.1)
        assert result == ["A", "A", "B", "B"]

    def test_hold_respects_tolerance(self) -> None:
        timeline = np.array([0.0, 0.5, 1.0])
        messages = [StampedValue(0.0, "A")]
        result = resample_hold(timeline, messages, tolerance_s=0.1)
        # t=0.0 within tolerance, t=0.5 and t=1.0 are too far
        assert result == ["A", None, None]

    def test_hold_empty_messages(self) -> None:
        timeline = np.array([0.0, 0.033])
        result = resample_hold(timeline, [], tolerance_s=0.1)
        assert result == [None, None]

    def test_hold_messages_after_timeline(self) -> None:
        timeline = np.array([0.0, 0.033])
        messages = [StampedValue(1.0, "future")]
        result = resample_hold(timeline, messages, tolerance_s=0.1)
        assert result == [None, None]

    def test_hold_single_message_covers_range(self) -> None:
        timeline = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
        messages = [StampedValue(0.0, "X")]
        result = resample_hold(timeline, messages, tolerance_s=0.05)
        assert result == ["X", "X", "X", "X", "X"]


# ---------------------------------------------------------------------------
# Tests: nearest policy
# ---------------------------------------------------------------------------


class TestNearestPolicy:
    def test_basic_nearest(self) -> None:
        timeline = np.array([0.0, 0.03, 0.07, 0.1])
        messages = [
            StampedValue(0.0, "A"),
            StampedValue(0.05, "B"),
            StampedValue(0.1, "C"),
        ]
        result = resample_nearest(timeline, messages, tolerance_s=0.1)
        # t=0.0 -> A (exact), t=0.03 -> A or B (A closer), t=0.07 -> B (closer), t=0.1 -> C
        assert result[0] == "A"
        assert result[2] == "B"
        assert result[3] == "C"

    def test_nearest_respects_tolerance(self) -> None:
        timeline = np.array([0.0, 1.0])
        messages = [StampedValue(0.0, "A")]
        result = resample_nearest(timeline, messages, tolerance_s=0.1)
        assert result[0] == "A"
        assert result[1] is None

    def test_nearest_empty_messages(self) -> None:
        timeline = np.array([0.0])
        result = resample_nearest(timeline, [], tolerance_s=0.1)
        assert result == [None]


# ---------------------------------------------------------------------------
# Tests: drop policy
# ---------------------------------------------------------------------------


class TestDropPolicy:
    def test_drop_returns_none_for_missing(self) -> None:
        timeline = np.array([0.0, 0.5, 1.0])
        messages = [StampedValue(0.0, "A"), StampedValue(1.0, "B")]
        result = resample_drop(timeline, messages, tolerance_s=0.1)
        assert result[0] == "A"
        assert result[1] is None  # too far from both A and B
        assert result[2] == "B"


# ---------------------------------------------------------------------------
# Tests: multi-topic synchronization
# ---------------------------------------------------------------------------


class TestMultiTopicSync:
    def test_sync_two_topics(self) -> None:
        timeline = np.array([0.0, 0.033, 0.066, 0.1])
        topic_msgs = {
            "/joints": [
                StampedValue(0.0, [1.0, 2.0]),
                StampedValue(0.05, [3.0, 4.0]),
            ],
            "/camera": [
                StampedValue(0.0, "img_A"),
                StampedValue(0.04, "img_B"),
            ],
        }
        result = synchronize_topics(
            timeline, topic_msgs, policy="hold", tolerance_s=0.1
        )
        # All ticks should be valid (both topics have data within tolerance)
        assert all(r is not None for r in result)
        assert result[0]["/joints"] == [1.0, 2.0]
        assert result[0]["/camera"] == "img_A"

    def test_sync_drops_when_topic_missing(self) -> None:
        timeline = np.array([0.0, 1.0])
        topic_msgs = {
            "/joints": [StampedValue(0.0, [1.0])],
            "/camera": [StampedValue(0.0, "img_A")],
        }
        result = synchronize_topics(
            timeline, topic_msgs, policy="hold", tolerance_s=0.1
        )
        assert result[0] is not None
        assert result[1] is None  # Both topics out of tolerance at t=1.0

    def test_sync_nearest_policy(self) -> None:
        timeline = np.array([0.0, 0.05])
        topic_msgs = {
            "/t1": [StampedValue(0.0, "a"), StampedValue(0.06, "b")],
            "/t2": [StampedValue(0.01, "x"), StampedValue(0.04, "y")],
        }
        result = synchronize_topics(
            timeline, topic_msgs, policy="nearest", tolerance_s=0.1
        )
        assert result[0] is not None
        assert result[1] is not None

    def test_sync_single_topic(self) -> None:
        timeline = np.array([0.0, 0.033])
        topic_msgs = {
            "/only": [StampedValue(0.0, 42)],
        }
        result = synchronize_topics(
            timeline, topic_msgs, policy="hold", tolerance_s=0.05
        )
        assert result[0] == {"/only": 42}
        assert result[1] == {"/only": 42}


# ---------------------------------------------------------------------------
# Tests: vectorised vs bisect reference (T9 regression)
# ---------------------------------------------------------------------------
# These tests guarantee that the numpy.searchsorted-based ``Resampler``
# returns bit-identical results to the classic per-frame bisect loop across
# random inputs for all three policies.

import math  # noqa: E402 (test-local helpers intentionally below class defs)

from rosbag2lerobot.resampler import (  # noqa: E402
    Resampler,
    _bisect_right_ts,
)


def _resample_reference(
    resampler: Resampler,
    messages: list[tuple[str, int, Any]],
    feature_keys: list[str],
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> list[dict[str, Any]]:
    """Classic bisect-per-frame reference implementation.

    Kept verbatim from the pre-vectorisation code path so the test can
    compare its output with the numpy-vectorised ``Resampler.resample``.
    """
    if not messages:
        return []
    if start_ns is None:
        start_ns = messages[0][1]
    if end_ns is None:
        end_ns = messages[-1][1]
    if end_ns < start_ns:
        raise ValueError("end_ns must be >= start_ns")

    key_timestamps: dict[str, list[int]] = {k: [] for k in feature_keys}
    key_values: dict[str, list[Any]] = {k: [] for k in feature_keys}
    for key, ts_ns, value in messages:
        if key in key_timestamps:
            key_timestamps[key].append(ts_ns)
            key_values[key].append(value)

    duration_s = (end_ns - start_ns) / 1e9
    n_frames = max(1, int(math.ceil(duration_s * resampler.fps)))
    frame_period_ns = int(1e9 / resampler.fps)
    tolerance_ns = int(resampler.tolerance_ms * 1e6)

    frames: list[dict[str, Any]] = []
    for fi in range(n_frames):
        frame_ns = start_ns + fi * frame_period_ns
        frame: dict[str, Any] = {
            "frame_index": fi,
            "timestamp": np.float32(fi / resampler.fps),
        }
        for k in feature_keys:
            ts_list = key_timestamps[k]
            val_list = key_values[k]
            if not ts_list:
                frame[k] = None
                continue
            idx = _bisect_right_ts(ts_list, frame_ns) - 1
            if resampler.policy == "hold":
                if idx < 0:
                    if ts_list and abs(ts_list[0] - frame_ns) <= tolerance_ns:
                        frame[k] = val_list[0]
                    else:
                        frame[k] = None
                else:
                    frame[k] = val_list[idx]
            else:  # nearest / drop — identical semantics
                best_val: Any = None
                best_dist = tolerance_ns + 1
                if 0 <= idx < len(ts_list):
                    d = abs(frame_ns - ts_list[idx])
                    if d < best_dist:
                        best_dist = d
                        best_val = val_list[idx]
                nxt = idx + 1
                if 0 <= nxt < len(ts_list):
                    d = abs(frame_ns - ts_list[nxt])
                    if d < best_dist:
                        best_dist = d
                        best_val = val_list[nxt]
                frame[k] = best_val if best_dist <= tolerance_ns else None
        frames.append(frame)
    return frames


def _assert_frames_equal(
    got: list[dict[str, Any]],
    want: list[dict[str, Any]],
    feature_keys: list[str],
) -> None:
    """Deep-compare two frame lists including types."""
    assert len(got) == len(want), f"frame count mismatch: {len(got)} vs {len(want)}"
    for i, (a, b) in enumerate(zip(got, want)):
        assert a["frame_index"] == b["frame_index"], f"frame {i} index"
        assert isinstance(a["frame_index"], int)
        # Timestamps: identical float32 bits expected.
        assert a["timestamp"].dtype == np.float32
        assert a["timestamp"] == b["timestamp"], f"frame {i} timestamp"
        for k in feature_keys:
            assert a[k] == b[k], f"frame {i} key {k}: {a[k]!r} vs {b[k]!r}"


class TestResamplerVectorisedEquivalence:
    """Bit-level parity between numpy.searchsorted and bisect implementations."""

    @staticmethod
    def _random_messages(
        rng: np.random.Generator,
        feature_keys: list[str],
        n_per_key: int,
        jitter_ns: int,
        period_ns: int,
        t0: int,
    ) -> list[tuple[str, int, Any]]:
        """Generate a sorted list of (key, ts_ns, value) tuples with jittered periods."""
        msgs: list[tuple[str, int, Any]] = []
        for k in feature_keys:
            base_ts = t0 + np.arange(n_per_key, dtype=np.int64) * period_ns
            jitter = rng.integers(
                -jitter_ns, jitter_ns + 1, size=n_per_key, dtype=np.int64
            )
            ts = base_ts + jitter
            ts.sort()
            for j, t in enumerate(ts.tolist()):
                # Use a unique token per (key, j) so any picking mistake is caught.
                msgs.append((k, int(t), f"{k}#{j}"))
        msgs.sort(key=lambda m: m[1])
        return msgs

    @pytest.mark.parametrize("policy", ["hold", "nearest", "drop"])
    def test_resample_searchsorted_matches_bisect_reference(self, policy: str) -> None:
        """Vectorised implementation matches the bisect reference across random inputs."""
        rng = np.random.default_rng(
            seed=42 if policy == "hold" else (1337 if policy == "nearest" else 2024)
        )
        feature_keys = [f"/k{i}" for i in range(5)]
        t0 = 1_700_000_000_000_000_000  # realistic ns epoch
        # Vary per-key rates so some keys are sparser than others.
        msgs = []
        for i, k in enumerate(feature_keys):
            period_ns = int(5e6 * (i + 1))  # 5 / 10 / 15 / 20 / 25 ms
            msgs += self._random_messages(
                rng,
                [k],
                n_per_key=2000,
                jitter_ns=int(0.3 * period_ns),
                period_ns=period_ns,
                t0=t0,
            )
        msgs.sort(key=lambda m: m[1])

        resampler = Resampler(fps=30, policy=policy, tolerance_ms=50.0)
        start_ns = t0
        # Enough duration to produce ~1000 frames at 30 fps.
        end_ns = t0 + int(1000 / 30 * 1e9)

        got = resampler.resample(msgs, feature_keys, start_ns=start_ns, end_ns=end_ns)
        want = _resample_reference(
            resampler, msgs, feature_keys, start_ns=start_ns, end_ns=end_ns
        )

        assert len(got) >= 900  # sanity: roughly 1000 frames.
        _assert_frames_equal(got, want, feature_keys)

    @pytest.mark.parametrize("policy", ["hold", "nearest", "drop"])
    def test_empty_key_all_none(self, policy: str) -> None:
        """A feature key with zero messages yields None across every frame."""
        feature_keys = ["/present", "/absent"]
        t0 = 1_700_000_000_000_000_000
        msgs: list[tuple[str, int, Any]] = [
            ("/present", t0 + i * int(10e6), f"v{i}") for i in range(50)
        ]
        resampler = Resampler(fps=30, policy=policy, tolerance_ms=50.0)
        got = resampler.resample(
            msgs,
            feature_keys,
            start_ns=t0,
            end_ns=t0 + int(0.5 * 1e9),
        )
        want = _resample_reference(
            resampler,
            msgs,
            feature_keys,
            start_ns=t0,
            end_ns=t0 + int(0.5 * 1e9),
        )
        _assert_frames_equal(got, want, feature_keys)
        assert all(f["/absent"] is None for f in got)

    @pytest.mark.parametrize("policy", ["hold", "nearest", "drop"])
    def test_edge_before_first_message(self, policy: str) -> None:
        """Frames before the first message follow policy-specific behaviour exactly."""
        t0 = 1_700_000_000_000_000_000
        # First message is 100ms into the episode; episode starts at t0.
        msgs: list[tuple[str, int, Any]] = [
            ("/a", t0 + int(100e6), "first"),
            ("/a", t0 + int(200e6), "second"),
        ]
        resampler = Resampler(fps=30, policy=policy, tolerance_ms=50.0)
        got = resampler.resample(msgs, ["/a"], start_ns=t0, end_ns=t0 + int(0.5 * 1e9))
        want = _resample_reference(
            resampler,
            msgs,
            ["/a"],
            start_ns=t0,
            end_ns=t0 + int(0.5 * 1e9),
        )
        _assert_frames_equal(got, want, ["/a"])


# ---------------------------------------------------------------------------
# Required-feature intersection window (cli._required_window / _process_episode)
# ---------------------------------------------------------------------------
# These cover the message-collection -> resample boundary added so the output
# frame grid spans only the range where every required feature has data
# (instead of the full bag time range). The pure-function tests need no
# config / reader; the integration tests drive the real ``_process_episode``
# through a fake BagReader.

from pathlib import Path  # noqa: E402
from unittest import mock  # noqa: E402

from rosbag2lerobot.cli import _required_window, _process_episode  # noqa: E402
from rosbag2lerobot.config import FeatureMapping, RobotConfig, ResamplingConfig  # noqa: E402


class TestRequiredWindow:
    """Pure-function tests for the required-feature intersection window."""

    def test_late_start_clips_head_and_early_end_clips_tail(self) -> None:
        """win = [max(per-key first), min(per-key last)]."""
        msgs: list[tuple[str, int, Any]] = []
        # /a spans 0..1000 ms; /b spans 300..700 ms.
        for ms in range(0, 1001, 100):
            msgs.append(("/a", ms * 1_000_000, "a"))
        for ms in range(300, 701, 100):
            msgs.append(("/b", ms * 1_000_000, "b"))
        win = _required_window(msgs, ["/a", "/b"])
        assert win == (300 * 1_000_000, 700 * 1_000_000)

    def test_unsorted_messages_handled(self) -> None:
        """min/max aggregation does not assume sorted input."""
        msgs: list[tuple[str, int, Any]] = [
            ("/a", 500, "a"),
            ("/a", 100, "a"),
            ("/a", 900, "a"),
            ("/b", 400, "b"),
            ("/b", 200, "b"),
        ]
        # /a span [100, 900], /b span [200, 400]
        # win = [max(100, 200), min(900, 400)] = [200, 400]
        assert _required_window(msgs, ["/a", "/b"]) == (200, 400)

    def test_missing_required_key_returns_none(self) -> None:
        """A required key with zero messages → empty episode (None)."""
        msgs: list[tuple[str, int, Any]] = [("/a", 100, "a"), ("/a", 200, "a")]
        assert _required_window(msgs, ["/a", "/b"]) is None

    def test_non_overlapping_spans_return_none(self) -> None:
        """win_end < win_start (disjoint spans) → empty episode (None)."""
        msgs: list[tuple[str, int, Any]] = [
            ("/a", 0, "a"),
            ("/a", 100, "a"),
            ("/b", 500, "b"),
            ("/b", 600, "b"),
        ]
        # win_start = max(0, 500) = 500 ; win_end = min(100, 600) = 100 < 500
        assert _required_window(msgs, ["/a", "/b"]) is None

    def test_no_required_keys_falls_back_to_message_span(self) -> None:
        """With no required keys, use the first/last message timestamps."""
        msgs: list[tuple[str, int, Any]] = [("/a", 100, "a"), ("/b", 700, "b")]
        assert _required_window(msgs, []) == (100, 700)

    def test_no_required_keys_empty_messages_returns_none(self) -> None:
        assert _required_window([], []) is None


class _FakeReader:
    """Minimal BagReader stand-in for _process_episode integration tests.

    Yields pre-built ``(topic, recv_ns, raw_msg)`` triples and reports a fixed
    full time range. ``raw_msg`` carries no header, so the real
    ``extract_header_stamp_ns`` returns None and the adopted timestamp is the
    bag receive time (matching ``stamp_source="receive"``).
    """

    def __init__(
        self, triples: list[tuple[str, int, object]], time_range: tuple[int, int]
    ) -> None:
        self._triples = triples
        self._time_range = time_range

    def __enter__(self) -> "_FakeReader":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def get_time_range(self) -> tuple[int, int]:
        return self._time_range

    def iter_messages(self, topics: list[str] | None = None):
        for topic, recv_ns, raw_msg in self._triples:
            if topics is None or topic in topics:
                yield topic, recv_ns, raw_msg


def _two_feature_config(align_to_required: bool) -> RobotConfig:
    """Config with two required numeric features on topics /a and /b."""
    return RobotConfig(
        robot_type="r",
        fps=10,
        task="t",
        observations=[
            FeatureMapping(
                key="observation.state",
                topic="/a",
                msg_type="sensor_msgs/msg/JointState",
                stamp_source="receive",
            ),
        ],
        actions=[
            FeatureMapping(
                key="action",
                topic="/b",
                msg_type="sensor_msgs/msg/JointState",
                stamp_source="receive",
            ),
        ],
        resampling=ResamplingConfig(
            default_policy="hold",
            align_to_required=align_to_required,
        ),
    )


class TestProcessEpisodeAlignment:
    """Integration tests for the _process_episode resample-window selection."""

    def _run(self, cfg: RobotConfig) -> list[dict[str, Any]]:
        """Drive _process_episode with a fake reader + stubbed decode."""
        t0 = 1_700_000_000_000_000_000
        triples: list[tuple[str, int, object]] = []
        # /a: 0..1000 ms ; /b: 300..700 ms  (10 fps grid period = 100 ms)
        for ms in range(0, 1001, 100):
            triples.append(("/a", t0 + ms * 1_000_000, object()))
        for ms in range(300, 701, 100):
            triples.append(("/b", t0 + ms * 1_000_000, object()))
        reader = _FakeReader(triples, time_range=(t0, t0 + 1000 * 1_000_000))
        resampler = Resampler(
            fps=cfg.fps, policy=cfg.resampling.default_policy, tolerance_ms=200.0
        )

        with (
            mock.patch("rosbag2lerobot.cli.convert.BagReader", return_value=reader),
            mock.patch(
                "rosbag2lerobot.cli.convert.decode",
                return_value=np.array([1.0], dtype=np.float32),
            ),
        ):
            return _process_episode(Path("/fake/bag"), cfg, resampler)

    def test_align_to_required_clips_to_intersection(self) -> None:
        """With align_to_required=True the grid spans only [300ms, 700ms]."""
        cfg = _two_feature_config(align_to_required=True)
        frames = self._run(cfg)
        # Window [300, 700] ms at 10 fps -> 4 frames (0.4 s * 10 = 4).
        assert len(frames) == 4
        # Both required features present in every retained frame.
        for f in frames:
            assert f["observation.state"] is not None
            assert f["action"] is not None

    def test_align_to_required_false_uses_full_range(self) -> None:
        """With align_to_required=False the legacy full bag range is used.

        The full [0, 1000] ms range is resampled, then ``hold`` carries the
        last /b value forward past 700 ms (the very out-of-range extrapolation
        that align_to_required eliminates), so trim_to_valid leaves a strictly
        wider frame extent than the intersection path's exact 4 frames. This
        pins the two code paths as distinct.
        """
        frames_false = self._run(_two_feature_config(align_to_required=False))
        frames_true = self._run(_two_feature_config(align_to_required=True))
        for f in frames_false:
            assert f["observation.state"] is not None
            assert f["action"] is not None
        assert len(frames_false) > len(frames_true)

    def test_non_overlapping_required_yields_empty(self) -> None:
        """Disjoint required spans → empty episode under align_to_required."""
        t0 = 1_700_000_000_000_000_000
        triples: list[tuple[str, int, object]] = [
            ("/a", t0 + 0, object()),
            ("/a", t0 + 100 * 1_000_000, object()),
            ("/b", t0 + 500 * 1_000_000, object()),
            ("/b", t0 + 600 * 1_000_000, object()),
        ]
        reader = _FakeReader(triples, time_range=(t0, t0 + 600 * 1_000_000))
        cfg = _two_feature_config(align_to_required=True)
        resampler = Resampler(fps=cfg.fps, policy="hold", tolerance_ms=50.0)
        with (
            mock.patch("rosbag2lerobot.cli.convert.BagReader", return_value=reader),
            mock.patch(
                "rosbag2lerobot.cli.convert.decode",
                return_value=np.array([1.0], dtype=np.float32),
            ),
        ):
            frames = _process_episode(Path("/fake/bag"), cfg, resampler)
        assert frames == []
