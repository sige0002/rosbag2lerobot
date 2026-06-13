"""Tests for the rosbag reader helpers."""

from __future__ import annotations

from types import SimpleNamespace

from bagel.reader import extract_header_stamp_ns


def _msg_with_stamp(sec: int, nanosec: int) -> SimpleNamespace:
    """Build a dummy message carrying a ``header.stamp`` like a ROS msg."""
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec))
    )


class TestExtractHeaderStampNs:
    """Unit tests for ``extract_header_stamp_ns``."""

    def test_header_present_returns_ns(self) -> None:
        msg = _msg_with_stamp(sec=12, nanosec=500_000_000)
        assert extract_header_stamp_ns(msg) == 12_500_000_000

    def test_sec_and_nanosec_combined(self) -> None:
        """sec contributes 1e9 ns each; nanosec adds the remainder."""
        msg = _msg_with_stamp(sec=3, nanosec=42)
        assert extract_header_stamp_ns(msg) == 3 * 1_000_000_000 + 42

    def test_nanosec_only(self) -> None:
        msg = _msg_with_stamp(sec=0, nanosec=250)
        assert extract_header_stamp_ns(msg) == 250

    def test_no_header_returns_none(self) -> None:
        msg = SimpleNamespace(data=[1.0, 2.0])  # no header attribute
        assert extract_header_stamp_ns(msg) is None

    def test_header_without_stamp_returns_none(self) -> None:
        msg = SimpleNamespace(header=SimpleNamespace(frame_id="base"))
        assert extract_header_stamp_ns(msg) is None

    def test_stamp_zero_returns_none(self) -> None:
        """An unset stamp (sec=0, nanosec=0) is treated as missing."""
        msg = _msg_with_stamp(sec=0, nanosec=0)
        assert extract_header_stamp_ns(msg) is None

    def test_missing_nanosec_returns_none(self) -> None:
        msg = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=5)))
        assert extract_header_stamp_ns(msg) is None

    def test_does_not_raise_on_garbage(self) -> None:
        """Non-numeric stamp fields yield None rather than raising."""
        msg = SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec="x", nanosec="y"))
        )
        assert extract_header_stamp_ns(msg) is None
