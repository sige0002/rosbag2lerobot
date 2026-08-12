"""Timestamp integrity guard for the conversion pipeline.

A ROS message carries two clocks: the ``header.stamp`` written by the
publisher, and the receive time the recorder stamped it with. The converter
adopts one of them per feature (``stamp_source``). When the two disagree by
more than transport latency can explain — an unsynchronised robot clock, a
driver stamping with a monotonic clock, sim time leaking into a real
recording — every sample time in the output is wrong, but nothing about the
resulting dataset *looks* broken.

This module holds the loud failure for that case: the threshold check, the
exception the pipeline raises, and the message it raises it with. The
threshold itself lives in ``config.TimestampsConfig``.

Design rules:

- :class:`StampSkewError` carries a single, already-formatted message so it
  survives the pickling round-trip a ``ProcessPoolExecutor`` worker puts an
  exception through (``--workers > 1``).
- :func:`format_skew_error` is pure, so the wording is testable without a bag.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

NS_PER_MS = 1_000_000


class StampSkewError(RuntimeError):
    """Raised when a message's header stamp diverges from its receive time.

    Handled by the per-episode failure machinery in ``cli.convert``: without
    ``--skip-failed`` the run aborts, with it the episode is recorded as failed
    and conversion continues.
    """


def _format_ns_as_utc(ts_ns: int) -> str:
    """Render a UNIX nanosecond timestamp as an ISO-8601 UTC string.

    Falls back to the raw nanosecond count when the value is outside the range
    ``datetime`` can represent — which is itself a symptom worth showing, since
    a clock that far off is exactly what this guard exists to catch.
    """
    try:
        return datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return f"{ts_ns} ns"


def format_skew_error(
    *,
    bag_path: Path | str,
    topic: str,
    feature_key: str,
    header_ns: int,
    receive_ns: int,
    threshold_ms: float,
) -> str:
    """Build the operator-facing message for a header/receive skew violation.

    Names the bag, the topic and feature that tripped the check, the observed
    skew, the threshold it exceeded, and both timestamps in human-readable
    form, then lists the ways out (fix the clock, switch the feature to receive
    stamps, raise or disable the threshold).

    Args:
        bag_path: Bag being converted when the violation was found.
        topic: ROS topic the offending message came from.
        feature_key: Config feature key that reads *topic* with
            ``stamp_source: header``.
        header_ns: The message's ``header.stamp`` in UNIX nanoseconds.
        receive_ns: The bag receive time in UNIX nanoseconds.
        threshold_ms: The configured ``max_header_receive_skew_ms``.

    Returns:
        A multi-line message suitable for a CLI error or a job summary entry.
    """
    skew_ms = abs(receive_ns - header_ns) / NS_PER_MS
    direction = "behind" if header_ns < receive_ns else "ahead of"
    return (
        f"Header/receive timestamp skew in {bag_path}: topic {topic!r} "
        f"(feature {feature_key!r}) has a header stamp {skew_ms:.0f} ms "
        f"{direction} its bag receive time, over the "
        f"timestamps.max_header_receive_skew_ms limit of {threshold_ms:g} ms. "
        f"header.stamp={_format_ns_as_utc(header_ns)}, "
        f"receive={_format_ns_as_utc(receive_ns)}. "
        "Converting this bag would put those wrong times in the dataset. "
        "Fix the clock on the recording/publishing host, or set "
        f"stamp_source: receive for {feature_key!r}, or raise "
        "timestamps.max_header_receive_skew_ms (set it to null to disable "
        "the check). To drop individual stale latched messages instead, use "
        "resampling.max_stamp_delay_ms."
    )
