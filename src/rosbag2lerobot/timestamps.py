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

TF inputs need their own check (:func:`first_tf_skew`). A ``TFMessage`` has no
header of its own — each transform inside it carries one — so the per-message
check cannot see them, while ``TransformLookup`` keys its timeline on exactly
those stamps. A skewed dynamic transform therefore pins a pose to whichever
end of its timeline is nearest, which looks like a valid, motionless frame.

Design rules:

- :class:`StampSkewError` carries a single, already-formatted message so it
  survives the pickling round-trip a ``ProcessPoolExecutor`` worker puts an
  exception through (``--workers > 1``).
- the format functions are pure, so the wording is testable without a bag.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    return (
        f"Header/receive timestamp skew in {bag_path}: topic {topic!r} "
        f"(feature {feature_key!r}) has "
        + _skew_phrase(header_ns, receive_ns, threshold_ms)
        + " Converting this bag would put those wrong times in the dataset. "
        "Fix the clock on the recording/publishing host, or set "
        f"stamp_source: receive for {feature_key!r}, or raise "
        "timestamps.max_header_receive_skew_ms (set it to null to disable "
        "the check). To drop individual stale latched messages instead, use "
        "resampling.max_stamp_delay_ms."
    )


def _skew_phrase(header_ns: int, receive_ns: int, threshold_ms: float) -> str:
    """Render the measurement shared by every skew message.

    Kept in one place so the number, the threshold and the two timestamps
    cannot drift apart between the per-feature and the TF wording.
    """
    skew_ms = abs(receive_ns - header_ns) / NS_PER_MS
    direction = "behind" if header_ns < receive_ns else "ahead of"
    return (
        f"a header stamp {skew_ms:.0f} ms {direction} its bag receive time, "
        "over the timestamps.max_header_receive_skew_ms limit of "
        f"{threshold_ms:g} ms. "
        f"header.stamp={_format_ns_as_utc(header_ns)}, "
        f"receive={_format_ns_as_utc(receive_ns)}."
    )


def first_tf_skew(
    tf_msg: Any,
    receive_ns: int,
    limit_ns: float,
) -> tuple[str, str, int] | None:
    """Find the first transform in *tf_msg* whose header stamp diverges too far.

    A ``tf2_msgs/msg/TFMessage`` carries no header of its own — each transform
    inside it has one — so the per-feature guard, which reads ``msg.header``,
    never sees a TF message at all. This is the TF equivalent.

    Args:
        tf_msg: A deserialized ``TFMessage``.
        receive_ns: The bag receive time of that message.
        limit_ns: ``max_header_receive_skew_ms`` expressed in nanoseconds.

    Returns:
        ``(parent_frame, child_frame, header_ns)`` for the first offending
        transform, or ``None`` when every transform is within the limit.
        Transforms with an unset stamp (``sec`` and ``nanosec`` both 0) are
        skipped, matching :func:`reader.extract_header_stamp_ns`.
    """
    for transform in getattr(tf_msg, "transforms", ()) or ():
        header = getattr(transform, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is None:
            continue
        try:
            sec = int(stamp.sec)
            nanosec = int(stamp.nanosec)
        except (AttributeError, TypeError, ValueError):
            continue
        if sec == 0 and nanosec == 0:
            continue
        header_ns = sec * 1_000_000_000 + nanosec
        if abs(receive_ns - header_ns) > limit_ns:
            return (
                str(getattr(header, "frame_id", "")),
                str(getattr(transform, "child_frame_id", "")),
                header_ns,
            )
    return None


def format_tf_skew_error(
    *,
    bag_path: Path | str,
    topic: str,
    parent_frame: str,
    child_frame: str,
    header_ns: int,
    receive_ns: int,
    threshold_ms: float,
) -> str:
    """Build the operator-facing message for a skewed dynamic transform.

    Separate wording from :func:`format_skew_error` because the way out is
    different: a TF feature has no ``stamp_source`` to switch to — the
    transform timeline is always keyed on header stamps — and the damage is
    different too (a frozen pose rather than shifted sample times).

    Args:
        bag_path: Bag being converted when the violation was found.
        topic: The dynamic TF topic (``tf_topic``, normally ``/tf``).
        parent_frame: ``header.frame_id`` of the offending transform.
        child_frame: ``child_frame_id`` of the offending transform.
        header_ns: That transform's header stamp, in UNIX nanoseconds.
        receive_ns: The bag receive time of the message carrying it.
        threshold_ms: The configured ``max_header_receive_skew_ms``.

    Returns:
        A message suitable for a CLI error or a job summary entry.
    """
    return (
        f"Header/receive timestamp skew in {bag_path}: topic {topic!r} "
        f"transform {parent_frame!r} -> {child_frame!r} has "
        + _skew_phrase(header_ns, receive_ns, threshold_ms)
        + " TF features are sampled from the transform timeline by header "
        "stamp, so converting this bag would silently pin that transform to "
        "whichever end of its timeline is nearest — a pose that looks valid "
        "and never moves. Fix the clock on the recording/publishing host, or "
        "raise timestamps.max_header_receive_skew_ms (set it to null to "
        "disable the check). Static transforms are not affected: their stamps "
        "are discarded rather than used to look poses up."
    )
