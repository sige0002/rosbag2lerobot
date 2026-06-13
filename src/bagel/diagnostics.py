"""Diagnostic helpers for bagel.

Pure functions that support the ``inspect --fps-stats``,
``inspect --suggest-image-size``, and ``validate-config`` CLI features.

All public functions here are I/O-free except ``detect_image_shape`` and
``validate_config_against_bag`` which consume a :class:`BagReader` but
never produce CLI output themselves. This separation lets unit tests
exercise the diagnostic logic without involving Click.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

from bagel.config import FeatureMapping, RobotConfig
from bagel.decoders import decode
from bagel.reader import BagReader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# F1. Topic FPS / gap / lag statistics
# ---------------------------------------------------------------------------


def compute_topic_fps_report(
    ts_ns: np.ndarray,
    bag_start_ns: int,
    bag_end_ns: int,
    msg_type: str,
    msg_count: int,
    gap_threshold_ms: float,
    head_n: int,
) -> dict[str, Any]:
    """Compute per-topic FPS, gap, and head/tail lag statistics.

    Parameters
    ----------
    ts_ns:
        Sorted array of message timestamps in nanoseconds. May be empty
        or length 1.
    bag_start_ns, bag_end_ns:
        Bag time-range bounds in nanoseconds.
    msg_type:
        ROS2 message type string for the topic.
    msg_count:
        Number of messages recorded for the topic.
    gap_threshold_ms:
        Inter-arrival gaps larger than this are flagged in ``gaps``.
    head_n:
        Number of head sample intervals (ms) to include in the report.

    Returns
    -------
    dict:
        Report dict. Keys: ``msg_type``, ``msg_count``, ``fps``,
        ``head_lag_ms``, ``tail_lag_ms``, ``gaps``, ``head_intervals_ms``.
        ``fps["mean"]`` is the effective rate over the topic's span
        (intervals / elapsed time), so it is not skewed by near-duplicate
        timestamps; ``min``/``max``/``std``/``p01``/``p99`` describe the
        instantaneous per-interval fps distribution.
    """
    report: dict[str, Any] = {
        "msg_type": msg_type,
        "msg_count": int(msg_count),
        "fps": {
            "mean": None,
            "min": None,
            "max": None,
            "std": None,
            "p01": None,
            "p99": None,
        },
        "head_lag_ms": None,
        "tail_lag_ms": None,
        "gaps": [],
        "head_intervals_ms": [],
    }

    if ts_ns.size == 0:
        return report

    ts_ns = np.asarray(ts_ns, dtype=np.int64)

    # head/tail lag relative to bag start / end.
    report["head_lag_ms"] = float((ts_ns[0] - bag_start_ns) / 1e6)
    report["tail_lag_ms"] = float((bag_end_ns - ts_ns[-1]) / 1e6)

    if ts_ns.size < 2:
        return report

    intervals_ns = np.diff(ts_ns).astype(np.int64)
    # Guard against zero-interval duplicate timestamps — rare but possible.
    positive = intervals_ns[intervals_ns > 0]
    if positive.size == 0:
        return report

    intervals_s = positive.astype(np.float64) / 1e9
    fps_arr = 1.0 / intervals_s

    # ``mean`` is the topic's effective rate over its whole span:
    # (#intervals) / (total elapsed time). This is robust to near-duplicate
    # timestamps — a single dt of a few microseconds (common when a publisher
    # emits a burst of messages sharing almost the same stamp) would blow up
    # ``mean(1/dt)`` to tens of thousands of fps, so we never average the
    # per-interval reciprocals. ``min``/``max``/``std``/``p01``/``p99`` keep
    # their meaning as the instantaneous per-interval fps distribution.
    span_s = float(positive.sum()) / 1e9
    report["fps"] = {
        "mean": float(positive.size / span_s),
        "min": float(np.min(fps_arr)),
        "max": float(np.max(fps_arr)),
        "std": float(np.std(fps_arr)),
        "p01": float(np.percentile(fps_arr, 1)),
        "p99": float(np.percentile(fps_arr, 99)),
    }

    # Gaps: intervals whose duration exceeds the threshold.
    gap_threshold_ns = int(gap_threshold_ms * 1e6)
    gap_positions = np.where(intervals_ns > gap_threshold_ns)[0]
    gaps: list[dict[str, float]] = []
    for idx in gap_positions:
        gap_ns = int(intervals_ns[idx])
        # idx is the interval between ts_ns[idx] and ts_ns[idx+1]; flag at
        # the later sample so "at_s" reflects when the silence ends.
        at_ns = int(ts_ns[idx])
        gaps.append(
            {
                "at_s": float((at_ns - bag_start_ns) / 1e9),
                "duration_ms": float(gap_ns / 1e6),
            }
        )
    report["gaps"] = gaps

    head_count = min(int(head_n), intervals_ns.size)
    report["head_intervals_ms"] = [
        float(x / 1e6) for x in intervals_ns[:head_count].tolist()
    ]

    return report


# ---------------------------------------------------------------------------
# F3. Image-shape detection
# ---------------------------------------------------------------------------


def detect_image_shape(
    reader: BagReader,
    fm: FeatureMapping,
    n_samples: int,
) -> Optional[tuple[int, int, int]]:
    """Decode up to ``n_samples`` frames and return the consensus shape.

    Returns ``None`` when there are no samples, decoding fails, or
    samples disagree on shape. A debug log entry is emitted on any
    irregularity so CLI callers can surface a meaningful warning.
    """
    if n_samples <= 0:
        return None

    shapes: list[tuple[int, int, int]] = []
    collected = 0
    for _topic, _ts_ns, raw_msg in reader.iter_messages(topics=[fm.topic]):
        try:
            # Intentionally pass an empty config so the resize step in the
            # image decoder is skipped — we want the native decoded shape.
            decoded = decode(
                msg_type=fm.msg_type,
                deserialized_msg=raw_msg,
                selector=None,
                config={},
            )
        except Exception as exc:
            logger.debug(
                "detect_image_shape: decode failed on %s: %s",
                fm.topic,
                exc,
            )
            return None

        shape = _shape_of_decoded(decoded)
        if shape is None:
            return None
        shapes.append(shape)
        collected += 1
        if collected >= n_samples:
            break

    if not shapes:
        return None

    first = shapes[0]
    if all(s == first for s in shapes):
        return first

    logger.debug(
        "detect_image_shape: inconsistent shapes on %s: %s",
        fm.topic,
        shapes,
    )
    return None


def _shape_of_decoded(decoded: Any) -> Optional[tuple[int, int, int]]:
    """Return ``(H, W, C)`` for a PIL image or ndarray, else ``None``."""
    # PIL Image exposes (W, H); numpy array exposes (H, W, C) or (H, W).
    if hasattr(decoded, "size") and hasattr(decoded, "mode"):
        w, h = decoded.size
        mode = decoded.mode
        channels = {"RGB": 3, "RGBA": 4, "L": 1}.get(mode, 3)
        return (int(h), int(w), int(channels))
    if hasattr(decoded, "shape"):
        shape = tuple(int(x) for x in decoded.shape)
        if len(shape) == 2:
            return (shape[0], shape[1], 1)
        if len(shape) == 3:
            return shape  # type: ignore[return-value]
    return None


# ---------------------------------------------------------------------------
# F4. Config validation
# ---------------------------------------------------------------------------


@dataclass
class MsgTypeMismatch:
    """YAML-declared msg_type differs from the bag's recorded msg_type."""

    topic: str
    yaml: str
    bag: str


@dataclass
class ImageShapeMismatch:
    """YAML ``image_size`` differs from the shape seen in decoded samples."""

    key: str
    topic: str
    yaml: Optional[list[int]]
    decoded: Optional[list[int]]


@dataclass
class ValidationReport:
    """Structured output of :func:`validate_config_against_bag`."""

    missing_required_topics: list[str] = field(default_factory=list)
    missing_optional_topics: list[str] = field(default_factory=list)
    msg_type_mismatches: list[MsgTypeMismatch] = field(default_factory=list)
    image_shape_mismatches: list[ImageShapeMismatch] = field(default_factory=list)
    unused_bag_topics: list[str] = field(default_factory=list)
    verdict: str = "OK"
    exit_code: int = 0

    def has_errors(self) -> bool:
        """True when any fatal mismatch was detected."""
        return bool(self.missing_required_topics) or bool(self.msg_type_mismatches)

    def has_warnings(self) -> bool:
        """True when any non-fatal mismatch was detected."""
        return bool(self.image_shape_mismatches)

    def has_infos(self) -> bool:
        """True when any informational-only mismatch was detected."""
        return bool(self.missing_optional_topics) or bool(self.unused_bag_topics)

    def apply_verdict(self, strict: bool) -> None:
        """Populate ``verdict`` / ``exit_code`` from the mismatch lists.

        With ``strict`` true, warnings and infos (except missing-optional)
        also escalate to a failing exit code.
        """
        if self.has_errors():
            self.verdict = "FAIL"
            self.exit_code = 1
            return
        if strict and (self.has_warnings() or self.unused_bag_topics):
            self.verdict = "FAIL"
            self.exit_code = 1
            return
        self.verdict = "OK"
        self.exit_code = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {
            "missing_required_topics": list(self.missing_required_topics),
            "missing_optional_topics": list(self.missing_optional_topics),
            "msg_type_mismatches": [asdict(m) for m in self.msg_type_mismatches],
            "image_shape_mismatches": [asdict(m) for m in self.image_shape_mismatches],
            "unused_bag_topics": list(self.unused_bag_topics),
            "verdict": self.verdict,
            "exit_code": int(self.exit_code),
        }


def validate_config_against_bag(
    cfg: RobotConfig,
    reader: BagReader,
    samples: int,
) -> ValidationReport:
    """Cross-check a :class:`RobotConfig` against the contents of a bag.

    Populates a :class:`ValidationReport` with every discrepancy the
    checker finds. Callers are expected to call
    :meth:`ValidationReport.apply_verdict` with the CLI ``--strict`` flag
    after this function returns; that split keeps the pure function free
    of CLI-specific policy.
    """
    report = ValidationReport()

    bag_info = reader.get_topics_info()
    bag_topics = set(bag_info.keys())
    optional_topics = cfg.optional_topics
    declared_topics = set(cfg.all_topics)

    for topic in cfg.all_topics:
        if topic in bag_topics:
            continue
        if topic in optional_topics:
            report.missing_optional_topics.append(topic)
        else:
            report.missing_required_topics.append(topic)

    # msg_type consistency — only check topics that actually exist in the bag.
    topic_to_fms = cfg.topic_to_features
    seen_mismatch_topics: set[str] = set()
    for topic, fms in topic_to_fms.items():
        if topic not in bag_info:
            continue
        bag_msg_type = bag_info[topic].msg_type
        for fm in fms:
            if fm.msg_type != bag_msg_type and topic not in seen_mismatch_topics:
                report.msg_type_mismatches.append(
                    MsgTypeMismatch(
                        topic=topic,
                        yaml=fm.msg_type,
                        bag=bag_msg_type,
                    )
                )
                seen_mismatch_topics.add(topic)

    # Image shape — only attempt when the topic is present and msg_type matches.
    mismatched_topics = {m.topic for m in report.msg_type_mismatches}
    for fm in cfg.image_features:
        if fm.topic not in bag_info or fm.topic in mismatched_topics:
            continue
        detected = detect_image_shape(reader, fm, samples)
        if detected is None:
            continue
        yaml_shape = _normalize_yaml_image_size(fm.image_size)
        detected_list = [int(x) for x in detected]
        if yaml_shape is None or yaml_shape != detected_list:
            report.image_shape_mismatches.append(
                ImageShapeMismatch(
                    key=fm.key,
                    topic=fm.topic,
                    yaml=list(fm.image_size) if fm.image_size is not None else None,
                    decoded=detected_list,
                )
            )

    # Unused bag topics — in the bag but not referenced by the config.
    for topic in sorted(bag_topics - declared_topics):
        report.unused_bag_topics.append(topic)

    return report


def _normalize_yaml_image_size(
    image_size: Optional[list[int]],
) -> Optional[list[int]]:
    """Normalize YAML ``image_size`` (``[H, W]`` or ``[H, W, C]``) to 3-tuple.

    Returns ``None`` when no image size is declared so callers can treat
    the missing case explicitly.
    """
    if image_size is None:
        return None
    if len(image_size) == 2:
        return [int(image_size[0]), int(image_size[1]), 3]
    if len(image_size) == 3:
        return [int(x) for x in image_size]
    return None
