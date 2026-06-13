"""Parse and validate per-bag ``task.json`` sidecar files.

Each rosbag directory may contain a ``task.json`` that declares:

- ``task`` (optional str): overall episode task string. If missing or empty,
  callers fall back to the CLI ``--task`` override or the YAML ``config.task``.
- ``subtasks`` (optional list of ``{start, end, subtask}``): time-ranged
  sub-task annotations, seconds relative to the LeRobot ``timestamp=0.0``
  (i.e. the first kept frame after ``trim_to_valid``).

If ``subtasks`` is present for at least one bag in the conversion, the writer
emits ``meta/subtasks.parquet`` and a per-frame ``subtask_index`` column.
If every bag omits ``subtasks``, neither is produced.

Coverage rules (enforced at write time when subtasks are present):

- The first span must start at ``0.0``.
- Consecutive spans must touch exactly (``subtasks[i].end == subtasks[i+1].start``).
- The last span's ``end`` must reach ``episode_duration = frame_count / fps``.

Gaps, overlaps, or shortfalls raise :class:`ValueError`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TASK_SIDECAR_FILENAME = "task.json"

_COVERAGE_TOL = 1e-6


@dataclass(frozen=True)
class SubtaskSpan:
    """Time range within an episode annotated with a single subtask string."""

    start: float
    end: float
    subtask: str


@dataclass
class TaskSpec:
    """Parsed contents of a bag's ``task.json``.

    ``task`` is ``None`` when the JSON file is absent or omits the field.
    ``subtasks`` is an empty list when no annotations were provided.
    """

    task: str | None = None
    subtasks: list[SubtaskSpan] = field(default_factory=list)


def load_task_json(bag_path: Path) -> TaskSpec | None:
    """Load ``<bag_path>/task.json`` if present.

    Returns ``None`` when the file does not exist. Raises
    :class:`json.JSONDecodeError` on malformed JSON and :class:`ValueError`
    on schema violations (wrong types, missing required subtask fields).
    """
    sidecar = bag_path / TASK_SIDECAR_FILENAME
    if not sidecar.is_file():
        return None

    raw = sidecar.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(
            f"{sidecar}: top-level JSON must be an object, got {type(data).__name__}"
        )

    task = data.get("task")
    if task is not None:
        if not isinstance(task, str):
            raise ValueError(
                f"{sidecar}: 'task' must be a string, got {type(task).__name__}"
            )
        task = task.strip() or None

    subtasks_raw = data.get("subtasks", [])
    if not isinstance(subtasks_raw, list):
        raise ValueError(
            f"{sidecar}: 'subtasks' must be a list, got {type(subtasks_raw).__name__}"
        )

    subtasks: list[SubtaskSpan] = []
    for i, entry in enumerate(subtasks_raw):
        subtasks.append(_parse_span(entry, sidecar, i))

    return TaskSpec(task=task, subtasks=subtasks)


def _parse_span(entry: Any, sidecar: Path, idx: int) -> SubtaskSpan:
    """Validate one element of the ``subtasks`` array."""
    if not isinstance(entry, dict):
        raise ValueError(
            f"{sidecar}: subtasks[{idx}] must be an object, got {type(entry).__name__}"
        )
    for key in ("start", "end", "subtask"):
        if key not in entry:
            raise ValueError(f"{sidecar}: subtasks[{idx}] missing required key '{key}'")

    start = entry["start"]
    end = entry["end"]
    name = entry["subtask"]

    if not isinstance(start, (int, float)) or isinstance(start, bool):
        raise ValueError(f"{sidecar}: subtasks[{idx}].start must be a number")
    if not isinstance(end, (int, float)) or isinstance(end, bool):
        raise ValueError(f"{sidecar}: subtasks[{idx}].end must be a number")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"{sidecar}: subtasks[{idx}].subtask must be a non-empty string"
        )
    if float(end) <= float(start):
        raise ValueError(
            f"{sidecar}: subtasks[{idx}] end ({end}) must be greater than start ({start})"
        )

    return SubtaskSpan(start=float(start), end=float(end), subtask=name.strip())


def resolve_task(bag_path: Path, fallback: str) -> tuple[str, list[SubtaskSpan]]:
    """Return ``(task_str, subtasks)`` for ``bag_path``.

    Task priority (highest first):

    1. ``task.json``'s non-empty ``task`` field
    2. ``fallback`` (caller-collapsed CLI + YAML precedence)

    Subtasks come solely from ``task.json``. Absent file → no subtasks.
    """
    spec = load_task_json(bag_path)
    if spec is None:
        return fallback, []
    task = spec.task if spec.task else fallback
    return task, spec.subtasks


def validate_subtask_coverage(
    subtasks: list[SubtaskSpan],
    episode_duration: float,
    *,
    context: str = "",
) -> None:
    """Ensure ``subtasks`` tile ``[0, episode_duration]`` without gaps/overlaps.

    Rules:

    - First span starts at ``0.0`` (within ``_COVERAGE_TOL``).
    - Consecutive spans touch exactly.
    - Last span's ``end`` reaches ``episode_duration``.

    Raises :class:`ValueError` on any violation. ``context`` is prepended to
    the message so callers can identify the offending episode / bag.
    """
    if not subtasks:
        return

    prefix = f"{context}: " if context else ""

    first = subtasks[0]
    if not math.isclose(first.start, 0.0, abs_tol=_COVERAGE_TOL):
        raise ValueError(f"{prefix}first subtask must start at 0.0 (got {first.start})")

    for i in range(len(subtasks) - 1):
        gap = subtasks[i + 1].start - subtasks[i].end
        if not math.isclose(gap, 0.0, abs_tol=_COVERAGE_TOL):
            raise ValueError(
                f"{prefix}subtasks[{i}].end ({subtasks[i].end}) must equal "
                f"subtasks[{i + 1}].start ({subtasks[i + 1].start})"
            )

    last_end = subtasks[-1].end
    if last_end + _COVERAGE_TOL < episode_duration:
        raise ValueError(
            f"{prefix}last subtask ends at {last_end} but episode_duration is "
            f"{episode_duration}; full-time coverage required"
        )


def subtask_for_timestamp(
    subtasks: list[SubtaskSpan],
    timestamp: float,
) -> str:
    """Return the subtask string whose span contains ``timestamp``.

    The matching interval is ``[start, end)`` except for the final span, where
    ``timestamp == end`` still matches (to cover a last frame exactly at the
    declared boundary). Raises :class:`ValueError` if no span contains it —
    callers should have already run :func:`validate_subtask_coverage`.
    """
    if not subtasks:
        raise ValueError("subtasks is empty")

    for i, span in enumerate(subtasks):
        is_last = i == len(subtasks) - 1
        if span.start - _COVERAGE_TOL <= timestamp < span.end:
            return span.subtask
        if is_last and math.isclose(timestamp, span.end, abs_tol=_COVERAGE_TOL):
            return span.subtask
    raise ValueError(f"timestamp {timestamp} not covered by any subtask span")
