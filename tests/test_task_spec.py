"""Tests for ``bagel.task_spec`` — task.json parsing + subtasks.

Covers:

- Task priority: ``task.json`` > fallback (CLI/YAML collapse)
- task.json presence / absence / empty / malformed
- Subtask schema: type checks, required fields, ordering invariants
- Coverage validation: start==0, contiguous spans, last.end >= duration
- ``subtask_for_timestamp`` lookup semantics
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bagel.task_spec import (
    SubtaskSpan,
    TASK_SIDECAR_FILENAME,
    load_task_json,
    resolve_task,
    subtask_for_timestamp,
    validate_subtask_coverage,
)


def _write_json(bag: Path, payload: dict) -> None:
    bag.mkdir(parents=True, exist_ok=True)
    (bag / TASK_SIDECAR_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


class TestResolveTask:
    def test_missing_sidecar_falls_back(self, tmp_path: Path) -> None:
        bag = tmp_path / "bag_a"
        bag.mkdir()
        task, subtasks = resolve_task(bag, "yaml-default")
        assert task == "yaml-default"
        assert subtasks == []

    def test_sidecar_task_wins(self, tmp_path: Path) -> None:
        bag = tmp_path / "bag_b"
        _write_json(bag, {"task": "pick up red block"})
        task, subtasks = resolve_task(bag, "yaml-default")
        assert task == "pick up red block"
        assert subtasks == []

    def test_empty_task_falls_back(self, tmp_path: Path) -> None:
        bag = tmp_path / "bag_c"
        _write_json(bag, {"task": "   "})
        task, _ = resolve_task(bag, "yaml-default")
        assert task == "yaml-default"

    def test_task_missing_field_falls_back(self, tmp_path: Path) -> None:
        bag = tmp_path / "bag_d"
        _write_json(bag, {"subtasks": []})
        task, subtasks = resolve_task(bag, "cli-or-yaml-default")
        assert task == "cli-or-yaml-default"
        assert subtasks == []

    def test_subtasks_parsed(self, tmp_path: Path) -> None:
        bag = tmp_path / "bag_e"
        _write_json(
            bag,
            {
                "task": "fold towel",
                "subtasks": [
                    {"start": 0.0, "end": 1.5, "subtask": "grasp"},
                    {"start": 1.5, "end": 3.0, "subtask": "lift"},
                ],
            },
        )
        task, subtasks = resolve_task(bag, "fallback")
        assert task == "fold towel"
        assert subtasks == [
            SubtaskSpan(0.0, 1.5, "grasp"),
            SubtaskSpan(1.5, 3.0, "lift"),
        ]

    def test_unknown_fields_ignored(self, tmp_path: Path) -> None:
        bag = tmp_path / "bag_f"
        _write_json(bag, {"task": "a", "description": "extra", "version": 2})
        task, subtasks = resolve_task(bag, "fallback")
        assert task == "a"
        assert subtasks == []


class TestLoadTaskJsonSchemaErrors:
    def test_top_level_not_object(self, tmp_path: Path) -> None:
        bag = tmp_path / "bag"
        bag.mkdir()
        (bag / TASK_SIDECAR_FILENAME).write_text('["a","b"]', encoding="utf-8")
        with pytest.raises(ValueError, match="top-level JSON must be an object"):
            load_task_json(bag)

    def test_task_wrong_type(self, tmp_path: Path) -> None:
        bag = tmp_path / "bag"
        _write_json(bag, {"task": 42})
        with pytest.raises(ValueError, match="'task' must be a string"):
            load_task_json(bag)

    def test_subtasks_not_list(self, tmp_path: Path) -> None:
        bag = tmp_path / "bag"
        _write_json(bag, {"subtasks": "oops"})
        with pytest.raises(ValueError, match="'subtasks' must be a list"):
            load_task_json(bag)

    def test_subtask_missing_field(self, tmp_path: Path) -> None:
        bag = tmp_path / "bag"
        _write_json(bag, {"subtasks": [{"start": 0, "subtask": "x"}]})
        with pytest.raises(ValueError, match="missing required key 'end'"):
            load_task_json(bag)

    def test_subtask_start_not_number(self, tmp_path: Path) -> None:
        bag = tmp_path / "bag"
        _write_json(bag, {"subtasks": [{"start": "0", "end": 1.0, "subtask": "x"}]})
        with pytest.raises(ValueError, match="start must be a number"):
            load_task_json(bag)

    def test_subtask_empty_name(self, tmp_path: Path) -> None:
        bag = tmp_path / "bag"
        _write_json(bag, {"subtasks": [{"start": 0.0, "end": 1.0, "subtask": " "}]})
        with pytest.raises(ValueError, match="subtask must be a non-empty string"):
            load_task_json(bag)

    def test_subtask_end_le_start(self, tmp_path: Path) -> None:
        bag = tmp_path / "bag"
        _write_json(bag, {"subtasks": [{"start": 1.0, "end": 1.0, "subtask": "x"}]})
        with pytest.raises(ValueError, match="end .* must be greater than start"):
            load_task_json(bag)

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        bag = tmp_path / "bag"
        bag.mkdir()
        (bag / TASK_SIDECAR_FILENAME).write_text("{not valid", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_task_json(bag)


class TestValidateSubtaskCoverage:
    def test_empty_is_noop(self) -> None:
        validate_subtask_coverage([], 5.0)

    def test_full_coverage_passes(self) -> None:
        spans = [
            SubtaskSpan(0.0, 1.0, "a"),
            SubtaskSpan(1.0, 2.5, "b"),
            SubtaskSpan(2.5, 5.0, "c"),
        ]
        validate_subtask_coverage(spans, 5.0)

    def test_first_not_zero_fails(self) -> None:
        spans = [SubtaskSpan(0.1, 5.0, "a")]
        with pytest.raises(ValueError, match="first subtask must start at 0.0"):
            validate_subtask_coverage(spans, 5.0, context="ep0")

    def test_gap_between_spans_fails(self) -> None:
        spans = [
            SubtaskSpan(0.0, 1.0, "a"),
            SubtaskSpan(1.5, 5.0, "b"),  # gap 1.0 -> 1.5
        ]
        with pytest.raises(ValueError, match="must equal"):
            validate_subtask_coverage(spans, 5.0)

    def test_overlap_fails(self) -> None:
        spans = [
            SubtaskSpan(0.0, 2.0, "a"),
            SubtaskSpan(1.5, 5.0, "b"),  # overlaps 1.5-2.0
        ]
        with pytest.raises(ValueError, match="must equal"):
            validate_subtask_coverage(spans, 5.0)

    def test_short_tail_fails(self) -> None:
        spans = [
            SubtaskSpan(0.0, 3.0, "a"),
        ]
        with pytest.raises(ValueError, match="full-time coverage required"):
            validate_subtask_coverage(spans, 5.0)

    def test_tail_beyond_duration_is_ok(self) -> None:
        """If the annotator padded the last span past the episode, accept."""
        spans = [SubtaskSpan(0.0, 999.0, "a")]
        validate_subtask_coverage(spans, 5.0)


class TestSubtaskForTimestamp:
    def test_first_span(self) -> None:
        spans = [SubtaskSpan(0.0, 1.0, "a"), SubtaskSpan(1.0, 2.0, "b")]
        assert subtask_for_timestamp(spans, 0.0) == "a"
        assert subtask_for_timestamp(spans, 0.5) == "a"

    def test_boundary_goes_to_next(self) -> None:
        """At exactly the boundary, the later span wins (half-open [start, end))."""
        spans = [SubtaskSpan(0.0, 1.0, "a"), SubtaskSpan(1.0, 2.0, "b")]
        assert subtask_for_timestamp(spans, 1.0) == "b"

    def test_final_span_end_is_inclusive(self) -> None:
        """A frame exactly at the final span's end should still match it."""
        spans = [SubtaskSpan(0.0, 1.0, "a"), SubtaskSpan(1.0, 2.0, "b")]
        assert subtask_for_timestamp(spans, 2.0) == "b"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="subtasks is empty"):
            subtask_for_timestamp([], 0.0)
