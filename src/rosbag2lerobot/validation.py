"""Structural validator for generated LeRobot v3.0 datasets.

This module inspects a finished dataset directory (the tree produced by
:class:`rosbag2lerobot.writer.DatasetWriter`) and verifies that its files,
``meta/info.json`` keys, and parquet schemas conform to the LeRobot v3.0
layout that :mod:`rosbag2lerobot.writer` emits.

The check is **read-only** and never raises on a validation failure: every
discrepancy is collected as a :class:`ValidationIssue` so a single pass
reports the complete picture. The only exceptions raised are for genuinely
unreadable inputs (a parquet file that cannot be opened, etc.), which the
CLI layer maps to a setup-error exit code.

Source-of-truth references:

- ``meta/info.json`` keys and values: :meth:`rosbag2lerobot.writer.DatasetWriter._write_info_json`.
- data parquet column types: :meth:`rosbag2lerobot.writer.DatasetWriter._build_data_table`.
- episodes parquet columns: :meth:`rosbag2lerobot.writer.DatasetWriter._write_episodes_parquet`.

Scope is deliberately structural: it does not decode videos or cross-check
mp4 frame counts (that is :mod:`rosbag2lerobot.quality`'s job). It answers the
question "is this a well-formed LeRobot v3.0 dataset on disk?".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from rosbag2lerobot.audit import (
    _collect_episodes_parquet,
    _discover_video_keys,
    _ensure_columns_present,
)
from rosbag2lerobot.writer import _CODEBASE_VERSION

__all__ = [
    "ValidationIssue",
    "DatasetValidationReport",
    "validate_dataset",
    "video_feature_keys",
]


def video_feature_keys(info: dict[str, Any]) -> list[str]:
    """Return ``dtype == "video"`` feature keys from ``info`` in declaration order.

    Shared predicate for the inline ``features`` filters previously duplicated
    across :mod:`rosbag2lerobot.quality`, :mod:`rosbag2lerobot.preview`, and this module. Lives
    here (the lowest-coupling module: stdlib + pyarrow only) so importing it
    introduces no new heavy dependency chain or import cycle.

    Args:
        info: Parsed ``meta/info.json`` (or any mapping with a ``features``
            sub-mapping of ``{key: feature_spec}``).

    Returns:
        Video feature keys in ``info["features"]`` iteration (declaration)
        order. Non-dict feature specs are skipped.
    """
    return [
        k
        for k, v in info.get("features", {}).items()
        if isinstance(v, dict) and v.get("dtype") == "video"
    ]


# Severity labels for issues. ``ERROR`` always fails; ``WARN`` only fails
# under ``--strict``.
_ERROR = "ERROR"
_WARN = "WARN"


# info.json keys the writer always emits (see DatasetWriter._write_info_json).
_REQUIRED_INFO_KEYS = (
    "codebase_version",
    "robot_type",
    "total_episodes",
    "total_frames",
    "total_tasks",
    "chunks_size",
    "data_files_size_in_mb",
    "video_files_size_in_mb",
    "fps",
    "splits",
    "data_path",
    "video_path",
    "features",
)

# Bookkeeping data-parquet columns and their pyarrow types
# (DatasetWriter._build_data_table). ``subtask_index`` is conditional.
_DATA_BOOKKEEPING_TYPES: dict[str, pa.DataType] = {
    "index": pa.int64(),
    "timestamp": pa.float32(),
    "frame_index": pa.int64(),
    "episode_index": pa.int64(),
    "task_index": pa.int64(),
}

# Required per-video-key columns in episodes parquet
# (DatasetWriter.save_episode -> ep_meta).
_VIDEO_KEY_SUFFIXES = (
    "chunk_index",
    "file_index",
    "from_timestamp",
    "to_timestamp",
)

# Required scalar columns in episodes parquet and their pyarrow types.
_EPISODES_SCALAR_TYPES: dict[str, pa.DataType] = {
    "episode_index": pa.int64(),
    "length": pa.int64(),
    "data/chunk_index": pa.int64(),
    "data/file_index": pa.int64(),
    "dataset_from_index": pa.int64(),
    "dataset_to_index": pa.int64(),
}


@dataclass
class ValidationIssue:
    """One structural discrepancy found in a dataset.

    Attributes:
        severity: ``"ERROR"`` (always fails) or ``"WARN"`` (fails only under
            ``--strict``).
        kind: A short machine-readable category, e.g. ``"missing_file"``,
            ``"missing_info_key"``, ``"codebase_version"``, ``"splits"``,
            ``"column_type"``, ``"missing_column"``, ``"extra_column"``,
            ``"count_mismatch"``.
        location: The file / column / key the issue pertains to (relative to
            the dataset root when it is a path).
        message: Human-readable description of the discrepancy.
    """

    severity: str
    kind: str
    location: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetValidationReport:
    """Aggregate result of :func:`validate_dataset`.

    Mirrors the :class:`rosbag2lerobot.diagnostics.ValidationReport` idiom: callers
    collect issues, then call :meth:`apply_verdict` with the CLI ``--strict``
    flag to populate ``verdict`` / ``exit_code``.
    """

    dataset: str
    issues: list[ValidationIssue] = field(default_factory=list)
    verdict: str = "OK"
    exit_code: int = 0

    def add(self, severity: str, kind: str, location: str, message: str) -> None:
        """Append a :class:`ValidationIssue` to the report."""
        self.issues.append(
            ValidationIssue(
                severity=severity,
                kind=kind,
                location=location,
                message=message,
            )
        )

    def errors(self) -> list[ValidationIssue]:
        """Return all ``ERROR``-severity issues."""
        return [i for i in self.issues if i.severity == _ERROR]

    def warnings(self) -> list[ValidationIssue]:
        """Return all ``WARN``-severity issues."""
        return [i for i in self.issues if i.severity == _WARN]

    def apply_verdict(self, strict: bool) -> None:
        """Populate ``verdict`` / ``exit_code`` from the collected issues.

        Any ``ERROR`` fails. Under ``strict`` a ``WARN`` also fails.
        """
        if self.errors():
            self.verdict = "FAIL"
            self.exit_code = 1
            return
        if strict and self.warnings():
            self.verdict = "FAIL"
            self.exit_code = 1
            return
        self.verdict = "OK"
        self.exit_code = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {
            "dataset": self.dataset,
            "issues": [i.to_dict() for i in self.issues],
            "n_errors": len(self.errors()),
            "n_warnings": len(self.warnings()),
            "verdict": self.verdict,
            "exit_code": int(self.exit_code),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rel(dataset_dir: Path, path: Path) -> str:
    """Return ``path`` relative to ``dataset_dir`` for tidy issue locations."""
    try:
        return str(path.relative_to(dataset_dir))
    except ValueError:
        return str(path)


def _load_info(
    dataset_dir: Path,
    report: DatasetValidationReport,
) -> dict[str, Any] | None:
    """Load ``meta/info.json``; record a missing/unreadable issue and return None."""
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        report.add(_ERROR, "missing_file", "meta/info.json", "info.json is missing")
        return None
    try:
        with open(info_path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        report.add(
            _ERROR,
            "unreadable_file",
            "meta/info.json",
            f"info.json could not be parsed: {exc}",
        )
        return None


def _expected_data_type(feat_spec: dict[str, Any]) -> pa.DataType | None:
    """Return the expected pyarrow type for a feature in the data parquet.

    Mirrors :meth:`DatasetWriter._build_data_table`: ``float32`` features
    become ``fixed_size_list<float32, shape[0]>``, ``int64`` features stay
    ``int64``. ``video`` features must not appear (returns ``None``).
    """
    dtype = feat_spec.get("dtype", "float32")
    if dtype == "video":
        return None
    if dtype == "int64":
        return pa.int64()
    # float32 (default)
    shape = feat_spec.get("shape", [1])
    dim = shape[0] if shape else 1
    return pa.list_(pa.float32(), dim)


def _check_required_files(
    dataset_dir: Path,
    info: dict[str, Any] | None,
    report: DatasetValidationReport,
) -> None:
    """Check presence of the mandatory files / directories."""
    for rel in ("meta/stats.json", "meta/tasks.parquet"):
        if not (dataset_dir / rel).is_file():
            report.add(_ERROR, "missing_file", rel, f"{rel} is missing")

    # episodes parquet (>= 1)
    episodes = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    if not episodes:
        report.add(
            _ERROR,
            "missing_file",
            "meta/episodes/**/*.parquet",
            "no episodes parquet files found under meta/episodes/",
        )

    # data parquet (>= 1)
    data_files = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not data_files:
        report.add(
            _ERROR,
            "missing_file",
            "data/**/*.parquet",
            "no data parquet files found under data/",
        )

    # videos: >= 1 mp4 per dtype=="video" feature key.
    for key in video_feature_keys(info) if isinstance(info, dict) else []:
        vdir = dataset_dir / "videos" / key
        mp4s = sorted(vdir.rglob("*.mp4")) if vdir.is_dir() else []
        if not mp4s:
            report.add(
                _ERROR,
                "missing_file",
                f"videos/{key}/**/*.mp4",
                f"no mp4 files found for video feature {key!r}",
            )


def _check_info_keys(
    info: dict[str, Any],
    report: DatasetValidationReport,
) -> None:
    """Validate required info.json keys, codebase_version, and splits."""
    for key in _REQUIRED_INFO_KEYS:
        if key not in info:
            report.add(
                _ERROR,
                "missing_info_key",
                f"meta/info.json:{key}",
                f"required info.json key {key!r} is missing",
            )

    cbv = info.get("codebase_version")
    if cbv is not None and cbv != _CODEBASE_VERSION:
        report.add(
            _ERROR,
            "codebase_version",
            "meta/info.json:codebase_version",
            f"codebase_version is {cbv!r}, expected {_CODEBASE_VERSION!r}",
        )

    # splits must partition [0, total_episodes): contiguous, gap-free,
    # non-overlapping ranges covering the whole episode space. Accepts both the
    # legacy single split ({"train": "0:N"}) and multi-way train/val/test.
    splits = info.get("splits")
    total_episodes = info.get("total_episodes")
    if isinstance(splits, dict) and total_episodes is not None:
        _check_splits_partition(splits, int(total_episodes), report)


def _check_splits_partition(
    splits: dict[str, Any],
    total_episodes: int,
    report: DatasetValidationReport,
) -> None:
    """Validate that ``splits`` partitions ``[0, total_episodes)`` exactly.

    Parses every ``"a:b"`` value to ``(int, int)``, rejecting malformed
    strings, ``a < 0`` / ``b > total_episodes`` / ``a >= b``. Sorts the
    intervals by start and asserts they are contiguous with no gap or overlap:
    the first start is ``0``, the last end is ``total_episodes``, and each
    interval's start equals the previous interval's end. All discrepancies are
    recorded under the ``"splits"`` issue kind.
    """
    intervals: list[tuple[int, int, str]] = []
    for name, value in splits.items():
        if not isinstance(value, str) or value.count(":") != 1:
            report.add(
                _ERROR,
                "splits",
                f"meta/info.json:splits.{name}",
                f"split {name!r} value {value!r} is not a valid 'a:b' range",
            )
            return
        a_str, b_str = value.split(":", 1)
        try:
            a, b = int(a_str), int(b_str)
        except ValueError:
            report.add(
                _ERROR,
                "splits",
                f"meta/info.json:splits.{name}",
                f"split {name!r} value {value!r} has non-integer bounds",
            )
            return
        if a < 0 or b > total_episodes or a >= b:
            report.add(
                _ERROR,
                "splits",
                f"meta/info.json:splits.{name}",
                f"split {name!r} range {value!r} is out of bounds for "
                f"total_episodes={total_episodes}",
            )
            return
        intervals.append((a, b, name))

    if not intervals:
        report.add(
            _ERROR,
            "splits",
            "meta/info.json:splits",
            f"splits is empty; expected a partition of [0, {total_episodes})",
        )
        return

    intervals.sort(key=lambda iv: iv[0])
    prev_end = 0
    for a, b, name in intervals:
        if a != prev_end:
            report.add(
                _ERROR,
                "splits",
                f"meta/info.json:splits.{name}",
                f"split {name!r} starts at {a}, expected {prev_end} "
                "(gap or overlap in splits partition)",
            )
            return
        prev_end = b
    if prev_end != total_episodes:
        report.add(
            _ERROR,
            "splits",
            "meta/info.json:splits",
            f"splits cover [0, {prev_end}) but total_episodes is {total_episodes}",
        )


def _check_data_parquet(
    dataset_dir: Path,
    info: dict[str, Any],
    report: DatasetValidationReport,
) -> None:
    """Validate the data parquet schema against info.json features."""
    data_files = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not data_files:
        return  # already reported as a missing file.

    features = info.get("features", {})
    has_subtask = "subtask_index" in features

    # Build the expected column -> type map from info features.
    expected: dict[str, pa.DataType] = dict(_DATA_BOOKKEEPING_TYPES)
    if has_subtask:
        expected["subtask_index"] = pa.int64()

    video_keys: set[str] = set()
    for key, feat_spec in features.items():
        if not isinstance(feat_spec, dict):
            continue
        if key in expected:
            continue  # bookkeeping handled above.
        if feat_spec.get("dtype") == "video":
            video_keys.add(key)
            continue
        etype = _expected_data_type(feat_spec)
        if etype is not None:
            expected[key] = etype

    for path in data_files:
        loc = _rel(dataset_dir, path)
        schema = pq.read_schema(path)
        present = set(schema.names)

        # Missing required columns -> ERROR.
        for col, etype in expected.items():
            if col not in present:
                report.add(
                    _ERROR,
                    "missing_column",
                    f"{loc}:{col}",
                    f"data parquet is missing required column {col!r}",
                )
                continue
            actual = schema.field(col).type
            if not actual.equals(etype):
                report.add(
                    _ERROR,
                    "column_type",
                    f"{loc}:{col}",
                    f"column {col!r} has type {actual}, expected {etype}",
                )

        # Video features must not appear in data parquet -> ERROR.
        for vk in video_keys:
            if vk in present:
                report.add(
                    _ERROR,
                    "column_type",
                    f"{loc}:{vk}",
                    f"video feature {vk!r} must not be a data parquet column",
                )

        # Unexpected extra columns -> WARN.
        allowed = set(expected) | video_keys
        for col in present:
            if col not in allowed:
                report.add(
                    _WARN,
                    "extra_column",
                    f"{loc}:{col}",
                    f"unexpected extra column {col!r} in data parquet",
                )


def _check_tasks_parquet(
    dataset_dir: Path,
    report: DatasetValidationReport,
) -> None:
    """Validate tasks.parquet has task_index:int64 + task:string."""
    path = dataset_dir / "meta" / "tasks.parquet"
    if not path.is_file():
        return  # already reported.
    schema = pq.read_schema(path)
    names = set(schema.names)
    for col, etype in (("task_index", pa.int64()), ("task", pa.string())):
        if col not in names:
            report.add(
                _ERROR,
                "missing_column",
                f"meta/tasks.parquet:{col}",
                f"tasks.parquet is missing required column {col!r}",
            )
            continue
        actual = schema.field(col).type
        if not actual.equals(etype):
            report.add(
                _ERROR,
                "column_type",
                f"meta/tasks.parquet:{col}",
                f"tasks.parquet column {col!r} has type {actual}, expected {etype}",
            )


def _check_episodes_parquet(
    dataset_dir: Path,
    info: dict[str, Any],
    report: DatasetValidationReport,
) -> None:
    """Validate episodes parquet columns + cross-check counts against info."""
    try:
        parquet_paths = _collect_episodes_parquet(dataset_dir)
    except FileNotFoundError:
        return  # already reported as a missing file.

    tables = [pq.read_table(p) for p in parquet_paths]
    merged = pa.concat_tables(tables, promote_options="default")
    columns = merged.column_names
    column_set = set(columns)

    # Scalar columns + their types.
    for col, etype in _EPISODES_SCALAR_TYPES.items():
        if col not in column_set:
            report.add(
                _ERROR,
                "missing_column",
                f"meta/episodes:{col}",
                f"episodes parquet is missing required column {col!r}",
            )
            continue
        actual = merged.schema.field(col).type
        if not actual.equals(etype):
            report.add(
                _ERROR,
                "column_type",
                f"meta/episodes:{col}",
                f"episodes column {col!r} has type {actual}, expected {etype}",
            )

    # Per video-key columns. Cross-check against info.json video features so a
    # declared video feature missing its episodes columns is an ERROR.
    info_video_keys = set(video_feature_keys(info))
    present_vkeys = set(_discover_video_keys(columns))
    for vk in info_video_keys:
        try:
            _ensure_columns_present(column_set, vk)
        except ValueError as exc:
            report.add(
                _ERROR,
                "missing_column",
                f"meta/episodes:videos/{vk}",
                str(exc),
            )
    # Stray video keys present in episodes but not declared in info -> WARN.
    for vk in present_vkeys - info_video_keys:
        report.add(
            _WARN,
            "extra_column",
            f"meta/episodes:videos/{vk}",
            f"episodes parquet has video key {vk!r} not declared in info.json",
        )

    # Cross-check row count and length sum against info.json totals.
    if "episode_index" in column_set:
        n_rows = merged.num_rows
        total_episodes = info.get("total_episodes")
        if total_episodes is not None and n_rows != total_episodes:
            report.add(
                _ERROR,
                "count_mismatch",
                "meta/episodes",
                f"episodes parquet has {n_rows} rows but "
                f"info.total_episodes is {total_episodes}",
            )
    if "length" in column_set:
        length_sum = int(sum(int(x) for x in merged.column("length").to_pylist()))
        total_frames = info.get("total_frames")
        if total_frames is not None and length_sum != total_frames:
            report.add(
                _ERROR,
                "count_mismatch",
                "meta/episodes:length",
                f"sum(length)={length_sum} but info.total_frames is {total_frames}",
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_dataset(dataset_dir: Path) -> DatasetValidationReport:
    """Validate the structure of a generated LeRobot v3.0 dataset.

    Performs read-only structural checks (required files, ``info.json`` keys
    and values, parquet schemas, and episode count cross-checks) and returns
    a populated :class:`DatasetValidationReport`. Validation failures are
    collected as issues rather than raised; only genuinely unreadable inputs
    surface as exceptions from the underlying parquet reader.

    Callers should invoke :meth:`DatasetValidationReport.apply_verdict` with
    the CLI ``--strict`` flag to populate ``verdict`` / ``exit_code``.

    Args:
        dataset_dir: Root of a LeRobot v3.0 dataset (the directory that
            contains ``meta/``, ``data/``, and ``videos/``).

    Returns:
        A :class:`DatasetValidationReport` with all detected issues. The
        verdict is left at its default ``"OK"`` until ``apply_verdict`` is
        called.
    """
    dataset_dir = Path(dataset_dir)
    report = DatasetValidationReport(dataset=str(dataset_dir))

    info = _load_info(dataset_dir, report)

    _check_required_files(dataset_dir, info, report)
    if info is not None:
        _check_info_keys(info, report)
        _check_data_parquet(dataset_dir, info, report)
        _check_episodes_parquet(dataset_dir, info, report)
    _check_tasks_parquet(dataset_dir, report)

    return report
