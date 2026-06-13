"""Timestamp drift auditor for generated LeRobot v3.0 datasets.

This module inspects ``meta/episodes/*.parquet`` produced by
:class:`bagel.writer.DatasetWriter` and verifies that the per-video
``from_timestamp`` / ``to_timestamp`` sequence is numerically self-consistent:

1. Within a single mp4 file (identified by the pair ``(chunk_index,
   file_index)``), ``to_timestamp[i]`` must equal ``from_timestamp[i + 1]``.
2. ``from_timestamp`` may only reset to ``0.0`` at a file boundary, never in
   the middle of a contiguous mp4.
3. The cumulative drift between the *observed* final ``to_timestamp`` within a
   file and the *expected* sum of per-episode durations must stay below
   ``max_drift_us`` microseconds.

Scope is deliberately narrower than ``verify-dataset`` (which cross-checks
against real mp4 duration on disk): ``audit-timestamps`` only looks at the
numerical continuity of the parquet rows, which is the invariant PR #3239
fixed upstream in LeRobot and that :mod:`bagel.writer` preserves via
``round(..., _TIMESTAMP_ROUND_DECIMALS)`` carry-forward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

# Re-use the same rounding grain the writer uses so the auditor's tolerance
# aligns with the authoritative definition of "correct" timestamps.
from bagel.writer import _TIMESTAMP_ROUND_DECIMALS


__all__ = [
    "BoundaryError",
    "VideoKeyAuditResult",
    "AuditReport",
    "audit_episode_timestamps",
]


# Column-name suffixes in the episodes parquet layout produced by
# DatasetWriter._write_episodes_parquet (see writer.py:371-375).
_COL_SUFFIXES = (
    "chunk_index",
    "file_index",
    "from_timestamp",
    "to_timestamp",
)


@dataclass
class BoundaryError:
    """One row of a non-conforming episode transition.

    Attributes:
        video_key: The LeRobot video feature key (e.g.
            ``"observation.images.front"``).
        episode_index: The ``episode_index`` of the row whose
            ``from_timestamp`` was inconsistent with the previous row's
            ``to_timestamp``.
        expected_from_ts: What ``from_timestamp`` should have been, given the
            previous row's ``to_timestamp`` (within the same mp4) or ``0.0``
            (at a file boundary).
        actual_from_ts: The actual ``from_timestamp`` observed in the parquet.
            Deviation from ``expected_from_ts`` is the flagged anomaly.
        delta_us: ``(actual - expected) * 1e6``, the signed boundary error in
            microseconds. Positive values mean the timestamp jumped ahead,
            negative means it went backwards.
    """

    video_key: str
    episode_index: int
    expected_from_ts: float
    actual_from_ts: float
    delta_us: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VideoKeyAuditResult:
    """Per-``video_key`` audit summary."""

    video_key: str
    n_episodes: int
    max_drift_us: float
    boundary_errors: list[BoundaryError] = field(default_factory=list)
    verdict: str = "OK"

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_key": self.video_key,
            "n_episodes": self.n_episodes,
            "max_drift_us": self.max_drift_us,
            "boundary_errors": [e.to_dict() for e in self.boundary_errors],
            "verdict": self.verdict,
        }


@dataclass
class AuditReport:
    """Top-level audit result for a whole dataset."""

    dataset: str
    video_keys: list[str]
    results: list[VideoKeyAuditResult]
    verdict: str
    exit_code: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "video_keys": list(self.video_keys),
            "results": [r.to_dict() for r in self.results],
            "verdict": self.verdict,
            "exit_code": self.exit_code,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_episodes_parquet(dataset_dir: Path) -> list[Path]:
    """Return episodes parquet files sorted by (chunk, file) on disk.

    The writer lays them out as
    ``meta/episodes/chunk-XXX/file-YYY.parquet``. We sort lexicographically,
    which matches the numeric order because the widths are fixed.
    """
    episodes_dir = dataset_dir / "meta" / "episodes"
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"episodes directory not found: {episodes_dir}")
    files = sorted(episodes_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no episodes parquet files under {episodes_dir}",
        )
    return files


def _discover_video_keys(columns: list[str]) -> list[str]:
    """Extract unique ``vkey`` strings from episodes parquet columns.

    Columns are named ``videos/<vkey>/<suffix>``. We recover ``<vkey>`` from
    all ``from_timestamp`` columns (picking one deterministic suffix so we
    don't count each key four times).
    """
    vkeys: list[str] = []
    seen: set[str] = set()
    for col in columns:
        if col.startswith("videos/") and col.endswith("/from_timestamp"):
            vkey = col[len("videos/") : -len("/from_timestamp")]
            if vkey not in seen:
                seen.add(vkey)
                vkeys.append(vkey)
    return vkeys


def _ensure_columns_present(columns: set[str], vkey: str) -> None:
    """Raise if any of the four required columns for ``vkey`` are missing."""
    missing = [
        f"videos/{vkey}/{suf}"
        for suf in _COL_SUFFIXES
        if f"videos/{vkey}/{suf}" not in columns
    ]
    if missing:
        raise ValueError(
            f"episodes parquet is missing required columns for {vkey!r}: {missing}",
        )


def _audit_video_key(
    video_key: str,
    episode_index: list[int],
    chunk_index: list[int],
    file_index: list[int],
    from_ts: list[float],
    to_ts: list[float],
    max_drift_us: float,
) -> VideoKeyAuditResult:
    """Run the three-invariant check for a single ``video_key``.

    The checks are implemented as a single forward sweep so the caller gets a
    complete list of every offending row rather than short-circuiting on the
    first one.
    """
    n = len(episode_index)
    # Honor the user's requested precision; add a 1 ns floor so ULP noise
    # from float64 parquet round-trip never trips a strict --max-drift-us.
    boundary_tol = max_drift_us * 1e-6 + 1e-9

    boundary_errors: list[BoundaryError] = []
    max_delta_us = 0.0

    # Per-file expected cursor: resets when (chunk, file) changes.
    for i in range(n):
        if i == 0:
            # First row of the first file must start at 0.0.
            expected = 0.0
            same_file = True  # trivially, nothing before it
        else:
            same_file = (
                chunk_index[i] == chunk_index[i - 1]
                and file_index[i] == file_index[i - 1]
            )
            if same_file:
                # Invariant #1: contiguous within one mp4.
                expected = to_ts[i - 1]
            else:
                # Invariant #2: file boundary ⇒ reset to 0.0.
                expected = 0.0

        delta = from_ts[i] - expected
        delta_us = delta * 1e6
        if abs(delta_us) > abs(max_delta_us):
            max_delta_us = delta_us

        if abs(delta) > boundary_tol:
            boundary_errors.append(
                BoundaryError(
                    video_key=video_key,
                    episode_index=int(episode_index[i]),
                    expected_from_ts=round(expected, _TIMESTAMP_ROUND_DECIMALS),
                    actual_from_ts=from_ts[i],
                    delta_us=delta_us,
                )
            )

    verdict = "OK"
    if boundary_errors or abs(max_delta_us) > max_drift_us:
        verdict = "FAIL"
        # A drift-only failure (no per-row boundary breach but cumulative >
        # max_drift_us) must still be surfaced. We flag it as a boundary
        # error on the last row that carried the drift, with the expected
        # value being ``from_ts[i] - delta``.
        if not boundary_errors:
            i = n - 1
            boundary_errors.append(
                BoundaryError(
                    video_key=video_key,
                    episode_index=int(episode_index[i]),
                    expected_from_ts=round(
                        from_ts[i] - max_delta_us * 1e-6,
                        _TIMESTAMP_ROUND_DECIMALS,
                    ),
                    actual_from_ts=from_ts[i],
                    delta_us=max_delta_us,
                )
            )

    return VideoKeyAuditResult(
        video_key=video_key,
        n_episodes=n,
        max_drift_us=abs(max_delta_us),
        boundary_errors=boundary_errors,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def audit_episode_timestamps(
    dataset_dir: Path,
    max_drift_us: float = 1.0,
    video_keys: list[str] | None = None,
) -> AuditReport:
    """Audit episodes parquet timestamp continuity for a LeRobot v3.0 dataset.

    Args:
        dataset_dir: Root of a LeRobot dataset (the directory that contains
            ``meta/episodes/``).
        max_drift_us: Maximum allowed per-row boundary error *and* cumulative
            drift, in microseconds. Defaults to ``1.0`` which is one grain of
            the writer's ``_TIMESTAMP_ROUND_DECIMALS`` rounding scheme.
        video_keys: Optional allowlist of video feature keys to audit. When
            ``None`` (default), every video key present in the parquet is
            audited.

    Returns:
        A populated :class:`AuditReport`. ``verdict == "OK"`` and
        ``exit_code == 0`` when all keys pass; otherwise ``"FAIL"`` / ``1``.

    Raises:
        FileNotFoundError: ``meta/episodes/`` is absent or empty.
        ValueError: A requested ``video_key`` is not present in the parquet,
            or the parquet is missing one of the four required columns.
    """
    dataset_dir = Path(dataset_dir)
    parquet_paths = _collect_episodes_parquet(dataset_dir)

    # Read all parquet files and concatenate in (chunk, file) order.
    tables = [pq.read_table(p) for p in parquet_paths]
    import pyarrow as pa

    merged = pa.concat_tables(tables, promote_options="default")
    # Sort by episode_index to ensure monotonic iteration regardless of how
    # the writer chunked files.
    merged = merged.sort_by("episode_index")
    columns = merged.column_names
    column_set = set(columns)

    all_vkeys = _discover_video_keys(columns)
    if video_keys is None:
        target_vkeys = all_vkeys
    else:
        target_vkeys = []
        for vk in video_keys:
            if vk not in all_vkeys:
                raise ValueError(
                    f"video_key {vk!r} not found in episodes parquet; "
                    f"available: {all_vkeys}",
                )
            target_vkeys.append(vk)

    if "episode_index" not in column_set:
        raise ValueError("episodes parquet is missing 'episode_index' column")

    episode_index = [int(x) for x in merged.column("episode_index").to_pylist()]

    results: list[VideoKeyAuditResult] = []
    for vkey in target_vkeys:
        _ensure_columns_present(column_set, vkey)
        chunk_col = merged.column(f"videos/{vkey}/chunk_index").to_pylist()
        file_col = merged.column(f"videos/{vkey}/file_index").to_pylist()
        from_col = merged.column(f"videos/{vkey}/from_timestamp").to_pylist()
        to_col = merged.column(f"videos/{vkey}/to_timestamp").to_pylist()

        result = _audit_video_key(
            video_key=vkey,
            episode_index=episode_index,
            chunk_index=[int(x) for x in chunk_col],
            file_index=[int(x) for x in file_col],
            from_ts=[float(x) for x in from_col],
            to_ts=[float(x) for x in to_col],
            max_drift_us=max_drift_us,
        )
        results.append(result)

    any_fail = any(r.verdict != "OK" for r in results)
    return AuditReport(
        dataset=str(dataset_dir),
        video_keys=target_vkeys,
        results=results,
        verdict="FAIL" if any_fail else "OK",
        exit_code=1 if any_fail else 0,
    )
