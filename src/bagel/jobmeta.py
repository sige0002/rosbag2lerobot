"""Job summary helpers (plan.md D-3).

Pure aggregation for the ``meta/job_summary.json`` file written after a
conversion run, plus the per-episode result record the CLI accumulates while
iterating bags. The summary feeds both the human progress/summary output and
the conversion manifest (:mod:`bagel.manifest`), which reads per-bag
``frame_count`` / ``processing_time_s`` from the same :class:`EpisodeResult`
list.

Design rules (CLAUDE.md):

- :meth:`JobSummary.to_dict` is **pure**: the run wall time is injected, so
  tests pin it and assert deterministic output.
- I/O is isolated to :func:`dir_bytes` (recursive size on disk).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpisodeResult:
    """Outcome of processing one bag (= one episode).

    Attributes:
        index: Original bag index (deterministic output order).
        bag_path: Source bag path as a string.
        worker: Stable worker ordinal (0 for the serial path).
        success: ``True`` if the episode decoded/resampled without error.
        n_frames: Number of frames produced (0 on failure / empty episode).
        processing_time_s: Wall time spent on this bag.
        error: Error string when ``success`` is ``False``, else ``None``.
    """

    index: int
    bag_path: str
    worker: int
    success: bool
    n_frames: int
    processing_time_s: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of this result."""
        return {
            "index": self.index,
            "bag_path": self.bag_path,
            "worker": self.worker,
            "success": self.success,
            "n_frames": self.n_frames,
            "processing_time_s": self.processing_time_s,
            "error": self.error,
        }


@dataclass
class JobSummary:
    """Accumulates :class:`EpisodeResult` records into a run summary.

    Attributes:
        input_bytes: Total size on disk of the input bags (filled by the CLI
            via :func:`dir_bytes`).
        output_bytes: Total size on disk of the output dataset.
        episodes: All recorded episode results, in arrival order.
    """

    input_bytes: int = 0
    output_bytes: int = 0
    episodes: list[EpisodeResult] = field(default_factory=list)

    def add(self, result: EpisodeResult) -> None:
        """Append one episode result to the summary."""
        self.episodes.append(result)

    def to_dict(self, *, wall_time_s: float) -> dict[str, Any]:
        """Compute the summary dict (pure; wall time injected).

        Args:
            wall_time_s: Total wall-clock time of the run, in seconds.

        Returns:
            A JSON-serializable summary with success/failure counts, frame
            totals, throughput, byte sizes, a per-worker breakdown, and the
            full episode list.
        """
        n_success = sum(1 for e in self.episodes if e.success)
        n_failed = sum(1 for e in self.episodes if not e.success)
        total_frames = sum(e.n_frames for e in self.episodes if e.success)
        frames_per_min = (total_frames / wall_time_s * 60.0) if wall_time_s > 0 else 0.0

        per_worker: dict[int, dict[str, Any]] = {}
        for e in self.episodes:
            w = per_worker.setdefault(
                e.worker,
                {"worker": e.worker, "n_episodes": 0, "n_frames": 0, "time_s": 0.0},
            )
            w["n_episodes"] += 1
            if e.success:
                w["n_frames"] += e.n_frames
            w["time_s"] += e.processing_time_s

        return {
            "n_episodes": len(self.episodes),
            "n_success": n_success,
            "n_failed": n_failed,
            "total_frames": total_frames,
            "wall_time_s": wall_time_s,
            "frames_per_min": frames_per_min,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "workers": [per_worker[k] for k in sorted(per_worker)],
            "episodes": [e.to_dict() for e in self.episodes],
        }


def dir_bytes(path: str | Path) -> int:
    """Return the total size in bytes of all files under *path*.

    Recurses into subdirectories. A non-existent path yields 0; a single file
    yields its own size.

    Args:
        path: Directory or file to measure.

    Returns:
        Total file size in bytes.
    """
    p = Path(path)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
