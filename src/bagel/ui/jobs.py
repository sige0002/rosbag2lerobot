"""Async ``bagel convert`` subprocess tracking for the ``bagel ui`` backend.

``convert`` is the one long-running verb, so the API runs it as a detached
``bagel convert ... --json --quiet`` subprocess (never in-process) and returns a
job id immediately. The FE then polls ``GET /api/convert/{job_id}``; progress is
read from the partial ``<output>/meta/job_summary.json`` the converter
checkpoints after each episode (see ``cli.convert``), so ``done/total`` advances
live.

Concurrency: a :class:`JobRegistry` guards its dict with a lock. Each
:class:`ConvertJob` owns one :class:`subprocess.Popen`; its terminal state
(``done`` / ``failed``) is derived lazily on poll from the process return code,
so no background reaper thread is needed.

The child's stderr is redirected to a temp file (not an undrained ``PIPE``)
so the converter never blocks once its stderr exceeds the OS pipe buffer; the
file is read back when reporting errors and unlinked on completion/shutdown.
The per-convert config tempfile (written by the API under the output root) is
also tracked on the job and unlinked in the same path so it does not leak.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


def read_progress(output_dir: Path, total: int) -> dict[str, int]:
    """Derive ``{done, total, n_failed}`` from a partial job_summary.json.

    Pure read of the checkpointed summary the converter writes incrementally.
    ``done`` counts both successes and recorded failures (``n_success +
    n_failed``) against the up-front ``total`` bag count.

    Args:
        output_dir: The conversion ``--output`` directory.
        total: Total number of bags known up front (``len(bag_paths)``).

    Returns:
        ``{"done", "total", "n_failed"}``. ``done`` is 0 until the first
        checkpoint exists or if the file is mid-write / malformed.
    """
    summary_path = output_dir / "meta" / "job_summary.json"
    done = 0
    n_failed = 0
    try:
        data = json.loads(summary_path.read_text())
        n_success = int(data.get("n_success", 0))
        n_failed = int(data.get("n_failed", 0))
        done = n_success + n_failed
    except (OSError, ValueError, TypeError):
        # No checkpoint yet, or a partial write — report no progress.
        done = 0
        n_failed = 0
    return {"done": done, "total": total, "n_failed": n_failed}


def read_summary(output_dir: Path) -> Optional[dict[str, Any]]:
    """Return the parsed job_summary.json, or ``None`` if unavailable."""
    summary_path = output_dir / "meta" / "job_summary.json"
    try:
        return json.loads(summary_path.read_text())
    except (OSError, ValueError):
        return None


@dataclass
class ConvertJob:
    """One tracked ``bagel convert`` subprocess.

    Attributes:
        job_id: Opaque uuid handed to the FE.
        output_dir: The conversion output directory (polled for progress).
        total: Total bag count, for the ``done/total`` ratio.
        command: The equivalent ``bagel ...`` invocation (for display).
        proc: The running subprocess.
        stderr_path: Temp file the child's stderr is redirected to (drained on
            terminal status, then unlinked).
        config_path: The per-convert config tempfile to unlink on completion
            (or ``None`` when the caller manages it).
    """

    job_id: str
    output_dir: Path
    total: int
    command: str
    proc: subprocess.Popen[bytes]
    stderr_path: Path
    config_path: Optional[Path] = None
    _stderr: str = field(default="", init=False)
    _cleaned: bool = field(default=False, init=False)

    def status(self) -> dict[str, Any]:
        """Return the current job status dict for the API.

        Lazily resolves the terminal state from the process return code:
        ``running`` while alive, ``done`` on exit 0, ``failed`` otherwise. On
        terminal states the stderr temp file is read once for diagnostics and
        the per-job temp artifacts (stderr file + config tempfile) are cleaned
        up; ``error`` carries the stderr text on failure.

        Returns:
            ``{"state", "progress", "summary", "error"}`` per the API contract.
        """
        progress = read_progress(self.output_dir, self.total)
        returncode = self.proc.poll()

        if returncode is None:
            return {
                "state": "running",
                "progress": progress,
                "summary": None,
                "error": None,
            }

        # Terminal: read the stderr file once, then clean up temp artifacts.
        if not self._stderr:
            try:
                self._stderr = self.stderr_path.read_text(errors="replace")
            except OSError:
                self._stderr = ""
        self.cleanup()

        summary = read_summary(self.output_dir)
        if returncode == 0:
            return {
                "state": "done",
                "progress": progress,
                "summary": summary,
                "error": None,
            }
        return {
            "state": "failed",
            "progress": progress,
            "summary": summary,
            "error": self._stderr.strip() or f"convert exited with code {returncode}",
        }

    def cleanup(self) -> None:
        """Unlink the stderr temp file and the config tempfile (idempotent)."""
        if self._cleaned:
            return
        self._cleaned = True
        self.stderr_path.unlink(missing_ok=True)
        if self.config_path is not None:
            self.config_path.unlink(missing_ok=True)


class JobRegistry:
    """Thread-safe registry of in-flight / completed convert jobs."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._jobs: dict[str, ConvertJob] = {}
        self._lock = threading.Lock()

    def launch(
        self,
        argv: list[str],
        output_dir: Path,
        total: int,
        command: str,
        config_path: Optional[Path] = None,
    ) -> str:
        """Start a convert subprocess and register it under a new job id.

        Args:
            argv: The full subprocess argument vector (``bagel`` first).
            output_dir: Conversion output directory (polled for progress).
            total: Up-front bag count for the progress ratio.
            command: Human-readable equivalent CLI string for display.
            config_path: The per-convert config tempfile to unlink when the
                job completes (``None`` if the caller manages it).

        Returns:
            The new job id (uuid4 hex-with-dashes string).
        """
        # Redirect stderr to a temp file rather than an undrained PIPE: a PIPE
        # that nobody reads until exit deadlocks the child once its stderr
        # exceeds the OS pipe buffer. The file is read back on terminal status.
        fd, stderr_name = tempfile.mkstemp(prefix="bagel_ui_stderr_", suffix=".log")
        stderr_path = Path(stderr_name)
        stderr_file = os.fdopen(fd, "wb")
        try:
            # stdin is closed (never grabs the TTY); stderr goes to the file.
            proc = subprocess.Popen(  # noqa: S603 - argv is fully constructed by the API layer
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
            )
        except BaseException:
            stderr_file.close()
            stderr_path.unlink(missing_ok=True)
            if config_path is not None:
                config_path.unlink(missing_ok=True)
            raise
        finally:
            # The child inherits its own dup of the fd; close our handle so the
            # file is fully released once the child exits.
            stderr_file.close()

        job_id = str(uuid.uuid4())
        job = ConvertJob(
            job_id=job_id,
            output_dir=output_dir,
            total=total,
            command=command,
            proc=proc,
            stderr_path=stderr_path,
            config_path=config_path,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job_id

    def get(self, job_id: str) -> Optional[ConvertJob]:
        """Return the job for ``job_id``, or ``None`` if unknown."""
        with self._lock:
            return self._jobs.get(job_id)

    def shutdown(self) -> None:
        """Terminate and reap any still-running jobs (best-effort, on stop).

        Each job is terminated if alive, then waited on so no zombie remains,
        and its temp artifacts (stderr file + config tempfile) are unlinked.
        """
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.proc.poll() is None:
                job.proc.terminate()
                try:
                    job.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    job.proc.kill()
                    job.proc.wait()
            job.cleanup()
