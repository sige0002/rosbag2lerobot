"""Machine-readable conversion progress (``meta/progress.json``).

A tqdm bar is the right answer at a terminal and the wrong one everywhere
else: in ``docker logs`` it is carriage-return soup, and a supervisor process
cannot read it at all. This module provides the two non-TTY answers instead —
a small heartbeat file that is rewritten atomically while a conversion runs,
and plain progress log lines at a coarse cadence.

The heartbeat file is transient run state, not part of the dataset: nothing
in the LeRobot v3.0 layout references it, and ``convert`` deletes it once the
run finishes successfully (a leftover file therefore means the run died, and
its contents say where).

Schema (``<output>/meta/progress.json``)::

    {
      "episode_index": 3,          # 0-based, the episode this update is about
      "episode_total": 40,         # episodes this run will convert
      "messages_done": 12000,      # bag messages read for that episode so far
      "messages_total": 19500,     # from the bag's metadata.yaml, or null
      "updated_at": "2026-08-13T04:05:06.123456+00:00"
    }
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

PROGRESS_FILENAME = "progress.json"

# Heartbeat cadence. The clock is only consulted every ``MESSAGE_STRIDE``
# messages so the per-message cost stays one comparison; a file write then
# needs ``WRITE_INTERVAL_S`` to have elapsed as well. The effective cadence is
# therefore whichever of the two is coarser.
WRITE_INTERVAL_S = 2.0
MESSAGE_STRIDE = 500

# Log cadence for the non-TTY progress lines: whichever comes first, so slow
# episodes still report and fast ones do not spam.
LOG_INTERVAL_S = 30.0
LOG_FRACTION = 0.10


@dataclass
class ProgressState:
    """One heartbeat sample. See the module docstring for the JSON schema."""

    episode_index: int
    episode_total: int
    messages_done: int
    messages_total: Optional[int]
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable form written to ``progress.json``."""
        return asdict(self)


def bag_message_count(
    bag_path: Path | str,
    topics: Optional[list[str]] = None,
) -> Optional[int]:
    """Return the recorded message count for *bag_path* from its metadata.

    Reads ``metadata.yaml`` only — no storage file is opened, so this stays
    cheap enough to call before every episode. When *topics* is given, only
    those topics' counts are summed, which is what the converter actually
    reads (it subscribes to the configured topics, not the whole bag).

    Args:
        bag_path: Bag directory, or the ``metadata.yaml`` file itself.
        topics: Restrict the sum to these topic names. ``None`` sums the
            whole bag.

    Returns:
        The message count, or ``None`` when it cannot be determined (no or
        unreadable ``metadata.yaml``, e.g. a bare directory of ``.mcap``
        files) or when it is zero (nothing to measure progress against).
    """
    meta = Path(bag_path)
    if meta.is_dir():
        meta = meta / "metadata.yaml"
    if not meta.is_file():
        return None
    try:
        raw = yaml.safe_load(meta.read_text())
    except (OSError, yaml.YAMLError):
        return None
    info = (raw or {}).get("rosbag2_bagfile_information")
    if not isinstance(info, dict):
        return None

    per_topic = info.get("topics_with_message_count")
    if isinstance(per_topic, list) and per_topic:
        wanted = None if topics is None else set(topics)
        total = 0
        for entry in per_topic:
            if not isinstance(entry, dict):
                continue
            name = (entry.get("topic_metadata") or {}).get("name")
            if wanted is not None and name not in wanted:
                continue
            try:
                total += int(entry.get("message_count") or 0)
            except (TypeError, ValueError):
                continue
        return total or None

    if topics is not None:
        # Only a whole-bag total is available; it would overstate the subset
        # the converter actually reads, so report "unknown" rather than lie.
        return None
    try:
        return int(info.get("message_count") or 0) or None
    except (TypeError, ValueError):
        return None


class ProgressReporter:
    """Maintains ``meta/progress.json`` and optional plain progress logging.

    Lifecycle per episode, from the serial conversion path::

        reporter.start_episode(idx, messages_total)
        ...  reporter.advance()  # once per message read
        reporter.finish_episode()

    The parallel path decodes episodes in worker processes that share no
    state with this object, so it reports whole episodes instead via
    :meth:`episode_completed` — the file then advances once per episode
    rather than continuously.

    All writes are atomic (temp file + ``os.replace``), so a poller never
    observes a half-written file.
    """

    def __init__(
        self,
        path: Path | str,
        episode_total: int,
        *,
        log_fn: Optional[Callable[[str], None]] = None,
        write_interval_s: float = WRITE_INTERVAL_S,
        message_stride: int = MESSAGE_STRIDE,
        log_interval_s: float = LOG_INTERVAL_S,
        log_fraction: float = LOG_FRACTION,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a reporter writing to *path*.

        Args:
            path: Destination file (``<output>/meta/progress.json``).
            episode_total: Number of episodes this run will convert.
            log_fn: Called with a one-line progress string when a log is due;
                ``None`` disables logging (e.g. when a tqdm bar is showing).
            write_interval_s: Minimum wall time between file writes.
            message_stride: Messages between clock checks.
            log_interval_s: Maximum wall time between log lines.
            log_fraction: Fraction of an episode's messages between log lines.
            clock: Monotonic clock, injectable for tests.
        """
        self.path = Path(path)
        self.episode_total = episode_total
        self._log_fn = log_fn
        self._write_interval_s = write_interval_s
        self._message_stride = max(1, message_stride)
        self._log_interval_s = log_interval_s
        self._log_fraction = log_fraction
        self._clock = clock

        self._episode_index = 0
        self._messages_total: Optional[int] = None
        self._messages_done = 0
        self._stride_mark = 0
        self._last_write = -float("inf")
        self._last_log = -float("inf")
        self._last_log_messages = 0

    # ----- episode lifecycle -----

    def start_episode(self, episode_index: int, messages_total: Optional[int]) -> None:
        """Begin tracking *episode_index* and write an immediate update."""
        self._episode_index = episode_index
        self._messages_total = messages_total
        self._messages_done = 0
        self._stride_mark = 0
        self._last_log_messages = 0
        now = self._clock()
        # The log cadence restarts with the episode: an episode that has just
        # begun has nothing to report beyond the line the caller already logged.
        self._last_log = now
        self._write(now)

    def advance(self, n: int = 1) -> None:
        """Count *n* more messages read, writing/logging when due.

        Called once per message, so the fast path is a single comparison; the
        clock is only read every ``message_stride`` messages.
        """
        self._messages_done += n
        if self._messages_done - self._stride_mark < self._message_stride:
            return
        self._stride_mark = self._messages_done
        now = self._clock()
        if now - self._last_write >= self._write_interval_s:
            self._write(now)
        self._maybe_log(now)

    def finish_episode(self) -> None:
        """Write the final update for the current episode."""
        self._write(self._clock())

    def episode_completed(
        self,
        episode_index: int,
        messages: Optional[int],
    ) -> None:
        """Record a whole episode as finished (used by the parallel path).

        Args:
            episode_index: The completed episode's 0-based index.
            messages: Messages the episode contained, or ``None`` if unknown;
                a finished episode reports ``messages_done == messages_total``.
        """
        self._episode_index = episode_index
        self._messages_total = messages
        self._messages_done = messages or 0
        self._stride_mark = self._messages_done
        self._write(self._clock())

    def remove(self) -> None:
        """Delete the heartbeat file, ignoring an already-absent one."""
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    # ----- internals -----

    def state(self) -> ProgressState:
        """Return the current state (also what :meth:`_write` serializes)."""
        return ProgressState(
            episode_index=self._episode_index,
            episode_total=self.episode_total,
            messages_done=self._messages_done,
            messages_total=self._messages_total,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _write(self, now: float) -> None:
        """Atomically replace the heartbeat file with the current state."""
        self._last_write = now
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(self.state().to_dict(), fh, indent=2)
        os.replace(tmp, self.path)

    def _maybe_log(self, now: float) -> None:
        """Emit a progress line when either log cadence has been reached."""
        if self._log_fn is None:
            return
        by_time = now - self._last_log >= self._log_interval_s
        by_count = (
            self._messages_total is not None
            and self._messages_done - self._last_log_messages
            >= self._log_fraction * self._messages_total
        )
        if not (by_time or by_count):
            return
        self._last_log = now
        self._last_log_messages = self._messages_done
        self._log_fn(
            format_progress_line(
                episode_index=self._episode_index,
                episode_total=self.episode_total,
                messages_done=self._messages_done,
                messages_total=self._messages_total,
            )
        )


def format_progress_line(
    *,
    episode_index: int,
    episode_total: int,
    messages_done: int,
    messages_total: Optional[int],
) -> str:
    """Render one plain-text progress line, e.g.::

        episode 3/40: 62% (12000/19500 messages)

    Episode numbers are 1-based for reading; ``episode_index`` is 0-based.
    Without a known total the percentage is omitted rather than guessed.
    """
    head = f"episode {episode_index + 1}/{episode_total}"
    if messages_total:
        pct = 100.0 * messages_done / messages_total
        return f"{head}: {pct:.0f}% ({messages_done}/{messages_total} messages)"
    return f"{head}: {messages_done} messages"
