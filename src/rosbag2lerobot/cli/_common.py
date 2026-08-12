"""Shared helpers for the rosbag2lerobot CLI commands."""

from __future__ import annotations

import functools
import json
import logging
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import click

logger = logging.getLogger("rosbag2lerobot")


_NVENC_ENCODERS = ("h264_nvenc", "hevc_nvenc", "av1_nvenc")

# 256x256 is the probe frame size: NVENC refuses anything below its minimum
# frame dimension ("Frame Dimension less than the minimum supported value" —
# 128x128 is already too small on current hardware), and a probe that fails
# on a healthy GPU would be worse than no probe at all.
_NVENC_PROBE_CMD = [
    "ffmpeg",
    "-nostdin",
    "-hide_banner",
    "-loglevel",
    "error",
    "-f",
    "lavfi",
    "-i",
    "color=black:s=256x256:d=0.1",
    "-frames:v",
    "1",
    "-c:v",
    "h264_nvenc",
    "-f",
    "null",
    "-",
]


def _ffmpeg_lists_nvenc() -> bool:
    """Return True if ffmpeg was built with any NVENC encoder.

    Scans ``ffmpeg -encoders``. This only proves the encoder was *compiled
    in*, which is why :func:`_detect_nvenc` does not stop here.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            stdin=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return any(enc in result.stdout for enc in _NVENC_ENCODERS)


def _nvenc_probe_error() -> str | None:
    """Encode one frame with NVENC; return why it failed, or ``None`` if it worked.

    A listed encoder is not a working one: a container without the NVIDIA
    runtime still advertises ``h264_nvenc`` and then dies at the first frame
    with ``Cannot load libcuda.so.1``. Actually opening the encoder is the
    only way to tell the two apart, and one 256x256 frame costs well under a
    second.

    Returns:
        ``None`` when the test encode succeeded; otherwise a single-line
        reason, or why the probe could not be run. The *first* stderr line is
        the useful one: ffmpeg reports the root cause ("Cannot load
        libcuda.so.1") and then cascades into generic follow-ups, ending on
        the useless "Nothing was written into output file".
    """
    try:
        result = subprocess.run(
            _NVENC_PROBE_CMD,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return "ffmpeg not found"
    except subprocess.TimeoutExpired:
        return "test encode timed out"
    if result.returncode == 0:
        return None
    lines = [ln.strip() for ln in (result.stderr or "").splitlines() if ln.strip()]
    return lines[0] if lines else f"ffmpeg exited {result.returncode}"


@functools.lru_cache(maxsize=1)
def _detect_nvenc() -> bool:
    """Return True if NVENC is not just present but actually usable here.

    Checked in two steps, cached for the life of the process (the answer
    cannot change mid-run, and the probe costs a subprocess):

    1. ``ffmpeg -encoders`` lists an NVENC encoder — cheap, and skips the
       probe entirely on machines that were never built for GPU encoding.
    2. A one-frame test encode succeeds — the part that catches an ffmpeg
       built with NVENC running where the driver is not reachable, e.g. a
       container started without ``--gpus all``. Without this the run would
       select ``h264_nvenc`` and then die at the first frame.

    A failed probe is logged as a warning with ffmpeg's own reason: falling
    back to a CPU codec is the right call, but doing it silently would leave
    an operator wondering why their GPU host encodes at CPU speed.

    Returns:
        ``True`` only when NVENC both exists and encodes.
    """
    if not _ffmpeg_lists_nvenc():
        return False
    reason = _nvenc_probe_error()
    if reason is None:
        return True
    logger.warning(
        "NVENC is listed by ffmpeg but cannot encode here (%s); using a CPU codec. "
        "In a container, NVENC needs the NVIDIA runtime (docker run --gpus all).",
        reason,
    )
    return False


def _setup_logging(verbose: bool = False) -> None:
    """Configure root logger format and level.

    Args:
        verbose: If True, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _make_progress(total: int, disable: bool) -> Any:
    """Return a tqdm progress bar over *total* episodes, or ``None``.

    A bar is only rendered for an interactive run. tqdm does not detect that
    itself unless asked, and its carriage returns are unreadable once stdout
    is a pipe — a log file or ``docker logs`` fills up with redrawn bars. When
    stdout is not a TTY the caller falls back to plain progress log lines and
    ``meta/progress.json`` instead.

    Args:
        total: Number of episodes to track.
        disable: When True, returns ``None`` (no bar) — used for ``--quiet``
            and ``--json`` so machine-readable output stays uncluttered.

    Returns:
        A configured ``tqdm`` instance, or ``None`` when *disable* is set or
        stdout is not a terminal.
    """
    if disable or not sys.stdout.isatty():
        return None
    from tqdm import tqdm

    return tqdm(total=total, unit="ep", desc="convert")


def _emit_report(
    payload: dict[str, Any],
    *,
    json_stdout: bool,
    json_out: Optional[str],
    human_fn: Callable[[dict[str, Any]], None],
) -> None:
    """Emit a report verb's *payload* per the uniform output precedence.

    Precedence (independent of one another):

    - ``--json`` (``json_stdout``): print ``json.dumps(payload, indent=2)`` to
      stdout and SUPPRESS the human summary. Logging stays on stderr so the
      stdout JSON is clean for machine consumers.
    - ``--json-out`` / ``-o`` FILE (``json_out``): write the payload as JSON to
      the file. This is the back-compat P0 file flag; it is independent of
      ``--json`` (both may be set: file is written AND stdout JSON is emitted).
    - Neither / file-only: render the human summary via *human_fn*.

    Args:
        payload: JSON-serializable report dict.
        json_stdout: Value of the verb's ``--json`` flag.
        json_out: Value of the verb's existing ``--json-out`` / ``-o`` FILE
            flag, or ``None`` when the verb has none / it was not set.
        human_fn: Callback that renders the human summary from *payload*.
    """
    if json_out is not None:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w") as fh:
            json.dump(payload, fh, indent=2)

    if json_stdout:
        click.echo(json.dumps(payload, indent=2))
        return

    if json_out is not None:
        click.echo(f"Wrote JSON report to {json_out}")
    human_fn(payload)


def _fmt(val: Any) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)
