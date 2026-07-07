"""Shared helpers for the rosbag2lerobot CLI commands."""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import click

logger = logging.getLogger("rosbag2lerobot")


def _detect_nvenc() -> bool:
    """Return True if ffmpeg has at least one ``*_nvenc`` encoder available.

    Implemented as a pure function that shells out to ``ffmpeg -encoders``
    and scans the stdout for the NVENC encoder names. The subprocess call
    is isolated so unit tests can mock :func:`subprocess.run`.

    Returns:
        ``True`` if any of ``h264_nvenc``, ``hevc_nvenc``, or
        ``av1_nvenc`` appears in ffmpeg's encoder list; ``False`` when
        ffmpeg is missing, times out, or reports no NVENC encoder.
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
    return any(
        enc in result.stdout for enc in ("h264_nvenc", "hevc_nvenc", "av1_nvenc")
    )


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

    Args:
        total: Number of episodes to track.
        disable: When True, returns ``None`` (no bar) — used for ``--quiet``
            and ``--json`` so machine-readable output stays uncluttered.

    Returns:
        A configured ``tqdm`` instance, or ``None`` when *disable* is set.
    """
    if disable:
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
