"""``validate-msg`` command: check a .msg file for syntactic correctness."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import click

from rosbag2lerobot.cli._common import _emit_report


@click.command("validate-msg")
@click.option(
    "--msg",
    "msg_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a .msg file to validate.",
)
@click.option(
    "--json",
    "json_stdout",
    is_flag=True,
    default=False,
    help="Emit the result dict as JSON to stdout (suppresses the human summary).",
)
def validate_msg(msg_path: str, json_stdout: bool) -> None:
    """Validate a ROS2 .msg file for syntax correctness."""
    from rosbags.typesys import get_types_from_msg

    msg_file = Path(msg_path)
    msg_text = msg_file.read_text()

    # Derive a dummy type name from the filename
    type_name = f"validation_pkg/msg/{msg_file.stem}"

    valid = True
    n_types = 0
    error: Optional[str] = None
    try:
        types = get_types_from_msg(msg_text, type_name)
        n_types = len(types) if types else 0
    except Exception as exc:  # noqa: BLE001 - reported as an invalid result
        valid = False
        error = str(exc)

    payload = {
        "msg": str(msg_file),
        "type_name": type_name,
        "valid": valid,
        "n_types": n_types,
        "error": error,
    }

    def _human(_p: dict[str, Any]) -> None:
        if valid:
            click.echo(f"Valid .msg file: {msg_file.name}")
            click.echo(f"  Registered type: {type_name}")
            if n_types:
                click.echo(f"  Fields defined: {n_types} type(s)")
            click.secho("  OK", fg="green")
        else:
            click.secho(f"  INVALID: {error}", fg="red")

    _emit_report(payload, json_stdout=json_stdout, json_out=None, human_fn=_human)

    if not valid:
        sys.exit(1)
