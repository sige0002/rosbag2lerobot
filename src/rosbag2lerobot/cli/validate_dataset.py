"""``validate-dataset`` command: validate a dataset against LeRobot v3.0."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import click

from rosbag2lerobot.cli._common import _emit_report


@click.command("validate-dataset")
@click.option(
    "--dataset",
    "dataset_path",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Root directory of a generated LeRobot v3.0 dataset.",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Treat WARN-level issues (extra columns) as failures.",
)
@click.option(
    "--json-out",
    "json_out",
    default=None,
    type=click.Path(dir_okay=False),
    help="If set, write the validation report as JSON to this path.",
)
@click.option(
    "--json",
    "json_stdout",
    is_flag=True,
    default=False,
    help="Emit the report dict as JSON to stdout (suppresses the human summary).",
)
def validate_dataset_cmd(
    dataset_path: str,
    strict: bool,
    json_out: Optional[str],
    json_stdout: bool,
) -> None:
    """Validate the structure of a generated LeRobot v3.0 dataset.

    Checks required files, ``meta/info.json`` keys/values, parquet schemas,
    and episode-count cross-checks. Exits 1 on any ERROR (or any WARN under
    ``--strict``); 2 on a setup error such as an unreadable parquet file.
    """
    import pyarrow.lib as pa_lib

    from rosbag2lerobot.validation import validate_dataset

    try:
        report = validate_dataset(Path(dataset_path))
    except (OSError, ValueError, pa_lib.ArrowInvalid) as exc:
        click.secho(f"validate-dataset: {exc}", fg="red")
        sys.exit(2)

    report.apply_verdict(strict=strict)
    payload = report.to_dict()

    _emit_report(
        payload,
        json_stdout=json_stdout,
        json_out=json_out,
        human_fn=_print_dataset_validation_summary,
    )

    if report.exit_code != 0:
        sys.exit(report.exit_code)


def _print_dataset_validation_summary(payload: dict[str, Any]) -> None:
    """Render a DatasetValidationReport dict as a colorized CLI summary."""
    click.echo(f"Dataset : {payload['dataset']}")
    click.echo("")
    for issue in payload["issues"]:
        color = "red" if issue["severity"] == "ERROR" else "yellow"
        click.secho(
            f"  [{issue['severity']:5s}] {issue['kind']} @ {issue['location']}",
            fg=color,
        )
        click.echo(f"            {issue['message']}")

    verdict = payload["verdict"]
    fg = "green" if verdict == "OK" else "red"
    click.echo("")
    click.secho(
        f"Verdict: {verdict} "
        f"({payload['n_errors']} error, {payload['n_warnings']} warning)",
        fg=fg,
        bold=True,
    )
