"""``audit-timestamps`` command: audit timestamp continuity of a dataset."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import click

from rosbag2lerobot.cli._common import _emit_report


@click.command("audit-timestamps")
@click.option(
    "--dataset",
    "dataset_path",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Root directory of a generated LeRobot v3.0 dataset.",
)
@click.option(
    "--max-drift-us",
    default=1.0,
    type=float,
    show_default=True,
    help="Maximum allowed per-row / cumulative drift in microseconds.",
)
@click.option(
    "--json-out",
    default=None,
    type=click.Path(dir_okay=False),
    help="If set, write the audit report as JSON to this path.",
)
@click.option(
    "--json",
    "json_stdout",
    is_flag=True,
    default=False,
    help="Emit the report dict as JSON to stdout (suppresses the human summary).",
)
@click.option(
    "--video-key",
    "video_key",
    default=None,
    help="Audit only this video_key (default: all video_keys present).",
)
def audit_timestamps(
    dataset_path: str,
    max_drift_us: float,
    json_out: Optional[str],
    json_stdout: bool,
    video_key: Optional[str],
) -> None:
    """Audit meta/episodes/*.parquet timestamp continuity for drift.

    Reads every episodes parquet file under the dataset's ``meta/episodes/``
    tree and verifies that ``to_timestamp[i] == from_timestamp[i + 1]`` inside
    each mp4 file and that ``from_timestamp`` only resets to ``0.0`` at mp4
    file boundaries. Exits with status 1 on any violation.
    """
    from rosbag2lerobot.audit import audit_episode_timestamps

    vkeys = [video_key] if video_key else None
    try:
        report = audit_episode_timestamps(
            Path(dataset_path),
            max_drift_us=max_drift_us,
            video_keys=vkeys,
        )
    except (FileNotFoundError, ValueError) as exc:
        click.secho(f"audit-timestamps: {exc}", fg="red")
        sys.exit(2)

    payload = report.to_dict()

    _emit_report(
        payload,
        json_stdout=json_stdout,
        json_out=json_out,
        human_fn=lambda p: _print_audit_summary(p, max_drift_us),
    )

    if report.verdict != "OK":
        sys.exit(report.exit_code)


def _print_audit_summary(payload: dict[str, Any], max_drift_us: float) -> None:
    """Render an AuditReport dict as a compact, colorized CLI summary."""
    click.echo(f"Dataset : {payload['dataset']}")
    click.echo(f"Keys    : {', '.join(payload['video_keys']) or '(none)'}")
    click.echo(f"Max drift threshold: {max_drift_us:.3f} us")
    click.echo("")
    click.echo(
        f"{'VIDEO_KEY':40s} {'EPS':>5s} {'MAX_DRIFT_US':>14s} {'ERRS':>5s}  VERDICT"
    )
    for r in payload["results"]:
        verdict_color = "green" if r["verdict"] == "OK" else "red"
        line = (
            f"{r['video_key']:40s} {r['n_episodes']:>5d} "
            f"{r['max_drift_us']:>14.3f} {len(r['boundary_errors']):>5d}  "
        )
        click.echo(line, nl=False)
        click.secho(r["verdict"], fg=verdict_color)
        for err in r["boundary_errors"][:10]:
            click.secho(
                f"    ep={err['episode_index']:<4d} "
                f"expected={err['expected_from_ts']:.6f} "
                f"actual={err['actual_from_ts']:.6f} "
                f"delta_us={err['delta_us']:+.3f}",
                fg="red",
            )
        if len(r["boundary_errors"]) > 10:
            click.secho(
                f"    ... and {len(r['boundary_errors']) - 10} more",
                fg="red",
            )

    click.echo("")
    fg = "green" if payload["verdict"] == "OK" else "red"
    click.secho(f"Verdict: {payload['verdict']}", fg=fg, bold=True)
