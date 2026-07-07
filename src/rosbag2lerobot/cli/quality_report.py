"""``quality-report`` command: score the data quality of a dataset."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import click

from rosbag2lerobot.cli._common import _emit_report


@click.command("quality-report")
@click.option(
    "--dataset",
    "dataset_path",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Root directory of a generated LeRobot v3.0 dataset.",
)
@click.option(
    "-o",
    "--report",
    "report_out",
    default=None,
    type=click.Path(dir_okay=False),
    help="If set, write the quality report as JSON to this path.",
)
@click.option(
    "--freeze-std-eps",
    default=1e-3,
    type=float,
    show_default=True,
    help="Per-pair std threshold for freeze-frame detection.",
)
@click.option(
    "--range-tol",
    default=0.0,
    type=float,
    show_default=True,
    help="Absolute tolerance added to stats.json min/max for out-of-range.",
)
@click.option(
    "--score-threshold",
    default=0.95,
    type=float,
    show_default=True,
    help="Minimum quality score for an OK verdict.",
)
@click.option(
    "--json",
    "json_stdout",
    is_flag=True,
    default=False,
    help="Emit the report dict as JSON to stdout (suppresses the human summary).",
)
def quality_report_cmd(
    dataset_path: str,
    report_out: Optional[str],
    freeze_std_eps: float,
    range_tol: float,
    score_threshold: float,
    json_stdout: bool,
) -> None:
    """Compute a data-quality report for a generated LeRobot v3.0 dataset.

    Reports per-feature null/NaN/out-of-range rates, freeze frames, and
    video/data frame reconciliation, condensed into a 0..1 score. Exits 1
    when the score is below ``--score-threshold`` or any video has a frame
    mismatch; 2 on a setup error (missing/unreadable metadata).
    """
    from rosbag2lerobot.quality import compute_quality_report

    try:
        report = compute_quality_report(
            Path(dataset_path),
            freeze_std_eps=freeze_std_eps,
            range_tol=range_tol,
            score_threshold=score_threshold,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        click.secho(f"quality-report: {exc}", fg="red")
        sys.exit(2)

    payload = report.to_dict()

    _emit_report(
        payload,
        json_stdout=json_stdout,
        json_out=report_out,
        human_fn=_print_quality_summary,
    )

    if report.exit_code != 0:
        sys.exit(report.exit_code)


def _print_quality_summary(payload: dict[str, Any]) -> None:
    """Render a QualityReport dict as a compact, colorized CLI summary."""
    click.echo(f"Dataset : {payload['dataset']}")
    click.echo("")
    click.echo(f"{'FEATURE':40s} {'NULL_RATE':>10s} {'NAN':>8s} {'OOR_RATE':>10s}")
    for f in payload["features"]:
        click.echo(
            f"{f['feature']:40s} {f['null_rate']:>10.4f} "
            f"{f['n_nan']:>8d} {f['oor_rate']:>10.4f}"
        )

    if payload["videos"]:
        click.echo("")
        click.echo(
            f"{'VIDEO_KEY':40s} {'EXPECTED':>9s} {'MP4':>9s} "
            f"{'MISMATCH':>9s} {'FREEZE':>7s}"
        )
        for v in payload["videos"]:
            mismatch_color = "green" if v["frame_mismatch"] == 0 else "red"
            line = (
                f"{v['video_key']:40s} {v['expected_frames']:>9d} "
                f"{v['mp4_frames']:>9d} "
            )
            click.echo(line, nl=False)
            click.secho(f"{v['frame_mismatch']:>9d}", fg=mismatch_color, nl=False)
            click.echo(f" {v['n_freeze']:>7d}")

    click.echo("")
    click.echo(
        f"Score: {payload['score']:.4f} (threshold {payload['score_threshold']:.4f})"
    )
    verdict = payload["verdict"]
    fg = "green" if verdict == "OK" else "red"
    click.secho(f"Verdict: {verdict}", fg=fg, bold=True)
