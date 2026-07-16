"""``validate-video-metadata`` command: reconcile mp4 frames with episode metadata."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import click

from rosbag2lerobot.cli._common import _emit_report


@click.command("validate-video-metadata")
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
    help=(
        "厳密チェック: 全 mp4 の全フレーム PTS を取得し、全 data 行について"
        "フレーム範囲と PTS 許容誤差・index 連続性・timestamp 単調性を検査"
        "（デコードが走るため遅い）。既定は高速チェック（各 episode の"
        "min/max timestamp とヘッダフレーム数のみ・即時）。"
    ),
)
@click.option(
    "--tolerance-s",
    "tolerance_s",
    default=None,
    type=float,
    help=(
        "厳密チェックの PTS 許容誤差（秒）。既定は 0.5/fps。学習時の LeRobot "
        "既定値を再現するには 1e-4 を指定。"
    ),
)
@click.option(
    "--full-decode",
    "full_decode",
    is_flag=True,
    default=False,
    help=(
        "厳密チェックに加え、全 mp4 を ffmpeg -xerror で末尾までデコードして"
        "ストリーム破損を検出する（--strict を含意）。"
    ),
)
@click.option(
    "--max-errors",
    "max_errors",
    default=50,
    type=int,
    show_default=True,
    help="記録するエラー/警告の上限（総数はカウントされ truncated で通知）。",
)
@click.option(
    "--json-out",
    default=None,
    type=click.Path(dir_okay=False),
    help="If set, write the reconciliation report as JSON to this path.",
)
@click.option(
    "--json",
    "json_stdout",
    is_flag=True,
    default=False,
    help="Emit the report dict as JSON to stdout (suppresses the human summary).",
)
def validate_video_metadata_cmd(
    dataset_path: str,
    strict: bool,
    tolerance_s: Optional[float],
    full_decode: bool,
    max_errors: int,
    json_out: Optional[str],
    json_stdout: bool,
) -> None:
    """Check LeRobot's video reference conditions via FFmpeg (torch-free).

    Reproduces the frame lookup LeRobot performs at training time —
    ``round((from_timestamp + row_timestamp) * avg_frame_rate)`` per data row —
    and verifies it lands inside the real mp4, so ``Invalid frame index=N must
    be less than M`` errors are caught before training. Fast mode checks each
    episode's extreme rows against the container header; ``--strict``
    validates every row against decoded per-frame PTS.

    Exit codes: 0 = consistent, 1 = inconsistencies found (including
    missing/unreadable mp4s), 2 = setup error (missing metadata/parquet,
    invalid fps, ffprobe not installed).
    """
    from rosbag2lerobot.video_reconciliation import (
        SetupError,
        validate_video_metadata,
    )

    try:
        report = validate_video_metadata(
            Path(dataset_path),
            strict=strict,
            tolerance_s=tolerance_s,
            full_decode=full_decode,
            max_errors=max_errors,
        )
    except SetupError as exc:
        click.secho(f"validate-video-metadata: [{exc.code}] {exc}", fg="red")
        sys.exit(2)
    except (OSError, ValueError) as exc:
        click.secho(f"validate-video-metadata: {exc}", fg="red")
        sys.exit(2)

    payload = report.to_dict()

    _emit_report(
        payload,
        json_stdout=json_stdout,
        json_out=json_out,
        human_fn=_print_reconciliation_summary,
    )

    if report.exit_code != 0:
        sys.exit(report.exit_code)


# (label, payload key, format) rows for the per-issue block. None values are
# skipped, so each status prints only its relevant fields (§6.7).
_ISSUE_FIELDS: list[tuple[str, str, str]] = [
    ("episode", "episode_index", "d"),
    ("video key", "video_key", "s"),
    ("video file", "video_path", "s"),
    ("dataset index", "dataset_index", "d"),
    ("row timestamp", "row_timestamp", "f"),
    ("from_timestamp", "from_timestamp", "f"),
    ("shifted ts", "shifted_timestamp", "f"),
    ("video avg fps", "video_average_fps", "f"),
    ("requested", "requested_frame", "d"),
    ("num frames", "video_frame_count", "d"),
    ("maximum valid", "max_valid_frame", "d"),
    ("overflow", "overflow", "d"),
    ("loaded PTS", "loaded_pts", "f"),
    ("ts error", "timestamp_error", "g"),
    ("tolerance", "tolerance_s", "g"),
    ("detail", "detail", "s"),
]


def _print_issue(issue: dict[str, Any], color: str) -> None:
    click.secho(f"[{issue['status']}]", fg=color, bold=True)
    for label, key, fmt in _ISSUE_FIELDS:
        val = issue.get(key)
        if val is None or val == "":
            continue
        if fmt == "f":
            text = f"{val:.6f}"
        elif fmt == "g":
            text = f"{val:.6g}"
        else:
            text = str(val)
        click.echo(f"  {label:<14}: {text}")


def _print_reconciliation_summary(payload: dict[str, Any]) -> None:
    """Render a VideoMetadataReport dict as a compact, colorized CLI summary."""
    mode = payload["mode"]
    if mode == "fast":
        mode_label = "fast (高速)"
    else:
        mode_label = f"strict (厳密, tolerance_s={payload['tolerance_s']:.6g})"
        if payload["full_decode"]:
            mode_label += " + full-decode"
    click.echo(f"Dataset : {payload['dataset']}")
    click.echo(f"Mode    : {mode_label}")
    click.echo("Checking LeRobot's video reference conditions via FFmpeg...")
    click.echo(f"Videos checked   : {payload['videos_checked']}")
    click.echo(f"Episodes checked : {payload['episodes_checked']}")
    click.echo(f"Mappings checked : {payload['mappings_checked']}")
    if mode == "strict":
        click.echo(f"Rows checked     : {payload['rows_checked']}")
    click.echo("")

    for w in payload["warnings"]:
        _print_issue(w, "yellow")
    if payload["warnings"]:
        click.echo("")

    errors = payload["errors"]
    if payload["total_errors"] == 0:
        click.secho("Verdict: OK", fg="green", bold=True)
        click.secho("No video ↔ metadata inconsistencies found.", fg="green")
        if payload["total_warnings"]:
            click.secho(
                f"({payload['total_warnings']} warning(s) — see above)", fg="yellow"
            )
        click.echo(
            "Checked LeRobot's video reference conditions via FFmpeg "
            "(not a training-success guarantee)."
        )
        return

    click.secho("Verdict: ERROR", fg="red", bold=True)
    click.secho(
        f"video ↔ metadata inconsistencies found: {payload['total_errors']}",
        fg="red",
    )
    for err in errors:
        _print_issue(err, "red")
    if payload["truncated"]:
        click.secho(
            f"... output truncated at --max-errors "
            f"(recorded {len(errors)}/{payload['total_errors']} errors)",
            fg="red",
        )
