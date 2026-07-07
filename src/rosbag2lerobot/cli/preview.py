"""``preview`` command: write a static HTML preview report for a dataset."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click


@click.command("preview")
@click.option(
    "--dataset",
    "dataset_path",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Root directory of a generated LeRobot v3.0 dataset.",
)
@click.option(
    "--n-frames",
    default=3,
    type=int,
    show_default=True,
    help="Number of sample frames to embed per video key.",
)
@click.option(
    "-o",
    "--out",
    "out_path",
    default=None,
    type=click.Path(dir_okay=False),
    help="Output HTML path (default: <dataset>/meta/preview.html).",
)
@click.option(
    "--sample-video/--no-sample-video",
    default=False,
    show_default=True,
    help="Decode mp4s to count freeze frames for the quality section.",
)
def preview_cmd(
    dataset_path: str,
    n_frames: int,
    out_path: Optional[str],
    sample_video: bool,
) -> None:
    """Write a self-contained static HTML preview report for a dataset.

    Renders the summary, the quality score and tables, a gallery of sampled
    video frames (inline base64), and the numeric per-feature statistics into
    a single self-contained HTML file (no external assets).
    """
    from rosbag2lerobot.preview import generate_preview

    dataset_dir = Path(dataset_path)
    try:
        html = generate_preview(
            dataset_dir,
            n_frames=n_frames,
            sample_video=sample_video,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        click.secho(f"preview: {exc}", fg="red")
        sys.exit(2)

    out = (
        Path(out_path)
        if out_path is not None
        else dataset_dir / "meta" / "preview.html"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    click.echo(f"Wrote preview to {out}")
