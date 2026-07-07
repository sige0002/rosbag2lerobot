"""``to-mcap`` command: convert ROS1 .bag recordings to ROS2 MCAP bags."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click

from rosbag2lerobot.bagconvert import (
    DEFAULT_DST_VERSION,
    convert_to_mcap,
    discover_ros1_bags,
    output_name,
)
from rosbag2lerobot.cli._common import _emit_report


@click.command("to-mcap")
@click.argument(
    "sources",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    "output_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Output base directory. Each bag is written to <output>/<name>/.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing output bag directories.",
)
@click.option(
    "--dst-version",
    "dst_version",
    default=DEFAULT_DST_VERSION,
    type=int,
    show_default=True,
    help="ROS2 bag format version to write.",
)
@click.option(
    "--json",
    "json_stdout",
    is_flag=True,
    default=False,
    help="Emit the result dict as JSON to stdout (suppresses the human summary).",
)
def to_mcap(
    sources: tuple[Path, ...],
    output_dir: Path,
    overwrite: bool,
    dst_version: int,
    json_stdout: bool,
) -> None:
    """Convert ROS1 .bag recordings to ROS2 MCAP bags.

    rosbag2lerobot itself only reads ROS2 bags (mcap/sqlite3). Use this command to
    pre-convert ROS1 .bag recordings (e.g. the airoa raw dataset) so they
    can be fed to `rosbag2lerobot convert`.

    SOURCES may be .bag files or directories (searched recursively for
    *.bag). Each input bag is written to <output>/<name>/, where <name> is
    the bag file's parent directory name (e.g. .../235210/data.bag ->
    <output>/235210/).
    """
    bags = discover_ros1_bags(list(sources))
    if not bags:
        click.secho("No ROS1 .bag files found in the given sources.", fg="yellow")
        sys.exit(1)

    if not json_stdout:
        click.echo(f"Found {len(bags)} ROS1 bag(s) to convert.")
    converted = 0
    failed = 0
    results: list[dict[str, Any]] = []
    for src in bags:
        dst = output_dir / output_name(src)
        try:
            convert_to_mcap(src, dst, dst_version=dst_version, overwrite=overwrite)
            results.append({"src": str(src), "dst": str(dst), "status": "OK"})
            if not json_stdout:
                click.secho(f"  OK  {src}  ->  {dst}", fg="green")
            converted += 1
        except FileExistsError as exc:
            results.append({"src": str(src), "dst": str(dst), "status": "SKIP"})
            if not json_stdout:
                click.secho(f"  SKIP {exc}", fg="yellow")
            failed += 1
        except Exception as exc:  # noqa: BLE001 - report and continue
            results.append(
                {"src": str(src), "dst": str(dst), "status": "FAIL", "error": str(exc)}
            )
            if not json_stdout:
                click.secho(f"  FAIL {src}: {exc}", fg="red")
            failed += 1

    payload = {
        "output_dir": str(output_dir),
        "results": results,
        "converted": converted,
        "failed": failed,
    }

    def _human(_p: dict[str, Any]) -> None:
        click.echo("")
        click.secho(
            f"Converted {converted}/{len(bags)} bag(s) to MCAP under {output_dir}",
            fg="green" if failed == 0 else "yellow",
            bold=True,
        )

    _emit_report(payload, json_stdout=json_stdout, json_out=None, human_fn=_human)

    if failed:
        sys.exit(1)
