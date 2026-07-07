"""``validate-config`` command: validate a YAML config against a rosbag."""

from __future__ import annotations

import sys
from typing import Any, Optional, TYPE_CHECKING

import click

from rosbag2lerobot.config import load_config
from rosbag2lerobot.reader import BagReader, discover_bags
from rosbag2lerobot.cli._common import _emit_report

if TYPE_CHECKING:
    from rosbag2lerobot.diagnostics import ValidationReport


@click.command("validate-config")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to robot_config.yaml.",
)
@click.option(
    "--bags",
    "bags_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to a bag directory or parent directory.",
)
@click.option(
    "--samples",
    default=5,
    type=int,
    show_default=True,
    help="Number of image frames to decode per topic for shape check.",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Treat warnings (image shape / unused topics) as failures.",
)
@click.option(
    "--json-out",
    "json_out",
    default=None,
    type=click.Path(dir_okay=False),
    help="If set, write the validation report JSON to this path.",
)
@click.option(
    "--ignore-unused-topics",
    is_flag=True,
    default=False,
    help="Do not report bag topics that the config does not reference.",
)
@click.option(
    "--suggest-fixes",
    is_flag=True,
    default=False,
    help="After the summary, print copy-pasteable image_size diffs for shape mismatches.",
)
@click.option(
    "--json",
    "json_stdout",
    is_flag=True,
    default=False,
    help="Emit the report dict as JSON to stdout (suppresses the human summary).",
)
def validate_config(
    config_path: str,
    bags_path: str,
    samples: int,
    strict: bool,
    json_out: Optional[str],
    ignore_unused_topics: bool,
    suggest_fixes: bool,
    json_stdout: bool,
) -> None:
    """Validate a YAML config against the contents of a rosbag."""
    from rosbag2lerobot.diagnostics import validate_config_against_bag

    cfg = load_config(config_path)
    bag_paths = discover_bags(bags_path)
    # Use the first discovered bag as the primary validation target. CI
    # pipelines typically point at a single representative bag anyway;
    # extending to multi-bag aggregation is out of scope here.
    bag_path = bag_paths[0]

    with BagReader(bag_path, cfg) as reader:
        report = validate_config_against_bag(cfg, reader, samples)

    if ignore_unused_topics:
        report.unused_bag_topics = []

    report.apply_verdict(strict=strict)

    payload = {
        "config": str(config_path),
        "bag": str(bag_path),
        "results": report.to_dict(),
    }

    def _human(p: dict[str, Any]) -> None:
        _print_validation_summary(p)
        if suggest_fixes:
            _print_suggested_fixes(report)

    _emit_report(payload, json_stdout=json_stdout, json_out=json_out, human_fn=_human)

    if report.exit_code != 0:
        sys.exit(report.exit_code)


def _print_validation_summary(payload: dict[str, Any]) -> None:
    """Render the validation report to the terminal."""
    results = payload["results"]
    click.echo(f"Config validation: {payload['config']} <-> {payload['bag']}\n")
    n_err = 0
    n_warn = 0
    n_info = 0

    for t in results["missing_required_topics"]:
        click.secho(f"  [ERROR]   Missing required topic: {t}", fg="red")
        n_err += 1
    for m in results["msg_type_mismatches"]:
        click.secho(f"  [ERROR]   msg_type mismatch on {m['topic']}", fg="red")
        click.echo(f"              YAML: {m['yaml']}")
        click.echo(f"              BAG:  {m['bag']}")
        n_err += 1
    for m in results["image_shape_mismatches"]:
        click.secho(
            f"  [WARN]    Image shape mismatch on {m['key']}",
            fg="yellow",
        )
        click.echo(f"              YAML image_size: {m['yaml']}")
        click.echo(f"              Decoded shape  : {m['decoded']}")
        n_warn += 1
    for t in results["missing_optional_topics"]:
        click.secho(f"  [INFO]    Missing optional topic: {t}", fg="cyan")
        n_info += 1
    if results["unused_bag_topics"]:
        click.secho(
            f"  [INFO]    Unused bag topics ({len(results['unused_bag_topics'])}): "
            f"{', '.join(results['unused_bag_topics'])}",
            fg="cyan",
        )
        n_info += 1

    verdict = results["verdict"]
    color = "green" if verdict == "OK" else "red"
    click.secho(
        f"\nVerdict: {verdict} ({n_err} error, {n_warn} warning, {n_info} info)",
        fg=color,
    )


def _print_suggested_fixes(report: "ValidationReport") -> None:
    """Print copy-pasteable ``image_size`` diffs for image-shape mismatches.

    For every :class:`~rosbag2lerobot.diagnostics.ImageShapeMismatch` the block shows
    the current YAML ``image_size`` and the measured ``[H, W, C]`` as a unified
    diff snippet the user can paste over the offending feature::

        observation.images.front  (/camera/front/image_raw)
        -   image_size: [480, 640, 3]  # current
        +   image_size: [720, 1280, 3]  # measured

    Args:
        report: The validation report whose ``image_shape_mismatches`` drive
            the suggestions. A no-op when that list is empty.
    """
    if not report.image_shape_mismatches:
        return
    click.echo("")
    click.secho("Suggested fixes:", bold=True)
    for m in report.image_shape_mismatches:
        click.echo(f"  {m.key}  ({m.topic})")
        click.secho(f"  -   image_size: {m.yaml}  # current", fg="red")
        click.secho(f"  +   image_size: {m.decoded}  # measured", fg="green")
