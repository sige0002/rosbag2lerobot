"""``push-to-hub`` command: upload a dataset to the HuggingFace Hub."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click


@click.command("push-to-hub")
@click.option(
    "--dataset",
    "dataset_path",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Root directory of a generated LeRobot v3.0 dataset.",
)
@click.option(
    "--repo-id",
    "repo_id",
    default=None,
    help="HuggingFace dataset repo id (default: info.json['repo_id']).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Plan the upload only: print repo_id + file count + card preview.",
)
@click.option(
    "--private",
    is_flag=True,
    default=False,
    help="Create the repo as private (ignored for --dry-run).",
)
@click.option(
    "--token",
    default=None,
    help="HuggingFace auth token (else the ambient login).",
)
@click.option(
    "--card-out",
    "card_out",
    default=None,
    type=click.Path(dir_okay=False),
    help="With --dry-run, also write the generated card to this path.",
)
def push_to_hub_cmd(
    dataset_path: str,
    repo_id: Optional[str],
    dry_run: bool,
    private: bool,
    token: Optional[str],
    card_out: Optional[str],
) -> None:
    """Push a generated dataset to the HuggingFace Hub (with a dataset card).

    ``--repo-id`` falls back to ``info.json['repo_id']``; if neither is set the
    command exits 2. With ``--dry-run`` nothing is uploaded — the planned
    repo_id, file count, and card preview are printed (and the card written to
    ``--card-out`` when given). Without ``--dry-run`` the dataset is uploaded
    and the card is placed at the repo root.
    """
    from rosbag2lerobot.hub import plan_push, push_to_hub
    from rosbag2lerobot.quality import _read_info

    dataset_dir = Path(dataset_path)

    effective_repo_id = repo_id
    if effective_repo_id is None:
        try:
            effective_repo_id = _read_info(dataset_dir).get("repo_id")
        except (OSError, ValueError) as exc:
            click.secho(f"push-to-hub: {exc}", fg="red")
            sys.exit(2)
    if not effective_repo_id:
        click.secho(
            "push-to-hub: no --repo-id given and info.json has no 'repo_id'.",
            fg="red",
        )
        sys.exit(2)

    if dry_run:
        plan = plan_push(dataset_dir, effective_repo_id)
        click.echo(f"[dry-run] repo_id : {plan.repo_id}")
        click.echo(f"[dry-run] files   : {len(plan.files)}")
        if card_out is not None:
            card_path = Path(card_out)
            card_path.parent.mkdir(parents=True, exist_ok=True)
            card_path.write_text(plan.card_text)
            click.echo(f"[dry-run] wrote card to {card_path}")
        click.echo("[dry-run] card preview:")
        click.echo(plan.card_text)
        return

    push_to_hub(
        dataset_dir,
        effective_repo_id,
        private=private,
        token=token,
    )
    click.secho(f"Pushed {dataset_dir} to {effective_repo_id}", fg="green", bold=True)
