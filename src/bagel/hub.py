"""HuggingFace Hub push + dataset-card generation for LeRobot v3.0 datasets.

Two concerns, cleanly split:

- :func:`build_dataset_card` is **pure**: ``info`` dict + task list in, a
  README.md markdown string out (HF dataset-card YAML front-matter + body). It
  performs no I/O, so the future UI can render the card preview directly.
- :func:`plan_push` enumerates the files that *would* be uploaded and builds
  the card without any network access (a dry-run plan), and
  :func:`push_to_hub` is the *only* function that touches the network.

The card's license / tags are hardcoded (apache-2.0, robotics, LeRobot/bagel)
per the spec — no speculative config knobs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from bagel.quality import _read_info

logger = logging.getLogger(__name__)

__all__ = [
    "PushPlan",
    "build_dataset_card",
    "plan_push",
    "push_to_hub",
]


# ---------------------------------------------------------------------------
# Pure card builder (no I/O — UI-reusable)
# ---------------------------------------------------------------------------


def _features_table(info: dict[str, Any]) -> str:
    """Render the features markdown table (key | dtype | shape) from ``info``."""
    lines = ["| key | dtype | shape |", "| --- | --- | --- |"]
    for key, spec in info.get("features", {}).items():
        if not isinstance(spec, dict):
            continue
        dtype = spec.get("dtype", "")
        shape = spec.get("shape", "")
        lines.append(f"| `{key}` | {dtype} | {shape} |")
    return "\n".join(lines)


def build_dataset_card(info: dict[str, Any], tasks: list[str]) -> str:
    """Build the HuggingFace dataset-card (README.md) markdown.

    Pure function: no I/O. Emits the HF dataset-card YAML front-matter
    (hardcoded ``license`` / ``task_categories`` / ``tags`` / ``configs``)
    followed by a summary table, the task list, and a features table — all
    derived from ``info`` and ``tasks``.

    Args:
        info: ``meta/info.json`` contents (robot_type / fps / totals /
            features / codebase_version).
        tasks: Task strings (the ``task`` column of ``meta/tasks.parquet``).

    Returns:
        A complete README.md markdown string.
    """
    front_matter = (
        "---\n"
        "license: apache-2.0\n"
        "task_categories:\n"
        "- robotics\n"
        "tags:\n"
        "- LeRobot\n"
        "- bagel\n"
        "configs:\n"
        "- config_name: default\n"
        "  data_files: data/*/*.parquet\n"
        "---\n"
    )

    robot = info.get("robot_type", "-")
    summary = (
        "## Summary\n\n"
        "| field | value |\n"
        "| --- | --- |\n"
        f"| robot_type | {robot} |\n"
        f"| fps | {info.get('fps', '-')} |\n"
        f"| episodes | {info.get('total_episodes', '-')} |\n"
        f"| frames | {info.get('total_frames', '-')} |\n"
        f"| tasks | {info.get('total_tasks', '-')} |\n"
        f"| codebase_version | {info.get('codebase_version', '-')} |\n"
    )

    if tasks:
        task_lines = "\n".join(f"- {t}" for t in tasks)
    else:
        task_lines = "_(none)_"
    tasks_section = f"## Tasks\n\n{task_lines}\n"

    features_section = f"## Features\n\n{_features_table(info)}\n"

    body = (
        f"# {robot} (LeRobot dataset)\n\n"
        "Converted from ROS2 rosbags with "
        "[bagel](https://github.com/) to the LeRobot Dataset v3.0 format.\n\n"
        f"{summary}\n"
        f"{tasks_section}\n"
        f"{features_section}"
    )

    return front_matter + "\n" + body


# ---------------------------------------------------------------------------
# Dry-run plan (no network)
# ---------------------------------------------------------------------------


@dataclass
class PushPlan:
    """A dry-run description of an upload.

    Attributes:
        repo_id: Target HuggingFace dataset repo id.
        dataset_dir: Local dataset root that would be uploaded.
        files: Relative paths (POSIX) of the dataset files to upload.
        card_text: The generated README.md dataset-card markdown.
    """

    repo_id: str
    dataset_dir: Path
    files: list[str]
    card_text: str


def _read_tasks(dataset_dir: Path) -> list[str]:
    """Read the ``task`` column of ``meta/tasks.parquet`` (empty if absent)."""
    tasks_path = dataset_dir / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        return []
    table = pq.read_table(tasks_path)
    if "task" not in table.column_names:
        return []
    return [str(t) for t in table.column("task").to_pylist()]


def plan_push(dataset_dir: Path, repo_id: str) -> PushPlan:
    """Build a :class:`PushPlan` without any network access.

    Reads ``meta/info.json`` and ``meta/tasks.parquet``, builds the dataset
    card, and enumerates the dataset files under ``data/`` / ``videos/`` /
    ``meta/`` that would be uploaded.

    Args:
        dataset_dir: Root of a LeRobot v3.0 dataset.
        repo_id: Target HuggingFace dataset repo id.

    Returns:
        A populated :class:`PushPlan`.
    """
    dataset_dir = Path(dataset_dir)
    info = _read_info(dataset_dir)
    tasks = _read_tasks(dataset_dir)
    card_text = build_dataset_card(info, tasks)

    files: list[str] = []
    for sub in ("data", "videos", "meta"):
        root = dataset_dir / sub
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file():
                files.append(p.relative_to(dataset_dir).as_posix())

    return PushPlan(
        repo_id=repo_id,
        dataset_dir=dataset_dir,
        files=files,
        card_text=card_text,
    )


# ---------------------------------------------------------------------------
# Networked push (the only function that touches the network)
# ---------------------------------------------------------------------------


def push_to_hub(
    dataset_dir: Path,
    repo_id: str,
    *,
    private: bool = False,
    token: str | None = None,
) -> None:
    """Upload a dataset to the HuggingFace Hub and place the card at the root.

    Creates (or reuses) the dataset repo, uploads the whole ``dataset_dir``,
    then writes the generated README.md card to the repo *root* via
    ``upload_file`` — the local dataset directory is never mutated.

    This is the only networked function in this module.

    Args:
        dataset_dir: Root of a LeRobot v3.0 dataset to upload.
        repo_id: Target HuggingFace dataset repo id.
        private: Whether to create the repo as private.
        token: Optional HuggingFace auth token (else the ambient login).
    """
    from huggingface_hub import HfApi

    dataset_dir = Path(dataset_dir)
    plan = plan_push(dataset_dir, repo_id)

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
    api.upload_folder(
        folder_path=str(dataset_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=plan.card_text.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    logger.info("Pushed %s to hub repo %s", dataset_dir, repo_id)
