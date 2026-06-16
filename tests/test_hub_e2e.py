"""Tests for :mod:`bagel.hub` (HuggingFace Hub push + dataset card).

Fast unit test covers the pure :func:`build_dataset_card`. CLI tests exercise
the ``push-to-hub --dry-run`` path and the ``repo_id`` fallback. NO test ever
performs a real network call: the dry-run test monkeypatches the three
networked ``HfApi`` methods to raise, proving the dry-run path never invokes
them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from bagel.cli import main
from bagel.hub import build_dataset_card, plan_push

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_DATASET = PROJECT_ROOT / "output" / "airoa_moma_mcap_hsr"


# ---------------------------------------------------------------------------
# Pure-function unit test (no I/O, no network)
# ---------------------------------------------------------------------------


def test_build_dataset_card_offline() -> None:
    if not REAL_DATASET.is_dir():
        pytest.skip(f"real dataset not present: {REAL_DATASET}")
    info = json.loads((REAL_DATASET / "meta" / "info.json").read_text())
    tasks_tbl = pq.read_table(REAL_DATASET / "meta" / "tasks.parquet")
    tasks = [str(t) for t in tasks_tbl.column("task").to_pylist()]

    card = build_dataset_card(info, tasks)

    # YAML front-matter with hardcoded categories/tags.
    assert card.startswith("---\n")
    assert "task_categories:" in card
    assert "license: apache-2.0" in card
    assert "data_files: data/*/*.parquet" in card
    # Summary rows.
    assert "robot_type" in card
    assert str(info["robot_type"]) in card
    assert "fps" in card
    assert "episodes" in card
    assert "frames" in card
    # Both video feature keys appear in the features table.
    assert "observation.images.head_rgb" in card
    assert "observation.images.hand" in card
    # Tasks listed.
    for t in tasks:
        assert t in card


# ---------------------------------------------------------------------------
# CLI dry-run (NO network — monkeypatched HfApi raises if called)
# ---------------------------------------------------------------------------


@pytest.fixture
def _hub_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every networked HfApi method raise, so any real call fails loudly."""
    import huggingface_hub

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network call attempted in a dry-run/offline test")

    monkeypatch.setattr(huggingface_hub.HfApi, "create_repo", _boom)
    monkeypatch.setattr(huggingface_hub.HfApi, "upload_folder", _boom)
    monkeypatch.setattr(huggingface_hub.HfApi, "upload_file", _boom)


def _tiny_meta_dataset(root: Path, repo_id: str | None = None) -> Path:
    """Write a minimal dataset dir with meta/info.json + tasks.parquet + files."""
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "observation.images.cam" / "chunk-000").mkdir(parents=True)

    info = {
        "codebase_version": "v3.0",
        "robot_type": "hsr",
        "fps": 10,
        "total_episodes": 1,
        "total_frames": 5,
        "total_tasks": 1,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [2]},
            "observation.images.cam": {"dtype": "video", "shape": [32, 32, 3]},
        },
    }
    if repo_id is not None:
        info["repo_id"] = repo_id
    (root / "meta" / "info.json").write_text(json.dumps(info))

    tbl = pa_table(["pick up the cube"])
    pq.write_table(tbl, root / "meta" / "tasks.parquet")

    # A couple of dummy data/video files so file enumeration is non-empty.
    (root / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"x")
    (
        root / "videos" / "observation.images.cam" / "chunk-000" / "file-000.mp4"
    ).write_bytes(b"y")
    return root


def pa_table(tasks: list[str]):
    import pyarrow as pa

    return pa.table({"task_index": list(range(len(tasks))), "task": tasks})


def test_cli_push_to_hub_dry_run(tmp_path: Path, _hub_no_network: None) -> None:
    dataset_dir = _tiny_meta_dataset(tmp_path / "ds")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "push-to-hub",
            "--dataset",
            str(dataset_dir),
            "--repo-id",
            "user/x",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "user/x" in result.output
    # File count is listed (data + videos + meta files).
    assert "files" in result.output
    # Card preview included.
    assert "task_categories" in result.output


def test_repo_id_fallback(tmp_path: Path, _hub_no_network: None) -> None:
    dataset_dir = _tiny_meta_dataset(tmp_path / "ds", repo_id="user/from_info")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["push-to-hub", "--dataset", str(dataset_dir), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "user/from_info" in result.output


def test_repo_id_missing_errors(tmp_path: Path, _hub_no_network: None) -> None:
    dataset_dir = _tiny_meta_dataset(tmp_path / "ds")  # no repo_id in info
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["push-to-hub", "--dataset", str(dataset_dir), "--dry-run"],
    )
    assert result.exit_code == 2, result.output


def test_plan_push_no_network(tmp_path: Path, _hub_no_network: None) -> None:
    dataset_dir = _tiny_meta_dataset(tmp_path / "ds")
    plan = plan_push(dataset_dir, "user/x")
    assert plan.repo_id == "user/x"
    # data + videos + meta files enumerated.
    assert any(f.startswith("data/") for f in plan.files)
    assert any(f.startswith("videos/") for f in plan.files)
    assert any(f.startswith("meta/") for f in plan.files)
    assert "license: apache-2.0" in plan.card_text
