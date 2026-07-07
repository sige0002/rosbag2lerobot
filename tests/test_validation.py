"""Tests for :mod:`rosbag2lerobot.validation` (P0-4 validate-dataset).

Fast unit tests build a tiny real dataset with :class:`DatasetWriter` (reusing
the helper style of ``tests/test_video_frame_alignment.py``) and assert that a
clean dataset validates OK, then tamper with individual files / metadata to
verify each check fires. An integration test runs against the real
``output/airoa_moma_mcap_hsr`` dataset.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from PIL import Image

from rosbag2lerobot.cli import main
from rosbag2lerobot.validation import DatasetValidationReport, validate_dataset
from rosbag2lerobot.writer import DatasetWriter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_DATASET = PROJECT_ROOT / "output" / "airoa_moma_mcap_hsr"


# ---------------------------------------------------------------------------
# Helpers (mirroring tests/test_video_frame_alignment.py)
# ---------------------------------------------------------------------------


def _video_features(
    keys: list[str],
    shape: tuple[int, int, int] = (32, 32, 3),
) -> dict[str, dict[str, Any]]:
    feats: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [2],
            "names": ["a", "b"],
        },
        "action": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for k in keys:
        feats[k] = {
            "dtype": "video",
            "shape": list(shape),
            "names": ["height", "width", "channels"],
        }
    return feats


def _random_image(rng: np.random.Generator, shape: tuple[int, int, int]) -> Image.Image:
    return Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8))


def _write_dataset(
    out_dir: Path,
    episode_lengths: list[int],
    video_keys: list[str],
    fps: int = 10,
    shape: tuple[int, int, int] = (32, 32, 3),
) -> None:
    feats = _video_features(video_keys, shape=shape)
    writer = DatasetWriter(
        out_dir,
        {"robot_type": "regression"},
        feats,
        fps=fps,
        video_codec="libx264",
    )
    rng = np.random.default_rng(0)
    for ep_len in episode_lengths:
        for i in range(ep_len):
            frame: dict[str, Any] = {
                "observation.state": np.array([float(i), 0.0], dtype=np.float32),
                "action": np.array([float(i), 0.0], dtype=np.float32),
                "task": "t",
            }
            for k in video_keys:
                frame[k] = _random_image(rng, shape)
            writer.add_frame(frame)
        writer.save_episode()
    writer.finalize()


@pytest.fixture(autouse=True)
def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")


@pytest.fixture
def clean_dataset(tmp_path: Path) -> Path:
    """A tiny, structurally valid LeRobot v3.0 dataset."""
    _write_dataset(
        tmp_path,
        episode_lengths=[5, 7],
        video_keys=["observation.images.cam"],
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_clean_dataset_validates_ok(clean_dataset: Path) -> None:
    report = validate_dataset(clean_dataset)
    report.apply_verdict(strict=False)
    assert report.verdict == "OK", [i.to_dict() for i in report.issues]
    assert report.errors() == []
    assert report.issues == []
    assert report.exit_code == 0


def test_missing_tasks_parquet_is_error(clean_dataset: Path) -> None:
    (clean_dataset / "meta" / "tasks.parquet").unlink()
    report = validate_dataset(clean_dataset)
    report.apply_verdict(strict=False)
    assert report.verdict == "FAIL"
    kinds = {(i.kind, i.location) for i in report.errors()}
    assert ("missing_file", "meta/tasks.parquet") in kinds


def test_corrupt_splits_is_error(clean_dataset: Path) -> None:
    info_path = clean_dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["splits"]["train"] = "0:99"
    info_path.write_text(json.dumps(info))

    report = validate_dataset(clean_dataset)
    report.apply_verdict(strict=False)
    assert report.verdict == "FAIL"
    splits_errors = [i for i in report.errors() if i.kind == "splits"]
    assert splits_errors, "expected a splits ERROR"


def test_valid_three_way_splits_ok(clean_dataset: Path) -> None:
    """A contiguous 3-way partition of [0, total_episodes) validates OK."""
    info_path = clean_dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    total = info["total_episodes"]  # clean fixture has 2 episodes
    info["splits"] = {"train": f"0:{total - 1}", "val": f"{total - 1}:{total}"}
    info_path.write_text(json.dumps(info))

    report = validate_dataset(clean_dataset)
    report.apply_verdict(strict=False)
    assert report.verdict == "OK", [i.to_dict() for i in report.issues]
    assert not [i for i in report.errors() if i.kind == "splits"]


def test_splits_gap_is_error(clean_dataset: Path) -> None:
    info_path = clean_dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    total = info["total_episodes"]
    # Leave a gap: skip index 0.
    info["splits"] = {"train": f"1:{total}"}
    info_path.write_text(json.dumps(info))

    report = validate_dataset(clean_dataset)
    report.apply_verdict(strict=False)
    assert report.verdict == "FAIL"
    assert any(i.kind == "splits" for i in report.errors())


def test_splits_overlap_is_error(clean_dataset: Path) -> None:
    info_path = clean_dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    total = info["total_episodes"]
    info["splits"] = {"train": f"0:{total}", "val": f"0:{total}"}
    info_path.write_text(json.dumps(info))

    report = validate_dataset(clean_dataset)
    report.apply_verdict(strict=False)
    assert report.verdict == "FAIL"
    assert any(i.kind == "splits" for i in report.errors())


def test_splits_out_of_bounds_is_error(clean_dataset: Path) -> None:
    info_path = clean_dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["splits"] = {"train": "0:999"}
    info_path.write_text(json.dumps(info))

    report = validate_dataset(clean_dataset)
    report.apply_verdict(strict=False)
    assert report.verdict == "FAIL"
    assert any(i.kind == "splits" for i in report.errors())


def test_wrong_codebase_version_is_error(clean_dataset: Path) -> None:
    info_path = clean_dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["codebase_version"] = "v2.0"
    info_path.write_text(json.dumps(info))

    report = validate_dataset(clean_dataset)
    report.apply_verdict(strict=False)
    assert report.verdict == "FAIL"
    assert any(i.kind == "codebase_version" for i in report.errors())


def test_wrong_column_type_is_error(clean_dataset: Path) -> None:
    """Rewrite a data parquet with a wrong-typed bookkeeping column."""
    data_files = sorted((clean_dataset / "data").rglob("*.parquet"))
    assert data_files
    path = data_files[0]
    table = pq.read_table(path)

    # Replace int64 ``index`` with a float64 column to trigger column_type.
    new_index = pa.array(
        [float(x) for x in table.column("index").to_pylist()],
        type=pa.float64(),
    )
    cols = {}
    for name in table.column_names:
        cols[name] = new_index if name == "index" else table.column(name)
    pq.write_table(pa.table(cols), path)

    report = validate_dataset(clean_dataset)
    report.apply_verdict(strict=False)
    assert report.verdict == "FAIL"
    col_errors = [
        i for i in report.errors() if i.kind == "column_type" and "index" in i.location
    ]
    assert col_errors, "expected a column_type ERROR on 'index'"


def test_extra_column_is_warn_only(clean_dataset: Path) -> None:
    """An unexpected extra column is a WARN: OK without --strict, FAIL with."""
    data_files = sorted((clean_dataset / "data").rglob("*.parquet"))
    path = data_files[0]
    table = pq.read_table(path)
    extra = pa.array(list(range(table.num_rows)), type=pa.int64())
    augmented = table.append_column("bogus_extra", extra)
    pq.write_table(augmented, path)

    report = validate_dataset(clean_dataset)
    report.apply_verdict(strict=False)
    assert report.verdict == "OK"
    assert any(i.kind == "extra_column" for i in report.warnings())

    report2 = validate_dataset(clean_dataset)
    report2.apply_verdict(strict=True)
    assert report2.verdict == "FAIL"


def test_missing_video_mp4_is_error(clean_dataset: Path) -> None:
    vdir = clean_dataset / "videos" / "observation.images.cam"
    for mp4 in vdir.rglob("*.mp4"):
        mp4.unlink()
    report = validate_dataset(clean_dataset)
    report.apply_verdict(strict=False)
    assert report.verdict == "FAIL"
    assert any(
        i.kind == "missing_file" and "mp4" in i.location for i in report.errors()
    )


def test_count_mismatch_is_error(clean_dataset: Path) -> None:
    info_path = clean_dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_frames"] = info["total_frames"] + 1
    info_path.write_text(json.dumps(info))

    report = validate_dataset(clean_dataset)
    report.apply_verdict(strict=False)
    assert report.verdict == "FAIL"
    assert any(i.kind == "count_mismatch" for i in report.errors())


def test_to_dict_roundtrips(clean_dataset: Path) -> None:
    report = validate_dataset(clean_dataset)
    report.apply_verdict(strict=False)
    d = report.to_dict()
    assert d["verdict"] == "OK"
    assert d["n_errors"] == 0
    assert d["n_warnings"] == 0
    assert isinstance(d["issues"], list)
    # JSON-serializable.
    json.dumps(d)


def test_cli_validate_dataset_ok(clean_dataset: Path, tmp_path: Path) -> None:
    out_json = tmp_path / "report.json"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "validate-dataset",
            "--dataset",
            str(clean_dataset),
            "--json-out",
            str(out_json),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_json.read_text())
    assert payload["verdict"] == "OK"


def test_cli_validate_dataset_fail_exit_1(clean_dataset: Path) -> None:
    (clean_dataset / "meta" / "tasks.parquet").unlink()
    runner = CliRunner()
    result = runner.invoke(main, ["validate-dataset", "--dataset", str(clean_dataset)])
    assert result.exit_code == 1, result.output


# ---------------------------------------------------------------------------
# Integration (real data)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_validate_real_dataset() -> None:
    if not REAL_DATASET.is_dir():
        pytest.skip(f"real dataset not present: {REAL_DATASET}")
    report: DatasetValidationReport = validate_dataset(REAL_DATASET)
    report.apply_verdict(strict=False)
    assert report.verdict == "OK", [i.to_dict() for i in report.errors()]
    assert len(report.errors()) == 0

    info = json.loads((REAL_DATASET / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 3
    assert info["total_frames"] == 4934
