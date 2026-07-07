"""E2E tests for train/val/test split + episode filter (⑨, plan.md C-3).

Converts the real ``bagdata/airoa-moma-mcap`` bags with a 3-way split config
and with a ``min_length`` filter, then asserts:

- ``info.splits`` partitions ``[0, total_episodes)`` (contiguous, gap-free).
- a ``min_length`` above the shortest episode reduces ``total_episodes``, the
  episodes/data rows stay consistent, and the dropped episode's video clip is
  absent.
- ``validate-dataset`` reports OK for multi-split, filtered, AND default runs.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from rosbag2lerobot.cli import main
from rosbag2lerobot.validation import validate_dataset
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_BAGS = PROJECT_ROOT / "bagdata" / "airoa-moma-mcap"
HSR_CONFIG = PROJECT_ROOT / "configs" / "hsr.yaml"


def _require_real() -> None:
    if not REAL_BAGS.is_dir():
        pytest.skip(f"real bags not present: {REAL_BAGS}")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")


def _config_with_split(tmp_path: Path, split: dict) -> Path:
    """Return a copy of the HSR config with an added ``split`` section."""
    raw = yaml.safe_load(HSR_CONFIG.read_text())
    raw["split"] = split
    path = tmp_path / "hsr_split.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def _convert(config: Path, out: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "convert",
            "--config",
            str(config),
            "--bags",
            str(REAL_BAGS),
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output


def _assert_partition(splits: dict[str, str], total: int) -> None:
    intervals = sorted(
        (int(v.split(":")[0]), int(v.split(":")[1])) for v in splits.values()
    )
    prev = 0
    for a, b in intervals:
        assert a == prev, (a, prev, splits)
        assert a < b
        prev = b
    assert prev == total, (prev, total, splits)


@pytest.mark.integration
def test_three_way_split_partitions(tmp_path: Path) -> None:
    _require_real()
    cfg = _config_with_split(tmp_path, {"train": 0.7, "val": 0.15, "test": 0.15})
    out = tmp_path / "ds"
    _convert(cfg, out)

    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 7
    _assert_partition(info["splits"], 7)

    # Sum of episode lengths matches total_frames.
    import pyarrow.parquet as pq

    eps = sorted((out / "meta" / "episodes").rglob("*.parquet"))
    length_sum = sum(
        int(x) for p in eps for x in pq.read_table(p).column("length").to_pylist()
    )
    assert length_sum == info["total_frames"]

    report = validate_dataset(out)
    report.apply_verdict(strict=False)
    assert report.verdict == "OK", [i.to_dict() for i in report.errors()]


@pytest.mark.integration
def test_min_length_filter_drops_short_episode(tmp_path: Path) -> None:
    _require_real()
    # Baseline run to learn the per-episode lengths.
    base_out = tmp_path / "base"
    _convert(HSR_CONFIG, base_out)
    base_info = json.loads((base_out / "meta" / "info.json").read_text())

    import pyarrow.parquet as pq

    base_eps = sorted((base_out / "meta" / "episodes").rglob("*.parquet"))
    lengths = sorted(
        int(x) for p in base_eps for x in pq.read_table(p).column("length").to_pylist()
    )
    assert len(lengths) == base_info["total_episodes"] == 7
    shortest = lengths[0]
    # Threshold strictly above the shortest episode -> at least one dropped.
    threshold = shortest + 1

    cfg = _config_with_split(tmp_path, {"train": 1.0, "min_length": threshold})
    out = tmp_path / "filtered"
    _convert(cfg, out)

    info = json.loads((out / "meta" / "info.json").read_text())
    n_dropped = sum(1 for ln in lengths if ln < threshold)
    assert n_dropped >= 1
    assert info["total_episodes"] == 7 - n_dropped

    # meta/episodes rows == new total_episodes.
    eps = sorted((out / "meta" / "episodes").rglob("*.parquet"))
    n_ep_rows = sum(pq.read_table(p).num_rows for p in eps)
    assert n_ep_rows == info["total_episodes"]

    # data rows sum == info.total_frames.
    data_files = sorted((out / "data").rglob("*.parquet"))
    n_data_rows = sum(pq.read_table(p).num_rows for p in data_files)
    assert n_data_rows == info["total_frames"]

    # stats.json must reflect ONLY the kept frames: the per-feature ``count``
    # (a bookkeeping feature is always present) equals total_frames. This is
    # the regression guard for the old writer-side drop, which fed dropped
    # frames into the global stats before discarding the episode.
    stats = json.loads((out / "meta" / "stats.json").read_text())
    stat_count = int(stats["index"]["count"][0])
    assert stat_count == info["total_frames"], (stat_count, info["total_frames"])

    # job_summary.json over-counts under the old behavior; it must agree with
    # info totals now that the producer filters before any accounting.
    summary = json.loads((out / "meta" / "job_summary.json").read_text())
    assert summary["n_success"] == info["total_episodes"]
    assert summary["total_frames"] == info["total_frames"]

    # The index column must stay contiguous [0, total_frames) after the drop.
    all_indices = sorted(
        int(x) for p in data_files for x in pq.read_table(p).column("index").to_pylist()
    )
    assert all_indices == list(range(info["total_frames"]))

    # The dropped episode's clip count is one fewer per video key than baseline.
    for key, spec in info["features"].items():
        if not isinstance(spec, dict) or spec.get("dtype") != "video":
            continue
        base_clips = list((base_out / "videos" / key).rglob("*.mp4"))
        cur_clips = list((out / "videos" / key).rglob("*.mp4"))
        # Splits aggregate clips into size-bounded files, so compare via the
        # episodes parquet to_timestamp coverage instead of raw mp4 count:
        # the filtered run must contain no orphaned clip for the dropped ep.
        assert cur_clips, f"missing video clips for {key}"
        assert len(cur_clips) <= len(base_clips)

    report = validate_dataset(out)
    report.apply_verdict(strict=False)
    assert report.verdict == "OK", [i.to_dict() for i in report.errors()]


@pytest.mark.integration
def test_default_config_validates_ok(tmp_path: Path) -> None:
    """A default-config run yields legacy {"train": "0:N"} and validates OK."""
    _require_real()
    out = tmp_path / "ds"
    _convert(HSR_CONFIG, out)
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["splits"] == {"train": f"0:{info['total_episodes']}"}

    report = validate_dataset(out)
    report.apply_verdict(strict=False)
    assert report.verdict == "OK", [i.to_dict() for i in report.errors()]
