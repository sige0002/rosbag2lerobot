"""E2E + unit tests for the job summary / progress (⑧, plan.md D-3).

Unit tests exercise :class:`bagel.jobmeta.JobSummary` math with injected wall
time (so ``to_dict`` is deterministic) over a mix of success/failure results.
The integration tests run the real converter with ``--workers 2`` and assert
the ``meta/job_summary.json`` contents, plus ``--json`` (stdout JSON) and
``--quiet`` (no progress bar) behavior on the real bags.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from bagel.cli import main
from bagel.jobmeta import EpisodeResult, JobSummary, dir_bytes

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_BAGS = PROJECT_ROOT / "bagdata" / "airoa-moma-mcap"
HSR_CONFIG = PROJECT_ROOT / "configs" / "hsr.yaml"


# ---------------------------------------------------------------------------
# Unit tests (pure; wall time injected)
# ---------------------------------------------------------------------------


def test_job_summary_math() -> None:
    js = JobSummary()
    js.add(EpisodeResult(0, "/b0", 0, True, 100, 2.0, None))
    js.add(EpisodeResult(1, "/b1", 1, True, 200, 4.0, None))
    js.add(EpisodeResult(2, "/b2", 0, False, 0, 1.0, "boom"))
    js.input_bytes = 1000
    js.output_bytes = 500

    d = js.to_dict(wall_time_s=60.0)
    assert d["n_episodes"] == 3
    assert d["n_success"] == 2
    assert d["n_failed"] == 1
    assert d["total_frames"] == 300
    assert d["wall_time_s"] == 60.0
    # 300 frames / 60 s * 60 = 300 frames/min.
    assert d["frames_per_min"] == pytest.approx(300.0)
    assert d["input_bytes"] == 1000
    assert d["output_bytes"] == 500
    # Per-worker breakdown: worker 0 has eps 0 (success) + 2 (fail), worker 1 ep 1.
    workers = {w["worker"]: w for w in d["workers"]}
    assert workers[0]["n_episodes"] == 2
    assert workers[0]["n_frames"] == 100  # failed ep contributes 0 frames
    assert workers[1]["n_frames"] == 200
    json.dumps(d)


def test_job_summary_zero_wall_time() -> None:
    js = JobSummary()
    js.add(EpisodeResult(0, "/b0", 0, True, 10, 0.0, None))
    d = js.to_dict(wall_time_s=0.0)
    assert d["frames_per_min"] == 0.0


def test_dir_bytes(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.bin").write_bytes(b"y" * 5)
    assert dir_bytes(tmp_path) == 15
    assert dir_bytes(tmp_path / "missing") == 0


# ---------------------------------------------------------------------------
# Integration (real bags)
# ---------------------------------------------------------------------------


def _require_real() -> None:
    if not REAL_BAGS.is_dir():
        pytest.skip(f"real bags not present: {REAL_BAGS}")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")


@pytest.mark.integration
def test_job_summary_real_parallel(tmp_path: Path) -> None:
    _require_real()
    out = tmp_path / "ds"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "convert",
            "--config",
            str(HSR_CONFIG),
            "--bags",
            str(REAL_BAGS),
            "--output",
            str(out),
            "--workers",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output

    summary = json.loads((out / "meta" / "job_summary.json").read_text())
    info = json.loads((out / "meta" / "info.json").read_text())

    assert summary["n_episodes"] == 7
    assert summary["n_success"] == 7
    assert summary["n_failed"] == 0
    assert summary["total_frames"] == info["total_frames"]
    assert summary["frames_per_min"] > 0
    assert summary["output_bytes"] > 0


@pytest.mark.integration
def test_job_summary_json_stdout(tmp_path: Path) -> None:
    _require_real()
    out = tmp_path / "ds"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "convert",
            "--config",
            str(HSR_CONFIG),
            "--bags",
            str(REAL_BAGS),
            "--output",
            str(out),
            "--json",
            "--max-episodes",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    # --json prints the summary dict (indent=2) to stdout; logging goes to
    # stderr, so the whole stdout is a single JSON object.
    payload = json.loads(result.output)
    assert payload["n_success"] == 3
    assert "total_frames" in payload


@pytest.mark.integration
def test_job_summary_quiet_no_bar(tmp_path: Path) -> None:
    _require_real()
    out = tmp_path / "ds"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "convert",
            "--config",
            str(HSR_CONFIG),
            "--bags",
            str(REAL_BAGS),
            "--output",
            str(out),
            "--quiet",
            "--max-episodes",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    # No tqdm progress bar artifacts (the bar renders an "ep" unit + a "%").
    assert "ep/s" not in result.output
    assert "convert:" not in result.output
    # Summary still persisted.
    assert (out / "meta" / "job_summary.json").is_file()
