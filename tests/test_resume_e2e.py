"""E2E tests for the ``convert --resume`` safe-rerun guard (plan.md P0#3).

P0 scope is the safety guard only (true work-skipping resume is a P1 item):
  - converting into a non-empty output without --resume aborts;
  - --resume on a finalized dataset is a no-op;
  - --resume on a crashed (non-finalized) output cleans + reconverts.

Uses the real HSR bag bagdata/airoa-moma-mcap/235210; skips when bagdata/
or ffmpeg is unavailable.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HSR_CONFIG = PROJECT_ROOT / "configs" / "hsr.yaml"
HSR_BAG = PROJECT_ROOT / "bagdata" / "airoa-moma-mcap" / "235210"


def _require() -> None:
    if not HSR_CONFIG.exists():
        pytest.skip("hsr.yaml not available")
    if not HSR_BAG.exists() or not (HSR_BAG / "metadata.yaml").exists():
        pytest.skip(f"real HSR bag not available at {HSR_BAG}")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")


def _convert_args(out: Path) -> list[str]:
    return [
        "convert",
        "--config",
        str(HSR_CONFIG),
        "--bags",
        str(HSR_BAG),
        "--output",
        str(out),
        "--video-codec",
        "libx264",
        "--max-episodes",
        "1",
    ]


@pytest.mark.integration
class TestResumeGuard:
    def test_overwrite_without_resume_errors(self, tmp_path: Path) -> None:
        _require()
        from bagel.cli import main

        runner = CliRunner()
        out = tmp_path / "ds"
        r1 = runner.invoke(main, _convert_args(out))
        assert r1.exit_code == 0, r1.output
        assert (out / "meta" / "info.json").exists()

        # Second run into the same non-empty dir without --resume must abort.
        r2 = runner.invoke(main, _convert_args(out))
        assert r2.exit_code != 0
        assert "not empty" in r2.output

    def test_resume_noop_on_complete(self, tmp_path: Path) -> None:
        _require()
        from bagel.cli import main

        runner = CliRunner()
        out = tmp_path / "ds"
        assert runner.invoke(main, _convert_args(out)).exit_code == 0
        info = out / "meta" / "info.json"
        before = info.read_bytes()

        r = runner.invoke(main, _convert_args(out) + ["--resume"])
        assert r.exit_code == 0, r.output
        assert "already complete" in r.output
        assert info.read_bytes() == before  # no-op: dataset untouched

    def test_resume_restarts_partial(self, tmp_path: Path) -> None:
        _require()
        from bagel.cli import main

        runner = CliRunner()
        out = tmp_path / "ds"
        # Simulate a crashed run: data/ exists with a stray file, no meta/.
        (out / "data").mkdir(parents=True)
        (out / "data" / "stray.parquet").write_bytes(b"garbage")

        r = runner.invoke(main, _convert_args(out) + ["--resume"])
        assert r.exit_code == 0, r.output
        assert (out / "meta" / "info.json").exists()
        # Partial artifacts were wiped before reconverting.
        assert not (out / "data" / "stray.parquet").exists()
