"""Tests for :mod:`bagel.audit` (F2 audit-timestamps).

Mirrors the style of :mod:`tests.test_writer` and reuses the
``TestTimestampRoundingAcrossEpisodes`` fixture pattern so the audit runs
against *real* writer output rather than synthetic parquet, which makes the
test an end-to-end drift-regression witness.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from bagel.audit import (
    AuditReport,
    BoundaryError,
    audit_episode_timestamps,
)
from bagel.cli import main
from bagel.writer import DatasetWriter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def features_with_video() -> dict:
    """Feature spec identical to tests/test_writer.py's fixture.

    Duplicated (not imported) so this file can run standalone without
    pulling the whole test_writer collection into the import graph.
    """
    return {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "observation.state": {
            "dtype": "float32",
            "shape": [2],
            "names": {"axes": ["j1", "j2"]},
        },
        "action": {
            "dtype": "float32",
            "shape": [2],
            "names": {"axes": ["j1", "j2"]},
        },
        "observation.images.cam": {
            "dtype": "video",
            "shape": [64, 64, 3],
            "names": ["height", "width", "channels"],
        },
    }


def _build_episodes_parquet(
    tmp_path: Path,
    features_with_video: dict,
    n_episodes: int,
    fps: int = 30,
    ep_len: int = 3,
) -> Path:
    """Drive DatasetWriter's timestamp bookkeeping for ``n_episodes``.

    Returns the dataset root directory, not the parquet path, so callers can
    pass it straight to :func:`audit_episode_timestamps`.

    Rather than invoking ffmpeg we emulate the writer's internal
    "register an episode segment" path (the same one exercised by
    ``TestTimestampRoundingAcrossEpisodes``) and then write the episodes
    parquet manually via the writer's private helper. This keeps the test
    fast and deterministic while covering the exact arithmetic that
    ``audit-timestamps`` guards.
    """
    writer = DatasetWriter(
        tmp_path,
        {"robot_type": "r"},
        features_with_video,
        fps=fps,
    )
    vkey = "observation.images.cam"

    for ep_idx in range(n_episodes):
        vm = writer._register_episode_video(vkey, ep_len)

        ep_meta = {
            "episode_index": ep_idx,
            "length": ep_len,
            "tasks": ["task"],
            "dataset_from_index": ep_idx * ep_len,
            "dataset_to_index": (ep_idx + 1) * ep_len,
            "data/chunk_index": 0,
            "data/file_index": 0,
            f"videos/{vkey}/chunk_index": vm["chunk_index"],
            f"videos/{vkey}/file_index": vm["file_index"],
            f"videos/{vkey}/from_timestamp": vm["from_timestamp"],
            f"videos/{vkey}/to_timestamp": vm["to_timestamp"],
        }
        writer._episodes_meta.append(ep_meta)

    # Invoke the private writer helper to drop the parquet(s) to disk.
    writer._write_episodes_parquet()
    return tmp_path


# ---------------------------------------------------------------------------
# Happy-path tests (writer output passes audit)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_episodes", [10, 50, 200])
def test_audit_passes_on_clean_writer_output(
    tmp_path: Path,
    features_with_video: dict,
    n_episodes: int,
) -> None:
    """Real writer output over 10/50/200 ep should audit with zero drift."""
    dataset_dir = _build_episodes_parquet(
        tmp_path,
        features_with_video,
        n_episodes=n_episodes,
    )

    report = audit_episode_timestamps(dataset_dir, max_drift_us=1.0)

    assert isinstance(report, AuditReport)
    assert report.verdict == "OK"
    assert report.exit_code == 0
    assert len(report.results) == 1
    (res,) = report.results
    assert res.video_key == "observation.images.cam"
    assert res.n_episodes == n_episodes
    assert res.max_drift_us < 1.0
    assert res.boundary_errors == []


def test_audit_report_to_dict_is_json_serialisable(
    tmp_path: Path,
    features_with_video: dict,
) -> None:
    """``AuditReport.to_dict()`` must yield a JSON-compatible dict."""
    import json

    dataset_dir = _build_episodes_parquet(
        tmp_path,
        features_with_video,
        n_episodes=5,
    )
    report = audit_episode_timestamps(dataset_dir)
    # Must round-trip through json.dumps without raising.
    _ = json.dumps(report.to_dict())


def test_audit_with_explicit_video_key(
    tmp_path: Path,
    features_with_video: dict,
) -> None:
    """Passing ``video_keys=[...]`` restricts the audit scope."""
    dataset_dir = _build_episodes_parquet(
        tmp_path,
        features_with_video,
        n_episodes=5,
    )
    report = audit_episode_timestamps(
        dataset_dir,
        video_keys=["observation.images.cam"],
    )
    assert [r.video_key for r in report.results] == ["observation.images.cam"]


def test_audit_rejects_unknown_video_key(
    tmp_path: Path,
    features_with_video: dict,
) -> None:
    """Requesting a key not present in the parquet raises ValueError."""
    dataset_dir = _build_episodes_parquet(
        tmp_path,
        features_with_video,
        n_episodes=3,
    )
    with pytest.raises(ValueError, match="not found"):
        audit_episode_timestamps(dataset_dir, video_keys=["does.not.exist"])


def test_audit_missing_episodes_dir(tmp_path: Path) -> None:
    """An empty tmp_path has no episodes dir → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        audit_episode_timestamps(tmp_path)


# ---------------------------------------------------------------------------
# Tamper tests (deliberately corrupt the parquet and expect FAIL)
# ---------------------------------------------------------------------------


def _tamper_to_timestamp(parquet_path: Path, vkey: str, row: int, bump: float) -> None:
    """Load parquet, add ``bump`` to ``to_timestamp[row]`` for ``vkey``, save."""
    import pyarrow as pa

    table = pq.read_table(parquet_path)
    col_name = f"videos/{vkey}/to_timestamp"
    series = table.column(col_name).to_pylist()
    series[row] = series[row] + bump

    new_col = pa.array(series, type=pa.float64())
    col_idx = table.column_names.index(col_name)
    table = table.set_column(col_idx, col_name, new_col)
    pq.write_table(table, parquet_path, compression="snappy")


def test_audit_flags_tampered_to_timestamp(
    tmp_path: Path,
    features_with_video: dict,
) -> None:
    """Bumping one ``to_timestamp`` by +1ms must be caught as a boundary error.

    The tampered row's ``to_timestamp`` (say row=3) disagrees with the next
    row's ``from_timestamp`` by 1000 us; audit should surface the next row
    (row=4) in ``boundary_errors`` and return verdict FAIL.
    """
    dataset_dir = _build_episodes_parquet(
        tmp_path,
        features_with_video,
        n_episodes=10,
    )
    parquet_paths = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    assert parquet_paths, "episodes parquet should have been written"

    _tamper_to_timestamp(
        parquet_paths[0],
        vkey="observation.images.cam",
        row=3,
        bump=0.001,  # 1 ms
    )

    report = audit_episode_timestamps(dataset_dir, max_drift_us=1.0)
    assert report.verdict == "FAIL"
    assert report.exit_code == 1
    (res,) = report.results
    assert res.verdict == "FAIL"
    assert len(res.boundary_errors) >= 1
    # The offending row is ep 4 (from_ts now lags prev to_ts by 1 ms).
    err = res.boundary_errors[0]
    assert isinstance(err, BoundaryError)
    assert err.episode_index == 4
    assert abs(err.delta_us + 1000.0) < 1.0  # actual < expected => negative


def test_audit_flags_orphan_zero_reset(
    tmp_path: Path,
    features_with_video: dict,
) -> None:
    """A mid-file ``from_timestamp`` reset to 0.0 without file boundary FAILs."""
    import pyarrow as pa

    dataset_dir = _build_episodes_parquet(
        tmp_path,
        features_with_video,
        n_episodes=6,
    )
    parquet_paths = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    table = pq.read_table(parquet_paths[0])
    vkey = "observation.images.cam"
    col_name = f"videos/{vkey}/from_timestamp"
    series = table.column(col_name).to_pylist()
    # Force a mid-file reset: ep index 3 starts at 0.0 despite prev to_ts > 0.
    series[3] = 0.0
    new_col = pa.array(series, type=pa.float64())
    col_idx = table.column_names.index(col_name)
    table = table.set_column(col_idx, col_name, new_col)
    pq.write_table(table, parquet_paths[0], compression="snappy")

    report = audit_episode_timestamps(dataset_dir, max_drift_us=1.0)
    assert report.verdict == "FAIL"
    assert any(e.episode_index == 3 for e in report.results[0].boundary_errors)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_audit_timestamps_ok(
    tmp_path: Path,
    features_with_video: dict,
) -> None:
    """CliRunner invocation on clean writer output should exit 0."""
    dataset_dir = _build_episodes_parquet(
        tmp_path,
        features_with_video,
        n_episodes=8,
    )
    json_out = tmp_path / "audit.json"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "audit-timestamps",
            "--dataset",
            str(dataset_dir),
            "--json-out",
            str(json_out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json_out.is_file()

    import json as _json

    payload = _json.loads(json_out.read_text())
    assert payload["verdict"] == "OK"
    assert payload["exit_code"] == 0
    assert payload["video_keys"] == ["observation.images.cam"]


def test_cli_audit_timestamps_fail(
    tmp_path: Path,
    features_with_video: dict,
) -> None:
    """Tampered parquet should make the CLI exit with status 1."""
    dataset_dir = _build_episodes_parquet(
        tmp_path,
        features_with_video,
        n_episodes=8,
    )
    parquet_paths = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    _tamper_to_timestamp(
        parquet_paths[0],
        vkey="observation.images.cam",
        row=2,
        bump=0.010,  # 10 ms  -> clearly > 1 us threshold
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["audit-timestamps", "--dataset", str(dataset_dir)],
    )
    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output


def test_cli_audit_timestamps_video_key_filter(
    tmp_path: Path,
    features_with_video: dict,
) -> None:
    """``--video-key`` limits the audit to the named key only."""
    dataset_dir = _build_episodes_parquet(
        tmp_path,
        features_with_video,
        n_episodes=3,
    )
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "audit-timestamps",
            "--dataset",
            str(dataset_dir),
            "--video-key",
            "observation.images.cam",
        ],
    )
    assert result.exit_code == 0, result.output


def test_cli_audit_timestamps_bad_dataset(tmp_path: Path) -> None:
    """Non-existent dataset → CLI exits 2 (reserved for setup errors)."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["audit-timestamps", "--dataset", str(tmp_path)],
    )
    assert result.exit_code == 2, result.output


def test_cli_audit_timestamps_help() -> None:
    """``bagel audit-timestamps --help`` must succeed."""
    runner = CliRunner()
    result = runner.invoke(main, ["audit-timestamps", "--help"])
    assert result.exit_code == 0
    assert "audit" in result.output.lower()
    assert "--dataset" in result.output
    assert "--max-drift-us" in result.output
    assert "--json-out" in result.output
    assert "--video-key" in result.output
