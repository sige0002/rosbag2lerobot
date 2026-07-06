"""Tests for bagel.writer module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from bagel.task_spec import SubtaskSpan
from bagel.writer import (
    _CHUNKS_SIZE,
    _CODEC_LABEL_MAP,
    _DATA_FILES_SIZE_IN_MB,
    _TIMESTAMP_ROUND_DECIMALS,
    _VIDEO_FILES_SIZE_IN_MB,
    DatasetWriter,
    _advance_chunk_file,
    _build_codec_args,
    _codec_label,
)


class TestAdvanceChunkFile:
    def test_increment_within_chunk(self) -> None:
        assert _advance_chunk_file(0, 0) == (0, 1)
        assert _advance_chunk_file(0, 5) == (0, 6)

    def test_rollover_to_next_chunk(self) -> None:
        assert _advance_chunk_file(0, _CHUNKS_SIZE - 1) == (1, 0)
        assert _advance_chunk_file(3, _CHUNKS_SIZE - 1) == (4, 0)


@pytest.fixture
def simple_features() -> dict:
    return {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "observation.state": {
            "dtype": "float32",
            "shape": [3],
            "names": {"axes": ["j1", "j2", "j3"]},
        },
        "action": {
            "dtype": "float32",
            "shape": [3],
            "names": {"axes": ["j1", "j2", "j3"]},
        },
    }


@pytest.fixture
def features_with_video() -> dict:
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


class TestDatasetWriterNumericOnly:
    """Test writer with numeric-only features (no video encoding)."""

    def test_single_episode(self, tmp_path: Path, simple_features: dict) -> None:
        config = {"robot_type": "test_robot"}
        writer = DatasetWriter(tmp_path, config, simple_features, fps=10)

        for i in range(5):
            writer.add_frame(
                {
                    "observation.state": np.array(
                        [float(i), float(i + 1), float(i + 2)]
                    ),
                    "action": np.array(
                        [float(i) * 0.1, float(i) * 0.2, float(i) * 0.3]
                    ),
                    "task": "pick_up_object",
                }
            )
        writer.save_episode()
        writer.finalize()

        # Check directory structure
        assert (tmp_path / "data" / "chunk-000" / "file-000.parquet").exists()
        assert (tmp_path / "meta" / "info.json").exists()
        assert (tmp_path / "meta" / "stats.json").exists()
        assert (tmp_path / "meta" / "tasks.parquet").exists()
        assert (
            tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        ).exists()

        # Validate info.json
        with open(tmp_path / "meta" / "info.json") as f:
            info = json.load(f)
        assert info["codebase_version"] == "v3.0"
        assert info["robot_type"] == "test_robot"
        assert info["total_episodes"] == 1
        assert info["total_frames"] == 5
        assert info["total_tasks"] == 1
        assert info["fps"] == 10
        assert info["splits"] == {"train": "0:1"}

        # Validate data parquet
        table = pq.read_table(tmp_path / "data" / "chunk-000" / "file-000.parquet")
        assert table.num_rows == 5
        assert "index" in table.column_names
        assert "timestamp" in table.column_names
        assert "observation.state" in table.column_names
        assert "action" in table.column_names

        # Check index values
        indices = table.column("index").to_pylist()
        assert indices == [0, 1, 2, 3, 4]

        # Check timestamps
        timestamps = table.column("timestamp").to_pylist()
        assert abs(timestamps[0] - 0.0) < 1e-5
        assert abs(timestamps[1] - 0.1) < 1e-5

        # Validate tasks parquet (lerobot-record: task strings are the pandas
        # Index named "task"; task_index is the only regular column)
        tasks_df = pd.read_parquet(tmp_path / "meta" / "tasks.parquet")
        assert tasks_df.index.name == "task"
        assert list(tasks_df.columns) == ["task_index"]
        assert tasks_df["task_index"].dtype.name == "int64"
        assert tasks_df.index.tolist() == ["pick_up_object"]

    def test_multiple_episodes(self, tmp_path: Path, simple_features: dict) -> None:
        config = {"robot_type": "test_robot"}
        writer = DatasetWriter(tmp_path, config, simple_features, fps=30)

        for ep in range(3):
            for i in range(10):
                writer.add_frame(
                    {
                        "observation.state": np.ones(3) * ep,
                        "action": np.zeros(3),
                        "task": f"task_{ep}",
                    }
                )
            writer.save_episode()
        writer.finalize()

        with open(tmp_path / "meta" / "info.json") as f:
            info = json.load(f)
        assert info["total_episodes"] == 3
        assert info["total_frames"] == 30
        assert info["total_tasks"] == 3

        # Check stats
        with open(tmp_path / "meta" / "stats.json") as f:
            stats = json.load(f)
        assert "observation.state" in stats
        assert "action" in stats

    def test_auto_finalize_unsaved_episode(
        self, tmp_path: Path, simple_features: dict
    ) -> None:
        """finalize() should flush the last unsaved episode."""
        config = {"robot_type": "r"}
        writer = DatasetWriter(tmp_path, config, simple_features, fps=10)
        writer.add_frame(
            {
                "observation.state": np.array([1.0, 2.0, 3.0]),
                "action": np.array([0.0, 0.0, 0.0]),
            }
        )
        # Don't call save_episode — finalize should handle it
        writer.finalize()

        with open(tmp_path / "meta" / "info.json") as f:
            info = json.load(f)
        assert info["total_episodes"] == 1
        assert info["total_frames"] == 1


class TestDatasetWriterWithVideo:
    """Test writer with video features (requires ffmpeg)."""

    @pytest.fixture(autouse=True)
    def _check_ffmpeg(self) -> None:
        import shutil

        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not available")

    def test_video_encoding(self, tmp_path: Path, features_with_video: dict) -> None:
        config = {"robot_type": "cam_robot"}
        writer = DatasetWriter(tmp_path, config, features_with_video, fps=10)

        for i in range(10):
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            writer.add_frame(
                {
                    "observation.state": np.array([float(i), float(i)]),
                    "action": np.array([0.0, 0.0]),
                    "observation.images.cam": img,
                }
            )
        writer.save_episode()
        writer.finalize()

        # Video file should exist
        video_path = (
            tmp_path
            / "videos"
            / "observation.images.cam"
            / "chunk-000"
            / "file-000.mp4"
        )
        assert video_path.exists()
        assert video_path.stat().st_size > 0

        # info.json should have video size
        with open(tmp_path / "meta" / "info.json") as f:
            info = json.load(f)
        assert info["video_files_size_in_mb"] >= 0

        # Episodes metadata should have video fields
        ep_table = pq.read_table(
            tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        )
        col_names = ep_table.column_names
        assert "videos/observation.images.cam/chunk_index" in col_names
        assert "videos/observation.images.cam/to_timestamp" in col_names

    def test_stats_include_video(
        self, tmp_path: Path, features_with_video: dict
    ) -> None:
        config = {"robot_type": "r"}
        writer = DatasetWriter(tmp_path, config, features_with_video, fps=10)

        for _ in range(5):
            img = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
            writer.add_frame(
                {
                    "observation.state": np.array([1.0, 2.0]),
                    "action": np.array([0.0, 0.0]),
                    "observation.images.cam": img,
                }
            )
        writer.save_episode()
        writer.finalize()

        with open(tmp_path / "meta" / "stats.json") as f:
            stats = json.load(f)
        assert "observation.images.cam" in stats
        cam = stats["observation.images.cam"]
        # Image stats are nested to (C, 1, 1) per channel (LeRobot v3.0).
        assert len(cam["mean"]) == 3
        for ch in range(3):
            assert np.array(cam["mean"][ch]).shape == (1, 1)
            # 128/255 ~ 0.502
            assert abs(cam["mean"][ch][0][0] - 128.0 / 255.0) < 0.01
        # count is a single-element list (shape (1,)) holding the frame count.
        assert cam["count"] == [5]
        # Numeric feature count is also a single-element list, not per-dim.
        assert stats["observation.state"]["count"] == [5]


class TestVideoFilePermissions:
    """The produced mp4 must be world rw (0o666).

    The streaming encoder writes the target mp4 directly (0644/0664 per
    umask); ``_close_video_encoder`` must normalise it to 0666 so other
    users / containers can read the dataset. Covered for both a single-
    episode file and a file aggregating multiple episodes.
    """

    @pytest.fixture(autouse=True)
    def _check_ffmpeg(self) -> None:
        import shutil

        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not available")

    @staticmethod
    def _mode(path: Path) -> int:
        import stat

        return stat.S_IMODE(path.stat().st_mode)

    def test_single_episode_file_is_0666(
        self, tmp_path: Path, features_with_video: dict
    ) -> None:
        """One episode → one output mp4."""
        writer = DatasetWriter(
            tmp_path, {"robot_type": "r"}, features_with_video, fps=10
        )
        for i in range(5):
            img = Image.fromarray(np.full((64, 64, 3), 100 + i, dtype=np.uint8))
            writer.add_frame(
                {
                    "observation.state": np.array([float(i), 0.0]),
                    "action": np.zeros(2),
                    "observation.images.cam": img,
                }
            )
        writer.save_episode()
        writer.finalize()

        video_path = (
            tmp_path
            / "videos"
            / "observation.images.cam"
            / "chunk-000"
            / "file-000.mp4"
        )
        assert video_path.exists()
        assert self._mode(video_path) == 0o666

    def test_multi_episode_aggregated_file_is_0666(
        self, tmp_path: Path, features_with_video: dict
    ) -> None:
        """Multiple episodes streamed into one aggregated output file."""
        writer = DatasetWriter(
            tmp_path, {"robot_type": "r"}, features_with_video, fps=10
        )
        for ep in range(3):
            for i in range(4):
                img = Image.fromarray(
                    np.full((64, 64, 3), 50 + ep * 10 + i, dtype=np.uint8)
                )
                writer.add_frame(
                    {
                        "observation.state": np.array([float(i), 0.0]),
                        "action": np.zeros(2),
                        "observation.images.cam": img,
                    }
                )
            writer.save_episode()
        writer.finalize()

        # All three episodes are far below the size threshold, so they land in
        # a single file-000.mp4 written by one continuous encoder.
        chunk_dir = tmp_path / "videos" / "observation.images.cam" / "chunk-000"
        mp4s = sorted(chunk_dir.glob("*.mp4"))
        assert [p.name for p in mp4s] == ["file-000.mp4"]
        assert self._mode(mp4s[0]) == 0o666


class TestEpisodeStatsSchema:
    """Verify the meta/episodes per-episode stats columns and self-pointers
    match the lerobot-record v3.0 layout (stats/<feature>/<stat>,
    meta/episodes/{chunk,file}_index)."""

    def _write_two_episodes(self, tmp_path: Path, features: dict) -> Path:
        writer = DatasetWriter(tmp_path, {"robot_type": "r"}, features, fps=10)
        for ep, task in enumerate(["task-a", "task-b"]):
            for i in range(5):
                img = Image.fromarray(np.full((64, 64, 3), 100 + i, dtype=np.uint8))
                writer.add_frame(
                    {
                        "observation.state": np.array([float(i), float(i) * 2]),
                        "action": np.array([0.0, 1.0]),
                        "observation.images.cam": img,
                        "task": task,
                    }
                )
            writer.save_episode()
        writer.finalize()
        return tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"

    def test_stats_columns_and_types(
        self, tmp_path: Path, features_with_video: dict
    ) -> None:
        table = pq.read_table(self._write_two_episodes(tmp_path, features_with_video))
        assert table.num_rows == 2
        sch = table.schema

        # Numeric float feature: per-dim list<double>, count single int64.
        assert sch.field("stats/observation.state/min").type == pa.list_(pa.float64())
        assert sch.field("stats/observation.state/count").type == pa.list_(pa.int64())
        # Int bookkeeping: min/max int64, mean/std double.
        assert sch.field("stats/index/min").type == pa.list_(pa.int64())
        assert sch.field("stats/index/mean").type == pa.list_(pa.float64())
        # timestamp is float32 -> double.
        assert sch.field("stats/timestamp/min").type == pa.list_(pa.float64())
        # Image feature: nested [C, 1, 1] double.
        assert sch.field("stats/observation.images.cam/min").type == pa.list_(
            pa.list_(pa.list_(pa.float64()))
        )
        assert sch.field("stats/observation.images.cam/count").type == pa.list_(
            pa.int64()
        )

        # Self-referential pointers to the episodes parquet file.
        assert "meta/episodes/chunk_index" in table.column_names
        assert "meta/episodes/file_index" in table.column_names

        # count reflects per-episode frame count (single element).
        row0 = table.slice(0, 1).to_pylist()[0]
        assert row0["stats/observation.state/count"] == [5]
        assert len(row0["stats/observation.state/min"]) == 2  # per dimension
        assert len(row0["stats/observation.images.cam/min"]) == 3  # per channel

    def test_tasks_named_index(self, tmp_path: Path, features_with_video: dict) -> None:
        self._write_two_episodes(tmp_path, features_with_video)
        df = pd.read_parquet(tmp_path / "meta" / "tasks.parquet")
        assert df.index.name == "task"
        assert list(df.columns) == ["task_index"]
        assert sorted(df.index.tolist()) == ["task-a", "task-b"]


class TestSizeBasedChunking:
    """Verify that multiple episodes aggregate into a single parquet file
    until the size threshold is crossed, and that ``info.json`` records
    the configured thresholds."""

    def test_multiple_episodes_share_data_file(
        self, tmp_path: Path, simple_features: dict
    ) -> None:
        """Small episodes (far below 100MB) should all land in file-000."""
        config = {"robot_type": "r"}
        writer = DatasetWriter(tmp_path, config, simple_features, fps=10)

        for ep in range(4):
            for i in range(3):
                writer.add_frame(
                    {
                        "observation.state": np.array([float(ep), float(i), 0.0]),
                        "action": np.zeros(3),
                    }
                )
            writer.save_episode()
        writer.finalize()

        data_dir = tmp_path / "data" / "chunk-000"
        files = sorted(p.name for p in data_dir.glob("*.parquet"))
        assert files == ["file-000.parquet"], (
            f"Expected a single aggregated data file, got: {files}"
        )

        # Episodes metadata should be a single grouped parquet too.
        ep_files = sorted(
            p.name
            for p in (tmp_path / "meta" / "episodes" / "chunk-000").glob("*.parquet")
        )
        assert ep_files == ["file-000.parquet"]
        ep_table = pq.read_table(
            tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        )
        assert ep_table.num_rows == 4
        assert ep_table.column("episode_index").to_pylist() == [0, 1, 2, 3]
        assert ep_table.column("data/chunk_index").to_pylist() == [0, 0, 0, 0]
        assert ep_table.column("data/file_index").to_pylist() == [0, 0, 0, 0]

        # The data parquet itself should contain all episodes' rows.
        data_table = pq.read_table(data_dir / "file-000.parquet")
        assert data_table.num_rows == 12

    def test_info_json_records_size_thresholds(
        self, tmp_path: Path, simple_features: dict
    ) -> None:
        writer = DatasetWriter(tmp_path, {"robot_type": "r"}, simple_features, fps=10)
        writer.add_frame(
            {
                "observation.state": np.array([1.0, 2.0, 3.0]),
                "action": np.zeros(3),
            }
        )
        writer.save_episode()
        writer.finalize()

        with open(tmp_path / "meta" / "info.json") as f:
            info = json.load(f)
        assert info["data_files_size_in_mb"] == _DATA_FILES_SIZE_IN_MB
        assert info["video_files_size_in_mb"] == _VIDEO_FILES_SIZE_IN_MB
        assert info["chunks_size"] == _CHUNKS_SIZE

    def test_data_file_rotates_when_size_exceeded(
        self, tmp_path: Path, simple_features: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a small threshold, successive episodes rotate to new files."""
        import bagel.writer as writer_mod

        # Force rotation after the very first episode by shrinking the threshold.
        monkeypatch.setattr(writer_mod, "_DATA_FILES_SIZE_IN_MB", 0)

        writer = DatasetWriter(tmp_path, {"robot_type": "r"}, simple_features, fps=10)
        for ep in range(3):
            for _ in range(2):
                writer.add_frame(
                    {
                        "observation.state": np.ones(3) * ep,
                        "action": np.zeros(3),
                    }
                )
            writer.save_episode()
        writer.finalize()

        files = sorted(
            p.name for p in (tmp_path / "data" / "chunk-000").glob("*.parquet")
        )
        assert files == [
            "file-000.parquet",
            "file-001.parquet",
            "file-002.parquet",
        ]


class TestSaveEpisodeEdgeCases:
    def test_empty_episode_warning(self, tmp_path: Path, simple_features: dict) -> None:
        writer = DatasetWriter(tmp_path, {"robot_type": "r"}, simple_features, fps=10)
        # Should not raise, just log warning
        writer.save_episode()
        writer.finalize()
        with open(tmp_path / "meta" / "info.json") as f:
            info = json.load(f)
        assert info["total_episodes"] == 0
        assert info["total_frames"] == 0

    def test_default_task(self, tmp_path: Path, simple_features: dict) -> None:
        writer = DatasetWriter(tmp_path, {"robot_type": "r"}, simple_features, fps=10)
        writer.add_frame(
            {
                "observation.state": np.array([1.0, 2.0, 3.0]),
                "action": np.array([0.0, 0.0, 0.0]),
                # No "task" key — should default
            }
        )
        writer.save_episode()
        writer.finalize()

        tasks_df = pd.read_parquet(tmp_path / "meta" / "tasks.parquet")
        assert list(tasks_df.columns) == ["task_index"]
        assert tasks_df.index.tolist() == ["default_task"]


# ============================================================================
# T1 regression: 50-episode cumulative rounding stability
# ============================================================================


class TestTimestampRoundingAcrossEpisodes:
    """T1 / LeRobot PR #3239: verify that per-episode from/to timestamps are
    rounded to microsecond precision and the cumulative offset does not drift
    due to float accumulation across many episodes."""

    def test_timestamp_rounded_at_microsecond_precision_over_50_episodes(
        self, tmp_path: Path, features_with_video: dict
    ) -> None:
        """All 50 episodes produce timestamps exactly representable at 1e-6
        resolution, and the running accumulator stays tight to the expected
        50 * ep_duration value.
        """
        fps = 30
        writer = DatasetWriter(
            tmp_path,
            {"robot_type": "r"},
            features_with_video,
            fps=fps,
        )
        vkey = "observation.images.cam"

        ep_len = 3  # 3 frames @ 30fps => ep_duration = 0.1s (exactly round-trippable)
        ep_duration_expected = round(ep_len / fps, _TIMESTAMP_ROUND_DECIMALS)

        from_ts_values: list[float] = []
        to_ts_values: list[float] = []

        # ``_register_episode_video`` performs the timestamp arithmetic in
        # isolation (no encoder is open, so no ffmpeg invocation and no size
        # rotation happens here).
        for _ in range(50):
            meta = writer._register_episode_video(vkey, ep_len)
            from_ts_values.append(meta["from_timestamp"])
            to_ts_values.append(meta["to_timestamp"])

        # Every returned value must be invariant under round(x, 6): i.e. the
        # float already encodes exactly 6 decimal places with no sub-uS noise.
        for x in from_ts_values + to_ts_values:
            assert abs(x - round(x, _TIMESTAMP_ROUND_DECIMALS)) < 1e-9, (
                f"Value {x!r} carries sub-microsecond float noise"
            )

        # The running accumulator must match the pure-integer expectation to
        # within 1e-6 (the rounding grain). Without rounding, floating-point
        # accumulation drifts by O(1e-14) per step * 50 ~ O(1e-12); rounding
        # deliberately snaps each step to the 1e-6 grid so this assertion is
        # the round-off vs. accumulation-drift witness.
        expected_final = 50 * ep_duration_expected
        actual_final = writer._video_file_duration[vkey]
        assert abs(actual_final - expected_final) < 1e-6, (
            f"Accumulator drifted: expected ~{expected_final}, got {actual_final}"
        )

        # Spot-check monotonicity and the first/last boundaries.
        assert from_ts_values[0] == 0.0
        assert to_ts_values[0] == ep_duration_expected
        for i in range(1, 50):
            assert from_ts_values[i] == to_ts_values[i - 1], (
                f"Ep{i} from_ts {from_ts_values[i]} != prev to_ts {to_ts_values[i - 1]}"
            )


# ============================================================================
# T2 regression: .staging directory is not left behind
# ============================================================================


class TestStagingDirectoryCleanup:
    """T2: both ``.staging/videos`` and its parent ``.staging`` must be gone
    after ``finalize()`` returns. Leaving these around pollutes the dataset
    output and breaks downstream tools that enumerate the top-level tree.
    """

    def test_staging_directory_removed_after_finalize(
        self, tmp_path: Path, simple_features: dict
    ) -> None:
        writer = DatasetWriter(
            tmp_path,
            {"robot_type": "r"},
            simple_features,
            fps=10,
        )
        for i in range(3):
            writer.add_frame(
                {
                    "observation.state": np.array([float(i), 0.0, 0.0]),
                    "action": np.zeros(3),
                }
            )
        writer.save_episode()
        writer.finalize()

        staging_parent = tmp_path / ".staging"
        staging_videos = tmp_path / ".staging" / "videos"
        assert not staging_videos.exists(), (
            f"{staging_videos} should have been removed by finalize()"
        )
        assert not staging_parent.exists(), (
            f"{staging_parent} should have been removed by finalize()"
        )


# ============================================================================
# T3 regression: codec-specific ffmpeg subprocess argument building
# ============================================================================


class TestBuildCodecArgs:
    """T3: per-codec defaults and overrides for preset / crf / cq / threads."""

    def test_libx264_uses_crf_and_threads(self) -> None:
        args = _build_codec_args("libx264", None, None)
        assert "-c:v" in args
        assert args[args.index("-c:v") + 1] == "libx264"
        assert "-crf" in args
        assert args[args.index("-crf") + 1] == "23"
        assert "-threads" in args
        assert args[args.index("-threads") + 1] == "0"

    def test_libsvtav1_uses_crf_and_threads(self) -> None:
        args = _build_codec_args("libsvtav1", None, None)
        assert args[args.index("-c:v") + 1] == "libsvtav1"
        assert "-crf" in args
        assert args[args.index("-crf") + 1] == "30"
        assert "-threads" in args
        assert args[args.index("-threads") + 1] == "0"

    def test_h264_nvenc_uses_cq_vbr_no_crf(self) -> None:
        args = _build_codec_args("h264_nvenc", None, None)
        assert args[args.index("-c:v") + 1] == "h264_nvenc"
        assert "-rc" in args
        assert args[args.index("-rc") + 1] == "vbr"
        assert "-cq" in args
        assert args[args.index("-cq") + 1] == "25"
        assert "-tune" in args
        assert args[args.index("-tune") + 1] == "hq"
        # NVENC must not use libx264's -crf flag
        assert "-crf" not in args

    def test_av1_nvenc_uses_cq_vbr(self) -> None:
        args = _build_codec_args("av1_nvenc", None, None)
        assert args[args.index("-c:v") + 1] == "av1_nvenc"
        assert "-rc" in args
        assert args[args.index("-rc") + 1] == "vbr"
        assert "-cq" in args
        assert args[args.index("-cq") + 1] == "25"
        assert "-crf" not in args

    def test_preset_override(self) -> None:
        args = _build_codec_args("libx264", "slow", None)
        assert "-preset" in args
        assert args[args.index("-preset") + 1] == "slow"

    def test_crf_override_maps_to_cq_on_nvenc(self) -> None:
        """-crf override supplied at the API level must become -cq on NVENC
        encoders (which reject -crf), not be passed through verbatim."""
        args = _build_codec_args("h264_nvenc", None, 18)
        assert "-cq" in args
        assert args[args.index("-cq") + 1] == "18"
        assert "-crf" not in args

    def test_crf_override_kept_on_cpu_encoder(self) -> None:
        """Sanity sibling to ensure the override path is actually reached."""
        args = _build_codec_args("libx264", None, 18)
        assert args[args.index("-crf") + 1] == "18"


# ============================================================================
# T5 regression: info.json video.codec label maps correctly for every codec
# ============================================================================


class TestCodecLabel:
    """T5: canonical LeRobot labels for each ffmpeg encoder.

    The label (``"h264"``/``"h265"``/``"av1"``) ends up inside
    ``info.json:features.<vkey>.info.video.codec`` and must reflect the
    actual on-disk compression family, not the ffmpeg encoder name.
    """

    def test_label_for_libx264(self) -> None:
        assert _codec_label("libx264") == "h264"

    def test_label_for_h264_nvenc(self) -> None:
        assert _codec_label("h264_nvenc") == "h264"

    def test_label_for_libx265(self) -> None:
        assert _codec_label("libx265") == "h265"

    def test_label_for_hevc_nvenc(self) -> None:
        assert _codec_label("hevc_nvenc") == "h265"

    def test_label_for_libsvtav1(self) -> None:
        assert _codec_label("libsvtav1") == "av1"

    def test_label_for_av1_nvenc(self) -> None:
        assert _codec_label("av1_nvenc") == "av1"

    def test_label_for_unknown_codec_falls_back_to_identity(self) -> None:
        """Unknown encoders must pass through unchanged so operators can
        inspect ``info.json`` and see what codec was actually requested."""
        assert _codec_label("custom_codec_xyz") == "custom_codec_xyz"

    def test_codec_label_map_is_complete(self) -> None:
        """Guard against future additions in ``_build_codec_args`` that
        forget to register a label mapping."""
        # Every encoder that _build_codec_args special-cases should have a
        # LeRobot label. libaom-av1 is included for completeness even though
        # we don't special-case it in _build_codec_args.
        for encoder in (
            "libx264",
            "libsvtav1",
            "h264_nvenc",
            "hevc_nvenc",
            "av1_nvenc",
        ):
            assert encoder in _CODEC_LABEL_MAP, (
                f"Encoder {encoder!r} missing from _CODEC_LABEL_MAP"
            )


class TestDatasetWriterSubtasks:
    """Subtask emission is gated on ``has_subtasks`` and per-episode spans."""

    @pytest.fixture
    def features_with_subtasks(self) -> dict:
        return {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "subtask_index": {"dtype": "int64", "shape": [1], "names": None},
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
        }

    def test_no_subtasks_parquet_when_flag_off(
        self,
        tmp_path: Path,
        simple_features: dict,
    ) -> None:
        """has_subtasks=False must not produce subtasks.parquet or the extra column."""
        writer = DatasetWriter(
            tmp_path,
            {"robot_type": "r"},
            simple_features,
            fps=10,
            has_subtasks=False,
        )
        for i in range(5):
            writer.add_frame(
                {
                    "observation.state": np.array([float(i)] * 3),
                    "action": np.zeros(3),
                    "task": "t",
                }
            )
        writer.save_episode()
        writer.finalize()

        assert not (tmp_path / "meta" / "subtasks.parquet").exists()
        table = pq.read_table(tmp_path / "data" / "chunk-000" / "file-000.parquet")
        assert "subtask_index" not in table.column_names

        with open(tmp_path / "meta" / "info.json") as f:
            info = json.load(f)
        assert "total_subtasks" not in info

    def test_full_coverage_writes_subtasks(
        self,
        tmp_path: Path,
        features_with_subtasks: dict,
    ) -> None:
        """Full subtask coverage → subtasks.parquet + subtask_index column + info."""
        writer = DatasetWriter(
            tmp_path,
            {"robot_type": "r"},
            features_with_subtasks,
            fps=10,
            has_subtasks=True,
        )
        spans = [SubtaskSpan(0.0, 0.3, "approach"), SubtaskSpan(0.3, 0.5, "grasp")]
        # 5 frames @ 10 fps → timestamps 0.0, 0.1, 0.2, 0.3, 0.4 → duration=0.5
        for i in range(5):
            frame = {
                "observation.state": np.array([float(i), 0.0]),
                "action": np.zeros(2),
                "task": "t",
            }
            if i == 0:
                frame["_episode_subtasks"] = spans
            writer.add_frame(frame)
        writer.save_episode()
        writer.finalize()

        # subtasks.parquet exists with index=subtask_str, column=subtask_index
        subtasks_df = pd.read_parquet(tmp_path / "meta" / "subtasks.parquet")
        assert subtasks_df.index.name is None
        assert list(subtasks_df.columns) == ["subtask_index"]
        assert set(subtasks_df.index.tolist()) == {"approach", "grasp"}

        # Data parquet carries per-frame subtask_index aligned with timestamps
        table = pq.read_table(tmp_path / "data" / "chunk-000" / "file-000.parquet")
        assert "subtask_index" in table.column_names
        approach_idx = subtasks_df.loc["approach", "subtask_index"]
        grasp_idx = subtasks_df.loc["grasp", "subtask_index"]
        indices = table.column("subtask_index").to_pylist()
        # ts 0.0, 0.1, 0.2 -> approach ; 0.3, 0.4 -> grasp
        assert indices == [approach_idx] * 3 + [grasp_idx] * 2

        with open(tmp_path / "meta" / "info.json") as f:
            info = json.load(f)
        assert info["total_subtasks"] == 2

        # Episode metadata has "subtasks" list column
        ep_df = pd.read_parquet(
            tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        )
        assert "subtasks" in ep_df.columns
        assert sorted(ep_df.iloc[0]["subtasks"]) == ["approach", "grasp"]

    def test_gap_in_subtask_coverage_raises(
        self,
        tmp_path: Path,
        features_with_subtasks: dict,
    ) -> None:
        """Short-tail spans must error out on save_episode."""
        writer = DatasetWriter(
            tmp_path,
            {"robot_type": "r"},
            features_with_subtasks,
            fps=10,
            has_subtasks=True,
        )
        bad_spans = [SubtaskSpan(0.0, 0.2, "short")]  # covers only up to 0.2, need 0.5
        for i in range(5):
            frame = {
                "observation.state": np.array([float(i), 0.0]),
                "action": np.zeros(2),
                "task": "t",
            }
            if i == 0:
                frame["_episode_subtasks"] = bad_spans
            writer.add_frame(frame)
        with pytest.raises(ValueError, match="full-time coverage required"):
            writer.save_episode()
