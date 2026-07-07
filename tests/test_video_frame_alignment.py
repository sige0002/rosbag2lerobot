"""Regression tests for per-camera mp4 frame-count alignment.

These tests verify the invariant that, for every video feature in a
LeRobot v3.0 dataset produced by :class:`DatasetWriter`, the number of
encoded mp4 frames equals the sum of the parquet ``length`` values of
all episodes that share the same mp4 file. Both the container-reported
``nb_frames`` and the actually-decoded frame count
(``ffprobe -count_frames``) are checked.

A drift between mp4 frames and parquet rows breaks LeRobot loaders that
slice videos by frame index or by ``(from_timestamp, to_timestamp)``, so
this invariant has to hold on every code path that produces video.

The investigation in
``docs/frame_alignment_investigation_ja.md`` documents the path that
motivated these tests.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

from rosbag2lerobot.writer import DatasetWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ffprobe_counts(path: Path) -> tuple[int | None, int]:
    """Return ``(container_nb_frames, decoded_nb_read_frames)`` for *path*.

    ``container_nb_frames`` is the value muxed into the mp4 ``stsz`` box,
    which is what fast-path decoders (TorchCodec, torchvision, pyav) tend
    to surface as ``video_decoder._num_frames``. It is ``None`` when the
    container does not record it.

    ``decoded_nb_read_frames`` is the authoritative count obtained by
    fully decoding every packet (``ffprobe -count_frames``).
    """
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_frames,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    stream = json.loads(out)["streams"][0]
    nb = stream.get("nb_frames")
    nb_int = int(nb) if nb not in (None, "N/A") else None
    return nb_int, int(stream["nb_read_frames"])


def _video_features(
    keys: list[str],
    shape: tuple[int, int, int] = (32, 32, 3),
) -> dict[str, dict[str, Any]]:
    """Build a minimal LeRobot v3.0 feature dict with *keys* as video features."""
    feats: dict[str, dict[str, Any]] = {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "observation.state": {
            "dtype": "float32",
            "shape": [1],
            "names": {"axes": ["x"]},
        },
        "action": {
            "dtype": "float32",
            "shape": [1],
            "names": {"axes": ["x"]},
        },
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
    codec: str = "libx264",
    shape: tuple[int, int, int] = (32, 32, 3),
) -> None:
    feats = _video_features(video_keys, shape=shape)
    writer = DatasetWriter(
        out_dir,
        {"robot_type": "regression"},
        feats,
        fps=fps,
        video_codec=codec,
    )
    rng = np.random.default_rng(0)
    for ep_len in episode_lengths:
        for i in range(ep_len):
            frame: dict[str, Any] = {
                "observation.state": np.array([float(i)], dtype=np.float32),
                "action": np.array([float(i)], dtype=np.float32),
                "task": "t",
            }
            for k in video_keys:
                frame[k] = _random_image(rng, shape)
            writer.add_frame(frame)
        writer.save_episode()
    writer.finalize()


def _episodes_meta(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for f in sorted((out_dir / "meta" / "episodes").rglob("*.parquet")):
        t = pq.read_table(f)
        for i in range(t.num_rows):
            rows.append({k: t.column(k)[i].as_py() for k in t.column_names})
    return rows


def _assert_alignment(out_dir: Path, video_keys: list[str], fps: int) -> None:
    """Check the three alignment invariants on *out_dir*.

    1. ``ffprobe -count_frames`` on each mp4 equals the sum of the
       ``length`` values of every episode that points at that mp4.
    2. The mp4 container's ``nb_frames`` (when present) equals the
       decoded count, so loaders that read the fast metadata path see the
       same number as those that decode every packet.
    3. The per-episode slice derived from the stored video timestamps
       (``(to_ts - from_ts) * fps``) matches the parquet ``length`` field.
    """
    eps = _episodes_meta(out_dir)
    assert eps, "no episode metadata produced"

    # Invariants 1 and 2: per mp4 file.
    for vk in video_keys:
        groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for ep in eps:
            ck = ep[f"videos/{vk}/chunk_index"]
            fk = ep[f"videos/{vk}/file_index"]
            groups[(ck, fk)].append(ep)
        for (ck, fk), members in groups.items():
            mp4 = out_dir / "videos" / vk / f"chunk-{ck:03d}" / f"file-{fk:03d}.mp4"
            assert mp4.exists(), f"missing mp4: {mp4}"
            nb, real = _ffprobe_counts(mp4)
            expected = sum(m["length"] for m in members)
            assert real == expected, (
                f"{vk} chunk={ck} file={fk}: ffprobe decoded {real} frames "
                f"but episodes sum to {expected} (eps="
                f"{[m['episode_index'] for m in members]})"
            )
            if nb is not None:
                assert nb == real, (
                    f"{vk} chunk={ck} file={fk}: container nb_frames={nb} "
                    f"diverges from decoded {real}"
                )

    # Invariant 3: per-episode timestamp slice.
    for ep in eps:
        ep_idx = ep["episode_index"]
        ep_len = ep["length"]
        for vk in video_keys:
            from_ts = ep[f"videos/{vk}/from_timestamp"]
            to_ts = ep[f"videos/{vk}/to_timestamp"]
            slice_frames = round((to_ts - from_ts) * fps)
            assert slice_frames == ep_len, (
                f"ep={ep_idx} {vk}: timestamp slice = {slice_frames} frames "
                f"but parquet length = {ep_len} "
                f"(from={from_ts:.6f} to={to_ts:.6f} fps={fps})"
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _require_ffmpeg_and_ffprobe() -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe not available")


class TestVideoFrameAlignment:
    """End-to-end frame-count parity between parquet rows and mp4 frames."""

    def test_single_episode_single_camera(self, tmp_path: Path) -> None:
        _write_dataset(
            tmp_path,
            episode_lengths=[40],
            video_keys=["observation.images.cam"],
        )
        _assert_alignment(tmp_path, ["observation.images.cam"], fps=10)

    def test_multi_episode_single_camera_aggregates_into_one_mp4(
        self,
        tmp_path: Path,
    ) -> None:
        """All episodes well below 200 MB end up in one aggregated mp4.

        Exercises the multi-episode streaming-encoder path, which must be
        free of boundary frame drops.
        """
        _write_dataset(
            tmp_path,
            episode_lengths=[40, 25, 35],
            video_keys=["observation.images.cam"],
        )
        _assert_alignment(tmp_path, ["observation.images.cam"], fps=10)

        # Sanity: there really is just one mp4 file (episodes aggregated).
        mp4s = list(
            (tmp_path / "videos" / "observation.images.cam").rglob("*.mp4"),
        )
        assert len(mp4s) == 1, f"expected one aggregated mp4, got {mp4s}"

    def test_multi_episode_multi_camera(self, tmp_path: Path) -> None:
        keys = [
            "observation.images.front",
            "observation.images.left",
            "observation.images.right",
        ]
        _write_dataset(
            tmp_path,
            episode_lengths=[30, 20, 25],
            video_keys=keys,
        )
        _assert_alignment(tmp_path, keys, fps=10)
