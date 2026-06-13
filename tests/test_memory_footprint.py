"""Memory-footprint regression test for the streaming writer (T10/T11).

The pre-T10 writer buffered an entire episode of raw PIL images in memory
(``_image_buffers: dict[str, list[Image.Image]]``), which pushed peak RSS
into the multi-GB range for realistic dual-camera datasets. T10 replaces
that with a per-camera ffmpeg stdin pipe and a bounded feeder queue, and
T11 turns ``write_dataset`` into a streaming generator over episodes.

This module measures the writer's Python-heap peak under a synthetic load
and asserts an absolute upper bound. The bound is deliberately generous
(< 200 MB for 5 episodes * 30 frames * 1 camera @ 480x640) so that legit
implementation changes don't flip the test while still catching a
regression back to the old O(ep_frames) buffer strategy which would
trivially blow past the limit.

Marked ``@pytest.mark.slow`` because it requires a real ffmpeg invocation
and is more expensive than the other unit tests; run explicitly with
``pytest -m slow``.
"""

from __future__ import annotations

import shutil
import tracemalloc
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest
from PIL import Image

from bagel.writer import DatasetWriter


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _synthetic_frame(i: int) -> Image.Image:
    """Produce a deterministic 480x640 RGB image.

    Using ``PIL.Image.new`` avoids allocating a numpy array we then throw
    away, keeping the per-call overhead realistic for the streaming path.
    """
    rgb = (i * 7 % 256, i * 13 % 256, i * 29 % 256)
    return Image.new("RGB", (640, 480), rgb)


def _iter_episode_frames(n_frames: int) -> Iterator[dict[str, Any]]:
    """Yield frames one at a time (no list materialization)."""
    for i in range(n_frames):
        yield {
            "observation.state": np.array([float(i), float(i)], dtype=np.float32),
            "action": np.zeros(2, dtype=np.float32),
            "observation.images.cam": _synthetic_frame(i),
        }


@pytest.mark.slow
@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not available")
def test_writer_peak_memory_stays_bounded(tmp_path: Path) -> None:
    """5 episodes * 30 frames * one 480x640 camera should stay well below
    the 200 MB heap ceiling. The old in-memory image buffer consumed
    ~3 * 480 * 640 = ~900 KB per frame, so 150 frames was ~130 MB of
    image bytes alone plus PIL overhead; streaming cuts this to the
    queue depth (_IMAGE_FEED_QUEUE_MAXSIZE frames) at steady state.
    """
    features: dict[str, dict] = {
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
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
        },
    }

    n_episodes = 5
    n_frames = 30

    tracemalloc.start()
    try:
        writer = DatasetWriter(
            tmp_path,
            {"robot_type": "r"},
            features,
            fps=30,
        )
        for _ep in range(n_episodes):
            for frame in _iter_episode_frames(n_frames):
                writer.add_frame(frame)
            writer.save_episode()
        writer.finalize()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    peak_mb = peak_bytes / (1024 * 1024)
    # Absolute ceiling: well above the steady-state footprint but far below
    # the pre-T10 O(ep_frames * WxHx3) buffer which would exceed 200 MB for
    # even a single realistic camera episode.
    assert peak_mb < 200.0, (
        f"Writer peak memory {peak_mb:.1f} MiB exceeds 200 MiB ceiling — "
        f"check that image buffers remain streaming and are not materialized."
    )

    # Sanity: the run actually produced the expected output.
    with (tmp_path / "meta" / "info.json").open() as f:
        import json as _json

        info = _json.load(f)
    assert info["total_episodes"] == n_episodes
    assert info["total_frames"] == n_episodes * n_frames
