"""E2E tests for the depth / raw-image decoders against real bagdata.

Covers plan.md B-1:
  - ``compressedDepth`` (RVL 16bit) decode from a real HSR bag.
  - raw ``sensor_msgs/msg/Image`` ``mono16`` / ``bgr8`` decode from sample_bags.
  - a full ``write_dataset`` convert of an HSR bag with the depth feature
    enabled, asserting the depth video + ``is_depth_map`` metadata.

bagdata/ is gitignored, so every test skips gracefully when the data is
absent (mirrors tests/test_topic_alignment_e2e.py conventions).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
BAGDATA_DIR = PROJECT_ROOT / "bagdata"

HSR_CONFIG = CONFIGS_DIR / "hsr.yaml"
HSR_BAG = BAGDATA_DIR / "airoa-moma-mcap" / "235210"  # has image_raw/compressedDepth
HSR_DEPTH_TOPIC = "/hsrb/head_rgbd_sensor/depth_registered/image_raw/compressedDepth"
SAMPLE_BAG = BAGDATA_DIR / "sample_bags" / "sample1"


def _require(bag: Path) -> None:
    if not bag.exists() or not (bag / "metadata.yaml").exists():
        pytest.skip(f"real bag not available at {bag}")


def _first_message(bag: Path, topic: str) -> Any:
    """Deserialize and return the first message on ``topic`` (or skip)."""
    from rosbags.highlevel import AnyReader

    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            pytest.skip(f"topic {topic} not in {bag}")
        for conn, _ts, raw in reader.messages(connections=conns):
            return reader.deserialize(raw, conn.msgtype)
    pytest.skip(f"no messages on {topic} in {bag}")


def _depth_config() -> Any:
    """Load hsr.yaml and append the optional head_depth feature for 235210."""
    from rosbag2lerobot.config import FeatureMapping, load_config

    cfg = load_config(HSR_CONFIG)
    cfg.fps = 10
    depth_fm = FeatureMapping(
        key="observation.images.head_depth",
        topic=HSR_DEPTH_TOPIC,
        msg_type="sensor_msgs/msg/CompressedImage",
        dtype="image",
        image_size=[480, 640],
        optional=True,
        stamp_source="header",
    )
    cfg.observations = list(cfg.observations) + [depth_fm]
    return cfg


def _ffprobe_nb_frames(path: Path) -> int:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    return int(out.stdout.strip())


# ---------------------------------------------------------------------------
# Layer 1 — single-frame decode against real bags
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDepthDecodeReal:
    def test_compressed_depth_rvl_real_frame(self) -> None:
        _require(HSR_BAG)
        from rosbag2lerobot.decoders.image import (
            _decode_compressed_depth,
            decode_compressed_image,
        )

        msg = _first_message(HSR_BAG, HSR_DEPTH_TOPIC)
        depth = _decode_compressed_depth(bytes(msg.data), msg.format.lower())
        assert depth.shape == (480, 640)
        assert depth.dtype == np.uint16
        valid = depth[depth > 0]
        assert valid.size > 0  # 実深度には有効値が存在する
        assert int(depth.max()) < 65535
        # 完全な PIL 経路（8bit RGB へ正規化）
        from PIL import Image

        pil = decode_compressed_image(msg, None, {})
        assert isinstance(pil, Image.Image)
        assert pil.mode == "RGB"
        assert pil.size == (640, 480)
        # resize 経路
        pil_small = decode_compressed_image(msg, None, {"image_size": [240, 320]})
        assert pil_small.size == (320, 240)

    def test_raw_mono16_and_bgr8_real(self) -> None:
        _require(SAMPLE_BAG)
        from rosbag2lerobot.decoders.image import decode_image

        depth_msg = _first_message(SAMPLE_BAG, "/camera/depth/image_raw0")
        assert depth_msg.encoding.lower() == "mono16"
        pil_d = decode_image(depth_msg, None, {})  # 以前は mono16 で例外だった
        assert pil_d.mode == "RGB"

        color_msg = _first_message(SAMPLE_BAG, "/camera/color/image_raw0")
        pil_c = decode_image(color_msg, None, {})
        assert pil_c.mode == "RGB"


# ---------------------------------------------------------------------------
# Layer 2 — full convert with the depth feature enabled
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDepthConvertReal:
    def test_convert_with_depth(self, tmp_path: Path) -> None:
        _require(HSR_BAG)
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            pytest.skip("ffmpeg/ffprobe not available")

        from rosbag2lerobot.cli import _process_episode
        from rosbag2lerobot.resampler import Resampler
        from rosbag2lerobot.writer import write_dataset

        cfg = _depth_config()
        resampler = Resampler(
            fps=cfg.fps,
            policy=cfg.resampling.default_policy,
            tolerance_ms=cfg.resampling.tolerance_ms,
        )

        def _episodes() -> Any:
            frames = _process_episode(HSR_BAG, cfg, resampler)
            for frame in frames:
                frame["task"] = cfg.task
            yield frames

        out = tmp_path / "hsr_depth"
        write_dataset(
            episodes=_episodes(),
            config=cfg,
            output_dir=out,
            video_codec="libx264",
            repo_id=cfg.repo_id,
        )

        info = json.loads((out / "meta" / "info.json").read_text())
        key = "observation.images.head_depth"
        assert key in info["features"], "depth feature missing from info.json"
        feat = info["features"][key]
        assert feat["dtype"] == "video"
        assert feat["shape"] == [480, 640, 3]
        assert feat["info"]["video.is_depth_map"] is True

        depth_mp4s = list((out / "videos" / key).rglob("*.mp4"))
        assert depth_mp4s, "no depth mp4 produced"
        assert depth_mp4s[0].stat().st_size > 100
        # 動画フレーム数とデータ行数の整合
        assert _ffprobe_nb_frames(depth_mp4s[0]) == info["total_frames"]
