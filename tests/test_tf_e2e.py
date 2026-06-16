"""End-to-end integration tests for TF lookup + quaternion->euler wiring.

These pin the behaviour of the TF feature pipeline (plan.md B-2 item ⑩) against
a *real* bag from ``bagdata/`` that carries ``/tf`` + ``/tf_static``:

  (a) A :class:`TransformLookup` built from the bag's real tf resolves
      ``lookup("base_link", "hand_palm_link", mid_stamp)`` to a finite 7-vector
      with a non-trivial translation.
  (b) A full convert of an HSR bag with an hsr-derived config that appends a TF
      FeatureMapping (``observation.ee_pose``, frame_from ``hand_palm_link``,
      frame_to ``base_link``, selector ``orientation.euler_xyz``) produces a
      parquet column of width 6 (``[tx,ty,tz,roll,pitch,yaw]``) with no nulls,
      and ``validate-dataset`` returns verdict OK.

Run with:  uv run pytest -m integration tests/test_tf_e2e.py -q

Real-data pairing (mirrors test_topic_alignment_e2e.py):
  config = configs/hsr.yaml (+ appended TF feature)
  bag    = bagdata/airoa-moma-mcap/000730  (HSR, ~44s, has /tf + /tf_static)
Verified frames present in 000730's tf tree: base_link, hand_palm_link.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths — mirror the conventions in test_topic_alignment_e2e.py
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
BAGDATA_DIR = PROJECT_ROOT / "bagdata"

HSR_CONFIG = CONFIGS_DIR / "hsr.yaml"
HSR_BAG = BAGDATA_DIR / "airoa-moma-mcap" / "000730"  # ~44s, has /tf + /tf_static

_TEST_FPS = 10
_FRAME_FROM = "hand_palm_link"
_FRAME_TO = "base_link"
_TF_KEY = "observation.ee_pose"


# ---------------------------------------------------------------------------
# Skip guards
# ---------------------------------------------------------------------------


def _require(bag: Path) -> None:
    """Skip if the real HSR bag / config is not present (bagdata/ is gitignored)."""
    if not HSR_CONFIG.exists():
        pytest.skip(f"hsr.yaml not available at {HSR_CONFIG}")
    if not bag.exists() or not (bag / "metadata.yaml").exists():
        pytest.skip(f"real HSR bag not available at {bag}")


def _load_tf_config() -> Any:
    """Load configs/hsr.yaml and append a TF FeatureMapping (in-memory only).

    configs/*.yaml are never mutated on disk: the appended feature and the fps
    override live on the in-memory RobotConfig.
    """
    from bagel.config import FeatureMapping, load_config

    cfg = load_config(HSR_CONFIG)
    cfg.fps = _TEST_FPS
    tf_fm = FeatureMapping(
        key=_TF_KEY,
        topic="/tf",
        msg_type="tf2_msgs/msg/TFMessage",
        frame_from=_FRAME_FROM,
        frame_to=_FRAME_TO,
        selector="orientation.euler_xyz",
    )
    cfg.observations.append(tf_fm)
    return cfg


# ===========================================================================
# (a) TransformLookup over the real bag tf
# ===========================================================================


@pytest.mark.integration
def test_real_tf_lookup_returns_finite_pose() -> None:
    """A TransformLookup built from the real bag resolves base_link<-hand_palm_link."""
    _require(HSR_BAG)
    from bagel.config import load_config
    from bagel.reader import BagReader
    from bagel.transforms import TransformLookup

    cfg = load_config(HSR_CONFIG)
    lookup = TransformLookup()
    stamps: list[int] = []
    with BagReader(HSR_BAG, cfg) as reader:
        for topic, ts_ns, msg in reader.iter_messages(topics=["/tf", "/tf_static"]):
            if topic == "/tf_static":
                lookup.add_static(msg)
            else:
                lookup.add_dynamic(msg)
                stamps.append(ts_ns)

    assert stamps, "no /tf messages found in the bag"

    # Real frames must be present in the tf tree.
    frames: set[str] = set()
    for parent, child in list(lookup._static.keys()) + list(lookup._dynamic.keys()):
        frames.add(parent)
        frames.add(child)
    assert _FRAME_TO in frames, f"{_FRAME_TO} missing from tf tree"
    assert _FRAME_FROM in frames, f"{_FRAME_FROM} missing from tf tree"

    mid_stamp = stamps[len(stamps) // 2]
    pose = lookup.lookup(_FRAME_TO, _FRAME_FROM, mid_stamp)
    assert pose.shape == (7,)
    assert np.all(np.isfinite(pose))
    # The hand is offset from the base, so the translation must be non-trivial.
    translation_norm = float(np.linalg.norm(pose[:3]))
    assert translation_norm > 0.05, (
        f"expected non-trivial translation, got norm={translation_norm}"
    )


# ===========================================================================
# (b) Full convert: TF feature column is width-6 euler with no nulls; OK verdict
# ===========================================================================


@pytest.mark.integration
def test_tf_feature_convert_writes_euler_column(tmp_path: Path) -> None:
    """Convert 000730 with a TF euler feature; parquet column is width 6, no nulls."""
    _require(HSR_BAG)
    import pandas as pd

    from bagel.cli import _process_episode
    from bagel.resampler import Resampler
    from bagel.validation import validate_dataset
    from bagel.writer import write_dataset

    cfg = _load_tf_config()
    resampler = Resampler(
        fps=cfg.fps,
        policy=cfg.resampling.default_policy,
        tolerance_ms=cfg.resampling.tolerance_ms,
    )

    frames = _process_episode(HSR_BAG, cfg, resampler)
    assert frames, "TF convert produced an empty episode"

    # Every retained frame must carry a width-6 TF value (no nulls).
    for f in frames:
        val = f.get(_TF_KEY)
        assert val is not None, f"TF feature {_TF_KEY} is None at a retained frame"
        arr = np.asarray(val)
        assert arr.shape == (6,), f"expected width-6 euler vector, got {arr.shape}"
        assert np.all(np.isfinite(arr))

    out = tmp_path / "hsr_tf"

    def _episodes() -> Any:
        for frame in frames:
            frame["task"] = cfg.task
        yield frames

    write_dataset(
        episodes=_episodes(),
        config=cfg,
        output_dir=out,
        video_codec="libx264",
        repo_id=cfg.repo_id,
    )

    # The dataset must declare the TF feature with shape [6].
    import json

    with open(out / "meta" / "info.json") as fh:
        info = json.load(fh)
    assert _TF_KEY in info["features"], "TF feature missing from info.json"
    assert info["features"][_TF_KEY]["shape"] == [6], (
        f"expected TF feature shape [6], got {info['features'][_TF_KEY]['shape']}"
    )

    # The parquet column must be width-6 with no nulls.
    data_files = list((out / "data").rglob("*.parquet"))
    assert data_files, "no data parquet written"
    df = pd.read_parquet(data_files[0])
    assert _TF_KEY in df.columns, f"{_TF_KEY} column missing from parquet"
    col = df[_TF_KEY]
    assert col.notna().all(), "TF feature column has nulls"
    widths = {len(np.asarray(v)) for v in col}
    assert widths == {6}, f"expected all TF rows width 6, got widths={widths}"

    # validate-dataset must return verdict OK.
    report = validate_dataset(out)
    report.apply_verdict(strict=False)
    assert report.verdict == "OK", (
        f"validate-dataset verdict={report.verdict}; issues={report.to_dict()}"
    )
