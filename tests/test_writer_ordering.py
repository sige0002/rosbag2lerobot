"""Regression test: meta ``names`` and parquet arrays must share config order.

Previously ``writer.py`` sorted sub-keyed ``observation.state.*`` /
``action.*`` entries alphabetically. Both the ``names`` list and the
concatenation order used the same sort, so they were consistent — but that
alphabetical order did *not* match the declaration order in the user's YAML
config, which was surprising and fragile. The fix switches to config order.

This test pins the new invariant: both the merged ``names`` and the
concatenated per-frame array follow the declaration order of
``config.observations`` / ``config.actions``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from bagel.config import FeatureMapping, RobotConfig
from bagel.writer import write_dataset


def _episode(num_frames: int = 3) -> list[dict]:
    """Build a single episode whose sub-key values are trivially distinguishable."""
    frames: list[dict] = []
    for i in range(num_frames):
        frames.append(
            {
                "timestamp": float(i) * 0.1,
                # Deliberately use distinct scalar magnitudes so the ordering
                # check can detect a swap of the two halves.
                "observation.state.z_last": np.array(
                    [100.0 + i, 101.0 + i], dtype=np.float32
                ),
                "observation.state.a_first": np.array(
                    [1.0 + i, 2.0 + i, 3.0 + i], dtype=np.float32
                ),
                "action.z_last": np.array([900.0 + i], dtype=np.float32),
                "action.a_first": np.array([9.0 + i, 8.0 + i], dtype=np.float32),
                "task": "ordering_probe",
            }
        )
    return frames


def test_meta_and_parquet_follow_config_order(tmp_path: Path) -> None:
    # Deliberately declare sub-keys in reverse-alphabetical order so that the
    # old alphabetical sort would reorder them and the new config-order code
    # path keeps them as declared.
    config = RobotConfig(
        robot_type="ordering_probe",
        fps=10,
        task="ordering_probe",
        observations=[
            FeatureMapping(
                key="observation.state.z_last",
                topic="/dummy_z",
                msg_type="std_msgs/msg/Float32MultiArray",
            ),
            FeatureMapping(
                key="observation.state.a_first",
                topic="/dummy_a",
                msg_type="std_msgs/msg/Float32MultiArray",
            ),
        ],
        actions=[
            FeatureMapping(
                key="action.z_last",
                topic="/dummy_action_z",
                msg_type="std_msgs/msg/Float32MultiArray",
            ),
            FeatureMapping(
                key="action.a_first",
                topic="/dummy_action_a",
                msg_type="std_msgs/msg/Float32MultiArray",
            ),
        ],
    )

    write_dataset([_episode()], config, tmp_path)

    with (tmp_path / "meta" / "info.json").open() as f:
        info = json.load(f)

    state_names = info["features"]["observation.state"]["names"]
    action_names = info["features"]["action"]["names"]

    # The z_last block (declared first) must come before a_first in both
    # merged name lists.
    assert state_names == [
        "z_last_0",
        "z_last_1",
        "a_first_0",
        "a_first_1",
        "a_first_2",
    ]
    assert action_names == ["z_last_0", "a_first_0", "a_first_1"]

    # The parquet vectors must agree with the names ordering. On frame 0,
    # z_last block values are [100.0, 101.0] and a_first block is [1.0, 2.0, 3.0].
    table = pq.read_table(tmp_path / "data" / "chunk-000" / "file-000.parquet")
    state_row0 = np.asarray(table.column("observation.state").to_pylist()[0])
    action_row0 = np.asarray(table.column("action").to_pylist()[0])

    np.testing.assert_allclose(state_row0, [100.0, 101.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(action_row0, [900.0, 9.0, 8.0])

    # Cross-check: element count in `names` must equal array length (would
    # catch a drift between schema and actual data width).
    assert len(state_names) == state_row0.shape[0]
    assert len(action_names) == action_row0.shape[0]
