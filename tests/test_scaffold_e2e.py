"""End-to-end integration tests for ``rosbag2lerobot scaffold`` on real bags.

Runs against unconfigured RealMan dual-arm bags from ``bagdata/`` (no shipped
config). These bags are large and gitignored, so each test guards on the bag
directory + its ``metadata.yaml`` and skips when absent.

Run with:  uv run pytest -m integration tests/test_scaffold_e2e.py -q

Real bags used:
  bagdata/sample_bags/sample1  (RealMan dual-arm, ~6.7 GB mcap)
  bagdata/bag/sample_001       (RealMan dual-arm, ~3.8 GB mcap)

Each test asserts the scaffold:
  * exits 0 and writes a config that reloads via load_config (round-trip),
  * maps /camera/color/image_raw0 and /camera/depth/image_raw0 to *distinct*
    image keys (collision disambiguation) and a JointState topic to
    observation.state*,
  * surfaces an rm_ros_interfaces/* topic only as a commented candidate, and
  * passes validate-config against the same bag (exit code not nonzero).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from rosbag2lerobot.cli import main
from rosbag2lerobot.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BAGDATA_DIR = PROJECT_ROOT / "bagdata"

SAMPLE1 = BAGDATA_DIR / "sample_bags" / "sample1"
SAMPLE_001 = BAGDATA_DIR / "bag" / "sample_001"


def _require(bag: Path) -> None:
    """Skip if the real bag (gitignored) is not present locally."""
    if not bag.exists() or not (bag / "metadata.yaml").exists():
        pytest.skip(f"real RealMan bag not available at {bag}")


@pytest.mark.integration
@pytest.mark.parametrize("bag", [SAMPLE1, SAMPLE_001])
def test_scaffold_real_bag(bag: Path, tmp_path: Path) -> None:
    _require(bag)
    out = tmp_path / "config.yaml"

    result = CliRunner().invoke(
        main,
        [
            "scaffold",
            "--bags",
            str(bag),
            "-o",
            str(out),
            "--robot-type",
            "realman_dual",
            "--task",
            "manipulation",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()

    # Round-trip: the generated config reloads cleanly.
    cfg = load_config(out)

    # Distinct image keys for color vs. depth on camera 0 (collision dedup).
    obs_by_topic = {fm.topic: fm for fm in cfg.observations}
    assert "/camera/color/image_raw0" in obs_by_topic
    assert "/camera/depth/image_raw0" in obs_by_topic
    color_key = obs_by_topic["/camera/color/image_raw0"].key
    depth_key = obs_by_topic["/camera/depth/image_raw0"].key
    assert color_key != depth_key
    assert "color" in color_key
    assert "depth" in depth_key

    # At least one observation.state* sourced from a JointState topic.
    state_fms = [
        fm
        for fm in cfg.observations
        if fm.key.startswith("observation.state")
        and fm.msg_type == "sensor_msgs/msg/JointState"
    ]
    assert state_fms, "expected an observation.state* from a JointState topic"

    # An rm_ros_interfaces/* topic appears only as a COMMENTED candidate
    # (never as an active mapping).
    raw = out.read_text()
    assert "rm_ros_interfaces/" in raw
    assert "decoder: NONE" in raw
    active_msg_types = {fm.msg_type for fm in cfg.observations + cfg.actions}
    assert not any(t.startswith("rm_ros_interfaces/") for t in active_msg_types)

    # validate-config on the generated config + bag must not fail.
    vresult = CliRunner().invoke(
        main,
        ["validate-config", "--config", str(out), "--bags", str(bag)],
    )
    assert vresult.exit_code == 0, vresult.output


@pytest.mark.integration
@pytest.mark.parametrize("bag", [SAMPLE1, SAMPLE_001])
def test_validate_config_suggest_fixes(bag: Path, tmp_path: Path) -> None:
    """`validate-config --suggest-fixes` prints an image_size diff on mismatch.

    Scaffold a real bag, mutate one image feature's ``image_size`` to a wrong
    value, then validate with ``--suggest-fixes``. The shape mismatch is a WARN
    (exit 0) and the suggested-fixes block must surface the real measured
    ``[H, W, C]`` on the ``+`` line.
    """
    _require(bag)
    out = tmp_path / "config.yaml"

    result = CliRunner().invoke(
        main,
        [
            "scaffold",
            "--bags",
            str(bag),
            "-o",
            str(out),
            "--robot-type",
            "realman_dual",
            "--task",
            "manipulation",
            "--no-validate",
        ],
    )
    assert result.exit_code == 0, result.output

    cfg = load_config(out)
    # Pick an image feature whose scaffold detected a real image_size.
    target = next(
        (fm for fm in cfg.image_features if fm.image_size is not None),
        None,
    )
    assert target is not None, "expected at least one image feature with a shape"
    measured = list(target.image_size)

    # Mutate that feature's image_size in the YAML to a deliberately wrong value.
    raw = yaml.safe_load(out.read_text())
    wrong = [measured[0] + 7, measured[1] + 11, measured[2] if len(measured) > 2 else 3]
    for entry in raw["observations"]:
        if entry.get("key") == target.key:
            entry["image_size"] = wrong
            break
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(raw))

    vresult = CliRunner().invoke(
        main,
        [
            "validate-config",
            "--config",
            str(mutated),
            "--bags",
            str(bag),
            "--suggest-fixes",
        ],
    )
    # Shape mismatch is a WARN, not an error -> exit 0.
    assert vresult.exit_code == 0, vresult.output
    assert "Suggested fixes:" in vresult.output
    # The real measured [H, W, C] appears on the suggested '+' (measured) line.
    plus_lines = [
        ln for ln in vresult.output.splitlines() if ln.lstrip().startswith("+")
    ]
    assert any(str(measured) in ln for ln in plus_lines), vresult.output
