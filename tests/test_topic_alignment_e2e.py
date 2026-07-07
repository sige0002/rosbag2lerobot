"""End-to-end integration tests for the topic-alignment / timestamp fixes.

These tests pin the behaviour of the three fixes landed for the topic-alignment
work and run against a *real* bag from ``bagdata/``:

  1. Generated mp4 permissions are 0o666 (rw-rw-rw-), including the
     multi-episode aggregated-mp4 path that previously produced 0o600.
  2. ``align_to_required`` (ResamplingConfig, default True) clips the output
     frame grid to the intersection of the required features' time spans, so
     toggling it True/False changes episode length (True <= False, and
     strictly < when ``trim_to_valid`` is off so the two mechanisms don't
     overlap).
  3. ``max_stamp_delay_ms`` drops stale latched messages (header lag > the
     threshold), reducing the retained frame count / emitting a drop log.
  4. The end of an episode has no long tail of hold-carried duplicate frames
     once ``align_to_required`` + ``trim_to_valid`` have run.

Run with:  uv run pytest -m integration tests/test_topic_alignment_e2e.py -q

Real-data pairing (established in Wave 1, verified in Wave 2):
  config  = configs/hsr.yaml
  bag     = bagdata/airoa-moma-mcap/235210   (HSR, ~10s, /hsrb/* topics)
  long    = bagdata/airoa-moma-mcap/000730   (HSR, ~44s) — used where a
            larger required-window delta makes the align effect unambiguous.
The multi-episode case reuses the bag directory's *parent*
(bagdata/airoa-moma-mcap/) with --max-episodes to force >1 episode into a
single aggregated mp4.

Measured Wave 2 reference values (fps=10), for context:
  - 235210: align True/False (trim on) -> 97 / 98 frames; max_stamp_delay_ms=50
    drops 297 stale msgs -> 96 frames (vs 97 baseline).
  - 000730: align True/False with trim_to_valid OFF -> 439 / 443 frames.
"""

from __future__ import annotations

import stat
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# rosbag2lerobot is imported lazily inside helpers/tests so module collection never
# imports the package at import time.

# ---------------------------------------------------------------------------
# Paths — mirror the conventions in test_e2e.py
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
BAGDATA_DIR = PROJECT_ROOT / "bagdata"

HSR_CONFIG = CONFIGS_DIR / "hsr.yaml"
HSR_BAG = BAGDATA_DIR / "airoa-moma-mcap" / "235210"  # ~10s
HSR_BAG_LONG = BAGDATA_DIR / "airoa-moma-mcap" / "000730"  # ~44s
HSR_BAG_PARENT = BAGDATA_DIR / "airoa-moma-mcap"  # parent => multi-episode

# Reduced fps keeps the tests fast (HSR config ships fps=10 already).
_TEST_FPS = 10


# ---------------------------------------------------------------------------
# Skip guards + small helpers
# ---------------------------------------------------------------------------


def _require(bag: Path) -> None:
    """Skip if a real HSR bag / config is not present (bagdata/ is gitignored)."""
    if not HSR_CONFIG.exists():
        pytest.skip(f"hsr.yaml not available at {HSR_CONFIG}")
    if not bag.exists() or not (bag / "metadata.yaml").exists():
        pytest.skip(f"real HSR bag not available at {bag}")


def _all_mp4s(dataset_dir: Path) -> list[Path]:
    return list((dataset_dir / "videos").rglob("*.mp4"))


def _mode_octal(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _load_hsr_config(
    *,
    align_to_required: bool = True,
    trim_to_valid: bool | None = None,
    max_stamp_delay_ms: float | None = None,
    fps: int = _TEST_FPS,
) -> Any:
    """Load configs/hsr.yaml and override the resampling knobs under test.

    configs/*.yaml are never mutated on disk: we tweak the in-memory
    RobotConfig (its ResamplingConfig is replaced via dataclasses.replace).
    """
    from rosbag2lerobot.config import load_config

    cfg = load_config(HSR_CONFIG)
    rs_kwargs: dict[str, Any] = {
        "align_to_required": align_to_required,
        "max_stamp_delay_ms": max_stamp_delay_ms,
    }
    if trim_to_valid is not None:
        rs_kwargs["trim_to_valid"] = trim_to_valid
    cfg.resampling = replace(cfg.resampling, **rs_kwargs)
    cfg.fps = fps
    return cfg


def _process(bag: Path, cfg: Any) -> list[dict]:
    """Run the production per-episode pipeline and return resampled frames."""
    from rosbag2lerobot.cli import _process_episode
    from rosbag2lerobot.resampler import Resampler

    resampler = Resampler(
        fps=cfg.fps,
        policy=cfg.resampling.default_policy,
        tolerance_ms=cfg.resampling.tolerance_ms,
    )
    return _process_episode(bag, cfg, resampler)


def _convert_inproc(
    bag: Path, cfg: Any, output_dir: Path, *, n_episodes: int = 1
) -> Path:
    """Convert ``bag`` to a LeRobot dataset at ``output_dir`` via ``write_dataset``.

    Replicates the production CLI path (cli.py ``convert`` -> ``write_dataset``):
    the RobotConfig is passed directly; the writer derives the feature spec and
    fps internally. Forces CPU encoding (libx264) for reproducibility. Writing
    the same bag ``n_episodes`` times yields ``n_episodes`` episodes that the
    writer aggregates into a single mp4 per camera — exactly the
    0o600-regression surface.
    """
    from rosbag2lerobot.writer import write_dataset

    def _episodes() -> Any:
        for _ in range(n_episodes):
            # Re-decode per episode: frames carry numpy/PIL objects the writer
            # consumes, so hand each episode an independent decode.
            ep = _process(bag, cfg)
            for frame in ep:
                frame["task"] = cfg.task
            yield ep

    write_dataset(
        episodes=_episodes(),
        config=cfg,
        output_dir=output_dir,
        video_codec="libx264",
        repo_id=cfg.repo_id,
    )
    return output_dir


def _info(dataset_dir: Path) -> dict[str, Any]:
    import json

    with open(dataset_dir / "meta" / "info.json") as f:
        return json.load(f)


# ===========================================================================
# Fix 1 — mp4 permissions are 0o666 (incl. multi-episode aggregated path)
# ===========================================================================


@pytest.mark.integration
class TestVideoPermissions:
    """Generated mp4 files must be world-rw (0o666)."""

    def test_single_episode_mp4_is_0o666(self, tmp_path: Path) -> None:
        """Convert one HSR bag and assert every produced mp4 is rw-rw-rw-."""
        _require(HSR_BAG)
        out = _convert_inproc(HSR_BAG, _load_hsr_config(), tmp_path / "hsr_single")

        mp4s = _all_mp4s(out)
        assert mp4s, f"no mp4 produced under {out}"
        for f in mp4s:
            assert _mode_octal(f) == 0o666, (
                f"{f} mode={oct(_mode_octal(f))}, expected 0o666"
            )

    def test_multi_episode_aggregated_mp4_is_0o666(self, tmp_path: Path) -> None:
        """Multiple episodes aggregated into a single mp4 must also be 0o666.

        Historical regression: the old concat path (mkstemp temp file ->
        ffmpeg concat filter -> shutil.move) left the merged mp4 at 0o600.
        Three episodes (well under the per-file size threshold) land in one
        aggregated mp4 per camera, driving _close_video_encoder + the chmod.
        """
        _require(HSR_BAG)
        out = _convert_inproc(
            HSR_BAG, _load_hsr_config(), tmp_path / "hsr_multi", n_episodes=3
        )

        info = _info(out)
        assert info["total_episodes"] >= 2, (
            f"expected >=2 episodes to exercise aggregation, got {info['total_episodes']}"
        )

        mp4s = _all_mp4s(out)
        assert mp4s, f"no mp4 produced under {out}"
        # Each camera should have streamed its 3 episodes into ONE file,
        # not one-file-per-episode.
        for cam_dir in (out / "videos").iterdir():
            cam_mp4s = list(cam_dir.rglob("*.mp4"))
            assert len(cam_mp4s) == 1, (
                f"{cam_dir.name}: expected a single aggregated mp4, got {len(cam_mp4s)}"
            )
        for f in mp4s:
            assert _mode_octal(f) == 0o666, (
                f"{f} mode={oct(_mode_octal(f))}, expected 0o666"
            )


# ===========================================================================
# Fix 2 — ffmpeg must not steal the TTY (-nostdin + stdin=DEVNULL)
# ===========================================================================


@pytest.mark.integration
class TestFfmpegNoStdin:
    """ffmpeg invocations are isolated from the controlling terminal.

    The interactive TTY-corruption symptom cannot be reliably reproduced in a
    non-interactive agent/CI environment, so this is split into:
      (a) a static check that every ffmpeg invocation pins stdin explicitly
          (the writer's streaming encoder feeds frames via stdin=PIPE and so
          never inherits the TTY; the NVENC-probe in cli.py is
          -nostdin + DEVNULL),
      (b) a smoke conversion that must complete cleanly.
    Final confirmation that the terminal stays usable after a real
    ``rosbag2lerobot convert`` is left to the user (see the Wave 2 runbook).
    """

    def test_ffmpeg_calls_pin_stdin_away_from_tty(self) -> None:
        """Static source check on writer.py's encoder ffmpeg call + cli probe."""
        writer_src = (PROJECT_ROOT / "src" / "rosbag2lerobot" / "writer.py").read_text()
        cli_src = (
            PROJECT_ROOT / "src" / "rosbag2lerobot" / "cli" / "_common.py"
        ).read_text()
        # The writer's only ffmpeg invocation is the streaming encoder, whose
        # stdin is the frame pipe (never the TTY).
        assert "stdin=subprocess.PIPE" in writer_src, (
            "writer.py encoder subprocess should set stdin=subprocess.PIPE"
        )
        # The ffmpeg -encoders probe in cli.py must also not grab the TTY.
        assert "-nostdin" in cli_src and "stdin=subprocess.DEVNULL" in cli_src, (
            "cli/_common.py ffmpeg -encoders probe should pass -nostdin + stdin=DEVNULL"
        )

    def test_conversion_completes_without_error(self, tmp_path: Path) -> None:
        """A full convert finishes and produces a readable dataset + mp4."""
        _require(HSR_BAG)
        out = _convert_inproc(HSR_BAG, _load_hsr_config(), tmp_path / "hsr_smoke")
        assert (out / "meta" / "info.json").exists()
        assert _all_mp4s(out), "smoke conversion produced no mp4"


# ===========================================================================
# Fix 3 — align_to_required changes episode length; no tail hold duplicates
# ===========================================================================


@pytest.mark.integration
class TestAlignToRequired:
    """align_to_required (default True) clips the grid to the required span."""

    def test_align_true_vs_false_changes_length(self) -> None:
        """align_to_required True yields <= the False frame count, and (with
        trim_to_valid off so the mechanisms don't overlap) strictly fewer.

        align=False spans the bag's full time range (reader.get_time_range);
        align=True clips to the intersection of the required features'
        [first,last] adopted-timestamp spans. The required topics in the HSR
        bag start after / end before the bag envelope, so True drops the
        boundary frames that lack full required coverage.
        """
        _require(HSR_BAG_LONG)

        # With trim_to_valid ON (production default): True <= False.
        true_trim = _process(HSR_BAG_LONG, _load_hsr_config(align_to_required=True))
        false_trim = _process(HSR_BAG_LONG, _load_hsr_config(align_to_required=False))
        assert true_trim, "align=True produced an empty episode"
        assert false_trim, "align=False produced an empty episode"
        assert len(true_trim) <= len(false_trim), (
            f"align=True ({len(true_trim)}) should be <= align=False ({len(false_trim)})"
        )

        # With trim_to_valid OFF the align effect is isolated and strict:
        # align=False keeps the leading/trailing frames that align=True drops.
        true_notrim = _process(
            HSR_BAG_LONG,
            _load_hsr_config(align_to_required=True, trim_to_valid=False),
        )
        false_notrim = _process(
            HSR_BAG_LONG,
            _load_hsr_config(align_to_required=False, trim_to_valid=False),
        )
        assert len(true_notrim) < len(false_notrim), (
            "align_to_required had no isolated effect (trim off): "
            f"True={len(true_notrim)} False={len(false_notrim)}"
        )

    def test_no_tail_hold_duplicates(self) -> None:
        """The episode does not end on a long plateau of hold-carried values.

        Under the `hold` policy a sparse/late-stopping feature could leave a
        tail of identical last-known values after its final genuine sample.
        align_to_required + trim_to_valid clip that tail. Assert the trailing
        run of byte-identical values on a required, reasonably dynamic feature
        is short.
        """
        _require(HSR_BAG_LONG)
        cfg = _load_hsr_config(align_to_required=True)
        frames = _process(HSR_BAG_LONG, cfg)
        assert len(frames) >= 10

        # observation.state (joint_states, ~50Hz) is dense and dynamic, so a
        # long trailing run of identical vectors would indicate a held tail.
        key = "observation.state"
        vals = [f.get(key) for f in frames]
        assert all(v is not None for v in vals), (
            f"required key {key} has gaps after trim_to_valid"
        )

        last = np.asarray(vals[-1])
        run = 0
        for v in reversed(vals):
            if np.array_equal(np.asarray(v), last):
                run += 1
            else:
                break
        # Allow the final frame itself plus at most a couple of genuinely-equal
        # samples; anything longer is a hold plateau the fix should have removed.
        assert run <= 3, (
            f"episode ends on {run} identical {key} frames (hold-tail not trimmed)"
        )


# ===========================================================================
# Fix 3 (cont.) — max_stamp_delay_ms stale-message drop
# ===========================================================================


@pytest.mark.integration
class TestMaxStampDelayDrop:
    """A strict max_stamp_delay_ms drops stale latched messages."""

    def test_strict_threshold_drops_frames_vs_none(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A tight max_stamp_delay_ms yields <= the None frame count and logs a drop.

        With max_stamp_delay_ms=None (default) no header-lag drop occurs. With
        a 50 ms threshold, messages whose header timestamp lags their bag
        receive time by more than 50 ms are dropped before decode. On the HSR
        bag this trips (camera/odom/joint head-lag >> 50 ms), so the converter
        emits a "dropped N stale message(s)" log and retains <= the baseline
        frame count.
        """
        import logging

        _require(HSR_BAG)

        baseline = _process(HSR_BAG, _load_hsr_config(max_stamp_delay_ms=None))
        assert baseline, "baseline (None) produced an empty episode"

        with caplog.at_level(logging.INFO, logger="rosbag2lerobot"):
            strict = _process(HSR_BAG, _load_hsr_config(max_stamp_delay_ms=50.0))

        assert len(strict) <= len(baseline), (
            f"strict ({len(strict)}) should be <= baseline ({len(baseline)})"
        )
        # The drop must be observable: a frame reduction OR an explicit log.
        dropped_logged = any(
            "stale message" in rec.getMessage() for rec in caplog.records
        )
        assert dropped_logged or len(strict) < len(baseline), (
            "max_stamp_delay_ms=50 neither reduced frames nor logged a stale "
            "drop on the HSR bag; if this bag has no laggy topics, switch the "
            "fixture to a bag with TRANSIENT_LOCAL latched topics."
        )


# ---------------------------------------------------------------------------
# Lightweight guards (pure config import + source string check; no bag I/O).
# ---------------------------------------------------------------------------


def test_resampling_config_has_alignment_fields() -> None:
    """ResamplingConfig exposes align_to_required + max_stamp_delay_ms.

    Defaults per config.py: align_to_required=True, max_stamp_delay_ms=None.
    """
    from rosbag2lerobot.config import ResamplingConfig

    rc = ResamplingConfig()
    assert hasattr(rc, "align_to_required")
    assert hasattr(rc, "max_stamp_delay_ms")
    assert rc.align_to_required is True
    assert rc.max_stamp_delay_ms is None


def test_video_permission_constant_is_world_rw() -> None:
    """writer.py normalises produced mp4 permissions to 0o666."""
    src = (PROJECT_ROOT / "src" / "rosbag2lerobot" / "writer.py").read_text()
    assert "0o666" in src, "writer.py should chmod produced mp4 to 0o666"
