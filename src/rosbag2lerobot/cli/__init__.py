"""CLI entry point for rosbag2lerobot.

Provides the following Click commands:

- ``convert``          -- Convert one or more ROS2 rosbags to a LeRobot v3.0
  dataset. Shows a tqdm ETA progress bar by default and writes
  ``meta/conversion_log.json`` (provenance) plus ``meta/job_summary.json``
  (run statistics). ``--json`` emits the job summary to stdout, ``--quiet``
  suppresses the progress bar / INFO chatter, and ``--skip-failed`` records a
  failed bag and continues instead of aborting the run. Supports ``--resume``
  as a safe re-run guard: converting into a non-empty ``--output`` without it
  aborts, and with it a crashed (non-finalized) output is wiped and rebuilt
  while a finalized one is left untouched.
- ``inspect``          -- Display topics, message counts, and time ranges of
  rosbags (with ``--fps-stats`` / ``--suggest-image-size`` diagnostics).
- ``scaffold``         -- Auto-generate a starter ``robot_config.yaml`` from an
  unknown robot's bag and (unless ``--no-validate``) validate it.
- ``validate-config``  -- Validate a YAML config against a rosbag's contents.
- ``validate-dataset`` -- Validate that a generated dataset conforms to the
  LeRobot Dataset v3.0 structure.
- ``quality-report``   -- Score the data quality of a generated dataset.
- ``audit-timestamps`` -- Audit timestamp continuity of a generated dataset.
- ``validate-msg``     -- Check a ``.msg`` file for syntactic correctness.
- ``preview``          -- Write a self-contained static HTML preview report
  (summary, quality score, sample frames, numeric stats) for a dataset.
- ``push-to-hub``      -- Upload a generated dataset to the HuggingFace Hub and
  generate a dataset card (opt-in; ``--dry-run`` plans the upload only).
- ``to-mcap``          -- Convert ROS1 ``.bag`` recordings to ROS2 MCAP bags.

All report commands (``validate-config`` / ``validate-dataset`` /
``quality-report`` / ``audit-timestamps`` / ``inspect`` / ``validate-msg`` /
``to-mcap``) accept ``--json`` to emit their report dict to stdout instead of
the human-readable summary.

Usage::

    rosbag2lerobot convert --config my_config.yaml --bags /bags/ --output /out/
    rosbag2lerobot scaffold --bags /bags/ -o robot_config.yaml
    rosbag2lerobot inspect --bags /bags/
    rosbag2lerobot validate-dataset --dataset /out/
    rosbag2lerobot quality-report --dataset /out/
    rosbag2lerobot validate-msg --msg msgs/MyType.msg
    rosbag2lerobot preview --dataset /out/
    rosbag2lerobot push-to-hub --dataset /out/ --dry-run
"""

from rosbag2lerobot.cli._common import _detect_nvenc
from rosbag2lerobot.cli.convert import (
    _iter_episodes_parallel,
    _iter_episodes_serial,
    _process_episode,
    _required_window,
)
from rosbag2lerobot.cli.main import main
from rosbag2lerobot.cli.scaffold import (
    _dedupe_key,
    _pick_target_fps,
    _scaffold_from_topics,
    _slug_from_topic,
    scaffold_config_yaml,
)

__all__ = [
    "_detect_nvenc",
    "_iter_episodes_parallel",
    "_iter_episodes_serial",
    "_process_episode",
    "_required_window",
    "main",
    "_dedupe_key",
    "_pick_target_fps",
    "_scaffold_from_topics",
    "_slug_from_topic",
    "scaffold_config_yaml",
]
