"""ROS1 ``.bag`` to ROS2 MCAP conversion for rosbag2lerobot.

rosbag2lerobot's reader only handles ROS2 bags (mcap/sqlite3).  ROS1 ``.bag``
recordings (e.g. the airoa raw dataset) must be converted to ROS2 MCAP
before they can be fed to ``rosbag2lerobot convert``.

This module wraps the conversion API of the ``rosbags`` library (already a
runtime dependency) -- the same code path used by the ``rosbags-convert``
CLI -- so no extra dependency and no ROS install is required.

Public API:

- ``is_ros1_bag()``        -- Magic-byte check for a ROS1 ``.bag`` file.
- ``discover_ros1_bags()`` -- Expand files/directories into ROS1 bag paths.
- ``convert_to_mcap()``    -- Convert one ROS1 bag to a ROS2 MCAP bag.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from rosbags.convert import convert as _rosbags_convert

logger = logging.getLogger(__name__)

# ROS1 bag files start with this magic string (rosbag format 2.0).
_ROS1_MAGIC = b"#ROSBAG V2.0"

# Default ROS2 bag format version written by rosbags-convert.
DEFAULT_DST_VERSION = 9


def is_ros1_bag(path: Path) -> bool:
    """Return True if *path* is a ROS1 ``.bag`` file (by magic bytes)."""
    if not path.is_file() or path.suffix != ".bag":
        return False
    try:
        with path.open("rb") as fh:
            return fh.read(len(_ROS1_MAGIC)) == _ROS1_MAGIC
    except OSError:
        return False


def discover_ros1_bags(sources: list[Path]) -> list[Path]:
    """Expand *sources* into a sorted list of ROS1 ``.bag`` files.

    Each source may be a ``.bag`` file or a directory; directories are
    searched recursively for ``*.bag`` files.  Non-ROS1 files are skipped.
    """
    found: list[Path] = []
    for src in sources:
        if src.is_dir():
            candidates = sorted(src.rglob("*.bag"))
        else:
            candidates = [src]
        for c in candidates:
            if is_ros1_bag(c):
                found.append(c)
            else:
                logger.warning("Skipping non-ROS1 file: %s", c)
    # De-duplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def output_name(src: Path) -> str:
    """Derive an output bag directory name for a ROS1 bag *src*.

    Generic filenames like ``data.bag`` carry no identity, so the parent
    directory name is used (e.g. ``.../235210/data.bag`` -> ``235210``).
    Otherwise the file stem is used (e.g. ``recording_01.bag`` -> ``recording_01``).
    """
    if src.stem in {"data", "bag", "rosbag"}:
        return src.parent.name
    return src.stem


def convert_to_mcap(
    src: Path,
    dst_dir: Path,
    *,
    dst_version: int = DEFAULT_DST_VERSION,
    overwrite: bool = False,
) -> Path:
    """Convert one ROS1 bag *src* into a ROS2 MCAP bag directory *dst_dir*.

    Returns the destination directory.  Raises ``FileExistsError`` if
    *dst_dir* already exists and *overwrite* is False.
    """
    if dst_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {dst_dir} (use --overwrite to replace)"
            )
        shutil.rmtree(dst_dir)

    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Converting %s -> %s (mcap)", src, dst_dir)
    _rosbags_convert(
        srcs=[src],
        dst=dst_dir,
        dst_storage="mcap",
        dst_version=dst_version,
        compress=None,
        compress_mode="file",
        default_typestore=None,
        typestore=None,
        exclude_topics=[],
        include_topics=[],
        exclude_msgtypes=[],
        include_msgtypes=[],
    )
    return dst_dir
