"""Conversion manifest helpers (plan.md D-2).

Pure, I/O-light building blocks for the ``meta/conversion_log.json`` file
written alongside a converted dataset. The provenance manifest records, per
run, the input bags (path + content hash + frame count + processing time),
the effective ffmpeg encode settings, the embedded config text and its hash,
and tool/ffmpeg versions.

Design rules (CLAUDE.md):

- :func:`build_manifest` is **pure**: it takes every value (including the
  ``run_timestamp``) as an argument and performs no I/O, so tests can pin a
  fixed timestamp and assert byte-for-byte output.
- I/O is isolated to :func:`sha256_of_path` (file hashing) and
  :func:`ffmpeg_version` (subprocess). Both keep ``stdin`` away from the TTY.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Files that describe the bag container itself rather than its recorded
# payload. They are excluded from the content hash so re-recording metadata
# (e.g. a re-serialized ``metadata.yaml``) does not change the digest.
_METADATA_FILENAME = "metadata.yaml"

# Read chunk size for streaming file hashing (1 MiB).
_HASH_CHUNK_BYTES = 1024 * 1024

# Manifest keys the tool writes itself: the writer-owned encode/output facts
# (``writer.DatasetWriter._write_conversion_log``) plus the provenance the CLI
# assembles (:func:`build_manifest`). Caller-supplied entries — e.g. the
# ``convert --manifest-extra`` JSON file — never overwrite these, so the
# manifest cannot be made to lie about how the dataset was produced.
BUILTIN_MANIFEST_KEYS = frozenset(
    {
        # writer-owned
        "codec",
        "codec_label",
        "ffmpeg_preset",
        "ffmpeg_crf",
        "fps",
        "total_episodes",
        "total_frames",
        "episode_lengths",
        # CLI-owned provenance
        "inputs",
        "config_snapshot",
        "config_sha256",
        "rosbag2lerobot_version",
        "ffmpeg_version",
        "run_timestamp",
    }
)


@dataclass
class ManifestInput:
    """One input bag's provenance entry in the conversion manifest.

    Attributes:
        path: Bag directory (or file) path as a string.
        sha256: Hex digest over the bag's storage files (see
            :func:`sha256_of_path`).
        frame_count: Number of frames the bag contributed to the dataset.
        processing_time_s: Wall time spent decoding/resampling this bag.
    """

    path: str
    sha256: str
    frame_count: int
    processing_time_s: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of this entry."""
        return asdict(self)


def sha256_of_path(path: str | Path) -> str:
    """Hash a bag's storage files into a single hex digest.

    For a directory, every contained file except ``metadata.yaml`` is hashed
    in name-sorted order (relative to *path*) into one rolling SHA-256, so the
    digest depends only on the recorded payload and is stable across runs. The
    relative file name is folded into the hash before its bytes to disambiguate
    files whose contents would otherwise concatenate ambiguously. For a single
    file, that file's bytes are hashed directly.

    Args:
        path: Bag directory or file to hash.

    Returns:
        A 64-character lowercase hex SHA-256 digest.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path to hash not found: {p}")

    digest = hashlib.sha256()
    if p.is_file():
        _update_with_file(digest, p)
        return digest.hexdigest()

    files = sorted(
        (f for f in p.rglob("*") if f.is_file() and f.name != _METADATA_FILENAME),
        key=lambda f: str(f.relative_to(p)),
    )
    for f in files:
        # Fold the relative name in first so structure is part of the digest.
        digest.update(str(f.relative_to(p)).encode("utf-8"))
        _update_with_file(digest, f)
    return digest.hexdigest()


def _update_with_file(digest: "hashlib._Hash", file_path: Path) -> None:
    """Stream-read *file_path* into *digest* in fixed-size chunks."""
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)


def ffmpeg_version() -> str | None:
    """Return ffmpeg's reported version string, or ``None`` if unavailable.

    Shells out to ``ffmpeg -version`` (mirroring ``cli._detect_nvenc``: TTY is
    detached via ``stdin=DEVNULL``) and returns the first stdout line. Returns
    ``None`` when ffmpeg is missing or the call times out.

    Returns:
        The first line of ``ffmpeg -version`` output (e.g.
        ``"ffmpeg version 6.0 ..."``), or ``None``.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            stdin=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    first_line = result.stdout.splitlines()[0].strip() if result.stdout else ""
    return first_line or None


def load_manifest_extra(path: str | Path) -> dict[str, Any]:
    """Load a caller-supplied manifest fragment from a JSON file.

    Used by ``convert --manifest-extra`` to let an operator (or an automation
    wrapping the CLI) record its own provenance — ticket ids, dataset labels,
    upstream job ids — inside ``meta/conversion_log.json``.

    Args:
        path: Path to a JSON file whose root is an object.

    Returns:
        The parsed object as a dict.

    Raises:
        ValueError: If the file cannot be read, is not valid JSON, or its root
            is not a JSON object. The message is phrased for direct display to
            the user, since the CLI surfaces it before conversion starts.
    """
    p = Path(path)
    try:
        text = p.read_text()
    except OSError as exc:
        raise ValueError(f"Cannot read manifest extra file {p}: {exc}") from exc
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest extra file {p} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Manifest extra file {p} must contain a JSON object at the root, "
            f"got {type(parsed).__name__}."
        )
    return parsed


def strip_builtin_keys(extra: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Remove keys owned by the tool from a caller-supplied manifest fragment.

    Built-ins win: a caller cannot overwrite ``codec``, ``config_sha256``,
    ``total_frames`` and friends, because those describe what actually
    happened during the run.

    Args:
        extra: Caller-supplied manifest fragment (see
            :func:`load_manifest_extra`).

    Returns:
        ``(filtered, dropped)`` — the fragment without built-in keys, and the
        sorted list of key names that were dropped (for a warning).
    """
    filtered = {k: v for k, v in extra.items() if k not in BUILTIN_MANIFEST_KEYS}
    dropped = sorted(k for k in extra if k in BUILTIN_MANIFEST_KEYS)
    return filtered, dropped


def build_manifest(
    *,
    inputs: list[ManifestInput],
    codec: str,
    ffmpeg_preset: str | None,
    ffmpeg_crf: int | None,
    total_episodes: int,
    total_frames: int,
    fps: int,
    config_snapshot: str,
    config_sha256: str,
    rosbag2lerobot_version: str,
    ffmpeg_version: str | None,
    run_timestamp: str,
) -> dict[str, Any]:
    """Build the conversion manifest dict (pure, no I/O).

    Every value — including the ``run_timestamp`` — is injected, so the result
    is a deterministic function of its arguments. The CLI computes the I/O-bound
    pieces (file hashes, wall-clock timestamp) and passes them in.

    Args:
        inputs: Per-bag provenance entries.
        codec: ffmpeg encoder name used for video (e.g. ``"libx264"``).
        ffmpeg_preset: Effective ffmpeg ``-preset`` (``None`` = codec default).
        ffmpeg_crf: Effective quality value (``None`` = codec default).
        total_episodes: Episodes written to the dataset.
        total_frames: Frames written to the dataset.
        fps: Dataset frames-per-second.
        config_snapshot: Full text of the config YAML used for the run.
        config_sha256: SHA-256 hex digest of the config YAML bytes.
        rosbag2lerobot_version: ``rosbag2lerobot.__version__`` of the running tool.
        ffmpeg_version: ffmpeg version line (``None`` if unavailable).
        run_timestamp: ISO-8601 UTC timestamp string for the run.

    Returns:
        A JSON-serializable manifest dict.
    """
    return {
        "rosbag2lerobot_version": rosbag2lerobot_version,
        "ffmpeg_version": ffmpeg_version,
        "run_timestamp": run_timestamp,
        "codec": codec,
        "ffmpeg_preset": ffmpeg_preset,
        "ffmpeg_crf": ffmpeg_crf,
        "fps": fps,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "config_sha256": config_sha256,
        "config_snapshot": config_snapshot,
        "inputs": [i.to_dict() if isinstance(i, ManifestInput) else i for i in inputs],
    }
