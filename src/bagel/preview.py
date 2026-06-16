"""Read-only static HTML preview report for LeRobot v3.0 datasets.

Renders a single self-contained HTML page summarising a finished dataset:
the summary (robot type / fps / episodes / frames / tasks), the quality score
and per-feature/per-video quality tables (reusing :mod:`bagel.quality`), a
gallery of sampled video frames (inline base64 JPEGs, no external requests),
and the numeric per-feature statistics from ``meta/stats.json``.

The page is intentionally *self-contained* — all CSS is inlined and every
image is a ``data:`` URI — so it can be opened straight from disk or shipped
to a reviewer without any server.

Design split (mirroring :mod:`bagel.diagnostics` / :mod:`bagel.audit`):

- :func:`build_preview_html` is **pure**: it takes plain dicts in and returns
  an HTML string. It performs no I/O and never touches the filesystem, so the
  future interactive UI can reuse it verbatim as its "results view".
- The I/O lives in the helpers (:func:`_grab_sample_frames`,
  :func:`_frames_to_base64_jpeg`) and the :func:`generate_preview`
  orchestrator, which read the dataset and call the pure builder.

Writing the HTML to disk is the caller's job; :func:`generate_preview` returns
the string so it stays testable without a filesystem.
"""

from __future__ import annotations

import base64
import html
import io
import itertools
import logging
from pathlib import Path
from typing import Any

import numpy as np

from bagel.quality import (
    _decode_video_frames,
    _load_stats,
    _read_info,
    compute_quality_report,
)
from bagel.validation import video_feature_keys

logger = logging.getLogger(__name__)

__all__ = [
    "build_preview_html",
    "generate_preview",
]


# ---------------------------------------------------------------------------
# Pure HTML builder (no I/O — UI-reusable)
# ---------------------------------------------------------------------------

_STYLE = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
       padding: 24px; color: #1a1a1a; background: #fafafa; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 28px 0 10px; border-bottom: 1px solid #ddd;
     padding-bottom: 4px; }
table { border-collapse: collapse; width: 100%; background: #fff;
        font-size: 13px; }
th, td { border: 1px solid #e0e0e0; padding: 5px 9px; text-align: right; }
th { background: #f0f0f0; text-align: left; }
td:first-child, th:first-child { text-align: left; }
.summary td:first-child { font-weight: 600; width: 180px; }
.summary td { text-align: left; }
.badge { display: inline-block; padding: 6px 14px; border-radius: 6px;
         color: #fff; font-weight: 700; font-size: 15px; }
.badge.ok { background: #2e7d32; }
.badge.fail { background: #c62828; }
.cam-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 12px; }
.cam-row img { height: 160px; border: 1px solid #ccc; border-radius: 4px; }
.cam-key { font-weight: 600; margin-top: 8px; }
.muted { color: #666; font-size: 12px; }
"""


def _fmt_num(value: Any) -> str:
    """Format a scalar/array stat cell compactly for the numeric table.

    Args:
        value: A scalar number, a 1-element sequence, or a multi-dim sequence.

    Returns:
        A short string: ``"-"`` when empty, a 4-dp float for scalars, or the
        per-dimension values joined by ``", "`` for vectors.
    """
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.size == 0:
        return "-"
    if arr.size == 1:
        return f"{float(arr[0]):.4f}"
    return ", ".join(f"{float(x):.4f}" for x in arr)


def _summary_section(info: dict[str, Any]) -> str:
    """Render the summary table from ``info.json`` fields."""
    rows = [
        ("Robot type", info.get("robot_type", "-")),
        ("FPS", info.get("fps", "-")),
        ("Episodes", info.get("total_episodes", "-")),
        ("Frames", info.get("total_frames", "-")),
        ("Tasks", info.get("total_tasks", "-")),
        ("Codebase version", info.get("codebase_version", "-")),
    ]
    body = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in rows
    )
    return f'<h2>Summary</h2>\n<table class="summary"><tbody>{body}</tbody></table>'


def _quality_section(quality: dict[str, Any]) -> str:
    """Render the quality score badge plus per-feature and per-video tables."""
    verdict = str(quality.get("verdict", "OK"))
    score = float(quality.get("score", 1.0))
    threshold = float(quality.get("score_threshold", 0.95))
    badge_cls = "ok" if verdict == "OK" else "fail"
    badge = (
        f'<span class="badge {badge_cls}">{html.escape(verdict)} '
        f"&middot; score {score:.4f}</span> "
        f'<span class="muted">threshold {threshold:.4f}</span>'
    )

    feat_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(f.get('feature', '')))}</td>"
        f"<td>{int(f.get('n_null', 0))}</td>"
        f"<td>{int(f.get('n_nan', 0))}</td>"
        f"<td>{float(f.get('null_rate', 0.0)):.4f}</td>"
        f"<td>{int(f.get('n_out_of_range', 0))}</td>"
        f"<td>{float(f.get('oor_rate', 0.0)):.4f}</td>"
        "</tr>"
        for f in quality.get("features", [])
    )
    feat_table = (
        "<table><thead><tr><th>feature</th><th>n_null</th><th>n_nan</th>"
        "<th>null_rate</th><th>n_out_of_range</th><th>oor_rate</th></tr></thead>"
        f"<tbody>{feat_rows}</tbody></table>"
    )

    vid_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(v.get('video_key', '')))}</td>"
        f"<td>{int(v.get('expected_frames', 0))}</td>"
        f"<td>{int(v.get('mp4_frames', 0))}</td>"
        f"<td>{int(v.get('frame_mismatch', 0))}</td>"
        f"<td>{int(v.get('n_freeze', 0))}</td>"
        f"<td>{float(v.get('freeze_rate', 0.0)):.4f}</td>"
        "</tr>"
        for v in quality.get("videos", [])
    )
    vid_table = (
        "<table><thead><tr><th>video_key</th><th>expected_frames</th>"
        "<th>mp4_frames</th><th>frame_mismatch</th><th>n_freeze</th>"
        "<th>freeze_rate</th></tr></thead>"
        f"<tbody>{vid_rows}</tbody></table>"
    )

    parts = [f"<h2>Quality</h2>\n<p>{badge}</p>", feat_table]
    if quality.get("videos"):
        parts.append("<h2>Video reconciliation</h2>")
        parts.append(vid_table)
    return "\n".join(parts)


def _cameras_section(frames_b64: dict[str, list[str]]) -> str:
    """Render one row of inline base64 JPEG thumbnails per video key."""
    if not frames_b64:
        return ""
    blocks = ["<h2>Cameras</h2>"]
    for video_key, imgs in frames_b64.items():
        blocks.append(f'<div class="cam-key">{html.escape(video_key)}</div>')
        cells = "".join(
            f'<img src="data:image/jpeg;base64,{b64}" alt="{html.escape(video_key)}">'
            for b64 in imgs
        )
        blocks.append(f'<div class="cam-row">{cells}</div>')
    return "\n".join(blocks)


def _stats_section(stats: dict[str, Any]) -> str:
    """Render the numeric per-feature statistics table from ``stats.json``."""
    rows = []
    for feature, s in stats.items():
        if not isinstance(s, dict) or "min" not in s:
            continue
        dim = int(np.asarray(s["min"], dtype=np.float64).ravel().size)
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(feature))}</td>"
            f"<td>{dim}</td>"
            f"<td>{_fmt_num(s.get('min'))}</td>"
            f"<td>{_fmt_num(s.get('max'))}</td>"
            f"<td>{_fmt_num(s.get('mean'))}</td>"
            f"<td>{_fmt_num(s.get('std'))}</td>"
            f"<td>{_fmt_num(s.get('q50'))}</td>"
            "</tr>"
        )
    body = "".join(rows)
    return (
        "<h2>Numeric statistics</h2>\n"
        "<table><thead><tr><th>feature</th><th>dim</th><th>min</th>"
        "<th>max</th><th>mean</th><th>std</th><th>q50</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def build_preview_html(
    info: dict[str, Any],
    stats: dict[str, Any],
    quality: dict[str, Any],
    frames_b64: dict[str, list[str]],
) -> str:
    """Build a self-contained HTML preview report from plain dicts.

    Pure function: no I/O, no :class:`~pathlib.Path`. All inputs are plain
    dicts so the future interactive UI can reuse this as its results view.

    Args:
        info: ``meta/info.json`` contents (robot_type / fps / totals /
            features).
        stats: ``meta/stats.json`` contents (per-feature min/max/mean/std/q50).
        quality: :meth:`bagel.quality.QualityReport.to_dict` output (score,
            verdict, per-feature and per-video tables).
        frames_b64: ``{video_key: [base64_jpeg, ...]}`` sample frames; each
            value becomes a row of inline ``data:image/jpeg;base64`` thumbnails.

    Returns:
        A complete, self-contained HTML document string (inline ``<style>``,
        no external scripts or stylesheets).
    """
    robot = html.escape(str(info.get("robot_type", "dataset")))
    sections = [
        _summary_section(info),
        _quality_section(quality),
        _cameras_section(frames_b64),
        _stats_section(stats),
    ]
    body = "\n".join(s for s in sections if s)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>bagel preview — {robot}</title>\n"
        f"<style>{_STYLE}</style>\n"
        "</head>\n<body>\n"
        f"<h1>bagel dataset preview — {robot}</h1>\n"
        f"{body}\n"
        "</body>\n</html>\n"
    )


# ---------------------------------------------------------------------------
# I/O helpers (kept out of the pure builder)
# ---------------------------------------------------------------------------


def _grab_sample_frames(
    dataset_dir: Path,
    video_keys: list[str],
    n_per_video: int,
) -> dict[str, list[np.ndarray]]:
    """Decode the first ``n_per_video`` frames of each video key's first mp4.

    Uses the streaming :func:`bagel.quality._decode_video_frames` generator and
    :func:`itertools.islice` so only the leading frames are decoded (a full
    decode is avoided). The first mp4 per key is the one at the lowest
    ``chunk-*/file-*`` lexicographic position.

    Args:
        dataset_dir: Root of a LeRobot v3.0 dataset.
        video_keys: Video feature keys (``info`` features with ``dtype ==
            "video"``).
        n_per_video: Number of leading frames to decode per video key.

    Returns:
        ``{video_key: [frame, ...]}`` of ``(H, W, 3)`` uint8 RGB ndarrays.
        Keys with no mp4 file are omitted.
    """
    out: dict[str, list[np.ndarray]] = {}
    for vk in video_keys:
        vid_root = dataset_dir / "videos" / vk
        mp4s = sorted(vid_root.rglob("*.mp4"))
        if not mp4s:
            logger.warning("preview: no mp4 found for video key %s", vk)
            continue
        gen = _decode_video_frames(mp4s[0])
        frames = list(itertools.islice(gen, n_per_video))
        # Close the generator deliberately: stopping early sends SIGPIPE to the
        # underlying ffmpeg, whose nonzero exit makes the generator's cleanup
        # raise. That broken-pipe error is expected when we only wanted the
        # first N frames, so swallow it (only the early-stop case).
        try:
            gen.close()
        except RuntimeError as exc:
            logger.debug("preview: ignoring early-stop ffmpeg cleanup: %s", exc)
        if frames:
            out[vk] = frames
    return out


def _frames_to_base64_jpeg(
    frames: dict[str, list[np.ndarray]],
    max_width: int = 320,
    quality: int = 70,
) -> dict[str, list[str]]:
    """Downscale frames to ~``max_width`` px wide and base64-encode as JPEG.

    Args:
        frames: ``{video_key: [(H, W, 3) uint8 RGB, ...]}`` decoded frames.
        max_width: Target width in pixels; frames wider than this are scaled
            down preserving aspect ratio (narrower frames are left as-is).
        quality: JPEG quality passed to PIL ``save``.

    Returns:
        ``{video_key: [base64_str, ...]}`` (raw base64, no ``data:`` prefix).
    """
    from PIL import Image

    out: dict[str, list[str]] = {}
    for vk, arrs in frames.items():
        encoded: list[str] = []
        for arr in arrs:
            img = Image.fromarray(np.asarray(arr, dtype=np.uint8))
            if img.width > max_width:
                new_h = round(img.height * max_width / img.width)
                img = img.resize((max_width, new_h), Image.BILINEAR)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=quality)
            encoded.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        out[vk] = encoded
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def generate_preview(
    dataset_dir: Path,
    n_frames: int = 3,
    sample_video: bool = False,
) -> str:
    """Build the HTML preview report for a generated dataset.

    Reads ``meta/info.json`` and ``meta/stats.json``, computes a quality
    report, samples a few leading frames per video, base64-encodes them, and
    assembles the page via the pure :func:`build_preview_html`. Returning the
    HTML string (rather than writing it) keeps this testable without a
    filesystem; persisting the result is the caller's responsibility.

    Args:
        dataset_dir: Root of a LeRobot v3.0 dataset.
        n_frames: Number of sample frames to embed per video key.
        sample_video: Forwarded to :func:`bagel.quality.compute_quality_report`
            for the *quality computation's* freeze-frame decode. The frame
            gallery is sampled independently regardless of this flag. Defaults
            to ``False``: the score stays meaningful and freeze metrics are
            informational only.

    Returns:
        A self-contained HTML document string.
    """
    dataset_dir = Path(dataset_dir)
    info = _read_info(dataset_dir)
    stats = _load_stats(dataset_dir)
    quality = compute_quality_report(
        dataset_dir, sample_video=sample_video, info=info, stats=stats
    ).to_dict()

    video_keys = video_feature_keys(info)
    raw_frames = _grab_sample_frames(dataset_dir, video_keys, n_frames)
    frames_b64 = _frames_to_base64_jpeg(raw_frames)

    return build_preview_html(info, stats, quality, frames_b64)
