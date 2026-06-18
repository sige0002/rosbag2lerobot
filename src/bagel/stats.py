"""Statistics computation for LeRobot Dataset v3.0.

Computes per-feature min, max, mean, std, count, and quantile estimates
(0.01, 0.10, 0.50, 0.90, 0.99), matching the algorithm used by lerobot's
``RunningQuantileStats`` (``lerobot/datasets/compute_stats.py``):

- mean / std via an incremental mean and mean-of-squares (population std),
- quantiles via per-dimension histograms with dynamic re-binning when the
  observed range expands, and in-bin linear interpolation.

Frames are buffered per feature and flushed to the running accumulator in
batches so that image features contribute *every pixel* as a per-channel
sample — exactly like lerobot — instead of a single per-frame spatial mean.
``_prepare_values`` reduces an image ``[H, W, C]`` to ``(H*W, C)`` (all pixels,
normalized to [0, 1], spatially downsampled like lerobot for large frames) and
a numeric vector to ``(1, D)``.  The reported ``count`` is the number of
*frames* (matching lerobot's per-episode count), even though image statistics
are computed over pixels.

The ``StatsComputer`` class accumulates statistics incrementally so the full
dataset never needs to reside in memory; each feature's buffer is bounded by
``_FLUSH_ROWS`` rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Number of histogram bins for quantile estimation (matches lerobot's
# ``RunningQuantileStats`` default ``num_quantile_bins``).
_HISTOGRAM_BINS = 5_000
# Quantile levels required by LeRobot v3.0.
_QUANTILE_LEVELS = (0.01, 0.10, 0.50, 0.90, 0.99)
# Flush a feature's frame buffer once it holds this many rows. Bounds memory
# for image features (many pixels per frame) while letting numeric features
# (one row per frame) accumulate and usually flush a single batch at compute().
_FLUSH_ROWS = 500_000
# Image downsampling parameters (mirror lerobot's ``auto_downsample_height_width``).
_DOWNSAMPLE_TARGET = 150
_DOWNSAMPLE_THRESHOLD = 300


def _downsample_image(img: np.ndarray) -> np.ndarray:
    """Spatially stride-downsample an ``[H, W, C]`` image (lerobot-compatible).

    Frames whose largest side is below ``_DOWNSAMPLE_THRESHOLD`` are returned
    unchanged.  Otherwise a stride factor is chosen so the larger side maps to
    roughly ``_DOWNSAMPLE_TARGET`` pixels, matching lerobot's
    ``auto_downsample_height_width`` (which operates on ``[C, H, W]``).

    Args:
        img: Image array of shape ``[H, W, C]``.

    Returns:
        The (possibly) downsampled image, same channel count.
    """
    h, w = img.shape[:2]
    if max(h, w) < _DOWNSAMPLE_THRESHOLD:
        return img
    factor = int(w / _DOWNSAMPLE_TARGET) if w > h else int(h / _DOWNSAMPLE_TARGET)
    factor = max(factor, 1)
    return img[::factor, ::factor]


@dataclass
class _FeatureAccumulator:
    """Running statistics for a single feature key.

    Holds an incremental mean / mean-of-squares, running min/max, and a
    per-dimension histogram for quantile estimation.  Pending frames are
    buffered and folded in as batches (see ``_FLUSH_ROWS``).
    """

    # Running state (populated on the first flush).
    count: int = 0  # number of rows seen (pixels for images)
    frames: int = 0  # number of frames (add_frame calls) — the reported count
    mean: np.ndarray | None = None
    mean_sq: np.ndarray | None = None  # incremental mean of squares
    min_val: np.ndarray | None = None
    max_val: np.ndarray | None = None
    hist: list[np.ndarray] | None = None  # per-dimension histogram counts
    edges: list[np.ndarray] | None = None  # per-dimension bin edges (len bins+1)
    # Pending frame buffer, flushed in batches.
    _buffer: list[np.ndarray] = field(default_factory=list)
    _buffer_rows: int = 0
    _buffer_frames: int = 0


class StatsComputer:
    """Accumulates running statistics per feature key.

    Supports both numeric (float32) vector features and image/video features.
    Image pixel values are normalized to [0, 1] before accumulation.
    """

    def __init__(self) -> None:
        self._accumulators: dict[str, _FeatureAccumulator] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_frame(self, feature_key: str, values: np.ndarray) -> None:
        """Add one frame of data for *feature_key*.

        Args:
            feature_key: Feature name, e.g. "observation.state".
            values: 1-D float32 array (numeric) or 3-D uint8/float array (image).
                    Images contribute every pixel as a per-channel sample.
        """
        batch = self._prepare_values(values)
        acc = self._accumulators.setdefault(feature_key, _FeatureAccumulator())
        acc._buffer.append(batch)
        acc._buffer_rows += batch.shape[0]
        acc._buffer_frames += 1
        if acc._buffer_rows >= _FLUSH_ROWS:
            self._flush(acc)

    def compute(self) -> dict[str, dict[str, list[float]]]:
        """Return the final statistics dictionary.

        Returns:
            ``{feature_key: {min, max, mean, std, count, q01, q10, q50, q90, q99}}``.
            Each stat (except ``count``) is a per-dimension list of floats;
            ``count`` is a single-element list with the frame count.
        """
        result: dict[str, dict[str, list[float]]] = {}
        for key, acc in self._accumulators.items():
            self._flush(acc)
            if acc.count == 0:
                continue
            variance = acc.mean_sq - acc.mean**2
            std = np.sqrt(np.maximum(0.0, variance))  # population std
            quantiles = self._estimate_quantiles(acc)
            result[key] = {
                "min": acc.min_val.tolist(),
                "max": acc.max_val.tolist(),
                "mean": acc.mean.tolist(),
                "std": std.tolist(),
                "count": [acc.frames],
                "q01": quantiles[0.01].tolist(),
                "q10": quantiles[0.10].tolist(),
                "q50": quantiles[0.50].tolist(),
                "q90": quantiles[0.90].tolist(),
                "q99": quantiles[0.99].tolist(),
            }
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_values(values: np.ndarray) -> np.ndarray:
        """Normalize a frame to a 2-D ``(N, D)`` float32 batch.

        Numeric vectors become a single row ``(1, D)``; images ``[H, W, C]``
        become ``(H*W, C)`` (every pixel, normalized to [0, 1], downsampled for
        large frames like lerobot).
        """
        arr = np.asarray(values)
        # Image: [H, W, C] -> every pixel as a per-channel sample.
        if arr.ndim == 3:
            arr = _downsample_image(arr)
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            else:
                arr = arr.astype(np.float32)
            return arr.reshape(-1, arr.shape[-1])
        # Numeric scalar or vector -> one row.
        arr = arr.astype(np.float32).ravel()
        return arr.reshape(1, -1)

    def _flush(self, acc: _FeatureAccumulator) -> None:
        """Fold the pending frame buffer into the running accumulator."""
        if not acc._buffer:
            return
        batch = np.concatenate(acc._buffer, axis=0)
        self._update(acc, batch)
        acc.frames += acc._buffer_frames
        acc._buffer.clear()
        acc._buffer_rows = 0
        acc._buffer_frames = 0

    def _update(self, acc: _FeatureAccumulator, batch: np.ndarray) -> None:
        """Update running mean/mean_sq/min/max and histograms with a batch.

        Mirrors lerobot's ``RunningQuantileStats.update``: incremental mean and
        mean-of-squares, min/max tracking, dynamic histogram re-binning when the
        range expands, then per-dimension histogram accumulation.
        """
        n, d = batch.shape
        batch_min = batch.min(axis=0)
        batch_max = batch.max(axis=0)
        batch_mean = batch.mean(axis=0, dtype=np.float64)
        batch_mean_sq = np.mean(batch.astype(np.float64) ** 2, axis=0)

        if acc.count == 0:
            acc.mean = batch_mean
            acc.mean_sq = batch_mean_sq
            acc.min_val = batch_min.copy()
            acc.max_val = batch_max.copy()
            acc.hist = [np.zeros(_HISTOGRAM_BINS, dtype=np.int64) for _ in range(d)]
            acc.edges = [
                np.linspace(
                    batch_min[i] - 1e-10, batch_max[i] + 1e-10, _HISTOGRAM_BINS + 1
                )
                for i in range(d)
            ]
            acc.count = n
        else:
            expanded = bool(
                np.any(batch_max > acc.max_val) or np.any(batch_min < acc.min_val)
            )
            acc.max_val = np.maximum(acc.max_val, batch_max)
            acc.min_val = np.minimum(acc.min_val, batch_min)
            if expanded:
                self._rebin(acc)
            acc.count += n
            weight = n / acc.count
            acc.mean += (batch_mean - acc.mean) * weight
            acc.mean_sq += (batch_mean_sq - acc.mean_sq) * weight

        # Histograms cover the (now up to date) global range, so no value is
        # dropped by np.histogram's outside-range clipping.
        for i in range(d):
            counts, _ = np.histogram(batch[:, i], bins=acc.edges[i])
            acc.hist[i] += counts

    def _rebin(self, acc: _FeatureAccumulator) -> None:
        """Re-bin every histogram to the expanded ``[min, max]`` range.

        Redistributes existing counts by mapping each old bin centre into the
        new bins (vectorized form of lerobot's ``_adjust_histograms``).
        """
        for i in range(len(acc.hist)):
            old_edges = acc.edges[i]
            old_hist = acc.hist[i]
            lo = acc.min_val[i]
            hi = acc.max_val[i]
            padding = (hi - lo) * 1e-10
            new_edges = np.linspace(lo - padding, hi + padding, _HISTOGRAM_BINS + 1)
            old_centers = (old_edges[:-1] + old_edges[1:]) / 2
            idx = np.clip(
                np.searchsorted(new_edges, old_centers) - 1, 0, _HISTOGRAM_BINS - 1
            )
            new_hist = np.zeros(_HISTOGRAM_BINS, dtype=np.int64)
            np.add.at(new_hist, idx, old_hist)
            acc.hist[i] = new_hist
            acc.edges[i] = new_edges

    def _estimate_quantiles(self, acc: _FeatureAccumulator) -> dict[float, np.ndarray]:
        """Estimate quantiles per dimension via histogram in-bin interpolation."""
        d = len(acc.mean)
        result: dict[float, np.ndarray] = {
            q: np.zeros(d, dtype=np.float64) for q in _QUANTILE_LEVELS
        }
        for i in range(d):
            hist = acc.hist[i]
            edges = acc.edges[i]
            cumsum = np.cumsum(hist)
            total = cumsum[-1]
            for q in _QUANTILE_LEVELS:
                if total == 0:
                    result[q][i] = float(acc.mean[i])
                else:
                    result[q][i] = self._single_quantile(edges, cumsum, q * total)
        return result

    @staticmethod
    def _single_quantile(edges: np.ndarray, cumsum: np.ndarray, target: float) -> float:
        """Interpolate a single quantile value from a histogram cumulative sum.

        Mirrors lerobot's ``_compute_single_quantile``.
        """
        idx = int(np.searchsorted(cumsum, target))
        if idx <= 0:
            return float(edges[0])
        if idx >= len(cumsum):
            return float(edges[-1])
        count_before = cumsum[idx - 1]
        count_in_bin = cumsum[idx] - count_before
        if count_in_bin == 0:
            return float(edges[idx])
        fraction = (target - count_before) / count_in_bin
        return float(edges[idx] + fraction * (edges[idx + 1] - edges[idx]))
