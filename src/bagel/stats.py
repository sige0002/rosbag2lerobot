"""Statistics computation for LeRobot Dataset v3.0.

Computes per-feature min, max, mean, std, count, and quantile estimates
using Welford's online algorithm for mean/variance and histogram-based
interpolation for quantiles (0.01, 0.10, 0.50, 0.90, 0.99).

The ``StatsComputer`` class accumulates statistics incrementally so that
the full dataset never needs to reside in memory.  Image features are
reduced to per-channel statistics before accumulation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Number of bins for histogram-based quantile estimation.
_HISTOGRAM_BINS = 10_000
# Quantile levels required by LeRobot v3.0.
_QUANTILE_LEVELS = (0.01, 0.10, 0.50, 0.90, 0.99)


@dataclass
class _FeatureAccumulator:
    """Running statistics for a single feature dimension set.

    Uses Welford's online algorithm for mean/variance and a fixed-bin
    histogram for quantile estimation.  Initial samples are buffered
    (up to ``_buffer_limit``) to determine the histogram range before
    constructing the bins.
    """

    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None  # Sum of squared deviations (Welford)
    min_val: np.ndarray | None = None
    max_val: np.ndarray | None = None
    # Histogram for quantile estimation: one histogram per dimension.
    hist_counts: np.ndarray | None = None
    hist_edges: np.ndarray | None = None
    # Track range for histogram re-binning.
    _range_min: np.ndarray | None = None
    _range_max: np.ndarray | None = None
    # Buffer initial values for deferred histogram construction.
    _buffer: list[np.ndarray] = field(default_factory=list)
    _buffer_limit: int = 500
    _histogram_initialized: bool = False


class StatsComputer:
    """Accumulates running statistics per feature key.

    Supports both numeric (float32) array features and image/video features.
    For image/video, pixel values are expected to be normalized to [0, 1].
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
                    Images are flattened per-channel for statistics.
        """
        values = self._prepare_values(feature_key, values)
        acc = self._accumulators.setdefault(feature_key, _FeatureAccumulator())
        self._update_running(acc, values)

    def compute(self) -> dict[str, dict[str, list[float]]]:
        """Return the final statistics dictionary.

        Returns:
            ``{feature_key: {min, max, mean, std, count, q01, q10, q50, q90, q99}}``
            Each stat value is a list of floats (one per dimension).
        """
        result: dict[str, dict[str, list[float]]] = {}
        for key, acc in self._accumulators.items():
            if acc.count == 0:
                continue
            std = np.sqrt(acc.m2 / acc.count)  # population std
            quantiles = self._estimate_quantiles(acc)
            result[key] = {
                "min": acc.min_val.tolist(),
                "max": acc.max_val.tolist(),
                "mean": acc.mean.tolist(),
                "std": std.tolist(),
                "count": [acc.count] * len(acc.mean),
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
    def _prepare_values(feature_key: str, values: np.ndarray) -> np.ndarray:
        """Normalize values to a 1-D float32 array."""
        if values.ndim == 0:
            values = values.reshape(1)
        # Image: [H, W, C] -> per-channel mean across spatial dims -> [C]
        if values.ndim == 3:
            # Normalize uint8 images to [0, 1]
            if values.dtype == np.uint8:
                values = values.astype(np.float32) / 255.0
            else:
                values = values.astype(np.float32)
            # Compute per-channel statistics (mean over spatial dimensions)
            values = values.reshape(-1, values.shape[-1]).mean(axis=0)
        return values.astype(np.float32).ravel()

    def _update_running(self, acc: _FeatureAccumulator, values: np.ndarray) -> None:
        """Perform a single-sample Welford update and buffer/update the histogram."""
        n = acc.count + 1

        if acc.mean is None:
            acc.mean = np.zeros_like(values)
            acc.m2 = np.zeros_like(values)
            acc.min_val = values.copy()
            acc.max_val = values.copy()
        else:
            acc.min_val = np.minimum(acc.min_val, values)
            acc.max_val = np.maximum(acc.max_val, values)

        delta = values - acc.mean
        acc.mean = acc.mean + delta / n
        delta2 = values - acc.mean
        acc.m2 = acc.m2 + delta * delta2
        acc.count = n

        # Buffer for histogram
        if not acc._histogram_initialized:
            acc._buffer.append(values.copy())
            if len(acc._buffer) >= acc._buffer_limit:
                self._init_histogram(acc)
        else:
            self._update_histogram(acc, values)

    def _init_histogram(self, acc: _FeatureAccumulator) -> None:
        """Build initial histogram from buffered samples and replay them."""
        buf = np.stack(acc._buffer)  # [N, D]
        d = buf.shape[1]
        acc.hist_counts = np.zeros((_HISTOGRAM_BINS, d), dtype=np.int64)
        acc._range_min = buf.min(axis=0)
        acc._range_max = buf.max(axis=0)
        # Slightly widen range to avoid edge issues
        eps = np.where(acc._range_max == acc._range_min, 1e-6, 0.0)
        acc._range_min = acc._range_min - eps
        acc._range_max = acc._range_max + eps

        for sample in acc._buffer:
            self._update_histogram(acc, sample)

        acc._buffer.clear()
        acc._histogram_initialized = True

    def _update_histogram(self, acc: _FeatureAccumulator, values: np.ndarray) -> None:
        """Increment histogram bins for each dimension."""
        assert acc._range_min is not None
        d = len(values)
        for dim in range(d):
            v = values[dim]
            lo, hi = acc._range_min[dim], acc._range_max[dim]
            if hi <= lo:
                bin_idx = 0
            else:
                bin_idx = int((v - lo) / (hi - lo) * _HISTOGRAM_BINS)
                bin_idx = max(0, min(_HISTOGRAM_BINS - 1, bin_idx))
            acc.hist_counts[bin_idx, dim] += 1

    def _estimate_quantiles(self, acc: _FeatureAccumulator) -> dict[float, np.ndarray]:
        """Estimate quantiles from the histogram via cumulative-sum interpolation.

        Falls back to exact ``np.quantile`` if the histogram was never
        initialized (fewer samples than ``_buffer_limit``).
        """
        result: dict[float, np.ndarray] = {}

        # Fallback if histogram was never initialised (very few samples)
        if not acc._histogram_initialized:
            if acc._buffer:
                buf = np.stack(acc._buffer)
                for q in _QUANTILE_LEVELS:
                    result[q] = np.quantile(buf, q, axis=0).astype(np.float32)
            else:
                for q in _QUANTILE_LEVELS:
                    result[q] = acc.mean.copy()
            return result

        d = acc.hist_counts.shape[1]
        for q in _QUANTILE_LEVELS:
            q_vals = np.zeros(d, dtype=np.float32)
            for dim in range(d):
                cumsum = np.cumsum(acc.hist_counts[:, dim])
                total = cumsum[-1]
                if total == 0:
                    q_vals[dim] = acc.mean[dim]
                    continue
                target = q * total
                idx = np.searchsorted(cumsum, target)
                idx = min(idx, _HISTOGRAM_BINS - 1)
                lo = acc._range_min[dim]
                hi = acc._range_max[dim]
                q_vals[dim] = lo + (idx + 0.5) / _HISTOGRAM_BINS * (hi - lo)
            result[q] = q_vals
        return result
