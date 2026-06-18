"""Tests for bagel.stats module."""

from __future__ import annotations

import numpy as np

from bagel.stats import StatsComputer


class TestStatsComputerNumeric:
    """Test statistics for numeric (float32) features."""

    def test_single_value(self) -> None:
        sc = StatsComputer()
        sc.add_frame("obs", np.array([1.0, 2.0, 3.0], dtype=np.float32))
        result = sc.compute()
        assert "obs" in result
        np.testing.assert_allclose(result["obs"]["min"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result["obs"]["max"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result["obs"]["mean"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result["obs"]["std"], [0.0, 0.0, 0.0])
        # count is a single-element list (LeRobot v3.0 shape (1,)), not per-dim.
        assert result["obs"]["count"] == [1]

    def test_multiple_values(self) -> None:
        sc = StatsComputer()
        vals = [np.array([float(i)], dtype=np.float32) for i in range(100)]
        for v in vals:
            sc.add_frame("x", v)
        result = sc.compute()
        np.testing.assert_allclose(result["x"]["min"], [0.0])
        np.testing.assert_allclose(result["x"]["max"], [99.0])
        np.testing.assert_allclose(result["x"]["mean"], [49.5], atol=0.1)
        assert result["x"]["count"] == [100]
        # std of 0..99 = sqrt((99*100*199)/(6*100^2)) ~ 28.87
        assert 28.0 < result["x"]["std"][0] < 30.0

    def test_quantiles_reasonable(self) -> None:
        sc = StatsComputer()
        rng = np.random.RandomState(42)
        for _ in range(1000):
            sc.add_frame("v", rng.randn(4).astype(np.float32))
        result = sc.compute()
        # q50 should be near 0 for standard normal
        for dim in range(4):
            assert abs(result["v"]["q50"][dim]) < 0.3
            assert result["v"]["q01"][dim] < result["v"]["q50"][dim]
            assert result["v"]["q50"][dim] < result["v"]["q99"][dim]

    def test_empty_compute(self) -> None:
        sc = StatsComputer()
        result = sc.compute()
        assert result == {}


class TestStatsComputerImage:
    """Test statistics for image features (3D arrays)."""

    def test_image_stats(self) -> None:
        sc = StatsComputer()
        # 4x4 RGB image, all white (255)
        img = np.full((4, 4, 3), 255, dtype=np.uint8)
        sc.add_frame("img", img)
        result = sc.compute()
        # After normalization, mean should be 1.0 per channel
        np.testing.assert_allclose(result["img"]["mean"], [1.0, 1.0, 1.0], atol=1e-5)

    def test_image_black(self) -> None:
        sc = StatsComputer()
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        sc.add_frame("img", img)
        result = sc.compute()
        np.testing.assert_allclose(result["img"]["mean"], [0.0, 0.0, 0.0], atol=1e-5)

    def test_multi_feature(self) -> None:
        sc = StatsComputer()
        sc.add_frame("state", np.array([1.0, 2.0], dtype=np.float32))
        sc.add_frame("img", np.full((2, 2, 3), 128, dtype=np.uint8))
        result = sc.compute()
        assert "state" in result
        assert "img" in result

    def test_image_stats_are_over_pixels(self) -> None:
        """Image std/min/max/quantiles must be computed over every pixel.

        Regression test: feeding frames whose per-frame spatial mean is constant
        but whose pixels vary widely used to collapse std to ~0 and min==max
        (frame-mean statistics). They must reflect the full pixel distribution.
        """
        sc = StatsComputer()
        h = w = 32
        base = np.tile(np.linspace(0, 255, w, dtype=np.float32), (h, 1))  # gradient
        all_px = []
        for i in range(50):
            g = np.roll(base, i, axis=1)  # spatial mean stays constant across frames
            img = np.stack([g, g[:, ::-1], g], axis=-1).astype(np.uint8)  # [H, W, 3]
            sc.add_frame("cam", img)
            all_px.append(img.reshape(-1, 3).astype(np.float32) / 255.0)
        result = sc.compute()["cam"]
        truth = np.concatenate(all_px)

        # count is the number of frames, not pixels.
        assert result["count"] == [50]
        # std reflects pixel spread (~0.3 for a uniform gradient), not ~0.
        np.testing.assert_allclose(result["std"], truth.std(axis=0), atol=1e-3)
        assert min(result["std"]) > 0.1
        # min/max span the full normalized range.
        np.testing.assert_allclose(result["min"], truth.min(axis=0), atol=1e-3)
        np.testing.assert_allclose(result["max"], truth.max(axis=0), atol=1e-3)
        # quantiles are ordered and bracket the median.
        for ch in range(3):
            assert result["q01"][ch] < result["q50"][ch] < result["q99"][ch]

    def test_image_quantiles_match_numpy(self) -> None:
        """Histogram quantiles should track exact numpy quantiles over pixels."""
        sc = StatsComputer()
        rng = np.random.RandomState(0)
        all_px = []
        for _ in range(20):
            img = rng.randint(0, 256, (40, 40, 3), dtype=np.uint8)
            sc.add_frame("cam", img)
            all_px.append(img.reshape(-1, 3).astype(np.float32) / 255.0)
        result = sc.compute()["cam"]
        truth = np.concatenate(all_px)
        for q, key in [(0.01, "q01"), (0.5, "q50"), (0.99, "q99")]:
            np.testing.assert_allclose(
                result[key], np.quantile(truth, q, axis=0), atol=0.02
            )
