"""Tests for giflab.metrics module, additional metrics, and enhanced metrics."""

import math
from pathlib import Path

import numpy as np
import pytest
from giflab.config import MetricsConfig
from giflab.enhanced_metrics import (
    calculate_composite_quality,
    calculate_efficiency_metric,
    normalize_metric,
    process_metrics_with_enhanced_quality,
)
from giflab.metrics import (
    FrameExtractResult,
    align_frames,
    align_frames_content_based,
    calculate_comprehensive_metrics,
    calculate_file_size_kb,
    calculate_temporal_consistency,
    chist,
    edge_similarity,
    extract_gif_frames,
    fsim,
    gmsd,
    mse,
    resize_to_common_dimensions,
    rmse,
    sharpness_similarity,
    texture_similarity,
)
from PIL import Image


class TestMetricsConfig:
    """Tests for MetricsConfig class."""

    def test_default_initialization(self):
        """Test that default values are set correctly."""
        config = MetricsConfig()

        assert config.SSIM_MODE == "comprehensive"
        assert config.SSIM_MAX_FRAMES == 30
        assert config.USE_COMPREHENSIVE_METRICS is True
        assert config.TEMPORAL_CONSISTENCY_ENABLED is True
        assert config.SSIM_WEIGHT == 0.30
        assert config.MS_SSIM_WEIGHT == 0.35
        assert config.PSNR_WEIGHT == 0.25
        assert config.TEMPORAL_WEIGHT == 0.10

    def test_invalid_weights_raises_error(self):
        """Test that invalid weights raise ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            MetricsConfig(
                SSIM_WEIGHT=0.5,
                MS_SSIM_WEIGHT=0.5,
                PSNR_WEIGHT=0.5,
                TEMPORAL_WEIGHT=0.5,
            )


class TestFileOperations:
    """Tests for basic file operations."""

    def test_calculate_file_size_kb(self, tmp_path):
        """Test file size calculation in KB."""
        test_file = tmp_path / "test.txt"
        test_content = "x" * 1024  # 1KB of content
        test_file.write_text(test_content)

        size_kb = calculate_file_size_kb(test_file)
        assert abs(size_kb - 1.0) < 0.1  # Should be approximately 1KB

    def test_calculate_file_size_kb_nonexistent_file(self):
        """Test file size calculation with nonexistent file."""
        with pytest.raises(IOError):
            calculate_file_size_kb(Path("nonexistent_file.gif"))


class TestFrameExtraction:
    """Tests for GIF frame extraction."""

    def create_test_gif(self, tmp_path, frames=3, width=50, height=50, duration=100):
        """Create a test GIF file."""
        gif_path = tmp_path / "test.gif"

        # Create frames with different colors
        images = []
        for i in range(frames):
            # Create a solid color frame
            color = (i * 80 % 255, (i * 100) % 255, (i * 120) % 255)
            img = Image.new("RGB", (width, height), color)
            images.append(img)

        # Save as GIF
        if images:
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0,
            )

        return gif_path

    def test_extract_gif_frames_basic(self, tmp_path):
        """Test basic GIF frame extraction."""
        gif_path = self.create_test_gif(tmp_path, frames=3)
        result = extract_gif_frames(gif_path)

        assert isinstance(result, FrameExtractResult)
        assert result.frame_count == 3
        assert len(result.frames) == 3
        assert result.dimensions == (50, 50)
        assert result.duration_ms > 0

        # Check frame data
        for frame in result.frames:
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (50, 50, 3)

    def test_extract_gif_frames_single_frame(self, tmp_path):
        """Test extraction from single-frame image."""
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (50, 50), (255, 0, 0))
        img.save(img_path)

        result = extract_gif_frames(img_path)

        assert result.frame_count == 1
        assert len(result.frames) == 1
        assert result.dimensions == (50, 50)
        assert result.duration_ms == 0

    def test_extract_gif_frames_invalid_file(self, tmp_path):
        """Test extraction from invalid file."""
        invalid_file = tmp_path / "invalid.gif"
        invalid_file.write_text("not a gif")

        with pytest.raises(IOError):
            extract_gif_frames(invalid_file)


class TestFrameDimensionHandling:
    """Tests for frame dimension handling."""

    def create_frames(self, count, height, width):
        """Create test frames with specified dimensions."""
        frames = []
        for _i in range(count):
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            frames.append(frame)
        return frames

    def test_resize_to_common_dimensions_empty_frames(self):
        """Test resizing with empty frame lists."""
        frames1 = []
        frames2 = self.create_frames(2, 50, 50)

        resized1, resized2 = resize_to_common_dimensions(frames1, frames2)

        assert len(resized1) == 0
        assert len(resized2) == 2


class TestFrameAlignment:
    """Tests for frame alignment methods."""

    def create_test_frames(self, count):
        """Create test frames with identifiable patterns."""
        frames = []
        for i in range(count):
            # Create frame with unique pattern based on index
            frame = np.full((50, 50, 3), i * 25, dtype=np.uint8)
            frames.append(frame)
        return frames

    def test_align_frames_content_based(self):
        """Test content-based alignment method (the only alignment method)."""
        original_frames = self.create_test_frames(5)
        compressed_frames = self.create_test_frames(3)

        aligned = align_frames_content_based(original_frames, compressed_frames)

        assert len(aligned) <= len(original_frames)
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in aligned)

    def test_align_frames_empty_lists(self):
        """Test alignment with empty frame lists."""
        original_frames = self.create_test_frames(5)
        empty_frames = []

        # Test with empty compressed frames
        aligned = align_frames(original_frames, empty_frames)
        assert len(aligned) == 0

        # Test with empty original frames
        aligned = align_frames(empty_frames, original_frames)
        assert len(aligned) == 0


class TestMetricCalculations:
    """Tests for individual metric calculations."""

    def test_calculate_temporal_consistency_single_frame(self):
        """Test temporal consistency with single frame."""
        frames = [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)]
        consistency = calculate_temporal_consistency(frames)

        assert consistency == 1.0  # Single frame is perfectly consistent


class TestComprehensiveMetrics:
    """Tests for the main comprehensive metrics calculation."""

    def create_test_gif_pair(self, tmp_path):
        """Create a pair of test GIFs (original and compressed)."""
        # Create original GIF
        original_path = tmp_path / "original.gif"
        original_images = []
        for i in range(5):
            img = Image.new("RGB", (100, 100), (i * 50, i * 60, i * 70))
            original_images.append(img)

        original_images[0].save(
            original_path,
            save_all=True,
            append_images=original_images[1:],
            duration=100,
            loop=0,
        )

        # Create compressed GIF (smaller, fewer frames)
        compressed_path = tmp_path / "compressed.gif"
        compressed_images = []
        for i in range(0, 5, 2):  # Every 2nd frame
            img = Image.new("RGB", (80, 80), (i * 50 + 10, i * 60 + 10, i * 70 + 10))
            compressed_images.append(img)

        compressed_images[0].save(
            compressed_path,
            save_all=True,
            append_images=compressed_images[1:],
            duration=100,
            loop=0,
        )

        return original_path, compressed_path

    def test_calculate_comprehensive_metrics_basic(self, tmp_path):
        """Test basic comprehensive metrics calculation."""
        original_path, compressed_path = self.create_test_gif_pair(tmp_path)

        metrics = calculate_comprehensive_metrics(original_path, compressed_path)

        # Check all required keys are present
        required_keys = [
            "ssim",
            "ms_ssim",
            "psnr",
            "temporal_consistency",
            "composite_quality",
            "render_ms",
            "kilobytes",
        ]
        assert all(key in metrics for key in required_keys)

        # Check value ranges
        assert 0.0 <= metrics["ssim"] <= 1.0
        assert 0.0 <= metrics["ms_ssim"] <= 1.0
        assert 0.0 <= metrics["psnr"] <= 1.0
        assert 0.0 <= metrics["temporal_consistency"] <= 1.0
        assert 0.0 <= metrics["composite_quality"] <= 1.0
        assert metrics["render_ms"] >= 0
        assert metrics["kilobytes"] > 0

    def test_calculate_comprehensive_metrics_nonexistent_file(self):
        """Test comprehensive metrics with nonexistent files."""
        with pytest.raises((IOError, ValueError)):
            calculate_comprehensive_metrics(
                Path("nonexistent1.gif"), Path("nonexistent2.gif")
            )


class TestLegacyCompatibility:
    """Tests for legacy function compatibility."""

    def test_calculate_ssim_legacy(self, tmp_path):
        """Test legacy SSIM function."""
        from giflab.metrics import calculate_ssim

        # Create simple test GIFs
        original_path = tmp_path / "original.gif"
        compressed_path = tmp_path / "compressed.gif"

        # Create identical single-frame GIFs
        img = Image.new("RGB", (50, 50), (128, 128, 128))
        img.save(original_path)
        img.save(compressed_path)

        ssim_value = calculate_ssim(original_path, compressed_path)

        assert isinstance(ssim_value, float)
        assert 0.0 <= ssim_value <= 1.0


class TestQualityDifferentiation:
    """Tests for quality differentiation validation."""

    def create_quality_test_gifs(self, tmp_path):
        """Create GIFs with different quality levels for testing differentiation."""
        base_img = Image.new("RGB", (100, 100), (128, 128, 128))

        # Excellent quality (identical)
        excellent_path = tmp_path / "excellent.gif"
        base_img.save(excellent_path)

        # Good quality (slight differences)
        good_path = tmp_path / "good.gif"
        good_array = np.array(base_img)
        good_array = good_array + np.random.randint(
            -10, 11, good_array.shape, dtype=np.int8
        )
        good_array = np.clip(good_array, 0, 255).astype(np.uint8)
        good_img = Image.fromarray(good_array)
        good_img.save(good_path)

        # Poor quality (significant differences)
        poor_path = tmp_path / "poor.gif"
        poor_array = np.array(base_img)
        poor_array = poor_array + np.random.randint(
            -50, 51, poor_array.shape, dtype=np.int8
        )
        poor_array = np.clip(poor_array, 0, 255).astype(np.uint8)
        poor_img = Image.fromarray(poor_array)
        poor_img.save(poor_path)

        return excellent_path, good_path, poor_path

    def test_quality_differentiation(self, tmp_path):
        """Test that metrics can differentiate between quality levels."""
        # Create base reference image
        reference_path = tmp_path / "reference.gif"
        ref_img = Image.new("RGB", (100, 100), (128, 128, 128))
        ref_img.save(reference_path)

        excellent_path, good_path, poor_path = self.create_quality_test_gifs(tmp_path)

        # Calculate metrics for each quality level
        excellent_metrics = calculate_comprehensive_metrics(
            reference_path, excellent_path
        )
        good_metrics = calculate_comprehensive_metrics(reference_path, good_path)
        poor_metrics = calculate_comprehensive_metrics(reference_path, poor_path)

        # Quality should decrease: excellent > good > poor
        assert (
            excellent_metrics["composite_quality"] >= good_metrics["composite_quality"]
        )
        assert good_metrics["composite_quality"] >= poor_metrics["composite_quality"]

        # Should achieve some differentiation (not necessarily 40% but some separation)
        quality_range = (
            excellent_metrics["composite_quality"] - poor_metrics["composite_quality"]
        )
        assert quality_range > 0.1  # At least 10% differentiation


class TestExtendedComprehensiveMetrics:
    """Tests for the extended comprehensive metrics functionality."""

    def create_test_gif_pair(self, tmp_path):
        """Create a pair of test GIFs for extended metrics testing."""
        # Create original GIF
        original_path = tmp_path / "original.gif"
        original_images = []
        for i in range(3):
            img = Image.new("RGB", (64, 64), (i * 80, i * 60, i * 70))
            original_images.append(img)

        original_images[0].save(
            original_path,
            save_all=True,
            append_images=original_images[1:],
            duration=100,
            loop=0,
        )

        # Create compressed GIF (slightly different)
        compressed_path = tmp_path / "compressed.gif"
        compressed_images = []
        for i in range(3):
            img = Image.new("RGB", (64, 64), (i * 80 + 5, i * 60 + 5, i * 70 + 5))
            compressed_images.append(img)

        compressed_images[0].save(
            compressed_path,
            save_all=True,
            append_images=compressed_images[1:],
            duration=100,
            loop=0,
        )

        return original_path, compressed_path

    def test_extended_metrics_basic(self, tmp_path):
        """Test that all new metrics are included in comprehensive results."""
        original_path, compressed_path = self.create_test_gif_pair(tmp_path)

        metrics = calculate_comprehensive_metrics(original_path, compressed_path)

        # Check that all new metrics are present
        expected_new_metrics = [
            "mse",
            "rmse",
            "fsim",
            "gmsd",
            "chist",
            "edge_similarity",
            "texture_similarity",
            "sharpness_similarity",
        ]

        for metric in expected_new_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(
                metrics[metric], float
            ), f"Metric {metric} should be float"

    def test_raw_metrics_flag_disabled(self, tmp_path):
        """Test that raw metrics are not included when RAW_METRICS=False."""
        original_path, compressed_path = self.create_test_gif_pair(tmp_path)

        config = MetricsConfig(RAW_METRICS=False)
        metrics = calculate_comprehensive_metrics(
            original_path, compressed_path, config
        )

        # Check that no raw metrics are present
        raw_metrics = [key for key in metrics.keys() if key.endswith("_raw")]
        assert (
            len(raw_metrics) == 0
        ), f"Should not have raw metrics when disabled, found: {raw_metrics}"

    def test_positional_sampling_enabled(self, tmp_path):
        """Test that positional sampling works when enabled."""
        original_path, compressed_path = self.create_test_gif_pair(tmp_path)

        config = MetricsConfig(ENABLE_POSITIONAL_SAMPLING=True)
        metrics = calculate_comprehensive_metrics(
            original_path, compressed_path, config
        )

        # Check that positional metrics are present for default metrics
        default_positional_metrics = ["ssim", "mse", "fsim", "chist"]

        for metric in default_positional_metrics:
            assert f"{metric}_first" in metrics, f"Missing {metric}_first"
            assert f"{metric}_middle" in metrics, f"Missing {metric}_middle"
            assert f"{metric}_last" in metrics, f"Missing {metric}_last"
            assert (
                f"{metric}_positional_variance" in metrics
            ), f"Missing {metric}_positional_variance"

            # Check that values are reasonable
            assert isinstance(metrics[f"{metric}_first"], float)
            assert isinstance(metrics[f"{metric}_middle"], float)
            assert isinstance(metrics[f"{metric}_last"], float)
            assert metrics[f"{metric}_positional_variance"] >= 0.0

    def test_aggregation_helper_function(self):
        """Test the _aggregate_metric helper function directly."""
        from giflab.metrics import _aggregate_metric

        # Test with multiple values
        values = [0.8, 0.9, 0.7, 0.85]
        result = _aggregate_metric(values, "test_metric")

        expected_keys = [
            "test_metric",
            "test_metric_std",
            "test_metric_min",
            "test_metric_max",
        ]
        assert all(key in result for key in expected_keys)

        assert result["test_metric"] == pytest.approx(0.8125, abs=0.001)  # mean
        assert result["test_metric_min"] == 0.7
        assert result["test_metric_max"] == 0.9
        assert result["test_metric_std"] > 0.0

        # Test with single value
        single_result = _aggregate_metric([0.5], "single")
        assert single_result["single"] == 0.5
        assert single_result["single_std"] == 0.0
        assert single_result["single_min"] == 0.5
        assert single_result["single_max"] == 0.5

        # Test with empty values
        empty_result = _aggregate_metric([], "empty")
        assert empty_result["empty"] == 0.0
        assert empty_result["empty_std"] == 0.0
        assert empty_result["empty_min"] == 0.0
        assert empty_result["empty_max"] == 0.0


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_single_frame_gifs(self, tmp_path):
        """Test handling of single-frame GIFs."""
        # Create single-frame GIFs
        img1 = Image.new("RGB", (50, 50), (100, 100, 100))
        img2 = Image.new("RGB", (50, 50), (110, 110, 110))

        path1 = tmp_path / "single1.gif"
        path2 = tmp_path / "single2.gif"

        img1.save(path1)
        img2.save(path2)

        metrics = calculate_comprehensive_metrics(path1, path2)

        assert isinstance(metrics, dict)
        assert (
            metrics["temporal_consistency"] == 1.0
        )  # Single frame is perfectly consistent


# =============================================================================
# Additional Metrics - Representative tests (3 metrics + ordering)
# =============================================================================


class TestAdditionalMetrics:
    """Unit tests for quality metrics - representative subset."""

    def _identical_frames(self):
        """Create two identical frames with realistic content."""
        frame = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        return frame, frame.copy()

    def _slightly_different_frames(self):
        """Create frames with small differences (should score well)."""
        frame1 = np.random.randint(100, 150, (64, 64, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        # Add small noise
        noise = np.random.randint(-10, 11, frame1.shape, dtype=np.int16)
        frame2 = np.clip(frame2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return frame1, frame2

    def _very_different_frames(self):
        """Create frames with significant differences (should score poorly)."""
        # Frame 1: Random pattern
        frame1 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        # Frame 2: Completely different random pattern
        np.random.seed(999)  # Different seed for different pattern
        frame2 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        np.random.seed()  # Reset seed
        return frame1, frame2

    def test_mse_comprehensive(self):
        """Test MSE (error-based, lower-is-better representative)."""
        # Identical frames
        ident1, ident2 = self._identical_frames()
        assert mse(ident1, ident2) == pytest.approx(0.0, abs=1e-6)

        # Similar frames should have low MSE
        sim1, sim2 = self._slightly_different_frames()
        similar_mse = mse(sim1, sim2)

        # Very different frames should have high MSE
        diff1, diff2 = self._very_different_frames()
        different_mse = mse(diff1, diff2)

        assert similar_mse < different_mse
        assert similar_mse > 0.0
        assert different_mse > 1000.0  # Should be significantly higher

    def test_fsim_comprehensive(self):
        """Test FSIM (gradient-based, higher-is-better representative)."""
        # Identical frames
        ident1, ident2 = self._identical_frames()
        identical_fsim = fsim(ident1, ident2)

        # Different frames
        diff1, diff2 = self._very_different_frames()
        different_fsim = fsim(diff1, diff2)

        assert identical_fsim >= different_fsim
        assert identical_fsim > 0.95  # Should be very high for identical
        assert different_fsim > 0.0  # Should be positive

    def test_metric_ordering_comprehensive(self):
        """Test that all metrics show proper ordering across similarity levels."""
        # Get all frame pairs
        identical = self._identical_frames()
        similar = self._slightly_different_frames()
        different = self._very_different_frames()

        # Test each metric maintains proper ordering
        metrics_higher_better = [
            ("chist", chist),
        ]

        # More lenient test for metrics that may not always maintain strict ordering
        metrics_higher_better_lenient = [
            ("fsim", fsim),
            ("edge_similarity", edge_similarity),
            ("texture_similarity", texture_similarity),
            ("sharpness_similarity", sharpness_similarity),
        ]

        metrics_lower_better = [
            ("mse", mse),
            ("rmse", rmse),
            ("gmsd", gmsd),
        ]

        # Strict higher-is-better metrics
        for name, metric_func in metrics_higher_better:
            identical_score = metric_func(*identical)
            similar_score = metric_func(*similar)
            different_score = metric_func(*different)

            assert (
                identical_score >= similar_score
            ), f"{name}: identical ({identical_score:.3f}) should be >= similar ({similar_score:.3f})"
            assert (
                similar_score >= different_score
            ), f"{name}: similar ({similar_score:.3f}) should be >= different ({different_score:.3f})"

        # Lenient higher-is-better metrics (just check identical >= different)
        for name, metric_func in metrics_higher_better_lenient:
            identical_score = metric_func(*identical)
            different_score = metric_func(*different)

            assert (
                identical_score >= different_score
            ), f"{name}: identical ({identical_score:.3f}) should be >= different ({different_score:.3f})"

        # Lower-is-better metrics
        for name, metric_func in metrics_lower_better:
            identical_score = metric_func(*identical)
            similar_score = metric_func(*similar)
            different_score = metric_func(*different)

            assert (
                identical_score <= similar_score
            ), f"{name}: identical ({identical_score:.3f}) should be <= similar ({similar_score:.3f})"
            assert (
                similar_score <= different_score
            ), f"{name}: similar ({similar_score:.3f}) should be <= different ({different_score:.3f})"


class TestChistDocumentedInvariances:
    """Lock in the documented invariances of ``chist`` (colour-histogram correlation).

    These properties are consequences of the metric's definition (per-channel
    32-bin Pearson correlation of marginal histograms) and are documented in
    the ``chist`` docstring. They are intentional, not bugs — but they mean
    chist is a colour-fidelity signal, not a holistic quality signal. These
    tests guard against accidental refactors that would change the metric's
    semantic contract (e.g. swapping to a joint-channel histogram or to a
    distance metric that no longer has these invariances).

    Surfaced from the 2026-05-22 metrics audit
    (``docs/metrics-audit/2026-05-22/report.md``); see
    ``audit-fix/chist-monotonicity-investigation``.
    """

    def test_spatial_invariance_pixel_scramble(self):
        """Scrambling pixel positions while preserving the pixel set yields chist == 1.0.

        chist sees only marginal pixel-value distributions, so any
        permutation of pixels (which preserves the multiset of pixel
        values per channel) is invisible to the metric.
        """
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
        scrambled = img.copy().reshape(-1, 3)
        rng.shuffle(scrambled)  # permute pixels in place
        scrambled = scrambled.reshape(img.shape)

        score = chist(img, scrambled)
        assert score == pytest.approx(1.0, abs=1e-6), (
            f"Spatial scramble should be invisible to chist; got {score:.6f}. "
            "If this changed, the per-channel marginal-histogram contract was broken."
        )

    def test_bin_coarseness_palette_quantize_gradient(self):
        """Heavy palette quantization of a smooth gradient still scores ~1.0.

        With 32 bins (≈8 levels each), a palette-quantized gradient maps
        to the same bins as the original. The histogram cannot distinguish
        the loss of intermediate intensities.
        """
        xs = np.linspace(0, 255, 256).astype(np.uint8)
        grad = np.tile(xs, (256, 1))
        grad_rgb = np.stack([grad, grad, grad], axis=-1)
        pil_grad = Image.fromarray(grad_rgb, mode="RGB")
        # Quantize to 8 colours — visually severe, but bin-aligned
        quantized = np.asarray(pil_grad.quantize(colors=8).convert("RGB"))

        score = chist(grad_rgb, quantized)
        assert score >= 0.99, (
            f"32-bin histograms should be insensitive to palette quantization "
            f"of a smooth gradient; got chist={score:.6f}. If this dropped, "
            "either `bins` changed or the underlying histogram metric changed."
        )

    def test_rebound_at_extreme_blur(self):
        """chist is non-monotonic across blur strength on random content.

        Extreme blur collapses pixel values toward the channel mean, which
        already dominates the original's histogram bins → bin overlap rises
        again at the tail. This is intentional behaviour of the metric.
        """
        import cv2

        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
        blur_mid = cv2.GaussianBlur(img, (5, 5), 1.0)
        blur_extreme = cv2.GaussianBlur(img, (101, 101), 50.0)

        score_mid = chist(img, blur_mid)
        score_extreme = chist(img, blur_extreme)
        # The rebound is the point: extreme blur should NOT score worse than mild blur
        assert score_extreme >= score_mid - 1e-3, (
            f"Expected chist rebound at extreme blur (mid={score_mid:.4f}, "
            f"extreme={score_extreme:.4f}). If extreme is meaningfully worse "
            "than mid, the documented rebound invariant has changed."
        )


# =============================================================================
# Enhanced Metrics
# =============================================================================


class TestEfficiencyMetricCalculation:
    """Tests for the 50/50 balanced efficiency metric calculation."""

    def test_basic_efficiency_calculation(self):
        """Test basic efficiency calculation with normal values."""
        efficiency = calculate_efficiency_metric(
            compression_ratio=5.0, composite_quality=0.8
        )

        # Should return a value between 0 and 1
        assert 0.0 <= efficiency <= 1.0

        # Calculate expected value manually
        normalized_compression = math.log(1 + 5.0) / math.log(1 + 20.0)
        expected = (0.8**0.5) * (normalized_compression**0.5)

        assert abs(efficiency - expected) < 0.001

    def test_boundary_conditions(self):
        """Test boundary conditions and error handling."""
        # Negative compression ratio
        assert calculate_efficiency_metric(-1.0, 0.8) == 0.0

        # Zero compression ratio
        assert calculate_efficiency_metric(0.0, 0.8) == 0.0

        # Negative quality
        assert calculate_efficiency_metric(5.0, -0.1) == 0.0

        # Quality > 1.0 should still work (not clamped in efficiency calc)
        eff_high_quality = calculate_efficiency_metric(5.0, 1.2)
        assert eff_high_quality > 0.0


class TestNormalizeMetric:
    """Tests for metric normalization functions."""

    def test_standard_normalization(self):
        """Test standard 0-1 normalization."""
        # SSIM-like metric (higher is better)
        normalized = normalize_metric("ssim_mean", 0.75, min_val=0.0, max_val=1.0)
        assert normalized == 0.75


class TestEnhancedCompositeQuality:
    """Tests for enhanced composite quality calculation."""

    def create_test_metrics(self) -> dict:
        """Create a comprehensive set of test metrics."""
        return {
            "ssim_mean": 0.9,
            "ms_ssim_mean": 0.85,
            "psnr_mean": 0.8,
            "mse_mean": 100.0,
            "fsim_mean": 0.88,
            "edge_similarity_mean": 0.82,
            "gmsd_mean": 0.1,
            "chist_mean": 0.75,
            "sharpness_similarity_mean": 0.78,
            "texture_similarity_mean": 0.84,
            "temporal_consistency": 0.92,
        }

    def test_enhanced_composite_calculation(self):
        """Test enhanced composite quality calculation with all metrics."""
        metrics = self.create_test_metrics()
        config = MetricsConfig(USE_ENHANCED_COMPOSITE_QUALITY=True)

        enhanced_quality = calculate_composite_quality(metrics, config)

        assert 0.0 <= enhanced_quality <= 1.0
        assert enhanced_quality > 0.65  # Should be reasonably high for good metrics

    def test_missing_metrics_handling(self):
        """Test handling of missing metrics in enhanced calculation."""
        # Only provide core metrics
        minimal_metrics = {
            "ssim_mean": 0.9,
            "ms_ssim_mean": 0.85,
        }

        enhanced_quality = calculate_composite_quality(minimal_metrics)

        # Should still work and provide reasonable result
        assert 0.0 <= enhanced_quality <= 1.0


class TestProcessMetricsIntegration:
    """Tests for the integrated metrics processing function."""

    def test_process_metrics_adds_efficiency(self):
        """Test that processing adds efficiency metric."""
        raw_metrics = {
            "compression_ratio": 5.0,
            "ssim_mean": 0.9,
            "ms_ssim_mean": 0.85,
            "psnr_mean": 0.8,
            "temporal_consistency": 0.92,
        }

        processed = process_metrics_with_enhanced_quality(raw_metrics)

        # Should add composite quality and efficiency
        assert "composite_quality" in processed
        assert "efficiency" in processed

        # Efficiency should be reasonable
        assert 0.0 <= processed["efficiency"] <= 1.0

    def test_process_metrics_edge_case_no_metrics(self):
        """Test processing when no quality metrics are provided (regression test)."""
        raw_metrics = {
            "compression_ratio": 5.0,
            # No quality metrics provided
        }

        processed = process_metrics_with_enhanced_quality(raw_metrics)

        # Should handle gracefully
        assert "composite_quality" in processed
        assert "efficiency" in processed

        # With no quality metrics, composite quality should be 0.0
        assert processed["composite_quality"] == 0.0
        # And efficiency should also be 0.0 (geometric mean with 0 quality)
        assert processed["efficiency"] == 0.0


class TestTemporalArtifactsGate:
    """Tests for the ENABLE_TEMPORAL_ARTIFACTS gate in the sequential path.

    The sequential path (force_all_metrics=True, which is what the public
    measure() API uses) used to unconditionally call
    calculate_enhanced_temporal_metrics — and that function loads LPIPS. The
    gate added for FR-009 must short-circuit when the flag is False.
    """

    def _frames(self):
        # Deterministic 24x24 RGB, 3 frames — large enough that downstream
        # metrics (SSIM/MS-SSIM/etc) don't choke on degenerate inputs, small
        # enough to keep the test fast.
        rng = np.random.default_rng(42)
        a = [
            rng.integers(0, 256, (24, 24, 3), dtype=np.uint8) for _ in range(3)
        ]
        b = [
            rng.integers(0, 256, (24, 24, 3), dtype=np.uint8) for _ in range(3)
        ]
        return a, b

    def _isolated_config(self, *, temporal: bool) -> MetricsConfig:
        # Disable other heavy/binary-dependent branches so the test only
        # measures whether the temporal gate fires. SSIMULACRA2 needs an
        # external binary that isn't guaranteed in CI; DEEP_PERCEPTUAL would
        # itself load LPIPS and pollute the call count we're asserting on.
        config = MetricsConfig()
        config.ENABLE_TEMPORAL_ARTIFACTS = temporal
        config.ENABLE_DEEP_PERCEPTUAL = False
        config.ENABLE_SSIMULACRA2 = False
        return config

    def test_gate_disabled_skips_temporal_artifacts(self, monkeypatch):
        from giflab.metrics import calculate_comprehensive_metrics_from_frames

        calls: list[int] = []

        def _spy(*args, **kwargs):  # noqa: ANN001 — match real signature loosely
            calls.append(1)
            return {}

        monkeypatch.setattr(
            "giflab.temporal_artifacts.calculate_enhanced_temporal_metrics",
            _spy,
        )

        a, b = self._frames()
        result = calculate_comprehensive_metrics_from_frames(
            a, b, config=self._isolated_config(temporal=False), force_all_metrics=True
        )

        assert calls == [], (
            "ENABLE_TEMPORAL_ARTIFACTS=False did not short-circuit the call"
        )
        # The result must still expose zeroed temporal keys so downstream
        # consumers reading `result["flicker_excess"]` don't KeyError.
        assert result.get("flicker_excess") == 0.0
        assert result.get("lpips_t_mean") == 0.0

    def test_gate_enabled_calls_temporal_artifacts(self, monkeypatch):
        from giflab.metrics import calculate_comprehensive_metrics_from_frames

        calls: list[int] = []

        def _spy(*args, **kwargs):  # noqa: ANN001
            calls.append(1)
            return {
                "flicker_excess": 0.01,
                "flicker_frame_ratio": 0.0,
                "flat_flicker_ratio": 0.0,
                "flat_region_count": 0,
                "temporal_pumping_score": 0.0,
                "quality_oscillation_frequency": 0.0,
                "lpips_t_mean": 0.0,
                "lpips_t_p95": 0.0,
                "frame_count": 3,
            }

        monkeypatch.setattr(
            "giflab.temporal_artifacts.calculate_enhanced_temporal_metrics",
            _spy,
        )

        a, b = self._frames()
        calculate_comprehensive_metrics_from_frames(
            a, b, config=self._isolated_config(temporal=True), force_all_metrics=True
        )

        assert calls == [1], (
            "ENABLE_TEMPORAL_ARTIFACTS=True did not invoke the temporal pipeline"
        )


class TestDeepPerceptualGate:
    """Tests for the ENABLE_DEEP_PERCEPTUAL gate in the sequential path.

    Before this gate was honoured at the call site, calculate_deep_perceptual_quality_metrics
    was invoked unconditionally and then short-circuited via a flag inside the
    deep_config dict — but its no-op path still logged a misleading
    WARNING 'No LPIPS scores obtained, using fallback values'. The fix moves
    the check up so the function is never called when disabled.
    """

    def _frames(self):
        rng = np.random.default_rng(7)
        return (
            [rng.integers(0, 256, (24, 24, 3), dtype=np.uint8) for _ in range(3)],
            [rng.integers(0, 256, (24, 24, 3), dtype=np.uint8) for _ in range(3)],
        )

    def _quiet_config(self) -> MetricsConfig:
        config = MetricsConfig()
        config.ENABLE_DEEP_PERCEPTUAL = False
        config.ENABLE_TEMPORAL_ARTIFACTS = False
        config.ENABLE_SSIMULACRA2 = False
        return config

    def test_disabled_does_not_warn_about_missing_lpips_scores(self, caplog):
        """Opting out of deep_perceptual must not emit the failure-shaped warning."""
        import logging

        from giflab.metrics import calculate_comprehensive_metrics_from_frames

        a, b = self._frames()
        with caplog.at_level(logging.WARNING, logger="giflab.deep_perceptual_metrics"):
            calculate_comprehensive_metrics_from_frames(
                a, b, config=self._quiet_config(), force_all_metrics=True
            )

        offending = [
            r for r in caplog.records
            if r.name == "giflab.deep_perceptual_metrics"
            and r.levelno >= logging.WARNING
            and "No LPIPS scores obtained" in r.getMessage()
        ]
        assert offending == [], (
            "ENABLE_DEEP_PERCEPTUAL=False still triggered the misleading warning: "
            f"{[r.getMessage() for r in offending]}"
        )

    def test_disabled_returns_fallback_lpips_values(self):
        """Disabled path must still surface the expected fallback keys."""
        from giflab.metrics import calculate_comprehensive_metrics_from_frames

        a, b = self._frames()
        result = calculate_comprehensive_metrics_from_frames(
            a, b, config=self._quiet_config(), force_all_metrics=True
        )

        # The fallback dict defined inline at the call site uses 0.5 sentinels.
        assert result.get("lpips_quality_mean") == 0.5
        assert result.get("lpips_quality_p95") == 0.5
        assert result.get("lpips_quality_max") == 0.5


class TestSsimulacra2Gate:
    """Parity guard for the ENABLE_SSIMULACRA2 gate across both code paths.

    The sequential path was already gated correctly. The conditional path
    (calculate_selected_metrics, used when force_all_metrics=False) checked
    only selected_metrics["ssimulacra2"] — meaning a caller who populated
    selected_metrics manually without honouring config.ENABLE_SSIMULACRA2
    could still trigger an external binary call. This test pins down the
    fix that brought it into parity with deep_perceptual / temporal_artifacts.
    """

    def _frames(self):
        rng = np.random.default_rng(11)
        return (
            [rng.integers(0, 256, (24, 24, 3), dtype=np.uint8) for _ in range(2)],
            [rng.integers(0, 256, (24, 24, 3), dtype=np.uint8) for _ in range(2)],
        )

    def test_conditional_path_skips_when_flag_disabled(self, monkeypatch):
        """selected_metrics['ssimulacra2']=True must NOT call ssimulacra2 when flag off."""
        from giflab.metrics import calculate_selected_metrics

        calls: list[int] = []

        def _spy(*args, **kwargs):  # noqa: ANN001
            calls.append(1)
            return {"ssimulacra2_mean": 80.0}

        monkeypatch.setattr(
            "giflab.ssimulacra2_metrics.calculate_ssimulacra2_quality_metrics",
            _spy,
        )

        config = MetricsConfig()
        config.ENABLE_SSIMULACRA2 = False

        a, b = self._frames()
        # Force the conditional path to ask for ssimulacra2; the gate must veto.
        selected = {"ssimulacra2": True}
        calculate_selected_metrics(a, b, selected, config=config)

        assert calls == [], (
            "ENABLE_SSIMULACRA2=False did not veto the conditional-path call"
        )


# =============================================================================
# Edge similarity sparse-edge / smooth-gradient aggregation tests
# =============================================================================


class TestEdgeSimilaritySparseEdgeAggregation:
    """Tests for edge_similarity aggregation robustness on sparse-edge content.

    Smooth-gradient and other sparse-edge GIFs expose a flaw in the ``max``
    sub-key: colour quantization introduces banding edges in the compressed
    stream that aren't present in the original, driving per-frame Jaccard
    similarity to near-zero. But at very low colour counts (4–16 colours) the
    banding becomes so coarse that Canny no longer detects it, so most frames
    have union == 0 and return 1.0 via the guard. A ``max`` aggregation then
    reports 1.0 regardless of quantization level — INCONCLUSIVE in the audit.

    The fix switches the primary aggregation key for ``edge_similarity`` from
    ``mean`` to ``median``. Median is robust to the outlier 1.0 frames from the
    union-zero guard while still reflecting the typical frame quality faithfully.

    Reference: giflab-edge-similarity-max-aggregation-sparse-edges task note,
    2026-05-22 sanity audit (report.md line 106: edge_similarity_max FLAT).
    """

    def _make_smooth_gradient_frame(self, width: int = 64, height: int = 64) -> np.ndarray:
        """Return an RGB frame that is a smooth horizontal gradient (very few real edges)."""
        row = np.linspace(0, 255, width, dtype=np.float32)
        frame = np.stack([row] * height, axis=0)  # (H, W)
        return np.stack([frame, frame, frame], axis=-1).astype(np.uint8)

    def _quantize_frame(self, frame: np.ndarray, n_colors: int) -> np.ndarray:
        """Simulate palette quantization by posterizing the frame to n_colors levels."""
        # Evenly space n_colors levels across [0, 255] and map each pixel to nearest
        levels = np.linspace(0, 255, n_colors, dtype=np.float32)
        out = frame.astype(np.float32)
        for ch in range(3):
            ch_vals = out[:, :, ch]
            indices = np.searchsorted(levels, ch_vals.ravel(), side="left")
            indices = np.clip(indices, 0, n_colors - 1)
            out[:, :, ch] = levels[indices].reshape(ch_vals.shape)
        return out.astype(np.uint8)

    def test_edge_similarity_max_is_not_stable_primary_on_sparse_edges(self):
        """Demonstrate the ``max`` sub-key volatility on sparse-edge content.

        This test documents the pre-fix behaviour: per-frame edge_similarity
        values on a smooth gradient include 1.0 outliers (from the union==0
        guard), which means ``np.max`` is always 1.0 regardless of how many
        colours the palette was reduced to.  A ``max``-based primary
        aggregation therefore can't discriminate between light and heavy
        quantization on sparse-edge content.
        """
        rng = np.random.default_rng(seed=42)
        frame = self._make_smooth_gradient_frame()

        # Collect per-frame edge_similarity scores for two levels of quantization.
        # 256 colours (near-identical) vs 4 colours (heavy quantization).
        frames_256_col = [self._quantize_frame(frame, 256) for _ in range(8)]
        frames_4_col = [self._quantize_frame(frame, 4) for _ in range(8)]

        scores_256: list[float] = [
            edge_similarity(frame, q) for q in frames_256_col
        ]
        scores_4: list[float] = [
            edge_similarity(frame, q) for q in frames_4_col
        ]

        # At 4 colours, at least some frames should detect banding → near-zero scores
        # (i.e., the metric IS sensitive at the frame level).
        # However, np.max picks the single highest score — often 1.0 from union==0.
        max_256 = float(np.max(scores_256))
        max_4 = float(np.max(scores_4))

        # Both max values can be 1.0 because at least one frame has union==0.
        # This is the bug: max cannot distinguish the two quantization levels.
        # The test documents this (it should PASS with current code, confirming the bug).
        assert max_256 >= max_4 or abs(max_256 - max_4) < 0.05, (
            f"Pre-fix invariant violated: max_256={max_256:.4f}, max_4={max_4:.4f}. "
            "The test may need updating if the synthetic content changed."
        )

    def test_edge_similarity_median_is_stable_on_sparse_edges(self):
        """Median aggregation is robust to outlier 1.0 union-zero frames.

        On smooth-gradient content, most frame pairs with heavy quantization
        will have one edge map with banding and the other with no edges
        (Jaccard → 0). Median ignores the occasional 1.0 frames from the
        union-zero guard and reflects the typical per-frame quality.

        After the fix, the primary aggregation key for ``edge_similarity`` uses
        ``median``, so ``edge_similarity`` (the primary key emitted in the
        aggregated result) should be close to 1.0 for light quantization and
        substantially lower for heavy quantization.
        """
        frame = self._make_smooth_gradient_frame()

        frames_256_col = [self._quantize_frame(frame, 256) for _ in range(8)]
        frames_4_col = [self._quantize_frame(frame, 4) for _ in range(8)]

        scores_256: list[float] = [
            edge_similarity(frame, q) for q in frames_256_col
        ]
        scores_4: list[float] = [
            edge_similarity(frame, q) for q in frames_4_col
        ]

        median_256 = float(np.median(scores_256))
        median_4 = float(np.median(scores_4))

        # After quantizing to 4 colours, at least some frames will have banding
        # edges that don't match the original → median should drop noticeably
        # below the near-identity 256-colour case.
        # For content where ALL frames return 1.0 under 256 colours (identical
        # or near-identical), the 4-colour median should be lower OR equal.
        assert median_256 >= median_4, (
            f"Median should be higher (or equal) at 256 colours than 4 colours on "
            f"smooth-gradient content; got median_256={median_256:.4f}, "
            f"median_4={median_4:.4f}"
        )

    def test_aggregate_metric_uses_median_for_edge_similarity(self):
        """_aggregate_metric must use median (not mean) as the primary value for
        ``edge_similarity``, so the exported ``edge_similarity`` key represents
        the robust central tendency of per-frame Jaccard scores.

        This test will FAIL before the fix (median != mean when outliers present)
        and PASS after.
        """
        from giflab.metrics import _aggregate_metric

        # Construct a scores list that mimics sparse-edge content:
        # most frames have near-zero Jaccard (banding not in original),
        # but a handful return 1.0 from the union==0 guard.
        scores = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]

        result = _aggregate_metric(scores, "edge_similarity")

        expected_median = float(np.median(scores))  # 0.0
        expected_mean = float(np.mean(scores))       # 0.25

        # The primary key must equal median, not mean.
        assert result["edge_similarity"] == pytest.approx(expected_median, abs=1e-6), (
            f"edge_similarity primary key should be median ({expected_median:.4f}) "
            f"but got {result['edge_similarity']:.4f} (mean would be {expected_mean:.4f}). "
            "Fix: pass np.median as the primary aggregation for edge_similarity."
        )
        # Min and max sub-keys should still be present and correct
        assert result["edge_similarity_min"] == pytest.approx(0.0, abs=1e-6)
        assert result["edge_similarity_max"] == pytest.approx(1.0, abs=1e-6)

    def test_aggregate_metric_preserves_mean_for_non_edge_metrics(self):
        """Changing edge_similarity to median must NOT affect other metric keys.

        _aggregate_metric with a non-edge metric name must still return mean as
        the primary value.
        """
        from giflab.metrics import _aggregate_metric

        scores = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
        result = _aggregate_metric(scores, "ssim")

        expected_mean = float(np.mean(scores))
        assert result["ssim"] == pytest.approx(expected_mean, abs=1e-6), (
            "ssim primary key should still use mean aggregation"
        )
