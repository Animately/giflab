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
            "temporal_consistency_compressed",
            "composite_quality",
            "render_ms",
            "kilobytes",
        ]
        assert all(key in metrics for key in required_keys)
        # Wave 7: bare ``temporal_consistency`` removed entirely.
        assert "temporal_consistency" not in metrics

        # Check value ranges
        assert 0.0 <= metrics["ssim"] <= 1.0
        assert 0.0 <= metrics["ms_ssim"] <= 1.0
        assert 0.0 <= metrics["psnr"] <= 1.0
        assert 0.0 <= metrics["temporal_consistency_compressed"] <= 1.0
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

        # Test with empty values: nothing was measured, so every key is NaN
        # ("not measured"), NOT 0.0 — a 0.0 here looks like a real worst-case
        # score downstream (audit-fix: NaN over fabricated values).
        import math

        empty_result = _aggregate_metric([], "empty")
        assert math.isnan(empty_result["empty"])
        assert math.isnan(empty_result["empty_std"])
        assert math.isnan(empty_result["empty_min"])
        assert math.isnan(empty_result["empty_max"])


class TestSsimClampBehaviour:
    """Regression tests documenting the SSIM clamp's behaviour.

    Investigated under the 2026-05-22 metrics audit follow-up
    ([[giflab-ssim-smooth-gradient-bump]]): the audit flagged ssim_min ticking
    UP at the strongest animately --lossy levels on smooth_gradient. These
    tests lock in the conclusion that the [0, 1] clamp in
    calculate_comprehensive_metrics_from_frames is a defensive guard, NOT the
    source of that bump-up — the bump-up is animately-side saturation.
    """

    def _gradient_pair(self):
        # Two 64x64 RGB gradient frames; comp differs slightly. Realistic SSIM
        # range for these is well inside [0, 1], so the clamp must be a no-op.
        x = np.tile(np.linspace(0, 255, 64, dtype=np.float32), (64, 1))
        orig = np.stack([x, x, x], axis=-1).astype(np.uint8)
        comp = np.clip(orig.astype(np.float32) + 5.0, 0, 255).astype(np.uint8)
        return [orig], [comp]

    def test_clamp_is_noop_for_typical_ssim_values(self):
        """SSIM on slightly-perturbed gradients is well inside [0, 1] — the
        clamp must not change the reported value.
        """
        from giflab.metrics import calculate_comprehensive_metrics_from_frames

        orig_frames, comp_frames = self._gradient_pair()
        config = MetricsConfig()
        if hasattr(config, "USE_PARALLEL"):
            config.USE_PARALLEL = False
        if hasattr(config, "ENABLE_PARALLEL_METRICS"):
            config.ENABLE_PARALLEL_METRICS = False

        m = calculate_comprehensive_metrics_from_frames(
            orig_frames, comp_frames, config=config
        )
        ssim_val = m.get("ssim_mean", m.get("ssim"))
        assert ssim_val is not None
        # On a tiny perturbation of a gradient, SSIM should be high but
        # strictly less than 1 — meaning the clamp's upper bound was not
        # actively reached.
        assert 0.5 < ssim_val < 1.0, (
            f"SSIM {ssim_val} hit a clamp boundary unexpectedly; the audit "
            f"smooth_gradient bump-up was NOT caused by clamping and this "
            f"test is the guard for that conclusion."
        )

    def test_identical_frames_score_at_clamp_upper_bound(self):
        """Sanity check: identical frames score SSIM == 1.0 (clamp permits
        the upper bound, doesn't truncate it).
        """
        from giflab.metrics import calculate_comprehensive_metrics_from_frames

        x = np.tile(np.linspace(0, 255, 64, dtype=np.float32), (64, 1))
        frame = np.stack([x, x, x], axis=-1).astype(np.uint8)
        config = MetricsConfig()
        if hasattr(config, "USE_PARALLEL"):
            config.USE_PARALLEL = False
        if hasattr(config, "ENABLE_PARALLEL_METRICS"):
            config.ENABLE_PARALLEL_METRICS = False

        m = calculate_comprehensive_metrics_from_frames(
            [frame], [frame.copy()], config=config
        )
        ssim_val = m.get("ssim_mean", m.get("ssim"))
        assert ssim_val == pytest.approx(1.0, abs=1e-6)


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
            metrics["temporal_consistency_compressed"] == 1.0
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
            "temporal_consistency_compressed": 0.92,
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
            "temporal_consistency_compressed": 0.92,
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
        a = [rng.integers(0, 256, (24, 24, 3), dtype=np.uint8) for _ in range(3)]
        b = [rng.integers(0, 256, (24, 24, 3), dtype=np.uint8) for _ in range(3)]
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

        assert (
            calls == []
        ), "ENABLE_TEMPORAL_ARTIFACTS=False did not short-circuit the call"
        # The result must still expose zeroed temporal keys so downstream
        # consumers don't KeyError. Wave 7: these single-stream signals are
        # keyed with the honest ``_compressed`` suffix (the bare keys are gone).
        assert result.get("flicker_excess_compressed") == 0.0
        assert result.get("lpips_t_mean_compressed") == 0.0
        # The bare keys must NOT be present any more.
        assert "flicker_excess" not in result
        assert "lpips_t_mean" not in result

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

        assert calls == [
            1
        ], "ENABLE_TEMPORAL_ARTIFACTS=True did not invoke the temporal pipeline"


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
            r
            for r in caplog.records
            if r.name == "giflab.deep_perceptual_metrics"
            and r.levelno >= logging.WARNING
            and "No LPIPS scores obtained" in r.getMessage()
        ]
        assert offending == [], (
            "ENABLE_DEEP_PERCEPTUAL=False still triggered the misleading warning: "
            f"{[r.getMessage() for r in offending]}"
        )

    def test_disabled_returns_fallback_lpips_values(self):
        """Disabled path must still surface the expected fallback keys.

        Audit-fix (NaN over sentinels): the disabled-LPIPS fallback now reports
        NaN score keys ("not measured") rather than the old 0.5 midpoint
        sentinel, which silently inflated composite_quality and corpus
        aggregates. The keys are still present (same shape), just honest.
        """
        import math

        from giflab.metrics import calculate_comprehensive_metrics_from_frames

        a, b = self._frames()
        result = calculate_comprehensive_metrics_from_frames(
            a, b, config=self._quiet_config(), force_all_metrics=True
        )

        # Keys present (same schema), values NaN ("not measured"), not 0.5.
        assert "lpips_quality_mean" in result
        assert math.isnan(result["lpips_quality_mean"])
        assert math.isnan(result["lpips_quality_p95"])
        assert math.isnan(result["lpips_quality_max"])

    def test_lpips_exception_path_returns_nan_not_sentinel(self, monkeypatch):
        """When the LPIPS entry point hits its exception path, the returned
        dict's score keys must be NaN (not the old 0.5 midpoint sentinel).

        Forces the exception via monkeypatch so no real LPIPS model is needed
        — pure functional layer.
        """
        from giflab.deep_perceptual_metrics import (
            calculate_deep_perceptual_quality_metrics,
        )

        def _boom(*args, **kwargs):  # noqa: ANN001
            raise RuntimeError("synthetic LPIPS validator failure")

        # The entry point calls _get_or_create_validator(...).calculate_...;
        # make the validator factory blow up to hit the outer except path.
        monkeypatch.setattr(
            "giflab.deep_perceptual_metrics._get_or_create_validator",
            _boom,
        )

        a, b = self._frames()
        result = calculate_deep_perceptual_quality_metrics(a, b, config={})

        assert math.isnan(result["lpips_quality_mean"])
        assert math.isnan(result["lpips_quality_p95"])
        assert math.isnan(result["lpips_quality_max"])
        assert result["deep_perceptual_device"] == "fallback"


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

        assert (
            calls == []
        ), "ENABLE_SSIMULACRA2=False did not veto the conditional-path call"


# =============================================================================
# NaN-over-sentinel hardening tests (audit-fix:
# giflab-metrics-nan-sentinel-hardening)
# =============================================================================


class TestAggregateMetricNaNAware:
    """``_aggregate_metric`` must be NaN-aware so a failed frame (now appended
    as NaN by the per-frame except blocks) doesn't drag the aggregate toward a
    fabricated 0.0. Mirrors the SSIMULACRA2 nanmean tests in
    tests/functional/test_ssimulacra2_metrics.py.
    """

    def test_mixed_nan_uses_nanmean(self):
        from giflab.metrics import _aggregate_metric

        values = [0.9, float("nan"), 0.8]
        result = _aggregate_metric(values, "ssim")

        # The surviving frames carry the score; NOT contaminated toward 0.
        expected = float(np.nanmean(values))
        assert result["ssim"] == pytest.approx(expected)
        assert result["ssim"] == pytest.approx(0.85)
        assert result["ssim_min"] == pytest.approx(0.8)
        assert result["ssim_max"] == pytest.approx(0.9)

    def test_all_nan_returns_nan(self):
        from giflab.metrics import _aggregate_metric

        result = _aggregate_metric([float("nan"), float("nan")], "ssim")
        assert math.isnan(result["ssim"])
        assert math.isnan(result["ssim_min"])
        assert math.isnan(result["ssim_max"])
        assert math.isnan(result["ssim_std"])

    def test_single_nan_returns_nan(self):
        from giflab.metrics import _aggregate_metric

        # A single NaN frame must produce NaN, not float(nan) silently cast to
        # a real-looking number (it stays NaN, which is correct).
        result = _aggregate_metric([float("nan")], "ssim")
        assert math.isnan(result["ssim"])

    def test_edge_similarity_uses_nanmedian(self):
        from giflab.metrics import _aggregate_metric

        values = [0.0, 0.0, float("nan"), 1.0, 0.0]
        result = _aggregate_metric(values, "edge_similarity")
        # nanmedian of [0,0,1,0] = 0.0 (NaN dropped); proves median path is
        # NaN-aware too (preserves _MEDIAN_AGGREGATED_METRICS behaviour).
        assert result["edge_similarity"] == pytest.approx(float(np.nanmedian(values)))


class TestPerFrameExceptionNaNPropagation:
    """A per-frame metric exception must append NaN and propagate through the
    aggregate as nanmean, NOT contaminate the mean toward 0.
    """

    def _frames(self):
        rng = np.random.default_rng(7)
        return (
            [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(3)],
            [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(3)],
        )

    def _quiet_config(self):
        config = MetricsConfig()
        # Disable expensive / external-binary metrics so this stays a fast
        # functional test exercising only the per-frame loop + aggregation.
        config.ENABLE_DEEP_PERCEPTUAL = False
        config.ENABLE_SSIMULACRA2 = False
        config.ENABLE_TEMPORAL_ARTIFACTS = False
        return config

    def test_gmsd_exception_on_one_frame_propagates_as_nanmean(self, monkeypatch):
        """Force ``gmsd`` to raise on one frame; the aggregated ``gmsd`` value
        must be the nanmean of the surviving frames, not be dragged toward the
        fabricated 0.0 a failed frame used to append.

        ``gmsd`` is chosen because (unlike SSIM, which ``calculate_ms_ssim``
        also calls) it is computed exactly once per frame in the sequential
        loop, so the failure targets the per-frame except block deterministically.
        The sequential ``_from_frames`` path emits the bare ``gmsd`` key (no
        ``_mean`` suffix; that rename only happens in calculate_selected_metrics).
        """
        import giflab.metrics as metrics_mod
        from giflab.metrics import calculate_comprehensive_metrics_from_frames

        real_gmsd = metrics_mod.gmsd
        a, b = self._frames()
        fail_idx = 1

        # Reference run: a clean 2-frame call over only the SURVIVING frames,
        # through the same resize+aggregate pipeline. The NaN-aware aggregate
        # of the 3-frame run (with frame `fail_idx` failing) must equal the
        # mean of this 2-frame survivor run — i.e. the NaN frame was dropped,
        # not averaged in as 0.
        survivors_a = [f for i, f in enumerate(a) if i != fail_idx]
        survivors_b = [f for i, f in enumerate(b) if i != fail_idx]
        survivor_result = calculate_comprehensive_metrics_from_frames(
            survivors_a,
            survivors_b,
            config=self._quiet_config(),
            force_all_metrics=True,
        )
        expected = survivor_result["gmsd"]

        state = {"n": 0}

        def flaky_gmsd(*args, **kwargs):  # noqa: ANN001
            idx = state["n"]
            state["n"] += 1
            if idx == fail_idx:
                raise RuntimeError("synthetic GMSD failure")
            return real_gmsd(*args, **kwargs)

        # Force the sequential loop so the per-frame except block is exercised
        # deterministically (the parallel path runs in subprocesses where this
        # monkeypatch wouldn't apply).
        monkeypatch.setenv("GIFLAB_ENABLE_PARALLEL_METRICS", "false")
        monkeypatch.setattr(metrics_mod, "gmsd", flaky_gmsd)

        result = calculate_comprehensive_metrics_from_frames(
            a, b, config=self._quiet_config(), force_all_metrics=True
        )

        gmsd_value = result["gmsd"]
        # Finite, and equal to the survivor-only mean — proves the NaN frame
        # was dropped. The old bug (append 0.0 + plain mean) would instead give
        # (s0 + 0 + s2)/3, materially lower; assert we're NOT near that.
        assert not math.isnan(gmsd_value)
        assert gmsd_value == pytest.approx(expected, rel=1e-6)
        contaminated = (expected * 2) / 3.0  # mean with a fabricated 0.0 frame
        assert abs(gmsd_value - expected) < abs(gmsd_value - contaminated)


class TestMainPathMeanKeysForComposite:
    """The main serial/parallel ``_from_frames`` path must emit ``{metric}_mean``
    aliases for the structural/signal metrics that ``calculate_composite_quality``
    (and storage / quality_validation) read.

    Audit-fix [[giflab-composite-quality-bare-vs-mean-key-mismatch]]: the main
    aggregation block (``_aggregate_metric``) emitted the primary statistic under
    the BARE key (``ssim`` not ``ssim_mean``), so on the default-config main path
    every ``"{m}_mean" in metrics`` lookup in ``calculate_composite_quality``
    MISSED — the 11-metric composite silently collapsed to the thin perceptual /
    temporal set (~10% of the weight) and ``total_weight`` renormalisation hid it.
    Phase 6 (``optimized_metrics.py``) already emits BOTH bare and ``_mean`` for
    ssim/mse/psnr (locked by ``test_phase6_schema_contract``); this brings the
    standard path to the same key shape.
    """

    # The ten structural/signal keys that calculate_composite_quality reads as
    # ``{name}_mean`` (enhanced_metrics.py:104-168). All MUST be present + finite
    # on the main path post-fix.
    _COMPOSITE_MEAN_KEYS = (
        "ssim_mean",
        "ms_ssim_mean",
        "psnr_mean",
        "mse_mean",
        "fsim_mean",
        "edge_similarity_mean",
        "gmsd_mean",
        "chist_mean",
        "sharpness_similarity_mean",
        "texture_similarity_mean",
    )

    # Bare → _mean aliases that share the SAME scale as their bare key. PSNR is
    # excluded here because its bare key is normalised 0-1 while ``psnr_mean``
    # must be RAW dB (see the dedicated PSNR test below).
    _SAME_SCALE_PAIRS = (
        "ssim",
        "ms_ssim",
        "mse",
        "rmse",
        "fsim",
        "gmsd",
        "chist",
        "edge_similarity",
        "texture_similarity",
        "sharpness_similarity",
    )

    def _frames(self):
        rng = np.random.default_rng(0)
        orig = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(4)]
        comp = [
            np.clip(f.astype(int) + rng.integers(-30, 30, (64, 64, 3)), 0, 255).astype(
                np.uint8
            )
            for f in orig
        ]
        return orig, comp

    def test_from_frames_emits_mean_keys_for_composite(self, monkeypatch):
        """FAILING-FIRST: every composite-read ``_mean`` key is present + finite
        on the default-config main path.

        Pre-fix the main path emits only bare ``ssim``/``mse``/``psnr``/... and
        the ten ``_mean`` keys are ABSENT, so this fails.
        """
        from giflab.metrics import calculate_comprehensive_metrics_from_frames

        # Force the deterministic serial path so the test is reproducible.
        monkeypatch.setenv("GIFLAB_ENABLE_PARALLEL_METRICS", "false")
        orig, comp = self._frames()
        result = calculate_comprehensive_metrics_from_frames(
            orig, comp, force_all_metrics=True
        )

        for key in self._COMPOSITE_MEAN_KEYS:
            assert key in result, f"{key} missing on main path"
            assert isinstance(result[key], float)
            assert not math.isnan(result[key]), f"{key} is NaN on a realistic pair"

    def test_from_frames_mean_aliases_match_bare(self, monkeypatch):
        """Scale guard (the load-bearing one).

        Same-scale ``_mean`` keys must EQUAL their bare key. ``psnr_mean`` must
        NOT equal the bare normalised ``psnr`` — it must be RAW dB so it feeds
        ``normalize_metric('psnr_mean', ...)`` (which divides by 50 dB) correctly
        rather than being double-normalised (0.47 → /50 → ~0.009).
        """
        from giflab.config import DEFAULT_METRICS_CONFIG
        from giflab.metrics import calculate_comprehensive_metrics_from_frames

        monkeypatch.setenv("GIFLAB_ENABLE_PARALLEL_METRICS", "false")
        orig, comp = self._frames()
        result = calculate_comprehensive_metrics_from_frames(
            orig, comp, force_all_metrics=True
        )

        for base in self._SAME_SCALE_PAIRS:
            assert result[f"{base}_mean"] == pytest.approx(
                result[base]
            ), f"{base}_mean must alias the same-scale bare {base} key"

        # PSNR: the trap. Bare psnr is normalised 0-1; psnr_mean is raw dB.
        psnr_bare = result["psnr"]
        psnr_mean = result["psnr_mean"]
        assert psnr_mean != pytest.approx(psnr_bare), (
            "psnr_mean must be RAW dB, distinct from the normalised bare psnr — "
            "aliasing the bare value would double-normalise in normalize_metric"
        )
        # On a realistic noisy pair PSNR is well above 1 dB.
        assert psnr_mean > 1.0
        # And psnr_mean ≈ bare_psnr * PSNR_MAX_DB (the un-normalisation), within
        # clamp tolerance (bare psnr is clamped to [0, 1] before scaling back).
        assert psnr_mean == pytest.approx(
            psnr_bare * float(DEFAULT_METRICS_CONFIG.PSNR_MAX_DB), rel=1e-3
        )

    def test_composite_uses_structural_metrics_on_from_frames(self, monkeypatch):
        """Behavioural: a structurally much-worse compression must score a lower
        ``composite_quality`` than a near-perfect one on the main path.

        Pre-fix the structural difference is INVISIBLE (the ten structural
        ``_mean`` keys are absent, so composite is driven only by the thin
        perceptual/temporal set); post-fix the structural collapse is counted
        and ordering reflects it. Assert ordering (an inequality), not absolute
        magnitudes, per CLAUDE.md "continuous, no cliff thresholds".
        """
        from giflab.metrics import calculate_comprehensive_metrics_from_frames

        monkeypatch.setenv("GIFLAB_ENABLE_PARALLEL_METRICS", "false")
        rng = np.random.default_rng(1)
        orig = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(4)]

        # Near-perfect: tiny perturbation (structure preserved).
        good = [
            np.clip(f.astype(int) + rng.integers(-2, 2, (64, 64, 3)), 0, 255).astype(
                np.uint8
            )
            for f in orig
        ]
        # Structurally destroyed: heavy noise wrecks ssim/fsim/edge/gmsd/...
        bad = [
            np.clip(
                f.astype(int) + rng.integers(-120, 120, (64, 64, 3)), 0, 255
            ).astype(np.uint8)
            for f in orig
        ]

        good_metrics = calculate_comprehensive_metrics_from_frames(
            orig, good, force_all_metrics=True
        )
        bad_metrics = calculate_comprehensive_metrics_from_frames(
            orig, bad, force_all_metrics=True
        )

        assert good_metrics["composite_quality"] > bad_metrics["composite_quality"], (
            "Structurally-destroyed compression must score lower composite_quality "
            "than a near-perfect one once the structural _mean keys are counted"
        )

    def test_psnr_mean_omitted_when_no_raw_values(self, monkeypatch):
        """NaN-honesty guard: if every frame's PSNR fails (raw list all-NaN),
        ``psnr_mean`` must be OMITTED — not present-with-NaN, not 0.0, not a
        crash.

        Omitting (rather than emitting NaN) is the load-bearing choice here:
        ``normalize_metric('psnr_mean', nan)`` returns 1.0 (``min(nan, 50)`` →
        ``nan`` → clamped to 1.0), so a present-but-NaN ``psnr_mean`` would
        silently award the full 20% PSNR weight as a PERFECT score — a
        fabricated value, worse than the honest "key absent → weight
        redistributed" path. This matches the skip-NaN policy for the
        structural aliases and the gmsd-exception NaN test.
        """
        import giflab.metrics as metrics_mod
        from giflab.metrics import calculate_comprehensive_metrics_from_frames

        # Force the sequential loop (parallel path runs in subprocesses where a
        # monkeypatch wouldn't apply).
        monkeypatch.setenv("GIFLAB_ENABLE_PARALLEL_METRICS", "false")

        def failing_psnr(*args, **kwargs):  # noqa: ANN001
            raise RuntimeError("synthetic PSNR failure")

        monkeypatch.setattr(metrics_mod, "calculate_safe_psnr", failing_psnr)

        orig, comp = self._frames()
        result = calculate_comprehensive_metrics_from_frames(
            orig, comp, force_all_metrics=True
        )

        assert "psnr_mean" not in result, (
            "psnr_mean must be omitted (not NaN) when no raw PSNR was measured, "
            "so composite redistributes weight instead of awarding a fabricated "
            "perfect PSNR score via normalize_metric('psnr_mean', nan) == 1.0"
        )
        # And composite must NOT be poisoned — finite, no fabricated-perfect
        # PSNR term sneaking in.
        assert "composite_quality" in result
        assert not math.isnan(result["composite_quality"])

    def test_dual_merge_preserves_bare_equals_mean_contract(self, monkeypatch):
        """Contract guard for the DUAL (white+black) worst-of merge.

        The existing ``test_from_frames_mean_aliases_match_bare`` runs on a
        SINGLE ``_from_frames`` pass, so it never exercises the merge that
        mutates ``_mean`` keys on the file-level transparent-GIF path. This test
        builds two real ``_from_frames`` results (a near-perfect and a degraded
        pass standing in for white/black), merges them, and asserts that AFTER
        the merge:

        * every same-scale stem still satisfies bare == ``_mean`` (the
          load-bearing alias the public ``measure()`` surface projects); and
        * ``psnr`` stays NORMALISED 0-1 while ``psnr_mean`` stays RAW dB, with
          ``psnr_mean ≈ psnr * PSNR_MAX_DB`` (the scale split survives the merge).

        Pre-fix the merge updated only ``X_mean`` and left bare ``X`` at the
        optimistic pass's value, so bare != ``_mean`` and ``measure().ssim`` /
        ``.gmsd`` / … reported the white-only score.
        """
        from giflab.config import DEFAULT_METRICS_CONFIG
        from giflab.metrics import (
            _merge_worst_of_dual_composite,
            calculate_comprehensive_metrics_from_frames,
        )

        monkeypatch.setenv("GIFLAB_ENABLE_PARALLEL_METRICS", "false")
        rng = np.random.default_rng(7)
        orig = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(4)]
        # "white" stand-in: near-perfect (small perturbation).
        white_comp = [
            np.clip(f.astype(int) + rng.integers(-3, 3, (64, 64, 3)), 0, 255).astype(
                np.uint8
            )
            for f in orig
        ]
        # "black" stand-in: degraded (heavy noise) so it wins (worst-of) most
        # stems, forcing the bare/companion family copy from the degraded pass.
        black_comp = [
            np.clip(f.astype(int) + rng.integers(-90, 90, (64, 64, 3)), 0, 255).astype(
                np.uint8
            )
            for f in orig
        ]

        white = calculate_comprehensive_metrics_from_frames(
            orig, white_comp, force_all_metrics=True
        )
        black = calculate_comprehensive_metrics_from_frames(
            orig, black_comp, force_all_metrics=True
        )
        merged = _merge_worst_of_dual_composite(white, black, DEFAULT_METRICS_CONFIG)

        for base in self._SAME_SCALE_PAIRS:
            assert merged[f"{base}_mean"] == pytest.approx(merged[base]), (
                f"after dual merge, {base}_mean must still alias the same-scale "
                f"bare {base} key (got bare={merged[base]}, "
                f"mean={merged[f'{base}_mean']})"
            )

        # PSNR scale split must survive the merge.
        psnr_bare = merged["psnr"]
        psnr_mean = merged["psnr_mean"]
        assert psnr_mean != pytest.approx(psnr_bare), (
            "after dual merge, psnr_mean must stay RAW dB, distinct from the "
            "normalised bare psnr"
        )
        assert psnr_mean > 1.0
        assert psnr_mean == pytest.approx(
            psnr_bare * float(DEFAULT_METRICS_CONFIG.PSNR_MAX_DB), rel=1e-3
        )

    def test_dual_merge_companion_keys_follow_winning_pass(self, monkeypatch):
        """Companion (non-sibling) keys must follow the SAME pass that won their
        stem — no stale white-provenance drift.

        ``ssimulacra2_p95``, the ``temporal_consistency_compressed`` pre/post/
        original/compressed cluster and the positional ``ssim_first/last/middle/
        positional_variance`` stats are NOT covered by the old
        ``_SIBLING_SUFFIXES`` set, so before the fix they stayed at the white
        value even when black won the stem. This asserts they now move with the
        winning pass by checking each companion equals the value from whichever
        pass supplied the merged ``_mean``/bare stem.
        """
        from giflab.config import DEFAULT_METRICS_CONFIG
        from giflab.metrics import (
            _merge_worst_of_dual_composite,
            calculate_comprehensive_metrics_from_frames,
        )

        monkeypatch.setenv("GIFLAB_ENABLE_PARALLEL_METRICS", "false")
        rng = np.random.default_rng(11)
        orig = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(4)]
        white_comp = [
            np.clip(f.astype(int) + rng.integers(-3, 3, (64, 64, 3)), 0, 255).astype(
                np.uint8
            )
            for f in orig
        ]
        black_comp = [
            np.clip(f.astype(int) + rng.integers(-90, 90, (64, 64, 3)), 0, 255).astype(
                np.uint8
            )
            for f in orig
        ]
        white = calculate_comprehensive_metrics_from_frames(
            orig, white_comp, force_all_metrics=True
        )
        black = calculate_comprehensive_metrics_from_frames(
            orig, black_comp, force_all_metrics=True
        )
        merged = _merge_worst_of_dual_composite(white, black, DEFAULT_METRICS_CONFIG)

        # For each stem with companions, the merged companion must equal the
        # value from whichever pass supplied the merged bare/stem value (no
        # half-white/half-black drift). Determine the winning pass per stem by
        # matching the merged bare value to white vs black.
        def winning_pass(stem):
            mv = merged.get(stem)
            if mv is None or (isinstance(mv, float) and math.isnan(mv)):
                return None
            if white.get(stem) == pytest.approx(mv):
                # Ambiguous when both passes agree — skip (no drift possible).
                if black.get(stem) == pytest.approx(mv):
                    return None
                return white
            if black.get(stem) == pytest.approx(mv):
                return black
            return None

        companions = {
            "ssim": [
                "ssim_first",
                "ssim_last",
                "ssim_middle",
                "ssim_positional_variance",
            ],
            "ssimulacra2": ["ssimulacra2_p95"],
            # Wave 7: the bare ``temporal_consistency`` stem is gone; use the
            # honest ``_compressed`` key as the representative of the family for
            # determining the winning pass.
            "temporal_consistency_compressed": [
                "temporal_consistency_pre",
                "temporal_consistency_post",
                "temporal_consistency_original",
                "temporal_consistency_compressed",
            ],
        }
        checked = 0
        for stem, comp_keys in companions.items():
            winner = winning_pass(stem)
            if winner is None:
                continue
            for ck in comp_keys:
                if ck not in winner or ck not in merged:
                    continue
                assert merged[ck] == pytest.approx(winner[ck]), (
                    f"companion {ck} must follow the pass that won stem {stem!r} "
                    f"(merged={merged[ck]}, winner={winner[ck]})"
                )
                checked += 1
        assert checked > 0, (
            "expected at least one companion key to be asserted; the synthetic "
            "passes should diverge enough to exercise the family copy"
        )


class TestSsimulacra2ErrorPathKeyShape:
    """Every SSIMULACRA2 error / disabled path must emit the full 5-key shape,
    not just ``ssimulacra2_mean`` (the calculate_selected_metrics partial-keys
    bug). NaN scores, 0.0 bookkeeping.
    """

    _KEYS = (
        "ssimulacra2_mean",
        "ssimulacra2_p95",
        "ssimulacra2_min",
        "ssimulacra2_frame_count",
        "ssimulacra2_triggered",
    )

    def _frames(self):
        rng = np.random.default_rng(3)
        return (
            [rng.integers(0, 256, (24, 24, 3), dtype=np.uint8) for _ in range(2)],
            [rng.integers(0, 256, (24, 24, 3), dtype=np.uint8) for _ in range(2)],
        )

    def test_selected_metrics_error_emits_full_5_key_shape(self, monkeypatch):
        from giflab.metrics import calculate_selected_metrics

        def _boom(*args, **kwargs):  # noqa: ANN001
            raise RuntimeError("synthetic SSIMULACRA2 failure")

        monkeypatch.setattr(
            "giflab.ssimulacra2_metrics.calculate_ssimulacra2_quality_metrics",
            _boom,
        )

        config = MetricsConfig()
        config.ENABLE_SSIMULACRA2 = True

        a, b = self._frames()
        result = calculate_selected_metrics(a, b, {"ssimulacra2": True}, config=config)

        for key in self._KEYS:
            assert key in result, f"missing {key} on selected-metrics error path"
        # Score keys NaN, bookkeeping keys 0.0.
        assert math.isnan(result["ssimulacra2_mean"])
        assert math.isnan(result["ssimulacra2_p95"])
        assert math.isnan(result["ssimulacra2_min"])
        assert result["ssimulacra2_frame_count"] == 0.0
        assert result["ssimulacra2_triggered"] == 0.0


class TestCompositeQualityLpipsNaNSafety:
    """``calculate_composite_quality`` must NOT add LPIPS weight or inflate the
    score when ``lpips_quality_mean`` is NaN. Without the guard,
    ``max(0.0, min(1.0, 1.0 - nan))`` evaluates to 1.0 (full LPIPS weight as
    perfect quality) and corrupts total_weight.
    """

    def _base_metrics(self):
        # A realistic set of non-LPIPS metrics so total_weight is well-defined.
        return {
            "ssim_mean": 0.8,
            "ms_ssim_mean": 0.8,
            "psnr_mean": 0.8,
            "mse_mean": 100.0,
            "fsim_mean": 0.8,
            "gmsd_mean": 0.1,
            "chist_mean": 0.9,
            "sharpness_similarity_mean": 0.8,
            "texture_similarity_mean": 0.8,
            "temporal_consistency_delta": 0.1,
        }

    def test_nan_lpips_does_not_inflate_composite(self):
        metrics_with_nan = dict(self._base_metrics())
        metrics_with_nan["lpips_quality_mean"] = float("nan")

        metrics_without = dict(self._base_metrics())  # no lpips key at all

        score_nan = calculate_composite_quality(metrics_with_nan)
        score_without = calculate_composite_quality(metrics_without)

        # A NaN LPIPS must be skipped exactly as if the key were absent —
        # identical composite, finite, not pinned to a perfect 1.0.
        assert not math.isnan(score_nan)
        assert score_nan == pytest.approx(score_without)
        assert 0.0 <= score_nan <= 1.0

    def test_real_lpips_still_contributes(self):
        """Guard must not accidentally drop a genuine (non-NaN) LPIPS score."""
        metrics_low_lpips = dict(self._base_metrics())
        metrics_low_lpips["lpips_quality_mean"] = 0.0  # identical -> best

        metrics_high_lpips = dict(self._base_metrics())
        metrics_high_lpips["lpips_quality_mean"] = 1.0  # very different -> worst

        score_low = calculate_composite_quality(metrics_low_lpips)
        score_high = calculate_composite_quality(metrics_high_lpips)

        # Lower LPIPS (more similar) must yield a higher composite — proves the
        # term is still wired in for finite values.
        assert score_low > score_high


class TestCompositeQualityNaNRedistribution:
    """``calculate_composite_quality`` must filter NaN inputs uniformly across
    ALL metric blocks (not just LPIPS/SSIMULACRA2), redistribute the surviving
    weight, and return ``float("nan")`` when the unmeasurable weight reaches
    ``COMPOSITE_NAN_THRESHOLD``.

    Without the fix a NaN metric flows through ``normalize_metric`` and the
    clamp ``max(0.0, min(1.0, nan))`` returns its upper bound 1.0 in CPython,
    fabricating a PERFECT contribution AND inflating ``total_weight`` — so a
    total measurement failure scores as perfect quality.

    Reference: giflab-composite-quality-nan-guard task note (accuracy-lens
    audit, PR #16). This generalises the per-metric LPIPS/SSIMULACRA2 guards
    (TestCompositeQualityLpipsNaNSafety) to every metric block.
    """

    def _base_metrics(self) -> dict:
        # ~10 finite metrics so total_weight is well-defined and the redistribution
        # behaviour is observable. Deliberately omits LPIPS/SSIMULACRA2 (already
        # guarded) to isolate the previously-unguarded blocks.
        return {
            "ssim_mean": 0.8,
            "ms_ssim_mean": 0.8,
            "psnr_mean": 0.8,
            "mse_mean": 100.0,
            "fsim_mean": 0.8,
            "gmsd_mean": 0.1,
            "chist_mean": 0.9,
            "sharpness_similarity_mean": 0.8,
            "texture_similarity_mean": 0.8,
            "temporal_consistency_delta": 0.1,
        }

    def test_one_metric_nan_behaves_like_absent_key(self):
        """(a) A single NaN metric must be redistributed away — the composite
        must EQUAL the composite of the same dict with that key absent, be
        finite, and not be the inflated value the old code produced.

        This is the assertion that fails on current main: today a NaN ssim
        flows through as a fabricated 1.0 contribution, inflating the score
        ABOVE the absent-key value rather than matching it.
        """
        metrics_with_nan = dict(self._base_metrics())
        metrics_with_nan["ssim_mean"] = float("nan")

        metrics_without = dict(self._base_metrics())
        del metrics_without["ssim_mean"]

        score_nan = calculate_composite_quality(metrics_with_nan)
        score_without = calculate_composite_quality(metrics_without)

        assert not math.isnan(score_nan)
        assert score_nan != 0.0
        assert 0.0 <= score_nan <= 1.0
        # NaN metric must behave EXACTLY like an absent key (proves
        # redistribution, not the current inflate-to-1.0 behaviour).
        assert score_nan == pytest.approx(score_without)

    def test_all_metrics_nan_returns_nan(self):
        """(b) When every present metric is NaN the composite is wholly
        unmeasurable and must return NaN — not a fabricated high score."""
        all_nan = {k: float("nan") for k in self._base_metrics()}
        result = calculate_composite_quality(all_nan)
        assert math.isnan(result)

    def test_missing_weight_at_or_above_threshold_returns_nan(self):
        """(c) When the NaN weight is >= COMPOSITE_NAN_THRESHOLD of the present
        weight, the composite is majority-missing and returns NaN rather than a
        misleadingly confident number.

        Present weights: ms_ssim 0.18 + ssim 0.15 + psnr 0.08 (NaN = 0.41)
        against fsim 0.07 + edge 0.06 + gmsd 0.04 + chist 0.04 (finite = 0.21).
        Missing fraction 0.41 / 0.62 ≈ 0.66 >= 0.5 → NaN.
        """
        metrics = {
            "ms_ssim_mean": float("nan"),
            "ssim_mean": float("nan"),
            "psnr_mean": float("nan"),
            "fsim_mean": 0.8,
            "edge_similarity_mean": 0.8,
            "gmsd_mean": 0.1,
            "chist_mean": 0.9,
        }
        result = calculate_composite_quality(metrics)
        assert math.isnan(result)

    def test_missing_weight_just_below_threshold_returns_finite(self):
        """Symmetric boundary pin: just under threshold → finite redistributed
        score, NOT NaN.

        Present weights: ssim 0.15 + psnr 0.08 (NaN = 0.23) against
        ms_ssim 0.18 + fsim 0.07 + edge 0.06 + gmsd 0.04 + chist 0.04
        (finite = 0.39). Missing fraction 0.23 / 0.62 ≈ 0.37 < 0.5 → finite.
        """
        metrics = {
            "ssim_mean": float("nan"),
            "psnr_mean": float("nan"),
            "ms_ssim_mean": 0.8,
            "fsim_mean": 0.8,
            "edge_similarity_mean": 0.8,
            "gmsd_mean": 0.1,
            "chist_mean": 0.9,
        }
        result = calculate_composite_quality(metrics)
        assert not math.isnan(result)
        assert 0.0 <= result <= 1.0

    def test_exactly_at_threshold_returns_nan(self):
        """The boundary is ``>=`` — exactly-half-missing weight returns NaN.

        Present weights: ssim 0.15 (NaN) against fsim 0.07 + edge 0.06 +
        gmsd 0.02-equiv... use a symmetric 50/50 split: ssim 0.15 NaN against
        edge 0.06 + gmsd 0.04 + chist 0.04 + sharpness 0.01-equiv. Simpler:
        ssim 0.15 NaN vs a finite set summing to exactly 0.15 →
        fsim 0.07 + chist 0.04 + gmsd 0.04 = 0.15. Missing fraction
        0.15 / 0.30 == 0.5 == threshold → NaN.
        """
        metrics = {
            "ssim_mean": float("nan"),
            "fsim_mean": 0.8,
            "chist_mean": 0.9,
            "gmsd_mean": 0.1,
        }
        result = calculate_composite_quality(metrics)
        assert math.isnan(result)

    def test_deltae_nan_also_redistributed(self):
        """``deltae_mean`` normalises NaN to 0.0 today (worst), unlike the other
        blocks that inflate to 1.0 — the uniform filter must make it behave like
        an absent key too, eliminating the asymmetry."""
        metrics_with_nan = dict(self._base_metrics())
        metrics_with_nan["deltae_mean"] = float("nan")

        metrics_without = dict(self._base_metrics())  # no deltae key

        score_nan = calculate_composite_quality(metrics_with_nan)
        score_without = calculate_composite_quality(metrics_without)

        assert not math.isnan(score_nan)
        assert score_nan == pytest.approx(score_without)

    def test_temporal_delta_nan_skips_temporal_does_not_fall_back(self):
        """A NaN ``temporal_consistency_delta`` must skip the temporal block
        entirely — NOT silently fall back to the single-stream
        ``temporal_consistency_compressed`` value (Wave 7 renamed the bare
        key), which would re-introduce the static-black-wins bug the delta was
        added to fix."""
        metrics = dict(self._base_metrics())
        metrics["temporal_consistency_delta"] = float("nan")
        # A deliberately perfect single-stream value that would inflate the
        # score if the code fell back to it.
        metrics["temporal_consistency_compressed"] = 1.0

        metrics_no_temporal = dict(self._base_metrics())
        del metrics_no_temporal["temporal_consistency_delta"]

        score = calculate_composite_quality(metrics)
        score_no_temporal = calculate_composite_quality(metrics_no_temporal)

        assert not math.isnan(score)
        # NaN delta must behave as if temporal were absent, NOT as if the
        # perfect legacy fallback contributed.
        assert score == pytest.approx(score_no_temporal)

    def test_none_value_treated_as_missing(self):
        """Defensive: a ``None`` value (not just NaN) must be treated as missing
        and redistributed, not crash the weighted sum."""
        metrics = dict(self._base_metrics())
        metrics["ssim_mean"] = None  # type: ignore[assignment]

        metrics_without = dict(self._base_metrics())
        del metrics_without["ssim_mean"]

        score = calculate_composite_quality(metrics)
        score_without = calculate_composite_quality(metrics_without)
        assert not math.isnan(score)
        assert score == pytest.approx(score_without)

    def test_empty_input_stays_zero_not_nan(self):
        """No quality keys present → total_weight 0 → return 0.0 (NOT NaN).

        Distinguishes 'nothing to measure' (0.0, preserves the existing
        ``test_process_metrics_edge_case_no_metrics`` contract) from 'keys
        present but all unmeasurable' (NaN)."""
        result = calculate_composite_quality({})
        assert result == 0.0
        assert not math.isnan(result)

    def test_schema_round_trips_nan_composite(self):
        """A NaN composite_quality must survive schema validation (CSV export
        path) — the prior ``ge=0.0, le=1.0`` Field constraint rejected NaN,
        which would crash every metrics export the moment a composite is
        unmeasurable. Finite out-of-range values must still be rejected."""
        from giflab.schema import validate_metric_record

        rec = {
            "render_ms": 10,
            "kilobytes": 5.0,
            "composite_quality": float("nan"),
        }
        model = validate_metric_record(rec)
        assert math.isnan(model.composite_quality)

        # Finite bounds still enforced.
        with pytest.raises(Exception):
            validate_metric_record(
                {"render_ms": 10, "kilobytes": 5.0, "composite_quality": 1.5}
            )


class TestLegacyCompositeQualityNaNSafety:
    """``calculate_legacy_composite_quality`` has the same anti-pattern:
    ``weight * nan = nan`` poisons the sum and ``max(0.0, min(1.0, nan))``
    returns 1.0 — a measurement failure scores as perfect. The fix filters NaN
    and applies the same threshold.
    """

    def _legacy_config(self) -> MetricsConfig:
        return MetricsConfig(USE_ENHANCED_COMPOSITE_QUALITY=False)

    def test_all_nan_legacy_returns_nan(self):
        config = self._legacy_config()
        metrics = {
            "ssim_mean": float("nan"),
            "ms_ssim_mean": float("nan"),
            "psnr_mean": float("nan"),
            "temporal_consistency_compressed": float("nan"),
        }
        from giflab.enhanced_metrics import calculate_legacy_composite_quality

        result = calculate_legacy_composite_quality(metrics, config)
        assert math.isnan(result)

    def test_one_nan_legacy_redistributes(self):
        """A single NaN (ssim, weight 0.30) against the rest finite is below the
        0.5 threshold of present weight → finite redistributed score, not the
        old fabricated 1.0."""
        config = self._legacy_config()
        from giflab.enhanced_metrics import calculate_legacy_composite_quality

        metrics = {
            "ssim_mean": float("nan"),
            "ms_ssim_mean": 0.8,
            "psnr_mean": 0.8,
            "temporal_consistency_compressed": 0.9,
        }
        result = calculate_legacy_composite_quality(metrics, config)
        assert not math.isnan(result)
        assert 0.0 <= result <= 1.0


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

    def _make_smooth_gradient_frame(
        self, width: int = 64, height: int = 64
    ) -> np.ndarray:
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
        frame = self._make_smooth_gradient_frame()

        # Collect per-frame edge_similarity scores for two levels of quantization.
        # 256 colours (near-identical) vs 4 colours (heavy quantization).
        frames_256_col = [self._quantize_frame(frame, 256) for _ in range(8)]
        frames_4_col = [self._quantize_frame(frame, 4) for _ in range(8)]

        scores_256: list[float] = [edge_similarity(frame, q) for q in frames_256_col]
        scores_4: list[float] = [edge_similarity(frame, q) for q in frames_4_col]

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

        scores_256: list[float] = [edge_similarity(frame, q) for q in frames_256_col]
        scores_4: list[float] = [edge_similarity(frame, q) for q in frames_4_col]

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
        expected_mean = float(np.mean(scores))  # 0.25

        # The primary key must equal median, not mean.
        assert result["edge_similarity"] == pytest.approx(expected_median, abs=1e-6), (
            f"edge_similarity primary key should be median ({expected_median:.4f}) "
            f"but got {result['edge_similarity']:.4f} (mean would be {expected_mean:.4f}). "
            "Fix: pass np.median as the primary aggregation for edge_similarity."
        )
        # Min and max sub-keys should still be present and correct
        assert result["edge_similarity_min"] == pytest.approx(0.0, abs=1e-6)
        assert result["edge_similarity_max"] == pytest.approx(1.0, abs=1e-6)

    def test_aggregate_metric_uses_median_for_texture_similarity(self):
        """_aggregate_metric must use median (not mean) for ``texture_similarity``.

        Same rationale as edge_similarity: LBP-histogram correlation is
        heavy-tailed toward 1.0 on flat / near-flat frames (intensity-inversion
        invariance), so a handful of 1.0 outliers drag the mean upward and make
        the exported score non-monotonic under palette reduction. Median ignores
        the outliers and reflects the typical frame quality.

        FAILS before the fix (texture_similarity uses mean), PASSES after.
        """
        from giflab.metrics import _aggregate_metric

        # Most frames near-zero; a couple return the LBP-identity outlier 1.0.
        scores = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]

        result = _aggregate_metric(scores, "texture_similarity")

        expected_median = float(np.median(scores))  # 0.0
        expected_mean = float(np.mean(scores))  # 0.25

        assert result["texture_similarity"] == pytest.approx(
            expected_median, abs=1e-6
        ), (
            f"texture_similarity primary key should be median "
            f"({expected_median:.4f}) but got {result['texture_similarity']:.4f} "
            f"(mean would be {expected_mean:.4f})."
        )
        assert result["texture_similarity_min"] == pytest.approx(0.0, abs=1e-6)
        assert result["texture_similarity_max"] == pytest.approx(1.0, abs=1e-6)

    def test_texture_similarity_is_median_aggregated(self):
        """texture_similarity must be a member of _MEDIAN_AGGREGATED_METRICS."""
        from giflab.metrics import _MEDIAN_AGGREGATED_METRICS

        assert "texture_similarity" in _MEDIAN_AGGREGATED_METRICS

    def test_aggregate_metric_preserves_mean_for_non_edge_metrics(self):
        """Changing edge_similarity to median must NOT affect other metric keys.

        _aggregate_metric with a non-edge metric name must still return mean as
        the primary value.
        """
        from giflab.metrics import _aggregate_metric

        scores = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
        result = _aggregate_metric(scores, "ssim")

        expected_mean = float(np.mean(scores))
        assert result["ssim"] == pytest.approx(
            expected_mean, abs=1e-6
        ), "ssim primary key should still use mean aggregation"


class TestExtractGifFramesAlphaCompositing:
    """Regression tests for the alpha-compositing bug.

    Bug: ``extract_gif_frames`` previously called ``Image.convert('RGB')``
    directly, which resolves transparent palette indices through the GIF's
    declared background colour. Animately (and other compressors) often
    rearrange the palette during recompression, so the "background colour"
    resolves differently for original vs. compressed copies of the same GIF
    — even when the visible content is identical. Naively-extracted RGB
    frames therefore disagree on the transparent regions and every metric
    collapses.

    These tests pin the fix: RGBA frames must be composited onto a fixed
    background (white) before being returned as RGB, so identity-pair
    extraction and identity-pair metrics behave as expected on
    transparency-bearing GIFs.

    Surfaced from audit/2026-05-22 — the corpus's #1 outlier
    (``8e172835-…-244425430cd5.gif``) collapsed to ssim=0.116 purely
    because of this rendering bug.
    """

    def _make_transparent_palette_gif(
        self,
        path: Path,
        background_index: int,
        n_frames: int = 2,
        size: tuple[int, int] = (32, 32),
    ) -> Path:
        """Build a palette-mode animated GIF with a transparent index.

        The visible content (a solid red square in the corner) is identical
        regardless of ``background_index``. The bug is that the *unused*
        background palette entry's COLOUR differs between two builds, so
        ``.convert('RGB')`` resolves the transparent pixels to different
        RGB values for files that look identical to a human.
        """
        # Construct a 4-entry palette: red, green, blue, yellow.
        # background_index picks which entry is treated as transparent —
        # callers vary this so the "transparent region resolves to red /
        # green / blue / yellow" between the two GIFs even though the
        # visible (non-transparent) content is the same.
        palette = [
            255,
            0,
            0,  # 0 = red
            0,
            255,
            0,  # 1 = green
            0,
            0,
            255,  # 2 = blue
            255,
            255,
            0,  # 3 = yellow
        ] + [0] * (256 * 3 - 12)

        frames = []
        for _ in range(n_frames):
            # Fill entire frame with the transparent index so the
            # transparent region dominates — mirrors the audit's
            # 76% transparent GIF.
            img = Image.new("P", size, background_index)
            img.putpalette(palette)
            # Stamp a small visible patch (index 0 = red) in the corner so
            # the GIF actually has visible content too.
            for y in range(0, 4):
                for x in range(0, 4):
                    img.putpixel((x, y), 0)
            frames.append(img)

        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
            transparency=background_index,
            disposal=2,
        )
        return path

    def test_extracted_frames_are_stable_across_palette_reorder(self, tmp_path):
        """Identical visible content with different palette → identical RGB frames.

        This is the core invariant the fix enforces: an originally-transparent
        pixel must composite to the SAME RGB value regardless of what colour
        happened to occupy the background palette index.
        """
        gif_a = self._make_transparent_palette_gif(
            tmp_path / "a.gif", background_index=1  # transparent ≈ green
        )
        gif_b = self._make_transparent_palette_gif(
            tmp_path / "b.gif", background_index=2  # transparent ≈ blue
        )

        frames_a = extract_gif_frames(gif_a).frames
        frames_b = extract_gif_frames(gif_b).frames

        assert len(frames_a) == len(frames_b)
        for fa, fb in zip(frames_a, frames_b, strict=False):
            # Frames must be byte-identical: same content, transparent
            # region composited onto the same background colour.
            assert np.array_equal(fa, fb), (
                "extract_gif_frames returned different RGB pixels for "
                "GIFs whose only difference is the unused background "
                "palette colour. The transparent region must composite "
                "to a fixed background regardless of palette ordering."
            )

    def test_identity_metrics_on_transparent_gif_hit_ceiling(self, tmp_path):
        """metric(gif, gif) must return ceiling values for a transparent GIF.

        Independent of palette weirdness — comparing a GIF against itself
        must yield perfect metric scores. This is the simplest possible
        regression test: any deviation from ceiling indicates the
        extraction pipeline is introducing noise.
        """
        gif = self._make_transparent_palette_gif(
            tmp_path / "transparent.gif", background_index=2
        )

        metrics = calculate_comprehensive_metrics(gif, gif)

        # Identity-pair must hit ceiling on the pair-comparison metrics.
        # (The high-quality tier in the optimisation pass may legitimately
        # short-circuit some metrics; we assert ceiling for everything that
        # IS reported plus the always-present composite.)
        assert metrics["ssim"] == pytest.approx(1.0, abs=1e-6), metrics
        assert metrics["mse"] == pytest.approx(0.0, abs=1e-6), metrics
        assert metrics["composite_quality"] == pytest.approx(1.0, abs=1e-6), metrics
        # ms_ssim and chist are conditionally computed; if present, ceiling.
        if "ms_ssim" in metrics:
            assert metrics["ms_ssim"] == pytest.approx(1.0, abs=1e-6), metrics
        if "chist" in metrics:
            assert metrics["chist"] == pytest.approx(1.0, abs=1e-6), metrics

    def test_metrics_invariant_to_palette_reorder(self, tmp_path):
        """Cross-pair metrics on visually-identical transparent GIFs hit ceiling.

        This is the audit-corpus scenario in miniature: two GIFs that look
        identical (same visible content, same transparent regions) but whose
        underlying palette ordering differs — exactly what animately's lossy
        re-encoding produces. Before the fix, the transparent regions
        composited to different RGB values and every metric collapsed
        (audit/2026-05-22 reported ssim=0.116 on the real corpus example).
        After the fix, metrics should treat the two as essentially identical.
        """
        gif_a = self._make_transparent_palette_gif(
            tmp_path / "a.gif", background_index=1
        )
        gif_b = self._make_transparent_palette_gif(
            tmp_path / "b.gif", background_index=2
        )

        metrics = calculate_comprehensive_metrics(gif_a, gif_b)

        # Visually-identical content should yield ceiling on the structural
        # similarity metrics. Pre-fix this collapsed to ~0.12 on real GIFs.
        assert metrics["ssim"] == pytest.approx(1.0, abs=1e-3), metrics
        if "ms_ssim" in metrics:
            assert metrics["ms_ssim"] == pytest.approx(1.0, abs=1e-3), metrics
        if "chist" in metrics:
            assert metrics["chist"] == pytest.approx(1.0, abs=1e-3), metrics
        assert metrics["composite_quality"] == pytest.approx(1.0, abs=1e-3), metrics

    # ------------------------------------------------------------------
    # Dual-composite (white + black) tests. White-only compositing biases
    # every pixel metric IN FAVOUR of dark-content GIFs: near-black detail on
    # a transparent field composited onto white is swamped (high ssim/psnr),
    # so a perturbation that would be obvious against a dark background is
    # invisible. The dual path also composites onto black and merges worst-of.
    # ------------------------------------------------------------------

    def _make_dark_on_transparent_gif(
        self,
        path: Path,
        content_rgb: tuple[int, int, int],
        n_frames: int = 2,
        size: tuple[int, int] = (128, 128),
        block: tuple[int, int] = (56, 72),
    ) -> Path:
        """Build an RGBA animated GIF: near-black content on a transparent field.

        A central ``block`` (``[lo, hi)`` on both axes) holds ``content_rgb``
        (opaque); the surrounding majority of the frame is fully transparent.
        Saved as an RGBA-source GIF so transparency is real per-pixel alpha
        (``_frame_has_transparency`` detects it). The transparent area
        dominates so that, on the white composite, the agreeing white field
        keeps global metrics high while the small dark patch perturbation is
        swamped.
        """
        frames = []
        lo, hi = block
        for _ in range(n_frames):
            frame = Image.new("RGBA", size, (0, 0, 0, 0))  # transparent field
            # Central opaque content block (the near-black "detail").
            for y in range(lo, hi):
                for x in range(lo, hi):
                    frame.putpixel((x, y), (*content_rgb, 255))
            frames.append(frame)

        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
            disposal=2,
        )
        return path

    def test_dual_composite_reveals_dark_region_difference(self, tmp_path):
        """HARD GATE: near-black content perturbation tanks the BLACK composite.

        Original near-black detail (~(8, 8, 8)) in a small patch on a
        mostly-transparent field; compressed perturbs the patch brightness.
        On the WHITE composite the dominant agreeing white field keeps the
        composite high (~0.90, hugging the ceiling). On the BLACK composite the
        near-black content merges toward the background, so the structural /
        edge / ssimulacra2 metrics collapse and the worst-of merged composite
        drops materially (~0.77).

        This MUST fail on ``main`` (white-only ~0.90 >= 0.85) and pass after
        the dual path (merged ~0.77 < 0.85): the bias against dark content is
        exactly what we're eliminating. Threshold 0.85 sits cleanly between the
        white-only and merged composites measured on this fixture.
        """
        original = self._make_dark_on_transparent_gif(
            tmp_path / "orig.gif", content_rgb=(8, 8, 8)
        )
        # Perturb the dark content. Against the dominant white field this is a
        # small global change; against black it is a large structural change.
        compressed = self._make_dark_on_transparent_gif(
            tmp_path / "comp.gif", content_rgb=(70, 70, 70)
        )

        metrics = calculate_comprehensive_metrics(original, compressed)

        # The dual (worst-of) composite must register the dark-region
        # difference rather than hugging the white-composited ceiling.
        assert metrics["composite_quality"] < 0.85, metrics

    def test_opaque_gif_single_pass_no_black_composite(self, tmp_path, monkeypatch):
        """Opaque GIF: exactly ONE extraction per file, no (0,0,0) second pass.

        Zero-cost guarantee — ``has_alpha is False`` short-circuits before any
        black extraction. We spy on ``extract_gif_frames`` and assert no call
        ever requests ``alpha_background=(0, 0, 0)``.
        """
        import giflab.metrics as metrics_mod

        # Fully-opaque animated GIF (RGB source, no transparency).
        gif = tmp_path / "opaque.gif"
        frames = [Image.new("RGB", (48, 48), (i * 40, 100, 200)) for i in range(3)]
        frames[0].save(
            gif, save_all=True, append_images=frames[1:], duration=100, loop=0
        )

        backgrounds_seen = []
        real_extract = metrics_mod.extract_gif_frames

        def _spy(path, max_frames=None, alpha_background=None):
            backgrounds_seen.append(alpha_background)
            return real_extract(path, max_frames, alpha_background)

        monkeypatch.setattr(metrics_mod, "extract_gif_frames", _spy)

        result = metrics_mod.calculate_comprehensive_metrics(gif, gif)

        # No black pass requested.
        assert (0, 0, 0) not in backgrounds_seen, backgrounds_seen
        # Exactly two extractions (original + compressed), both white/default.
        assert len(backgrounds_seen) == 2, backgrounds_seen
        # Identity opaque pair still hits ceiling.
        assert result["composite_quality"] == pytest.approx(1.0, abs=1e-6), result
        # has_alpha must be False on the extraction result.
        assert real_extract(gif).has_alpha is False

    def test_transparent_render_ms_reflects_dual_pass(self, tmp_path):
        """File-level render_ms covers BOTH passes (>= the larger inner pass)."""
        original = self._make_dark_on_transparent_gif(
            tmp_path / "orig.gif", content_rgb=(8, 8, 8)
        )
        compressed = self._make_dark_on_transparent_gif(
            tmp_path / "comp.gif", content_rgb=(40, 40, 40)
        )

        metrics = calculate_comprehensive_metrics(original, compressed)

        # render_ms is a non-negative int in the dict (true wall-clock total).
        assert "render_ms" in metrics
        assert isinstance(metrics["render_ms"], int)
        assert metrics["render_ms"] >= 0

    def test_merged_aggregation_stats_consistent(self, tmp_path):
        """Every present stem satisfies X_min <= X_mean <= X_max after merge."""
        original = self._make_dark_on_transparent_gif(
            tmp_path / "orig.gif", content_rgb=(8, 8, 8)
        )
        compressed = self._make_dark_on_transparent_gif(
            tmp_path / "comp.gif", content_rgb=(40, 40, 40)
        )

        metrics = calculate_comprehensive_metrics(original, compressed)

        # psnr is a documented schema exception (pre-existing, not introduced
        # by the merge): ``psnr_mean`` is RAW dB while its bare-key-derived
        # siblings ``psnr_min``/``psnr_max``/``psnr_std`` are the NORMALISED
        # [0, 1] PSNR. They live on different scales by design, so the
        # min<=mean<=max invariant does not (and is not meant to) hold there.
        scale_mismatch_stems = {"psnr"}

        for key in list(metrics):
            if not key.endswith("_mean"):
                continue
            stem = key[: -len("_mean")]
            if stem in scale_mismatch_stems:
                continue
            mean_v = metrics[key]
            min_v = metrics.get(f"{stem}_min")
            max_v = metrics.get(f"{stem}_max")
            if not isinstance(mean_v, int | float) or math.isnan(float(mean_v)):
                continue
            if isinstance(min_v, int | float) and not math.isnan(float(min_v)):
                assert float(min_v) <= float(mean_v) + 1e-6, (key, min_v, mean_v)
            if isinstance(max_v, int | float) and not math.isnan(float(max_v)):
                assert float(mean_v) <= float(max_v) + 1e-6, (key, mean_v, max_v)

    def test_warm_white_cache_still_triggers_black_pass(self, tmp_path, monkeypatch):
        """R2 regression: a warm WHITE cache hit must NOT no-op the black pass.

        Prime the white cache entry, then re-run the full metrics path on a
        transparent GIF. has_alpha must survive the cache round-trip so the
        black extraction (alpha_background=(0, 0, 0)) still runs.
        """
        import giflab.config as config_mod
        import giflab.metrics as metrics_mod
        from giflab.caching import frame_cache as fc_mod

        # Enable caching with a temp disk path.
        monkeypatch.setitem(config_mod.FRAME_CACHE, "enabled", True)
        monkeypatch.setitem(
            config_mod.FRAME_CACHE, "disk_path", tmp_path / "warm_cache.db"
        )

        # Enabling FRAME_CACHE triggers a dynamic import inside
        # extract_gif_frames that mutates the metrics-module globals
        # CACHING_ENABLED / get_frame_cache. Save and restore them so this
        # test does not pollute test_caching_architecture's expectation that
        # CACHING_ENABLED is False.
        saved_caching_enabled = metrics_mod.CACHING_ENABLED
        saved_get_frame_cache = metrics_mod.get_frame_cache
        fc_mod.reset_frame_cache()

        original = self._make_dark_on_transparent_gif(
            tmp_path / "orig.gif", content_rgb=(8, 8, 8)
        )
        compressed = self._make_dark_on_transparent_gif(
            tmp_path / "comp.gif", content_rgb=(40, 40, 40)
        )

        backgrounds_seen = []
        real_extract = metrics_mod.extract_gif_frames

        try:
            # Prime the WHITE cache entry for both files via a white extraction.
            white_o = real_extract(original, 30)
            white_c = real_extract(compressed, 30)
            assert white_o.has_alpha is True
            assert white_c.has_alpha is True

            def _spy(path, max_frames=None, alpha_background=None):
                backgrounds_seen.append(alpha_background)
                return real_extract(path, max_frames, alpha_background)

            monkeypatch.setattr(metrics_mod, "extract_gif_frames", _spy)
            metrics_mod.calculate_comprehensive_metrics(original, compressed)
        finally:
            fc_mod.reset_frame_cache()
            metrics_mod.CACHING_ENABLED = saved_caching_enabled
            metrics_mod.get_frame_cache = saved_get_frame_cache

        # The black pass still ran despite the warm white cache.
        assert (0, 0, 0) in backgrounds_seen, backgrounds_seen

    def test_single_frame_alpha_dual_path(self, tmp_path):
        """Single-frame alpha PNG + 1-frame transparent GIF identity -> ceiling.

        Exercises the single-frame extraction branch under BOTH backgrounds.
        """
        png = tmp_path / "single.png"
        img = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
        for y in range(10, 30):
            for x in range(10, 30):
                img.putpixel((x, y), (10, 10, 10, 255))
        img.save(png)

        # has_alpha detected on a single-frame alpha image.
        assert extract_gif_frames(png).has_alpha is True

        metrics = calculate_comprehensive_metrics(png, png)
        # Identity pair must hit ceiling on both composites -> worst-of ceiling.
        assert metrics["ssim"] == pytest.approx(1.0, abs=1e-6), metrics
        assert metrics["composite_quality"] == pytest.approx(1.0, abs=1e-6), metrics


class TestOptimizedTemporalFailureNaNHonesty:
    """The OPTIMIZED (conditional / Phase-6 high-tier) temporal block must emit
    NaN — not fabricated PERFECT temporal preservation — when the temporal
    consistency calculation raises.

    Audit-fix [[giflab-optimized-temporal-failure-nan]]: the optimized-branch
    temporal ``except`` handler in ``calculate_comprehensive_metrics_from_frames``
    previously wrote ``temporal_consistency_{pre,post,delta,compressed,original}``
    = ``1.0/1.0/0.0/1.0/1.0`` on failure. Because ``temporal_consistency_delta``
    (default ``USE_TEMPORAL_DELTA_FOR_COMPOSITE=True``) and the legacy
    ``temporal_consistency_compressed`` feed ``calculate_composite_quality``, a
    FAILED temporal calc silently inflated composite_quality on exactly the runs
    that lost the signal. Emitting NaN propagates the loss honestly: the consumer
    side is NaN-safe (``_is_missing`` filters it, ``_resolve_composite_from_
    contributions`` redistributes the 10% temporal weight), and the public
    validator reports the data as unavailable.

    Precedent: ``test_temporal_delta_nan_skips_temporal_does_not_fall_back`` (this
    file) already proves the honest NaN-delta composite path. This test exercises
    the OPTIMIZED path end-to-end — the INVERSE of
    ``test_psnr_mean_omitted_when_no_raw_values`` (which uses ``force_all_metrics=
    True`` to drive the STANDARD path); here we leave the force flag off and feed
    high-tier identical frames so the conditional optimized branch is taken.
    """

    _TEMPORAL_KEYS = (
        "temporal_consistency_pre",
        "temporal_consistency_post",
        "temporal_consistency_delta",
        "temporal_consistency_compressed",
        "temporal_consistency_original",
    )

    def test_optimized_temporal_failure_emits_nan_not_fabricated_perfect(
        self, monkeypatch
    ):
        import giflab.enhanced_metrics as enhanced_metrics
        import giflab.metrics as metrics_mod
        from giflab.metrics import calculate_comprehensive_metrics_from_frames

        # --- Step 1: force entry to the conditional/optimized block ---
        # Leave GIFLAB_ENABLE_CONDITIONAL_METRICS at its default (true) and ensure
        # NEITHER force flag is set, so ``use_conditional and not force_all_metrics``
        # holds and the optimized branch (not the standard path) runs.
        monkeypatch.delenv("GIFLAB_FORCE_ALL_METRICS", raising=False)
        monkeypatch.delenv("GIFLAB_ENABLE_CONDITIONAL_METRICS", raising=False)
        # Step 8: optimized conditional branch is in-process, but set this
        # defensively so a monkeypatch would apply even if a parallel path were
        # reached.
        monkeypatch.setenv("GIFLAB_ENABLE_PARALLEL_METRICS", "false")

        # --- Step 2: force the HIGH-tier gate ---
        # 3 IDENTICAL mid-grey frames -> infinite PSNR -> avg_psnr = 40.0 -> HIGH
        # tier, at which lpips + ssimulacra2 are deselected so the optimized gate
        # passes. Multi-frame so temporal is actually computed (not the trivial
        # single-frame 1.0). Identical streams keep gradient/color/ssimulacra2
        # sub-blocks succeeding so nothing else throws into the outer catch-all.
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        orig = [frame] * 3
        comp = [frame] * 3

        # --- Step 3: force the temporal except ---
        # The success branch calls the module-level ``calculate_temporal_consistency``
        # at its two call sites; patching the name on the module intercepts both so
        # the optimized-branch temporal ``except`` fires.
        def raising_temporal(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            raise RuntimeError("forced temporal failure")

        monkeypatch.setattr(
            metrics_mod, "calculate_temporal_consistency", raising_temporal
        )

        result = calculate_comprehensive_metrics_from_frames(orig, comp)

        # --- Step 4: POSITIVE path-taken assertion FIRST (load-bearing) ---
        # The optimized block is wrapped in an outer catch-all that silently demotes
        # to standard processing on ANY in-block error. Without this assertion a
        # mis-wired test would pass VACUOUSLY on the wrong path. ``optimization_
        # applied is True`` proves we reached the optimized return.
        opt_meta = result.get("_optimization_metadata", {})
        assert opt_meta.get("optimization_applied") is True, (
            "optimized path was NOT taken — the outer catch-all demoted to standard "
            f"processing (metadata: {opt_meta}). The NaN assertions below would be "
            "meaningless on the standard path."
        )
        assert opt_meta.get("quality_tier") == "high", (
            f"expected HIGH tier to reach the temporal except gate, got {opt_meta}"
        )

        # --- Step 5: NaN assertions (the PRIMARY before/after discriminator) ---
        # On the BUGGY code these are 1.0/0.0 (fabricated perfect) -> these
        # assertions FAIL. After the fix they are NaN -> PASS.
        for key in self._TEMPORAL_KEYS:
            assert key in result, f"{key} missing from optimized result"
            assert math.isnan(result[key]), (
                f"{key} must be NaN on the optimized temporal-failure path "
                f"(honest signal loss), not a fabricated value; got {result[key]!r}"
            )

        # --- Step 6: composite finite-guard ONLY (no-NaN-leak / no-crash) ---
        # NOT a de-inflation proof: on this identical-frame fixture the structural
        # metrics are genuinely perfect, so composite_quality is ~1.0 before AND
        # after the fix (the temporal weight is redistributed across still-perfect
        # metrics). This only guards that the NaN does not poison the composite.
        assert "composite_quality" in result
        assert math.isfinite(result["composite_quality"]), (
            "NaN temporal must be filtered by _resolve_composite_from_contributions "
            "(10% weight << 50% COMPOSITE_NAN_THRESHOLD), keeping composite finite"
        )

        # --- Step 7: validator honesty via the PUBLIC entry ---
        # The fixed NaN ``temporal_consistency_post`` must make the public validator
        # report the data as unavailable (and NOT raise a false animation_corruption
        # — ``disposal_artifacts_pre`` is unset on the optimized path, so the
        # cross-validation combination short-circuits before any NaN comparison).
        from giflab.meta import GifMetadata
        from giflab.optimization_validation.validation_checker import ValidationChecker

        # Precondition the validator keys on (enhanced_metrics-mirrored sentinel).
        assert enhanced_metrics._is_missing(result["temporal_consistency_post"]) is True

        checker = ValidationChecker(None)  # default config
        metadata = GifMetadata(
            gif_sha="test_sha",
            orig_filename="t.gif",
            orig_kilobytes=10.0,
            orig_width=100,
            orig_height=100,
            orig_frames=3,
            orig_fps=10.0,
            orig_n_colors=256,
            entropy=5.0,
            source_platform="test",
        )
        validation = checker.validate_compression_result(
            original_metadata=metadata,
            compression_metrics=result,
            pipeline_id="t",
            gif_name="t",
            content_type="test",
        )

        temporal_unavailable = [
            w
            for w in validation.warnings
            if w.category == "temporal_consistency" and "unavailable" in w.message
        ]
        assert temporal_unavailable, (
            "validator must emit a temporal_consistency 'data unavailable' WARNING "
            f"when temporal_consistency_post is NaN; warnings: {validation.warnings}"
        )
        # No false corruption issue from the NaN (disposal_pre short-circuit).
        assert not any(
            getattr(issue, "category", None) == "animation_corruption"
            for issue in validation.issues
        ), (
            "NaN temporal must NOT trip a false animation_corruption issue "
            f"(issues: {validation.issues})"
        )
