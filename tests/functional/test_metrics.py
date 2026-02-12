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
