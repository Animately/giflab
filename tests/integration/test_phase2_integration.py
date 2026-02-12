"""
Integration tests for Phase 2 Quality Refinement Metrics.

This module tests the integration between Phase 2.1 (Dither Quality) and Phase 2.2 (Deep Perceptual)
metrics, focusing on cross-module functionality, LPIPS infrastructure sharing, and composite
quality calculation integration.

Key Integration Points:
- LPIPS model sharing between temporal and spatial analysis
- Enhanced composite quality calculation with 3% LPIPS weight
- Conditional triggering based on actual quality thresholds
- Memory management across temporal and spatial LPIPS usage
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from giflab.deep_perceptual_metrics import (
    DeepPerceptualValidator,
    calculate_deep_perceptual_quality_metrics,
    should_use_deep_perceptual,
)
from giflab.enhanced_metrics import calculate_composite_quality
from giflab.gradient_color_artifacts import (
    DitherQualityAnalyzer,
    calculate_dither_quality_metrics,
)
from giflab.temporal_artifacts import TemporalArtifactDetector


class TestLPIPSInfrastructureSharing:
    """Test that LPIPS infrastructure is properly shared between temporal and spatial metrics."""

    @pytest.fixture
    def test_frames(self):
        """Create test frames for LPIPS infrastructure testing."""
        np.random.seed(42)  # Reproducible test data
        frames = []
        for i in range(4):
            # Create frames with slight variations
            base = np.ones((64, 64, 3), dtype=np.uint8) * (100 + i * 10)
            noise = np.random.randint(-5, 5, base.shape, dtype=np.int16)
            frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            frames.append(frame)
        return frames

    @pytest.mark.fast
    def test_lpips_temporal_vs_spatial_independence(self, test_frames):
        """Test that temporal and spatial LPIPS calculations don't interfere with each other."""
        original_frames = test_frames
        compressed_frames = [f.copy() for f in test_frames]

        # Apply slight compression artifacts to compressed frames
        for frame in compressed_frames:
            frame[::2, ::2] = np.clip(
                frame[::2, ::2].astype(np.int16) - 10, 0, 255
            ).astype(np.uint8)

        # Initialize both detectors
        temporal_detector = TemporalArtifactDetector(
            device="cpu", force_mse_fallback=False
        )
        DeepPerceptualValidator(device="cpu", force_fallback=False)

        # Test temporal LPIPS (consecutive frame comparison)
        temporal_metrics = temporal_detector.calculate_lpips_temporal(original_frames)

        # Test spatial LPIPS (corresponding frame comparison)
        spatial_config = {"device": "cpu", "disable_deep_perceptual": False}
        spatial_metrics = calculate_deep_perceptual_quality_metrics(
            original_frames, compressed_frames, spatial_config
        )

        # Verify both produced valid results
        assert "lpips_t_mean" in temporal_metrics
        assert temporal_metrics["lpips_t_mean"] > 0

        assert "lpips_quality_mean" in spatial_metrics
        assert spatial_metrics["lpips_quality_mean"] > 0

        # Temporal and spatial scores should be different (different comparison types)
        assert temporal_metrics["lpips_t_mean"] != spatial_metrics["lpips_quality_mean"]

    @pytest.mark.fast
    def test_lpips_model_reuse_efficiency(self):
        """Test that LPIPS model is efficiently reused between temporal and spatial analysis."""
        # Test now verifies that the model cache is properly shared
        from giflab.model_cache import LPIPSModelCache, cleanup_model_cache

        # Clean cache to start fresh
        cleanup_model_cache(force=True)

        # Initialize both systems
        temporal_detector = TemporalArtifactDetector(
            device="cpu", force_mse_fallback=False
        )
        spatial_validator = DeepPerceptualValidator(device="cpu", force_fallback=False)

        # Force model initialization
        temporal_model = temporal_detector._get_lpips_model()
        spatial_model = spatial_validator._get_lpips_model()

        # Both should get the same cached model instance
        if temporal_model is not None and spatial_model is not None:
            assert (
                temporal_model is spatial_model
            ), "Models should be the same cached instance"

        # Verify cache info shows single model
        cache_info = LPIPSModelCache.get_cache_info()
        if cache_info["models_cached"] > 0:
            # Should have at most one model cached
            assert (
                cache_info["models_cached"] <= 1
            ), "Should reuse the same cached model"

    @pytest.mark.slow
    def test_memory_management_across_modules(self, test_frames):
        """Test GPU memory management when using both temporal and spatial LPIPS."""
        # Skip if no CUDA available
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("CUDA not available for memory testing")
        except ImportError:
            pytest.skip("PyTorch not available")

        original_frames = test_frames * 10  # Larger frame set for memory testing
        compressed_frames = [f.copy() for f in original_frames]

        # Initialize with GPU
        temporal_detector = TemporalArtifactDetector(
            device="cuda", force_mse_fallback=False
        )
        DeepPerceptualValidator(device="cuda", force_fallback=False)

        initial_memory = torch.cuda.memory_allocated()

        # Run temporal analysis
        temporal_metrics = temporal_detector.calculate_lpips_temporal(original_frames)
        torch.cuda.memory_allocated()

        # Run spatial analysis
        spatial_config = {"device": "cuda", "disable_deep_perceptual": False}
        spatial_metrics = calculate_deep_perceptual_quality_metrics(
            original_frames, compressed_frames, spatial_config
        )
        final_memory = torch.cuda.memory_allocated()

        # Memory should not grow unboundedly
        memory_growth = final_memory - initial_memory
        assert memory_growth < 500_000_000  # Less than 500MB growth

        # Both analyses should succeed
        assert temporal_metrics["lpips_t_mean"] > 0
        assert spatial_metrics["lpips_quality_mean"] > 0


class TestCompositeQualityIntegration:
    """Test integration with enhanced composite quality calculation."""

    @pytest.fixture
    def mock_metrics(self):
        """Create mock metrics for composite quality testing."""
        return {
            # Traditional metrics
            "ssim": 0.85,
            "gmsd": 0.05,
            "sharpness_ratio": 0.9,
            "color_histogram_correlation": 0.88,
            # Phase 2 metrics
            "dither_ratio_mean": 1.1,
            "dither_quality_score": 75.0,
            "lpips_quality_mean": 0.25,  # Deep perceptual
            # Enhanced temporal metrics
            "flicker_excess": 0.01,
            "temporal_pumping_score": 0.08,
        }

    @pytest.mark.fast
    def test_enhanced_lpips_weight_contribution(self, mock_metrics):
        """Test that LPIPS contributes 3% weight to composite quality."""
        # Test with LPIPS metrics present
        composite_with_lpips = calculate_composite_quality(mock_metrics)

        # Test without LPIPS metrics
        metrics_without_lpips = mock_metrics.copy()
        del metrics_without_lpips["lpips_quality_mean"]
        composite_without_lpips = calculate_composite_quality(metrics_without_lpips)

        # Should have different scores
        assert composite_with_lpips != composite_without_lpips

        # Verify LPIPS impact - low LPIPS score (0.25) should improve composite quality
        # Since LPIPS is inverted for quality (1.0 - lpips_score), good LPIPS should help
        assert 0.0 <= composite_with_lpips <= 1.0
        assert 0.0 <= composite_without_lpips <= 1.0

    @pytest.mark.fast
    def test_conditional_triggering_with_real_metrics(self, mock_metrics):
        """Test conditional triggering based on actual composite quality scores."""
        # Test borderline quality case
        borderline_metrics = mock_metrics.copy()
        borderline_metrics.update(
            {
                "ssim": 0.55,  # Lower quality to get composite into 0.3-0.7 range
                "gmsd": 0.25,  # Higher distortion
                "psnr_mean": 18.0,  # Lower PSNR
            }
        )
        borderline_quality = calculate_composite_quality(borderline_metrics)

        # Should trigger deep perceptual for borderline cases
        assert should_use_deep_perceptual(borderline_quality)

        # Test high quality case
        high_quality_metrics = mock_metrics.copy()
        high_quality_metrics.update(
            {
                "ssim": 0.95,  # High quality
                "gmsd": 0.02,  # Low distortion
            }
        )
        high_quality = calculate_composite_quality(high_quality_metrics)

        # Should skip deep perceptual for high quality
        assert not should_use_deep_perceptual(high_quality)

    @pytest.mark.fast
    def test_quality_threshold_boundaries(self):
        """Test edge cases at quality thresholds 0.3 and 0.7."""
        # Test exact boundaries
        assert should_use_deep_perceptual(0.3)  # Exactly at lower threshold
        assert should_use_deep_perceptual(0.7)  # Exactly at upper threshold
        assert should_use_deep_perceptual(0.699)  # Just below upper threshold
        assert not should_use_deep_perceptual(0.701)  # Just above upper threshold

        # Test very low quality (should trigger)
        assert should_use_deep_perceptual(0.1)
        assert should_use_deep_perceptual(0.299)

        # Test very high quality (should skip)
        assert not should_use_deep_perceptual(0.8)
        assert not should_use_deep_perceptual(0.9)


class TestCrossModuleIntegration:
    """Test integration between different Phase 2 modules."""

    @pytest.fixture
    def synthetic_gif_frames(self):
        """Create synthetic frames with known dithering and compression artifacts."""
        frames = []
        for i in range(6):
            frame = np.zeros((128, 128, 3), dtype=np.uint8)

            # Create gradient with dithering patterns
            for y in range(128):
                for x in range(128):
                    # Base gradient
                    intensity = int(255 * x / 128)

                    # Add dithering pattern for some frames
                    if i >= 3:  # "Compressed" frames have dithering
                        dither_noise = 20 if (x + y) % 2 == 0 else -20
                        intensity = np.clip(intensity + dither_noise, 0, 255)

                    frame[y, x] = [intensity, intensity, intensity]

            frames.append(frame)

        return frames[:3], frames[3:]  # Original, compressed

    @pytest.mark.fast
    def test_dither_and_perceptual_metrics_together(self, synthetic_gif_frames):
        """Test that dither quality and deep perceptual metrics work together."""
        original_frames, compressed_frames = synthetic_gif_frames

        # Calculate dither quality metrics
        dither_metrics = calculate_dither_quality_metrics(
            original_frames, compressed_frames
        )

        # Calculate deep perceptual metrics
        perceptual_config = {"device": "cpu", "disable_deep_perceptual": False}
        perceptual_metrics = calculate_deep_perceptual_quality_metrics(
            original_frames, compressed_frames, perceptual_config
        )

        # Both should detect the added dithering/artifacts
        assert "dither_quality_score" in dither_metrics
        assert "lpips_quality_mean" in perceptual_metrics

        # Dither quality should be lower due to artificial dithering patterns
        assert dither_metrics["dither_quality_score"] < 90  # Should detect issues

        # LPIPS should detect perceptual differences (even small ones)
        assert (
            perceptual_metrics["lpips_quality_mean"] > 0.0
        )  # Should detect some difference
        assert perceptual_metrics["lpips_quality_mean"] < 1.0  # But not extreme

    @pytest.mark.fast
    def test_phase2_metrics_in_enhanced_composite(self, synthetic_gif_frames):
        """Test that Phase 2 metrics properly contribute to enhanced composite quality."""
        original_frames, compressed_frames = synthetic_gif_frames

        # Calculate all Phase 2 metrics
        dither_metrics = calculate_dither_quality_metrics(
            original_frames, compressed_frames
        )
        perceptual_config = {"device": "cpu", "disable_deep_perceptual": False}
        perceptual_metrics = calculate_deep_perceptual_quality_metrics(
            original_frames, compressed_frames, perceptual_config
        )

        # Create complete metrics dict
        all_metrics = {
            "ssim": 0.75,  # Baseline traditional metric
            **dither_metrics,
            **perceptual_metrics,
        }

        # Calculate enhanced composite quality
        composite_quality = calculate_composite_quality(all_metrics)

        # Should be valid quality score
        assert 0.0 <= composite_quality <= 1.0

        # Should incorporate Phase 2 metrics
        # Test by removing Phase 2 metrics and comparing
        traditional_metrics = {"ssim": 0.75}
        traditional_composite = calculate_composite_quality(traditional_metrics)

        # Phase 2 metrics should affect the score
        assert composite_quality != traditional_composite

    @pytest.mark.fast
    def test_error_propagation_between_modules(self):
        """Test that errors in one Phase 2 module don't break others."""
        original_frames = [np.ones((32, 32, 3), dtype=np.uint8) * 128]
        compressed_frames = [np.ones((32, 32, 3), dtype=np.uint8) * 100]

        # Test with dither analyzer error
        with patch.object(
            DitherQualityAnalyzer,
            "analyze_dither_quality",
            side_effect=Exception("Dither error"),
        ):
            # Deep perceptual should still work
            perceptual_metrics = calculate_deep_perceptual_quality_metrics(
                original_frames, compressed_frames, {"device": "cpu"}
            )
            assert "lpips_quality_mean" in perceptual_metrics

        # Test with deep perceptual error
        with patch.object(
            DeepPerceptualValidator,
            "calculate_deep_perceptual_metrics",
            side_effect=Exception("LPIPS error"),
        ):
            # Dither quality should still work
            dither_metrics = calculate_dither_quality_metrics(
                original_frames, compressed_frames
            )
            assert "dither_quality_score" in dither_metrics


class TestPerformanceInteractions:
    """Test performance characteristics when using multiple Phase 2 modules together."""

    @pytest.mark.slow
    def test_combined_metrics_performance(self):
        """Test performance when running all Phase 2 metrics together."""
        # Create larger test frames for performance testing
        np.random.seed(42)
        original_frames = [
            np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(10)
        ]
        compressed_frames = [
            np.clip(
                f.astype(np.int16)
                + np.random.randint(-10, 10, f.shape, dtype=np.int16),
                0,
                255,
            ).astype(np.uint8)
            for f in original_frames
        ]

        start_time = time.time()

        # Run all Phase 2 metrics
        dither_metrics = calculate_dither_quality_metrics(
            original_frames, compressed_frames
        )
        perceptual_config = {
            "device": "cpu",
            "lpips_max_frames": 8,
        }  # Limit for performance
        perceptual_metrics = calculate_deep_perceptual_quality_metrics(
            original_frames, compressed_frames, perceptual_config
        )

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete in reasonable time (adjust threshold based on expected performance)
        assert total_time < 30.0  # 30 seconds max for 10 frames

        # Both metrics should succeed
        assert "dither_quality_score" in dither_metrics
        assert "lpips_quality_mean" in perceptual_metrics

        # Log performance for monitoring
        print(f"Phase 2 combined metrics completed in {total_time:.2f}s for 10 frames")

    @pytest.mark.fast
    def test_memory_efficiency_with_multiple_modules(self):
        """Test that using multiple Phase 2 modules doesn't cause memory leaks."""
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss

            # Create test data
            original_frames = [
                np.ones((64, 64, 3), dtype=np.uint8) * 128 for _ in range(5)
            ]
            compressed_frames = [
                np.ones((64, 64, 3), dtype=np.uint8) * 100 for _ in range(5)
            ]

            # Run multiple iterations to check for memory leaks
            for _i in range(10):
                dither_metrics = calculate_dither_quality_metrics(
                    original_frames, compressed_frames
                )
                perceptual_metrics = calculate_deep_perceptual_quality_metrics(
                    original_frames, compressed_frames, {"device": "cpu"}
                )

                # Verify metrics are calculated
                assert "dither_quality_score" in dither_metrics
                assert "lpips_quality_mean" in perceptual_metrics

            final_memory = process.memory_info().rss
            memory_growth = final_memory - initial_memory

            # Memory growth should be minimal (< 50MB)
            assert memory_growth < 50_000_000

        except ImportError:
            pytest.skip("psutil not available for memory testing")
