"""Integration tests for frame sampling with metrics calculation."""

import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from giflab.config import DEFAULT_METRICS_CONFIG, FRAME_SAMPLING
from giflab.sampling.metrics_integration import (
    apply_sampling_to_frames,
    calculate_metrics_with_sampling,
    estimate_sampling_speedup,
)


class TestMetricsSamplingIntegration:
    """Test integration of sampling with metrics calculation."""
    
    @pytest.fixture
    def sample_frames(self):
        """Create sample frames for testing."""
        # Create 50 frames with gradual color change
        frames = []
        for i in range(50):
            # Create frame with gradient based on index
            frame = np.ones((100, 100, 3), dtype=np.uint8) * (50 + i * 3)
            # Add some noise for realism
            noise = np.random.randint(-5, 5, (100, 100, 3), dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            frames.append(frame)
        return frames
    
    @pytest.fixture
    def compressed_frames(self, sample_frames):
        """Create simulated compressed frames."""
        # Simulate compression by reducing colors and adding artifacts
        compressed = []
        for frame in sample_frames:
            # Quantize colors to simulate compression
            compressed_frame = (frame // 16) * 16
            compressed.append(compressed_frame)
        return compressed
    
    def test_metrics_with_uniform_sampling(self, sample_frames, compressed_frames):
        """Test metrics calculation with uniform sampling."""
        metrics = calculate_metrics_with_sampling(
            sample_frames,
            compressed_frames,
            sampling_enabled=True,
            sampling_strategy="uniform",
            return_sampling_info=True,
        )
        
        # Check that metrics were calculated
        assert "ssim" in metrics
        assert "psnr" in metrics
        assert "mse" in metrics
        
        # Check sampling info
        assert "_sampling_info" in metrics
        assert metrics["_sampling_info"]["sampling_applied"] is True
        assert metrics["_sampling_info"]["strategy_used"] == "uniform"
        assert metrics["_sampling_info"]["frames_sampled"] < 50
    
    def test_metrics_with_adaptive_sampling(self, sample_frames, compressed_frames):
        """Test metrics calculation with adaptive sampling."""
        metrics = calculate_metrics_with_sampling(
            sample_frames,
            compressed_frames,
            sampling_enabled=True,
            sampling_strategy="adaptive",
            return_sampling_info=True,
        )
        
        # Check metrics and sampling info
        assert "ssim" in metrics
        assert "_sampling_info" in metrics
        assert metrics["_sampling_info"]["strategy_used"] == "adaptive"
        
        # Adaptive should include motion metadata
        if "strategy_metadata" in metrics["_sampling_info"]:
            metadata = metrics["_sampling_info"]["strategy_metadata"]
            assert "motion_intensity" in metadata
    
    def test_metrics_with_progressive_sampling(self, sample_frames, compressed_frames):
        """Test metrics calculation with progressive sampling."""
        metrics = calculate_metrics_with_sampling(
            sample_frames,
            compressed_frames,
            sampling_enabled=True,
            sampling_strategy="progressive",
            return_sampling_info=True,
        )
        
        # Check metrics and sampling info
        assert "ssim" in metrics
        assert "_sampling_info" in metrics
        assert metrics["_sampling_info"]["strategy_used"] == "progressive"
        
        # Progressive should include iteration info
        if "strategy_metadata" in metrics["_sampling_info"]:
            metadata = metrics["_sampling_info"]["strategy_metadata"]
            assert "iterations" in metadata
    
    def test_metrics_with_scene_aware_sampling(self):
        """Test metrics calculation with scene-aware sampling."""
        # Create frames with distinct scenes
        frames = []
        compressed = []
        
        # Scene 1: Dark frames
        for _ in range(20):
            frame = np.ones((100, 100, 3), dtype=np.uint8) * 30
            frames.append(frame)
            compressed.append((frame // 16) * 16)
        
        # Scene 2: Bright frames
        for _ in range(20):
            frame = np.ones((100, 100, 3), dtype=np.uint8) * 200
            frames.append(frame)
            compressed.append((frame // 16) * 16)
        
        metrics = calculate_metrics_with_sampling(
            frames,
            compressed,
            sampling_enabled=True,
            sampling_strategy="scene_aware",
            return_sampling_info=True,
        )
        
        # Check metrics and sampling info
        assert "ssim" in metrics
        assert "_sampling_info" in metrics
        assert metrics["_sampling_info"]["strategy_used"] == "scene_aware"
        
        # Scene-aware should detect multiple scenes
        if "strategy_metadata" in metrics["_sampling_info"]:
            metadata = metrics["_sampling_info"]["strategy_metadata"]
            assert "num_scenes" in metadata
            assert metadata["num_scenes"] >= 2
    
    def test_sampling_disabled(self, sample_frames, compressed_frames):
        """Test that sampling can be disabled."""
        metrics = calculate_metrics_with_sampling(
            sample_frames,
            compressed_frames,
            sampling_enabled=False,
            return_sampling_info=True,
        )
        
        # Check that sampling was not applied
        assert "_sampling_info" in metrics
        assert metrics["_sampling_info"]["sampling_applied"] is False
        assert metrics["_sampling_info"]["reason"] == "disabled"
    
    def test_sampling_below_threshold(self):
        """Test sampling with frames below threshold."""
        # Create only 10 frames (below default threshold of 30)
        frames = [np.ones((50, 50, 3), dtype=np.uint8) * i for i in range(10)]
        compressed = [(f // 16) * 16 for f in frames]
        
        metrics = calculate_metrics_with_sampling(
            frames,
            compressed,
            sampling_enabled=True,
            return_sampling_info=True,
        )
        
        # Should not sample due to insufficient frames
        assert "_sampling_info" in metrics
        assert metrics["_sampling_info"]["sampling_applied"] is False
        assert metrics["_sampling_info"]["reason"] == "insufficient_frames"
    
    def test_sampling_with_different_frame_counts(self):
        """Test sampling when original and compressed have different frame counts."""
        # Original has 40 frames, compressed has 20 (frame reduction)
        original = [np.ones((50, 50, 3), dtype=np.uint8) * i for i in range(40)]
        compressed = [np.ones((50, 50, 3), dtype=np.uint8) * (i*2) for i in range(20)]
        
        metrics = calculate_metrics_with_sampling(
            original,
            compressed,
            sampling_enabled=True,
            sampling_strategy="uniform",
            return_sampling_info=True,
        )
        
        # Should still calculate metrics successfully
        assert "ssim_mean" in metrics
        assert "_sampling_info" in metrics
        assert metrics["_sampling_info"]["sampling_applied"] is True
    
    def test_sampling_performance_improvement(self, sample_frames, compressed_frames):
        """Test that sampling actually improves performance."""
        # Measure time with full processing
        start_full = time.perf_counter()
        metrics_full = calculate_metrics_with_sampling(
            sample_frames,
            compressed_frames,
            sampling_enabled=False,
        )
        time_full = time.perf_counter() - start_full
        
        # Measure time with sampling
        start_sampled = time.perf_counter()
        metrics_sampled = calculate_metrics_with_sampling(
            sample_frames,
            compressed_frames,
            sampling_enabled=True,
            sampling_strategy="uniform",
        )
        time_sampled = time.perf_counter() - start_sampled
        
        # Sampling should be faster (allow some variance)
        # Note: In unit tests this might not always be true due to overhead
        # but in real scenarios with large GIFs it should be
        if len(sample_frames) >= 30:
            # Only check speedup for sufficient frames
            assert time_sampled <= time_full * 1.2  # Allow 20% variance
        
        # Metrics should be similar (within reasonable tolerance)
        if "ssim" in metrics_full and "ssim" in metrics_sampled:
            ssim_diff = abs(metrics_full["ssim"] - metrics_sampled["ssim"])
            assert ssim_diff < 0.1  # Within 10% difference
    
    def test_apply_sampling_to_frames(self, sample_frames):
        """Test standalone frame sampling function."""
        sampled_frames, result = apply_sampling_to_frames(
            sample_frames,
            sampling_strategy="uniform",
            confidence_level=0.95,
        )
        
        assert len(sampled_frames) < len(sample_frames)
        assert len(sampled_frames) == result.num_sampled
        assert result.strategy_used == "uniform"
        
        # Check that sampled frames are from original
        for frame in sampled_frames:
            assert any(np.array_equal(frame, orig) for orig in sample_frames)
    
    def test_estimate_sampling_speedup(self):
        """Test speedup estimation function."""
        # Test with various frame counts
        speedup_10 = estimate_sampling_speedup(10, "uniform")
        speedup_50 = estimate_sampling_speedup(50, "uniform")
        speedup_100 = estimate_sampling_speedup(100, "uniform")
        
        # No speedup for small frame counts
        assert speedup_10 == 1.0
        
        # Increasing speedup with more frames
        assert speedup_50 > 1.0
        assert speedup_100 > 1.0
        
        # Test different strategies
        speedup_adaptive = estimate_sampling_speedup(100, "adaptive")
        speedup_progressive = estimate_sampling_speedup(100, "progressive")
        
        assert speedup_adaptive > 1.0
        assert speedup_progressive > 1.0


class TestSamplingWithConfig:
    """Test sampling with configuration settings."""
    
    def test_sampling_uses_config_defaults(self):
        """Test that sampling uses configuration defaults."""
        # Create frames
        frames = [np.ones((50, 50, 3), dtype=np.uint8) * i for i in range(50)]
        compressed = [(f // 16) * 16 for f in frames]
        
        # Don't specify strategy, should use config default
        metrics = calculate_metrics_with_sampling(
            frames,
            compressed,
            sampling_enabled=True,
            return_sampling_info=True,
        )
        
        # Should use default strategy from config
        default_strategy = FRAME_SAMPLING.get("default_strategy", "adaptive")
        assert metrics["_sampling_info"]["strategy_used"] == default_strategy
    
    def test_sampling_config_override(self):
        """Test that sampling parameters can override config."""
        frames = [np.ones((50, 50, 3), dtype=np.uint8) * i for i in range(50)]
        compressed = [(f // 16) * 16 for f in frames]
        
        # Override strategy
        metrics = calculate_metrics_with_sampling(
            frames,
            compressed,
            sampling_enabled=True,
            sampling_strategy="progressive",  # Override default
            return_sampling_info=True,
        )
        
        assert metrics["_sampling_info"]["strategy_used"] == "progressive"
    
    @patch.dict(FRAME_SAMPLING, {"enabled": False})
    def test_sampling_disabled_in_config(self):
        """Test that sampling respects config disable."""
        frames = [np.ones((50, 50, 3), dtype=np.uint8) * i for i in range(50)]
        compressed = [(f // 16) * 16 for f in frames]
        
        # Config has sampling disabled, don't override
        metrics = calculate_metrics_with_sampling(
            frames,
            compressed,
            return_sampling_info=True,
        )
        
        assert metrics["_sampling_info"]["sampling_applied"] is False
    
    @patch.dict(FRAME_SAMPLING, {"min_frames_threshold": 100})
    def test_sampling_threshold_from_config(self):
        """Test that minimum frame threshold is read from config."""
        frames = [np.ones((50, 50, 3), dtype=np.uint8) * i for i in range(50)]
        compressed = [(f // 16) * 16 for f in frames]
        
        # 50 frames is below the patched threshold of 100
        metrics = calculate_metrics_with_sampling(
            frames,
            compressed,
            sampling_enabled=True,
            return_sampling_info=True,
        )
        
        assert metrics["_sampling_info"]["sampling_applied"] is False
        assert metrics["_sampling_info"]["reason"] == "insufficient_frames"


class TestSamplingErrorHandling:
    """Test error handling in sampling integration."""
    
    def test_sampling_with_empty_frames(self):
        """Test handling of empty frame lists."""
        with pytest.raises(ValueError):
            calculate_metrics_with_sampling(
                [],  # Empty frames
                [],
                sampling_enabled=True,
            )
    
    def test_sampling_with_mismatched_dimensions(self):
        """Test handling of mismatched frame dimensions."""
        original = [np.ones((50, 50, 3), dtype=np.uint8)]
        compressed = [np.ones((100, 100, 3), dtype=np.uint8)]
        
        # Should handle resize internally
        metrics = calculate_metrics_with_sampling(
            original,
            compressed,
            sampling_enabled=True,
        )
        
        assert "ssim_mean" in metrics
    
    def test_sampling_failure_fallback(self):
        """Test that sampling failures fall back to full processing."""
        frames = [np.ones((50, 50, 3), dtype=np.uint8) * i for i in range(40)]
        compressed = [(f // 16) * 16 for f in frames]
        
        # Mock sampler to raise exception
        with patch('giflab.sampling.metrics_integration.create_sampler') as mock_create:
            mock_create.side_effect = Exception("Sampling failed")
            
            # Should fall back to full processing
            metrics = calculate_metrics_with_sampling(
                frames,
                compressed,
                sampling_enabled=True,
                return_sampling_info=True,
            )
            
            # Metrics should still be calculated (key may be 'ssim' or 'ssim_mean')
            assert "ssim" in metrics or "ssim_mean" in metrics
            
            # But sampling should not be applied
            if "_sampling_info" in metrics:
                assert metrics["_sampling_info"]["sampling_applied"] is False


class TestSamplingAccuracy:
    """Test accuracy of sampled metrics vs full metrics."""
    
    def create_test_frames(self, num_frames=100, pattern="gradient"):
        """Create test frames with specific patterns."""
        frames = []
        
        if pattern == "gradient":
            # Gradual color change
            for i in range(num_frames):
                frame = np.ones((100, 100, 3), dtype=np.uint8) * (i * 255 // num_frames)
                frames.append(frame)
        elif pattern == "alternating":
            # Alternating bright/dark
            for i in range(num_frames):
                value = 50 if i % 2 == 0 else 200
                frame = np.ones((100, 100, 3), dtype=np.uint8) * value
                frames.append(frame)
        elif pattern == "noise":
            # Random noise
            for _ in range(num_frames):
                frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
                frames.append(frame)
        
        return frames
    
    @pytest.mark.parametrize("pattern", ["gradient", "alternating", "noise"])
    def test_sampling_accuracy(self, pattern):
        """Test that sampled metrics are accurate for different patterns."""
        original = self.create_test_frames(100, pattern)
        # Simulate compression
        compressed = [(f // 32) * 32 for f in original]
        
        # Calculate full metrics
        metrics_full = calculate_metrics_with_sampling(
            original,
            compressed,
            sampling_enabled=False,
        )
        
        # Calculate sampled metrics with different strategies
        strategies = ["uniform", "adaptive", "progressive"]
        
        for strategy in strategies:
            metrics_sampled = calculate_metrics_with_sampling(
                original,
                compressed,
                sampling_enabled=True,
                sampling_strategy=strategy,
            )
            
            # Compare key metrics
            if "ssim" in metrics_full and "ssim" in metrics_sampled:
                ssim_error = abs(metrics_full["ssim"] - metrics_sampled["ssim"])
                # Allow up to 15% error for sampling
                assert ssim_error < 0.15, f"{strategy} sampling has high SSIM error for {pattern}"
            
            if "psnr" in metrics_full and "psnr" in metrics_sampled:
                psnr_error = abs(metrics_full["psnr"] - metrics_sampled["psnr"])
                # PSNR is normalized, so check relative error
                assert psnr_error < 0.2, f"{strategy} sampling has high PSNR error for {pattern}"
