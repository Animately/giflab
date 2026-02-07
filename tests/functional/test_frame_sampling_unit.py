"""Unit tests for frame sampling strategies."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from giflab.sampling import (
    FrameSampler,
    SamplingResult,
    SamplingStrategy,
    UniformSampler,
    AdaptiveSampler,
    ProgressiveSampler,
    SceneAwareSampler,
)
from giflab.sampling.frame_sampler import (
    FrameDifferenceCalculator,
    create_sampler,
)


class TestSamplingResult:
    """Test SamplingResult data structure."""
    
    def test_sampling_result_creation(self):
        """Test creating a SamplingResult."""
        result = SamplingResult(
            sampled_indices=[0, 5, 10],
            total_frames=15,
            sampling_rate=0.2,
            strategy_used="test",
        )
        
        assert result.num_sampled == 3
        assert result.total_frames == 15
        assert result.sampling_rate == 0.2
        assert result.strategy_used == "test"
        assert not result.is_full_sampling()
    
    def test_sampling_result_auto_rate(self):
        """Test automatic sampling rate calculation."""
        result = SamplingResult(
            sampled_indices=[0, 1, 2, 3, 4],
            total_frames=10,
            sampling_rate=0,  # Will be recalculated
        )
        
        assert result.sampling_rate == 0.5
    
    def test_full_sampling_detection(self):
        """Test detection of full sampling."""
        result = SamplingResult(
            sampled_indices=list(range(10)),
            total_frames=10,
            sampling_rate=1.0,
        )
        
        assert result.is_full_sampling()


class TestFrameDifferenceCalculator:
    """Test frame difference calculation utilities."""
    
    def test_calculate_mse(self):
        """Test MSE calculation between frames."""
        frame1 = np.ones((10, 10, 3), dtype=np.uint8) * 100
        frame2 = np.ones((10, 10, 3), dtype=np.uint8) * 110
        
        mse = FrameDifferenceCalculator.calculate_mse(frame1, frame2)
        assert mse == 100.0  # (110-100)^2 = 100
    
    def test_calculate_mse_identical(self):
        """Test MSE for identical frames."""
        frame = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        mse = FrameDifferenceCalculator.calculate_mse(frame, frame)
        assert mse == 0.0
    
    def test_calculate_histogram_difference(self):
        """Test histogram difference calculation."""
        # Create frames with different color distributions
        frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2 = np.ones((10, 10, 3), dtype=np.uint8) * 255
        
        diff = FrameDifferenceCalculator.calculate_histogram_difference(frame1, frame2)
        assert 0.9 < diff <= 1.0  # Should be close to maximum difference
    
    def test_histogram_difference_identical(self):
        """Test histogram difference for identical frames."""
        frame = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        diff = FrameDifferenceCalculator.calculate_histogram_difference(frame, frame)
        # Allow small floating point tolerance
        assert diff < 1e-6
    
    def test_detect_scene_change(self):
        """Test scene change detection."""
        # Create very different frames
        frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2 = np.ones((10, 10, 3), dtype=np.uint8) * 255
        
        is_scene_change = FrameDifferenceCalculator.detect_scene_change(
            frame1, frame2, threshold=0.3
        )
        assert is_scene_change
    
    def test_no_scene_change(self):
        """Test no scene change for similar frames."""
        # Create frames with slight noise variation
        np.random.seed(42)
        frame1 = np.ones((10, 10, 3), dtype=np.uint8) * 100
        # Add minimal noise to make slightly different but not a scene change
        frame2 = frame1.copy()
        frame2[0, 0, 0] = 101  # Change just one pixel slightly
        
        is_scene_change = FrameDifferenceCalculator.detect_scene_change(
            frame1, frame2, threshold=0.3
        )
        assert not is_scene_change


class TestUniformSampler:
    """Test uniform sampling strategy."""
    
    def test_uniform_sampling_basic(self):
        """Test basic uniform sampling."""
        frames = [np.zeros((10, 10, 3)) for _ in range(100)]
        sampler = UniformSampler(sampling_rate=0.3, min_frames_threshold=10)
        
        result = sampler.sample(frames)
        
        assert 25 <= result.num_sampled <= 35  # ~30% of 100
        assert result.sampled_indices[0] == 0  # First frame included
        assert result.sampled_indices[-1] == 99  # Last frame included
        assert result.strategy_used == "uniform"
    
    def test_uniform_below_threshold(self):
        """Test uniform sampling below minimum threshold."""
        frames = [np.zeros((10, 10, 3)) for _ in range(5)]
        sampler = UniformSampler(sampling_rate=0.3, min_frames_threshold=10)
        
        result = sampler.sample(frames)
        
        assert result.num_sampled == 5  # All frames
        assert result.is_full_sampling()
        assert result.strategy_used == "uniform (full)"
    
    def test_uniform_sampling_rate_limits(self):
        """Test sampling rate boundary conditions."""
        frames = [np.zeros((10, 10, 3)) for _ in range(50)]
        
        # Test minimum rate
        sampler = UniformSampler(sampling_rate=0.0, min_frames_threshold=10)
        result = sampler.sample(frames)
        assert result.num_sampled >= 2  # At least first and last
        
        # Test maximum rate
        sampler = UniformSampler(sampling_rate=1.0, min_frames_threshold=10)
        result = sampler.sample(frames)
        assert result.num_sampled == 50  # All frames
    
    def test_uniform_distribution(self):
        """Test that uniform sampling is evenly distributed."""
        frames = [np.zeros((10, 10, 3)) for _ in range(100)]
        sampler = UniformSampler(sampling_rate=0.2, min_frames_threshold=10)
        
        result = sampler.sample(frames)
        indices = result.sampled_indices
        
        # Check that gaps between samples are relatively uniform
        gaps = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        
        # Standard deviation should be small relative to mean for uniform distribution
        assert std_gap / mean_gap < 0.5


class TestAdaptiveSampler:
    """Test adaptive sampling strategy."""
    
    def test_adaptive_sampling_low_motion(self):
        """Test adaptive sampling with low motion frames."""
        # Create frames with minimal differences
        frames = []
        for i in range(50):
            frame = np.ones((10, 10, 3), dtype=np.uint8) * (100 + i % 2)
            frames.append(frame)
        
        sampler = AdaptiveSampler(
            base_rate=0.2,
            max_rate=0.8,
            min_frames_threshold=10
        )
        
        result = sampler.sample(frames)
        
        assert result.num_sampled < 25  # Should sample less than 50%
        assert result.strategy_used == "adaptive"
        assert "motion_intensity" in result.metadata
    
    def test_adaptive_sampling_high_motion(self):
        """Test adaptive sampling with high motion frames."""
        # Create frames with large differences
        frames = []
        for i in range(50):
            frame = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
            frames.append(frame)
        
        sampler = AdaptiveSampler(
            base_rate=0.2,
            max_rate=0.8,
            min_frames_threshold=10
        )
        
        # Mock random to ensure deterministic sampling
        with patch('numpy.random.random', return_value=0.5):
            result = sampler.sample(frames)
        
        assert result.num_sampled >= 10  # Should sample more for high motion
        assert "high_motion_frames" in result.metadata
    
    def test_adaptive_metadata(self):
        """Test that adaptive sampling provides motion metadata."""
        frames = [np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8) 
                  for _ in range(30)]
        sampler = AdaptiveSampler(min_frames_threshold=10)
        
        result = sampler.sample(frames)
        
        assert "motion_intensity" in result.metadata
        assert "high_motion_frames" in result.metadata
        assert "mean_difference" in result.metadata
        assert "std_difference" in result.metadata
        assert result.metadata["motion_intensity"] >= 0


class TestProgressiveSampler:
    """Test progressive sampling strategy."""
    
    def test_progressive_sampling_basic(self):
        """Test basic progressive sampling."""
        frames = [np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8) 
                  for _ in range(50)]
        
        sampler = ProgressiveSampler(
            initial_rate=0.1,
            increment_rate=0.1,
            max_iterations=3,
            min_frames_threshold=10
        )
        
        result = sampler.sample(frames)
        
        assert result.num_sampled >= 5  # At least 10% initial
        assert result.strategy_used == "progressive"
        assert "iterations" in result.metadata
        assert result.metadata["iterations"] <= 3
    
    def test_progressive_with_metric(self):
        """Test progressive sampling with custom metric function."""
        frames = [np.ones((10, 10, 3), dtype=np.uint8) * i 
                  for i in range(30)]
        
        def simple_metric(idx):
            return float(idx)  # Simple linear metric
        
        sampler = ProgressiveSampler(
            initial_rate=0.1,
            target_ci_width=100.0,  # Large CI width for quick convergence
            min_frames_threshold=10
        )
        
        result = sampler.sample(frames, metric_func=simple_metric)
        
        assert result.num_sampled >= 3
        assert result.confidence_interval is not None
        assert result.confidence_level == 0.95
    
    def test_progressive_convergence(self):
        """Test that progressive sampling converges."""
        frames = [np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8) 
                  for _ in range(100)]
        
        sampler = ProgressiveSampler(
            initial_rate=0.05,
            increment_rate=0.1,
            max_iterations=5,
            target_ci_width=0.01,  # Very tight CI to force max iterations
            min_frames_threshold=10
        )
        
        result = sampler.sample(frames)
        
        # Should hit max iterations with tight CI
        assert result.metadata["iterations"] == 5
        assert "final_ci_width" in result.metadata


class TestSceneAwareSampler:
    """Test scene-aware sampling strategy."""
    
    def test_scene_detection(self):
        """Test scene detection in frames."""
        frames = []
        # Create 3 distinct scenes
        for scene_val in [0, 128, 255]:
            for _ in range(10):
                frame = np.ones((10, 10, 3), dtype=np.uint8) * scene_val
                frames.append(frame)
        
        sampler = SceneAwareSampler(
            scene_threshold=0.3,
            min_frames_threshold=10
        )
        
        scenes = sampler.detect_scenes(frames)
        
        assert len(scenes) == 3  # Should detect 3 scenes
        assert scenes[0] == (0, 9)
        assert scenes[1] == (10, 19)
        assert scenes[2] == (20, 29)
    
    def test_scene_aware_sampling(self):
        """Test scene-aware sampling with multiple scenes."""
        frames = []
        # Create 2 distinct scenes
        for scene_val in [50, 200]:
            for _ in range(20):
                frame = np.ones((10, 10, 3), dtype=np.uint8) * scene_val
                frames.append(frame)
        
        sampler = SceneAwareSampler(
            scene_threshold=0.3,
            boundary_window=2,
            min_scene_samples=3,
            min_frames_threshold=10
        )
        
        result = sampler.sample(frames)
        
        assert result.strategy_used == "scene_aware"
        assert "num_scenes" in result.metadata
        assert result.metadata["num_scenes"] == 2
        
        # Check that scene boundaries are sampled
        assert 0 in result.sampled_indices  # Start of scene 1
        assert 19 in result.sampled_indices  # End of scene 1
        assert 20 in result.sampled_indices  # Start of scene 2
        assert 39 in result.sampled_indices  # End of scene 2
    
    def test_scene_boundary_sampling(self):
        """Test that scene boundaries are properly sampled."""
        frames = []
        # Create sharp scene transition
        for i in range(30):
            if i < 15:
                frame = np.zeros((10, 10, 3), dtype=np.uint8)
            else:
                frame = np.ones((10, 10, 3), dtype=np.uint8) * 255
            frames.append(frame)
        
        sampler = SceneAwareSampler(
            scene_threshold=0.3,
            boundary_window=3,
            min_frames_threshold=10
        )
        
        result = sampler.sample(frames)
        
        # Check boundary sampling
        indices = result.sampled_indices
        
        # Should sample around the boundary at frame 15
        boundary_samples = [i for i in indices if 12 <= i <= 17]
        assert len(boundary_samples) >= 4  # Should have multiple samples near boundary


class TestSamplerFactory:
    """Test sampler factory function."""
    
    def test_create_uniform_sampler(self):
        """Test creating uniform sampler via factory."""
        sampler = create_sampler(
            SamplingStrategy.UNIFORM,
            sampling_rate=0.5
        )
        assert isinstance(sampler, UniformSampler)
    
    def test_create_adaptive_sampler(self):
        """Test creating adaptive sampler via factory."""
        sampler = create_sampler(
            SamplingStrategy.ADAPTIVE,
            base_rate=0.3
        )
        assert isinstance(sampler, AdaptiveSampler)
    
    def test_create_progressive_sampler(self):
        """Test creating progressive sampler via factory."""
        sampler = create_sampler(
            SamplingStrategy.PROGRESSIVE,
            initial_rate=0.15
        )
        assert isinstance(sampler, ProgressiveSampler)
    
    def test_create_scene_aware_sampler(self):
        """Test creating scene-aware sampler via factory."""
        sampler = create_sampler(
            SamplingStrategy.SCENE_AWARE,
            scene_threshold=0.25
        )
        assert isinstance(sampler, SceneAwareSampler)
    
    def test_create_full_sampler(self):
        """Test creating full (no sampling) sampler."""
        sampler = create_sampler(SamplingStrategy.FULL)
        
        # Should create a uniform sampler with 100% rate
        assert isinstance(sampler, UniformSampler)
        
        # Test that it doesn't sample
        frames = [np.zeros((10, 10, 3)) for _ in range(20)]
        result = sampler.sample(frames)
        assert result.is_full_sampling()
    
    def test_invalid_strategy(self):
        """Test error handling for invalid strategy."""
        with pytest.raises(ValueError, match="Unknown sampling strategy"):
            create_sampler("invalid_strategy")  # type: ignore


class TestConfidenceIntervals:
    """Test confidence interval calculations."""
    
    def test_confidence_interval_calculation(self):
        """Test basic confidence interval calculation."""
        sampler = UniformSampler()
        
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci_lower, ci_upper = sampler.calculate_confidence_interval(samples, 0.95)
        
        # Check that CI contains the mean
        mean = np.mean(samples)
        assert ci_lower < mean < ci_upper
        
        # Check CI width is reasonable
        ci_width = ci_upper - ci_lower
        assert 0 < ci_width < 10
    
    def test_confidence_interval_single_sample(self):
        """Test CI calculation with single sample."""
        sampler = UniformSampler()
        
        samples = [5.0]
        ci_lower, ci_upper = sampler.calculate_confidence_interval(samples)
        
        # With single sample, CI should be the value itself
        assert ci_lower == 5.0
        assert ci_upper == 5.0
    
    def test_confidence_interval_levels(self):
        """Test different confidence levels."""
        sampler = UniformSampler()
        samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        ci_90 = sampler.calculate_confidence_interval(samples, 0.90)
        ci_95 = sampler.calculate_confidence_interval(samples, 0.95)
        ci_99 = sampler.calculate_confidence_interval(samples, 0.99)
        
        # Higher confidence levels should have wider intervals
        width_90 = ci_90[1] - ci_90[0]
        width_95 = ci_95[1] - ci_95[0]
        width_99 = ci_99[1] - ci_99[0]
        
        assert width_90 < width_95 < width_99


class TestSamplerIntegration:
    """Integration tests for sampling system."""
    
    def test_sampling_preserves_frame_order(self):
        """Test that sampling preserves frame order."""
        frames = [np.ones((10, 10, 3), dtype=np.uint8) * i for i in range(100)]
        
        for strategy in [SamplingStrategy.UNIFORM, SamplingStrategy.ADAPTIVE]:
            sampler = create_sampler(strategy, min_frames_threshold=10)
            result = sampler.sample(frames)
            
            # Check indices are in ascending order
            indices = result.sampled_indices
            assert indices == sorted(indices)
    
    def test_sampling_includes_endpoints(self):
        """Test that all strategies include first and last frames."""
        frames = [np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8) 
                  for _ in range(50)]
        
        strategies = [
            SamplingStrategy.UNIFORM,
            SamplingStrategy.ADAPTIVE,
            SamplingStrategy.PROGRESSIVE,
            SamplingStrategy.SCENE_AWARE,
        ]
        
        for strategy in strategies:
            sampler = create_sampler(strategy, min_frames_threshold=10)
            result = sampler.sample(frames)
            
            if not result.is_full_sampling():
                assert 0 in result.sampled_indices, f"{strategy} missing first frame"
                assert 49 in result.sampled_indices, f"{strategy} missing last frame"
    
    def test_sampling_respects_threshold(self):
        """Test that sampling respects minimum frame threshold."""
        # Create frames below threshold
        frames = [np.zeros((10, 10, 3)) for _ in range(15)]
        
        for strategy in [SamplingStrategy.UNIFORM, SamplingStrategy.ADAPTIVE]:
            sampler = create_sampler(strategy, min_frames_threshold=20)
            result = sampler.sample(frames)
            
            # Should not sample when below threshold
            assert result.is_full_sampling()
            assert result.num_sampled == 15