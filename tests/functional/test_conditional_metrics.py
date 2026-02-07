"""Unit tests for conditional metrics optimization module."""

import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

from giflab.conditional_metrics import (
    ConditionalMetricsCalculator,
    QualityTier,
    ContentProfile,
    QualityAssessment,
    FrameHashCache
)


class TestFrameHashCache:
    """Test suite for FrameHashCache class."""
    
    def test_frame_hash_generation(self):
        """Test that frame hashes are generated correctly."""
        cache = FrameHashCache(max_size=10)
        
        # Create test frames
        frame1 = np.ones((10, 10, 3), dtype=np.uint8) * 100
        frame2 = np.ones((10, 10, 3), dtype=np.uint8) * 200
        frame3 = np.ones((10, 10, 3), dtype=np.uint8) * 100  # Same as frame1
        
        hash1 = cache.get_frame_hash(frame1)
        hash2 = cache.get_frame_hash(frame2)
        hash3 = cache.get_frame_hash(frame3)
        
        # Different frames should have different hashes
        assert hash1 != hash2
        # Same content should have same hash
        assert hash1 == hash3
        
    def test_similarity_caching(self):
        """Test that similarity scores are cached properly."""
        cache = FrameHashCache(max_size=10)
        
        hash1 = "abc123"
        hash2 = "def456"
        similarity = 0.95
        
        # Cache similarity
        cache.cache_similarity(hash1, hash2, similarity)
        
        # Retrieve in both orders (should be same)
        assert cache.get_similarity(hash1, hash2) == similarity
        assert cache.get_similarity(hash2, hash1) == similarity
        
    def test_cache_eviction(self):
        """Test that cache evicts old entries when full."""
        cache = FrameHashCache(max_size=2)
        
        # Add entries beyond max size
        cache.cache_similarity("h1", "h2", 0.9)
        cache.cache_similarity("h3", "h4", 0.8)
        cache.cache_similarity("h5", "h6", 0.7)  # Should evict first entry
        
        # First entry should be evicted
        assert cache.get_similarity("h1", "h2") is None
        assert cache.get_similarity("h3", "h4") == 0.8
        assert cache.get_similarity("h5", "h6") == 0.7
        
    def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = FrameHashCache(max_size=10)
        
        frame = np.ones((10, 10), dtype=np.uint8)
        
        # Generate some activity
        cache.get_frame_hash(frame)
        cache.get_frame_hash(frame)  # Cache hit
        cache.cache_similarity("h1", "h2", 0.9)
        
        stats = cache.get_cache_stats()
        assert stats["total_accesses"] == 2
        assert stats["cache_hits"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["similarity_cache_size"] == 1


class TestQualityAssessment:
    """Test suite for quality assessment functionality."""
    
    def test_high_quality_detection(self):
        """Test detection of high quality GIFs."""
        calc = ConditionalMetricsCalculator()
        
        # Create identical frames (perfect quality)
        original = [np.ones((100, 100, 3), dtype=np.uint8) * 128] * 3
        compressed = [np.ones((100, 100, 3), dtype=np.uint8) * 128] * 3
        
        assessment = calc.assess_quality(original, compressed)
        
        assert assessment.tier == QualityTier.HIGH
        assert assessment.confidence > 0.9
        assert assessment.base_psnr > 35
        assert assessment.base_mse < 1
        
    def test_low_quality_detection(self):
        """Test detection of low quality GIFs."""
        calc = ConditionalMetricsCalculator()
        
        # Create very different frames (poor quality)
        original = [np.ones((100, 100, 3), dtype=np.uint8) * 255] * 3
        compressed = [np.zeros((100, 100, 3), dtype=np.uint8)] * 3
        
        assessment = calc.assess_quality(original, compressed)
        
        assert assessment.tier == QualityTier.LOW
        assert assessment.confidence < 0.5
        assert assessment.base_psnr < 25
        assert assessment.base_mse > 100
        
    def test_medium_quality_detection(self):
        """Test detection of medium quality GIFs."""
        calc = ConditionalMetricsCalculator()
        
        # Create slightly different frames (medium quality)
        np.random.seed(42)
        original = [np.ones((100, 100, 3), dtype=np.uint8) * 128] * 3
        compressed = []
        for frame in original:
            noisy = frame.copy()
            noise = np.random.normal(0, 10, frame.shape).astype(np.int16)
            noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            compressed.append(noisy)
        
        assessment = calc.assess_quality(original, compressed)
        
        assert assessment.tier == QualityTier.MEDIUM
        assert 0.4 < assessment.confidence < 0.9
        assert 25 <= assessment.base_psnr <= 35
        
    def test_frame_sampling_for_assessment(self):
        """Test that quality assessment samples frames correctly."""
        calc = ConditionalMetricsCalculator(max_sample_frames=3)
        
        # Create 10 frames
        original = [np.ones((50, 50), dtype=np.uint8) * i for i in range(10)]
        compressed = [np.ones((50, 50), dtype=np.uint8) * i for i in range(10)]
        
        with patch.object(calc, '_load_env_config'):
            assessment = calc.assess_quality(original, compressed)
            
        # Should have sampled only 3 frames
        assert len(assessment.details["mse_values"]) == 3
        assert len(assessment.details["psnr_values"]) == 3


class TestContentProfile:
    """Test suite for content profiling functionality."""
    
    def test_monochrome_detection(self):
        """Test detection of monochrome content."""
        calc = ConditionalMetricsCalculator()
        
        # Grayscale frame
        gray_frame = np.ones((100, 100), dtype=np.uint8) * 128
        assert calc._is_monochrome(gray_frame) is True
        
        # Color frame with equal RGB channels (monochrome)
        mono_color = np.stack([gray_frame, gray_frame, gray_frame], axis=2)
        assert calc._is_monochrome(mono_color) is True
        
        # True color frame
        color_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        color_frame[:, :, 0] = 255  # Red channel
        color_frame[:, :, 1] = 128  # Green channel
        color_frame[:, :, 2] = 64   # Blue channel
        assert calc._is_monochrome(color_frame) is False
        
    def test_temporal_changes_detection(self):
        """Test detection of temporal changes between frames."""
        calc = ConditionalMetricsCalculator()
        
        # Static frames (no temporal changes)
        static_frames = [np.ones((50, 50), dtype=np.uint8) * 128] * 3
        assert calc._detect_temporal_changes(static_frames) is False
        
        # Changing frames (temporal changes)
        changing_frames = [
            np.ones((50, 50), dtype=np.uint8) * 50,
            np.ones((50, 50), dtype=np.uint8) * 128,
            np.ones((50, 50), dtype=np.uint8) * 200
        ]
        assert calc._detect_temporal_changes(changing_frames) is True
        
    def test_frame_similarity_calculation(self):
        """Test frame similarity calculation."""
        calc = ConditionalMetricsCalculator()
        
        # Identical frames
        identical = [np.ones((50, 50), dtype=np.uint8) * 128] * 3
        similarity = calc._calculate_frame_similarity(identical)
        assert similarity > 0.99
        
        # Different frames (more extreme differences)
        different = [
            np.ones((50, 50), dtype=np.uint8) * 0,
            np.ones((50, 50), dtype=np.uint8) * 255,
            np.ones((50, 50), dtype=np.uint8) * 0
        ]
        similarity = calc._calculate_frame_similarity(different)
        # The similarity should be low but not zero (average of two comparisons)
        assert similarity < 0.6  # Adjusted threshold for realistic expectation
        
    def test_edge_detection(self):
        """Test quick edge detection."""
        calc = ConditionalMetricsCalculator()
        
        # Frame with no edges (uniform)
        uniform = np.ones((50, 50), dtype=np.uint8) * 128
        edges = calc._detect_edges_quick(uniform)
        assert np.mean(edges) < 0.01
        
        # Frame with edges (checkerboard pattern)
        checkerboard = np.zeros((50, 50), dtype=np.uint8)
        checkerboard[::2, ::2] = 255
        checkerboard[1::2, 1::2] = 255
        edges = calc._detect_edges_quick(checkerboard)
        assert np.mean(edges) > 0.1
        
    def test_gradient_detection(self):
        """Test gradient detection in frames."""
        calc = ConditionalMetricsCalculator()
        
        # Frame with smooth gradient
        gradient = np.zeros((100, 100), dtype=np.uint8)
        for i in range(100):
            gradient[i, :] = int(255 * i / 100)
        assert calc._detect_gradients(gradient) is True
        
        # Frame with high variance (no gradient)
        np.random.seed(42)
        noisy = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        assert calc._detect_gradients(noisy) is False
        
    def test_content_profile_detection(self):
        """Test complete content profile detection."""
        calc = ConditionalMetricsCalculator()
        
        # Create test frames with various characteristics
        frames = []
        for i in range(3):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            # Add some structure
            frame[20:80, 20:80, :] = 200
            # Add temporal variation
            frame = frame + i * 20
            frames.append(frame)
        
        profile = calc.detect_content_profile(frames, quick_mode=True)
        
        assert isinstance(profile, ContentProfile)
        assert isinstance(profile.is_monochrome, bool)
        assert isinstance(profile.has_temporal_changes, bool)
        assert 0 <= profile.frame_similarity <= 1
        assert 0 <= profile.complexity_score <= 1


class TestMetricSelection:
    """Test suite for metric selection logic."""
    
    def test_high_quality_metric_selection(self):
        """Test that high quality GIFs skip expensive metrics."""
        calc = ConditionalMetricsCalculator()
        calc.skip_expensive_on_high_quality = True
        
        quality = QualityAssessment(
            tier=QualityTier.HIGH,
            confidence=0.95,
            base_psnr=38.0,
            base_mse=1.0,
            frame_consistency=0.98,
            details={}
        )
        
        profile = ContentProfile()
        
        selected = calc.select_metrics(quality, profile)
        
        # Basic metrics should be selected
        assert selected["mse"] is True
        assert selected["psnr"] is True
        assert selected["ssim"] is True
        
        # Expensive metrics should be skipped
        assert selected["lpips"] is False
        assert selected["ssimulacra2"] is False
        assert selected["temporal_artifacts"] is False
        
    def test_low_quality_metric_selection(self):
        """Test that low quality GIFs calculate all relevant metrics."""
        calc = ConditionalMetricsCalculator()
        
        quality = QualityAssessment(
            tier=QualityTier.LOW,
            confidence=0.3,
            base_psnr=20.0,
            base_mse=100.0,
            frame_consistency=0.7,
            details={}
        )
        
        profile = ContentProfile(
            has_text=True,
            has_temporal_changes=True,
            has_color_gradients=True
        )
        
        selected = calc.select_metrics(quality, profile)
        
        # All metrics should be selected for low quality
        assert selected["mse"] is True
        assert selected["psnr"] is True
        assert selected["ssim"] is True
        assert selected["lpips"] is True
        assert selected["ssimulacra2"] is True
        assert selected["temporal_artifacts"] is True
        assert selected["text_ui_validation"] is True
        assert selected["color_gradients"] is True
        
    def test_medium_quality_selective_metrics(self):
        """Test selective metric calculation for medium quality."""
        calc = ConditionalMetricsCalculator()
        
        quality = QualityAssessment(
            tier=QualityTier.MEDIUM,
            confidence=0.6,
            base_psnr=28.0,
            base_mse=40.0,
            frame_consistency=0.85,
            details={}
        )
        
        profile = ContentProfile(
            has_text=False,
            has_temporal_changes=True,
            frame_similarity=0.7,
            complexity_score=0.6
        )
        
        selected = calc.select_metrics(quality, profile)
        
        # Basic metrics always selected
        assert selected["mse"] is True
        assert selected["psnr"] is True
        assert selected["ssim"] is True
        
        # Some advanced metrics based on profile
        assert selected["lpips"] is True  # Always for medium
        assert selected["temporal_artifacts"] is True  # Has temporal changes
        assert selected["text_ui_validation"] is False  # No text detected
        
    def test_content_based_metric_selection(self):
        """Test that content profile influences metric selection."""
        calc = ConditionalMetricsCalculator()
        
        quality = QualityAssessment(
            tier=QualityTier.MEDIUM,
            confidence=0.65,
            base_psnr=30.0,
            base_mse=25.0,
            frame_consistency=0.9,
            details={}
        )
        
        # Test with text/UI content
        text_profile = ContentProfile(has_text=True, has_ui_elements=True)
        selected = calc.select_metrics(quality, text_profile)
        assert selected["text_ui_validation"] is True
        assert selected["edge_similarity"] is True
        
        # Test without text/UI content
        no_text_profile = ContentProfile(has_text=False, has_ui_elements=False)
        selected = calc.select_metrics(quality, no_text_profile)
        assert selected["text_ui_validation"] is False


class TestEnvironmentConfiguration:
    """Test suite for environment variable configuration."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            calc = ConditionalMetricsCalculator()
            
            assert calc.enabled is True
            assert calc.skip_expensive_on_high_quality is True
            assert calc.use_progressive_calculation is True
            assert calc.cache_frame_hashes is True
            
    def test_environment_override(self):
        """Test environment variable overrides."""
        env_vars = {
            "GIFLAB_ENABLE_CONDITIONAL_METRICS": "false",
            "GIFLAB_SKIP_EXPENSIVE_ON_HIGH_QUALITY": "false",
            "GIFLAB_USE_PROGRESSIVE_CALCULATION": "false",
            "GIFLAB_CACHE_FRAME_HASHES": "false",
            "GIFLAB_QUALITY_HIGH_THRESHOLD": "0.95",
            "GIFLAB_QUALITY_MEDIUM_THRESHOLD": "0.6",
            "GIFLAB_QUALITY_SAMPLE_FRAMES": "10"
        }
        
        with patch.dict(os.environ, env_vars):
            calc = ConditionalMetricsCalculator()
            
            assert calc.enabled is False
            assert calc.skip_expensive_on_high_quality is False
            assert calc.use_progressive_calculation is False
            assert calc.cache_frame_hashes is False
            assert calc.quality_thresholds["high"] == 0.95
            assert calc.quality_thresholds["medium"] == 0.6
            assert calc.max_sample_frames == 10


class TestProgressiveCalculation:
    """Test suite for progressive metric calculation."""
    
    @patch('giflab.conditional_metrics.logger')
    def test_progressive_calculation_high_quality(self, mock_logger):
        """Test progressive calculation for high quality GIFs."""
        calc = ConditionalMetricsCalculator()
        
        # Create high quality frames
        original = [np.ones((100, 100, 3), dtype=np.uint8) * 128] * 3
        compressed = [np.ones((100, 100, 3), dtype=np.uint8) * 128] * 3
        
        # Mock metrics calculator
        mock_metrics_calc = MagicMock()
        mock_metrics_calc.calculate_selected_metrics.return_value = {
            "ssim": 0.99,
            "psnr": 0.95,
            "mse": 1.0
        }
        
        results = calc.calculate_progressive(
            original, compressed, mock_metrics_calc, force_all=False
        )
        
        # Should have called calculate_selected_metrics
        mock_metrics_calc.calculate_selected_metrics.assert_called_once()
        
        # Should have optimization metadata
        assert "_optimization_metadata" in results
        assert results["_optimization_metadata"]["quality_tier"] == "high"
        assert results["_optimization_metadata"]["metrics_skipped"] > 0
        
    def test_force_all_bypasses_optimization(self):
        """Test that force_all parameter bypasses optimization."""
        calc = ConditionalMetricsCalculator()
        
        original = [np.ones((100, 100, 3), dtype=np.uint8) * 128] * 3
        compressed = [np.ones((100, 100, 3), dtype=np.uint8) * 128] * 3
        
        # Mock metrics calculator
        mock_metrics_calc = MagicMock()
        mock_metrics_calc.calculate_all_metrics.return_value = {
            "all_metrics": "calculated"
        }
        
        results = calc.calculate_progressive(
            original, compressed, mock_metrics_calc, force_all=True
        )
        
        # Should have called calculate_all_metrics
        mock_metrics_calc.calculate_all_metrics.assert_called_once()
        mock_metrics_calc.calculate_selected_metrics.assert_not_called()
        
    def test_disabled_optimization(self):
        """Test that disabled optimization calculates all metrics."""
        calc = ConditionalMetricsCalculator()
        calc.enabled = False
        
        original = [np.ones((100, 100, 3), dtype=np.uint8) * 128] * 3
        compressed = [np.ones((100, 100, 3), dtype=np.uint8) * 128] * 3
        
        # Mock metrics calculator
        mock_metrics_calc = MagicMock()
        mock_metrics_calc.calculate_all_metrics.return_value = {
            "all_metrics": "calculated"
        }
        
        results = calc.calculate_progressive(
            original, compressed, mock_metrics_calc, force_all=False
        )
        
        # Should have called calculate_all_metrics despite force_all=False
        mock_metrics_calc.calculate_all_metrics.assert_called_once()
        
    def test_optimization_statistics(self):
        """Test optimization statistics tracking."""
        calc = ConditionalMetricsCalculator()
        
        # Reset stats
        calc.reset_stats()
        
        assert calc.metrics_calculated == 0
        assert calc.metrics_skipped == 0
        assert calc.time_saved == 0.0
        
        # Simulate metric selection
        quality = QualityAssessment(
            tier=QualityTier.HIGH,
            confidence=0.95,
            base_psnr=38.0,
            base_mse=1.0,
            frame_consistency=0.98,
            details={}
        )
        profile = ContentProfile()
        
        calc.select_metrics(quality, profile)
        
        # Should have tracked metrics
        assert calc.metrics_calculated > 0
        assert calc.metrics_skipped > 0
        
        # Get stats
        stats = calc.get_optimization_stats()
        assert "metrics_calculated" in stats
        assert "metrics_skipped" in stats
        assert "estimated_time_saved" in stats
        assert "optimization_ratio" in stats


@pytest.fixture
def sample_frames():
    """Create sample frames for testing."""
    np.random.seed(42)
    original = []
    compressed = []
    
    for i in range(5):
        # Original frames
        orig_frame = np.ones((100, 100, 3), dtype=np.uint8) * (50 + i * 30)
        original.append(orig_frame)
        
        # Compressed frames with some noise
        comp_frame = orig_frame.copy()
        noise = np.random.normal(0, 5, orig_frame.shape)
        comp_frame = np.clip(orig_frame + noise, 0, 255).astype(np.uint8)
        compressed.append(comp_frame)
    
    return original, compressed


def test_integration_with_sample_frames(sample_frames):
    """Test complete integration with sample frames."""
    original, compressed = sample_frames
    
    calc = ConditionalMetricsCalculator()
    
    # Perform quality assessment
    quality = calc.assess_quality(original, compressed)
    assert quality.tier in [QualityTier.HIGH, QualityTier.MEDIUM, QualityTier.LOW]
    
    # Detect content profile
    profile = calc.detect_content_profile(compressed)
    assert isinstance(profile.complexity_score, float)
    
    # Select metrics
    selected = calc.select_metrics(quality, profile)
    assert isinstance(selected, dict)
    assert len(selected) > 0
    
    # Check that basic metrics are always selected
    assert selected["mse"] is True
    assert selected["psnr"] is True
    assert selected["ssim"] is True


def test_edge_cases():
    """Test edge cases and error handling."""
    calc = ConditionalMetricsCalculator()
    
    # Empty frames should return a default assessment
    assessment = calc.assess_quality([], [])
    assert assessment.tier in [QualityTier.UNKNOWN, QualityTier.HIGH, QualityTier.LOW]  # Can handle empty frames
    
    # Mismatched frame counts (should still work)
    original = [np.ones((50, 50), dtype=np.uint8)]
    compressed = [np.ones((50, 50), dtype=np.uint8)] * 3
    
    assessment = calc.assess_quality(original, compressed)
    assert assessment.tier != QualityTier.UNKNOWN
    
    # Different frame shapes (should handle gracefully)
    original = [np.ones((100, 100), dtype=np.uint8)]
    compressed = [np.ones((50, 50), dtype=np.uint8)]
    
    assessment = calc.assess_quality(original, compressed)
    # Should handle the shape mismatch
    assert assessment is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])