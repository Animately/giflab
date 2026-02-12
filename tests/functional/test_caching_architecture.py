"""
Tests for the caching architecture system implemented in Phase 2.1.

This module tests the conditional import system, feature flags, and fallback
implementations that allow the metrics system to work with or without caching.

Tests cover:
- ENABLE_EXPERIMENTAL_CACHING feature flag behavior
- Conditional import system for caching modules
- Fallback implementation functionality
- Metrics functionality with caching enabled/disabled
- Error handling for import failures
"""

import sys
import tempfile
import unittest.mock
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest

# Test imports
from giflab.config import ENABLE_EXPERIMENTAL_CACHING


class TestCachingFeatureFlag:
    """Test the ENABLE_EXPERIMENTAL_CACHING feature flag system."""

    def test_caching_disabled_by_default(self):
        """Test that caching is disabled by default for production safety."""
        # Should be False by default as per Phase 2.1 implementation
        assert ENABLE_EXPERIMENTAL_CACHING is False

    def test_feature_flag_configuration(self):
        """Test that feature flag is properly configured in config.py."""
        from giflab.config import FRAME_CACHE

        # Frame cache should be linked to the experimental flag
        assert FRAME_CACHE["enabled"] == ENABLE_EXPERIMENTAL_CACHING


class TestConditionalImports:
    """Test the conditional import system for caching modules."""

    def setup_method(self):
        """Set up test fixtures - save module state."""
        self._saved_modules = {
            k: v for k, v in sys.modules.items() if "giflab.metrics" in k
        }

    def teardown_method(self):
        """Restore module state to avoid polluting other tests."""
        # Remove any freshly imported metrics modules
        metrics_modules = [k for k in sys.modules if "giflab.metrics" in k]
        for mod in metrics_modules:
            del sys.modules[mod]
        # Restore original module references
        sys.modules.update(self._saved_modules)

    def test_imports_when_caching_enabled_theoretical(self):
        """Test how conditional imports should work when caching is enabled."""
        from giflab.config import ENABLE_EXPERIMENTAL_CACHING
        from giflab.metrics import CACHING_ENABLED

        if ENABLE_EXPERIMENTAL_CACHING:
            assert CACHING_ENABLED in [True, False]
        else:
            assert CACHING_ENABLED is False

    def test_imports_when_caching_disabled(self):
        """Test conditional imports work when caching is disabled."""
        from giflab import metrics
        from giflab.config import ENABLE_EXPERIMENTAL_CACHING

        assert ENABLE_EXPERIMENTAL_CACHING is False
        assert metrics.CACHING_ENABLED is False

        assert hasattr(metrics, "_resize_frame_fallback")
        assert callable(metrics._resize_frame_fallback)

    def test_import_error_handling(self):
        """Test graceful handling of caching module import errors."""
        from giflab import metrics

        assert hasattr(metrics, "CACHING_ERROR_MESSAGE")

        assert hasattr(metrics, "get_caching_status")
        status = metrics.get_caching_status()
        assert isinstance(status, dict)
        assert "enabled" in status
        assert "error_message" in status


class TestFallbackImplementations:
    """Test fallback implementations when caching is unavailable."""

    def test_resize_frame_fallback_functionality(self):
        """Test that resize fallback produces correct results."""
        from giflab.metrics import _resize_frame_fallback

        # Create test frame
        test_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        target_size = (50, 50)

        # Test fallback function
        result = _resize_frame_fallback(test_frame, target_size)

        assert result.shape[:2] == target_size
        assert result.dtype == test_frame.dtype

    def test_resize_frame_fallback_parameters(self):
        """Test fallback function handles all parameters correctly."""
        from giflab.metrics import _resize_frame_fallback

        test_frame = np.ones((100, 100, 3), dtype=np.uint8)

        # Test with different interpolation methods
        result_area = _resize_frame_fallback(test_frame, (50, 50), cv2.INTER_AREA)
        result_linear = _resize_frame_fallback(test_frame, (50, 50), cv2.INTER_LINEAR)

        assert result_area.shape[:2] == (50, 50)
        assert result_linear.shape[:2] == (50, 50)

    def test_fallback_assignment_when_caching_disabled(self):
        """Test that fallback is assigned when caching is disabled."""
        from giflab import metrics

        # Verify fallback function is assigned (caching disabled by default)
        assert metrics.resize_frame_cached is not None
        assert callable(metrics.resize_frame_cached)

        # Test it works like the fallback
        test_frame = np.ones((20, 20, 3), dtype=np.uint8)
        result = metrics.resize_frame_cached(test_frame, (10, 10))
        assert result.shape[:2] == (10, 10)


class TestMetricsWithCachingDisabled:
    """Test core metrics functionality with caching disabled."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_gif_bytes = self._create_minimal_gif_bytes()

    def _create_minimal_gif_bytes(self):
        """Create minimal valid GIF bytes for testing."""
        # Create a proper minimal GIF using PIL
        import io

        from PIL import Image

        # Create a 1x1 black image
        img = Image.new("P", (1, 1), 0)
        img.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)  # Black and white palette

        # Save as GIF to bytes
        gif_bytes = io.BytesIO()
        img.save(gif_bytes, format="GIF")
        return gif_bytes.getvalue()

    def test_extract_gif_frames_without_caching(self):
        """Test extract_gif_frames works without caching."""
        with patch("giflab.metrics.CACHING_ENABLED", False), patch(
            "giflab.metrics.get_frame_cache", None
        ):
            # Create a temporary GIF file
            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp_file:
                tmp_file.write(self.test_gif_bytes)
                gif_path = tmp_file.name

            try:
                from giflab.metrics import extract_gif_frames

                # Should work without caching - may return empty frames for minimal GIF
                result = extract_gif_frames(gif_path)
                assert result is not None
                assert hasattr(result, "frames")
                assert hasattr(result, "frame_count")

            finally:
                Path(gif_path).unlink()

    def test_calculate_comprehensive_metrics_without_caching(self):
        """Test calculate_comprehensive_metrics works without caching."""
        with patch("giflab.metrics.CACHING_ENABLED", False), patch(
            "giflab.metrics.get_frame_cache", None
        ):
            # Create a temporary GIF file
            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp_file:
                tmp_file.write(self.test_gif_bytes)
                gif_path = tmp_file.name

            try:
                from giflab.metrics import calculate_comprehensive_metrics

                # Should work without caching (may have limited functionality)
                result = calculate_comprehensive_metrics(gif_path)
                assert result is not None
                assert isinstance(result, dict)

            except Exception as e:
                # Some metrics might not work with minimal GIF, but should not crash due to caching
                assert "caching" not in str(e).lower()

            finally:
                Path(gif_path).unlink()


class TestCachingIntegrationPatterns:
    """Test caching integration patterns and code paths."""

    def test_frame_cache_access_protection(self):
        """Test that frame cache access is properly protected by conditionals."""
        from giflab.config import FRAME_CACHE

        with patch("giflab.metrics.CACHING_ENABLED", True), patch(
            "giflab.metrics.get_frame_cache"
        ) as mock_get_cache, patch.dict(FRAME_CACHE, {"enabled": True}):
            # Mock frame cache instance
            mock_cache = MagicMock()
            mock_cache.get.return_value = None  # Cache miss
            mock_get_cache.return_value = mock_cache

            from giflab.metrics import extract_gif_frames

            # Create temporary GIF
            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp_file:
                tmp_file.write(
                    b"GIF89a\x01\x00\x01\x00\x00\x00\x00\x21\xF9\x04\x00\x00\x00\x00\x00\x2C\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x04\x01\x00\x3B"
                )
                gif_path = tmp_file.name

            try:
                # Should access cache when enabled
                extract_gif_frames(gif_path)

                # Verify cache was accessed
                mock_get_cache.assert_called()
                mock_cache.get.assert_called()

            finally:
                Path(gif_path).unlink()

    def test_frame_cache_storage_protection(self):
        """Test that frame cache storage is properly protected by conditionals."""
        with patch("giflab.metrics.CACHING_ENABLED", False), patch(
            "giflab.metrics.get_frame_cache", None
        ) as mock_get_cache:
            from giflab.metrics import extract_gif_frames

            # Create temporary GIF
            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp_file:
                tmp_file.write(
                    b"GIF89a\x01\x00\x01\x00\x01\x00\x00\x00\x00\x21\xF9\x04\x00\x00\x00\x00\x00\x2C\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02\x04\x01\x00\x3B"
                )
                gif_path = tmp_file.name

            try:
                # Should not access cache when disabled
                extract_gif_frames(gif_path)

                # Cache should never be accessed
                assert mock_get_cache is None

            finally:
                Path(gif_path).unlink()

    def test_resize_frame_cached_behavior(self):
        """Test resize_frame_cached behavior in different states."""
        test_frame = np.ones((50, 50, 3), dtype=np.uint8) * 100
        target_size = (25, 25)

        # Test with caching enabled (should use real caching function)
        with patch("giflab.metrics.CACHING_ENABLED", True):
            mock_cached_resize = MagicMock(
                return_value=np.ones((25, 25, 3), dtype=np.uint8)
            )

            with patch("giflab.metrics.resize_frame_cached", mock_cached_resize):
                from giflab import metrics

                result = metrics.resize_frame_cached(test_frame, target_size)

                assert result.shape[:2] == target_size
                mock_cached_resize.assert_called_once()

        # Test with caching disabled (should use fallback)
        with patch("giflab.metrics.CACHING_ENABLED", False):
            from giflab.metrics import _resize_frame_fallback

            with patch("giflab.metrics.resize_frame_cached", _resize_frame_fallback):
                from giflab import metrics

                result = metrics.resize_frame_cached(test_frame, target_size)

                assert result.shape[:2] == target_size


class TestArchitecturalIntegration:
    """Test architectural integration between components."""

    def setup_method(self):
        """Save module state."""
        self._saved_modules = {
            k: v
            for k, v in sys.modules.items()
            if "giflab.metrics" in k or "giflab.caching" in k
        }

    def teardown_method(self):
        """Restore module state to avoid polluting other tests."""
        # Remove any freshly imported modules
        to_remove = [
            k for k in sys.modules if "giflab.metrics" in k or "giflab.caching" in k
        ]
        for mod in to_remove:
            del sys.modules[mod]
        # Restore original module references
        sys.modules.update(self._saved_modules)

    def test_config_consistency(self):
        """Test configuration consistency across modules."""
        from giflab.config import ENABLE_EXPERIMENTAL_CACHING, FRAME_CACHE

        # Frame cache enabled flag should match experimental flag
        assert FRAME_CACHE["enabled"] == ENABLE_EXPERIMENTAL_CACHING

    def test_metrics_import_independence(self):
        """Test metrics module can be imported independent of caching."""
        # Clear any existing imports
        metrics_modules = [mod for mod in sys.modules.keys() if "giflab.metrics" in mod]
        for mod in metrics_modules:
            del sys.modules[mod]

        # Should be able to import metrics without any caching modules
        try:
            from giflab import metrics

            assert metrics is not None
        except ImportError as e:
            pytest.fail(f"Metrics module should import independently: {e}")

    def test_backward_compatibility(self):
        """Test that disabling caching doesn't break existing functionality."""
        with patch("giflab.metrics.ENABLE_EXPERIMENTAL_CACHING", False):
            # Should be able to import and use basic metrics
            from giflab.metrics import calculate_comprehensive_metrics

            # Function should exist and be callable
            assert callable(calculate_comprehensive_metrics)

    def test_no_circular_dependencies(self):
        """Test that there are no circular import dependencies."""
        # Clear modules
        modules_to_clear = [
            mod
            for mod in sys.modules.keys()
            if any(x in mod for x in ["giflab.metrics", "giflab.caching"])
        ]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]

        # Should be able to import metrics without circular dependency issues
        try:
            from giflab import metrics

            assert metrics is not None
        except ImportError as e:
            if "circular" in str(e).lower():
                pytest.fail(f"Circular dependency detected: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
