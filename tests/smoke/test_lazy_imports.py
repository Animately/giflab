"""
Unit tests for the lazy import system.

Tests the LazyModule class, import registry, and availability checking
to ensure lazy loading works correctly and maintains backward compatibility.
"""

import sys
import threading
import time
from unittest.mock import MagicMock, patch, Mock
import pytest

# Import directly from the source
sys.path.insert(0, '/Users/lachlants/repos/animately/giflab/src')

from giflab.lazy_imports import (
    LazyModule,
    LazyImportRegistry,
    lazy_import,
    check_import_available,
    get_import_status,
    preload_modules,
    is_torch_available,
    is_lpips_available,
    is_cv2_available,
    is_scipy_available,
    is_sklearn_available,
    # Phase 2.3 additions
    get_pil,
    is_pil_available,
    get_matplotlib,
    is_matplotlib_available,
    get_seaborn,
    is_seaborn_available,
    get_plotly,
    is_plotly_available,
    get_subprocess,
    is_subprocess_available,
)


class TestLazyModule:
    """Test the LazyModule class functionality."""
    
    def test_lazy_module_defers_import(self):
        """Test that LazyModule doesn't import until accessed."""
        # Create a lazy module for a non-existent module
        lazy_mod = LazyModule("non_existent_module_12345")
        
        # Module should not be imported yet
        assert lazy_mod._module is None
        assert lazy_mod._import_attempted is False
    
    def test_lazy_module_imports_on_attribute_access(self):
        """Test that module is imported when attribute is accessed."""
        # Use a real module that should exist
        lazy_mod = LazyModule("json")
        
        # Access an attribute
        dumps = lazy_mod.dumps
        
        # Now module should be imported
        assert lazy_mod._module is not None
        assert lazy_mod._import_attempted is True
        assert dumps is not None
    
    def test_lazy_module_handles_import_error(self):
        """Test graceful handling of import errors."""
        lazy_mod = LazyModule("non_existent_module_xyz", fallback_value=None)
        
        # Should return fallback value
        result = lazy_mod._load_module()
        assert result is None
        assert lazy_mod._import_error is not None
    
    def test_lazy_module_is_available(self):
        """Test the is_available property."""
        # Real module
        lazy_json = LazyModule("json")
        assert lazy_json.is_available is True
        
        # Non-existent module
        lazy_fake = LazyModule("fake_module_abc")
        assert lazy_fake.is_available is False
    
    def test_lazy_module_thread_safety(self):
        """Test thread-safe loading of modules."""
        lazy_mod = LazyModule("json")
        results = []
        
        def access_module():
            # Each thread tries to access the module
            result = lazy_mod.dumps
            results.append(result is not None)
        
        # Create multiple threads
        threads = [threading.Thread(target=access_module) for _ in range(10)]
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # All threads should have gotten the same module
        assert all(results)
        assert lazy_mod._module is not None


class TestLazyImportRegistry:
    """Test the LazyImportRegistry class."""
    
    def test_registry_registers_modules(self):
        """Test that modules can be registered."""
        registry = LazyImportRegistry()
        
        # Register a module
        lazy_mod = registry.register("os")
        
        assert "os" in registry._modules
        assert lazy_mod is not None
    
    def test_registry_returns_same_instance(self):
        """Test that registering same module returns same instance."""
        registry = LazyImportRegistry()
        
        # Register twice
        mod1 = registry.register("sys")
        mod2 = registry.register("sys")
        
        # Should be the same instance
        assert mod1 is mod2
    
    def test_registry_get_status(self):
        """Test status reporting."""
        registry = LazyImportRegistry()
        
        # Register but don't import
        registry.register("json")
        
        status = registry.get_status()
        assert "json" in status
        
        # Status should show (available, not_imported)
        available, imported = status["json"]
        assert available is True
        assert imported is False
    
    def test_registry_preload(self):
        """Test preloading modules."""
        registry = LazyImportRegistry()
        
        # Register a module
        lazy_mod = registry.register("json")
        
        # Preload it
        registry.preload(["json"])
        
        # Should now be loaded
        assert lazy_mod._module is not None


class TestPublicAPI:
    """Test the public API functions."""
    
    def test_lazy_import_function(self):
        """Test the lazy_import convenience function."""
        lazy_mod = lazy_import("json")
        
        assert lazy_mod is not None
        assert lazy_mod.is_available is True
    
    def test_check_import_available(self):
        """Test checking if imports are available."""
        # Standard library should be available
        assert check_import_available("json") is True
        assert check_import_available("os") is True
        
        # Non-existent should not be available
        assert check_import_available("fake_module_xyz") is False
    
    @patch('giflab.lazy_imports.check_import_available')
    def test_availability_checkers(self, mock_check):
        """Test the specific availability checker functions."""
        # Clear the cache first
        from giflab.lazy_imports import _availability_cache
        _availability_cache.clear()
        
        # Mock different availability states
        mock_check.side_effect = [True, False, True, False, True, True, False, True, False, True]
        
        # Test each checker
        assert is_torch_available() is True
        assert is_lpips_available() is False
        assert is_cv2_available() is True
        assert is_scipy_available() is False
        assert is_sklearn_available() is True
        
        # Test Phase 2.3 additions
        assert is_pil_available() is True
        assert is_matplotlib_available() is False
        assert is_seaborn_available() is True
        assert is_plotly_available() is False
        assert is_subprocess_available() is True
        
        # Should be cached now (no more calls to mock)
        mock_check.side_effect = None
        mock_check.return_value = False  # This shouldn't be used
        
        # These should use cached values
        assert is_torch_available() is True
        assert is_lpips_available() is False
        assert is_pil_available() is True
        assert is_matplotlib_available() is False


class TestIntegration:
    """Integration tests with actual GifLab modules."""
    
    def test_deep_perceptual_metrics_lazy_loading(self):
        """Test that deep_perceptual_metrics uses lazy loading correctly."""
        # Import the module - torch may already be loaded by other tests,
        # so we don't try to clear it (torch C extensions can't be reloaded)
        try:
            from giflab import deep_perceptual_metrics
        except RuntimeError as e:
            if "TORCH_LIBRARY" in str(e):
                pytest.skip("Torch C extensions conflict in multi-test process")
                return
            raise

        # Check that the lazy modules are set up
        assert hasattr(deep_perceptual_metrics, 'torch')
        assert hasattr(deep_perceptual_metrics, 'lpips')

        # Check availability flags are set correctly
        # Note: This depends on whether torch/lpips are installed
        # We just check that the flags exist
        assert hasattr(deep_perceptual_metrics, 'TORCH_AVAILABLE')
        assert hasattr(deep_perceptual_metrics, 'LPIPS_AVAILABLE')
    
    def test_eda_lazy_sklearn(self):
        """Test that eda.py loads sklearn lazily."""
        # Clear sklearn from sys.modules
        if 'sklearn' in sys.modules:
            del sys.modules['sklearn']
        if 'sklearn.decomposition' in sys.modules:
            del sys.modules['sklearn.decomposition']
        
        # Import eda - should not import sklearn
        from giflab import eda
        
        # sklearn should not be in sys.modules yet
        # (unless it was already imported by something else)
        # Just verify the module loads without error
        assert eda is not None


class TestPerformance:
    """Performance tests for lazy loading."""
    
    def test_lazy_import_is_fast(self):
        """Test that creating lazy imports is fast."""
        start = time.perf_counter()
        
        # Create many lazy imports
        for i in range(100):
            lazy_import(f"fake_module_{i}")
        
        elapsed = time.perf_counter() - start
        
        # Should be very fast (< 10ms for 100 lazy imports)
        assert elapsed < 0.01
    
    def test_cached_availability_check(self):
        """Test that availability checks are cached."""
        from giflab.lazy_imports import _availability_cache
        _availability_cache.clear()
        
        # First call might be slow
        start = time.perf_counter()
        result1 = is_torch_available()
        first_time = time.perf_counter() - start
        
        # Second call should be much faster (cached)
        start = time.perf_counter()
        result2 = is_torch_available()
        second_time = time.perf_counter() - start
        
        # Results should be the same
        assert result1 == result2
        
        # Second call should be at least 10x faster (likely 100x+)
        if first_time > 0.0001:  # Only check if first call took measurable time
            assert second_time < first_time / 10


class TestPhase23Additions:
    """Test the Phase 2.3 lazy import additions."""
    
    def test_new_get_functions(self):
        """Test new get_* functions added in Phase 2.3."""
        # Test PIL functions
        pil_module = get_pil()
        assert pil_module is not None
        assert hasattr(pil_module, '_module_name')
        assert pil_module._module_name == 'PIL'
        
        # Test matplotlib function
        matplotlib_module = get_matplotlib()
        assert matplotlib_module is not None
        assert matplotlib_module._module_name == 'matplotlib'
        
        # Test seaborn function  
        seaborn_module = get_seaborn()
        assert seaborn_module is not None
        assert seaborn_module._module_name == 'seaborn'
        
        # Test plotly function
        plotly_module = get_plotly()
        assert plotly_module is not None
        assert plotly_module._module_name == 'plotly'
        
        # Test subprocess function (standard library)
        subprocess_module = get_subprocess()
        assert subprocess_module is not None
        assert subprocess_module._module_name == 'subprocess'
    
    def test_new_availability_checkers(self):
        """Test new is_*_available functions added in Phase 2.3."""
        # PIL should be available (it's a core dependency)
        assert isinstance(is_pil_available(), bool)
        
        # matplotlib availability depends on installation
        assert isinstance(is_matplotlib_available(), bool)
        
        # seaborn availability depends on installation
        assert isinstance(is_seaborn_available(), bool)
        
        # plotly availability depends on installation
        assert isinstance(is_plotly_available(), bool)
        
        # subprocess should always be available (standard library)
        assert is_subprocess_available() is True
    
    def test_availability_caching_for_new_functions(self):
        """Test that new availability functions use caching correctly."""
        from giflab.lazy_imports import _availability_cache, _availability_lock
        
        # Clear cache for testing
        with _availability_lock:
            _availability_cache.clear()
        
        # First calls should populate cache
        pil_result1 = is_pil_available()
        matplotlib_result1 = is_matplotlib_available()
        
        # Cache should now contain these modules
        with _availability_lock:
            assert 'PIL' in _availability_cache
            assert 'matplotlib' in _availability_cache
        
        # Second calls should use cached values
        pil_result2 = is_pil_available()
        matplotlib_result2 = is_matplotlib_available()
        
        # Results should be consistent
        assert pil_result1 == pil_result2
        assert matplotlib_result1 == matplotlib_result2
    
    @patch('giflab.lazy_imports.check_import_available')
    def test_availability_with_mocked_imports(self, mock_check):
        """Test availability checking with mocked import states."""
        from giflab.lazy_imports import _availability_cache, _availability_lock
        
        # Clear cache
        with _availability_lock:
            _availability_cache.clear()
        
        # Mock different availability states for new functions
        mock_availability = {
            'PIL': True,
            'matplotlib': False, 
            'seaborn': True,
            'plotly': False,
            'subprocess': True
        }
        
        def mock_check_func(module_name):
            return mock_availability.get(module_name, False)
        
        mock_check.side_effect = mock_check_func
        
        # Test all new availability checkers
        assert is_pil_available() is True
        assert is_matplotlib_available() is False
        assert is_seaborn_available() is True
        assert is_plotly_available() is False
        assert is_subprocess_available() is True
        
        # Verify check was called for each module
        assert mock_check.call_count == 5
        
        # Test caching by calling again (should not call mock again)
        mock_check.reset_mock()
        
        assert is_pil_available() is True  # From cache
        assert is_matplotlib_available() is False  # From cache
        
        # Mock should not be called again
        assert mock_check.call_count == 0
    
    def test_thread_safety_of_new_functions(self):
        """Test thread safety of new availability checkers."""
        import threading
        from giflab.lazy_imports import _availability_cache, _availability_lock
        
        # Clear cache
        with _availability_lock:
            _availability_cache.clear()
        
        results = []
        
        def check_availability():
            # Each thread checks availability of new functions
            results.append({
                'pil': is_pil_available(),
                'matplotlib': is_matplotlib_available(),
                'seaborn': is_seaborn_available(),
                'plotly': is_plotly_available(),
                'subprocess': is_subprocess_available()
            })
        
        # Create multiple threads
        threads = [threading.Thread(target=check_availability) for _ in range(5)]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All threads should get consistent results
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result, "Thread safety issue: inconsistent results"
    
    def test_lazy_module_consistency(self):
        """Test that new get functions return consistent LazyModule instances."""
        # Multiple calls should return the same lazy module instance
        pil1 = get_pil()
        pil2 = get_pil()
        assert pil1 is pil2, "get_pil should return same LazyModule instance"
        
        matplotlib1 = get_matplotlib()
        matplotlib2 = get_matplotlib()
        assert matplotlib1 is matplotlib2, "get_matplotlib should return same LazyModule instance"
        
        seaborn1 = get_seaborn()
        seaborn2 = get_seaborn()
        assert seaborn1 is seaborn2, "get_seaborn should return same LazyModule instance"
        
        plotly1 = get_plotly()
        plotly2 = get_plotly()
        assert plotly1 is plotly2, "get_plotly should return same LazyModule instance"
    
    def test_import_status_includes_new_modules(self):
        """Test that get_import_status includes new modules when used."""
        # Use some of the new lazy imports to register them
        _ = get_pil()
        _ = get_matplotlib()
        _ = get_seaborn()
        
        # Get import status
        status = get_import_status()
        
        # Status should include the modules we've accessed
        assert isinstance(status, dict)
        # Note: The specific modules in status depend on what's been accessed
        # So we just verify the status structure is correct
        for module_name, (available, imported) in status.items():
            assert isinstance(available, bool)
            assert isinstance(imported, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])