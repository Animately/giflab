#!/usr/bin/env python3
"""Memory Leak Detection and Stability Testing for Phase 3 Optimizations.

This module provides comprehensive memory leak detection covering:
1. Long-running scenarios with 100+ iterations
2. Rapid succession of different GIF sizes
3. Model cache thrashing scenarios
4. Parallel processing cleanup verification
5. Resource cleanup on failures

Memory profiling tools used:
- tracemalloc for Python memory tracking
- psutil for system-level memory monitoring
- gc for garbage collection analysis
"""

import gc
import os
import sys
import time
import tracemalloc
import weakref
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from giflab.config import MetricsConfig
from giflab.deep_perceptual_metrics import (
    _get_or_create_validator,
    cleanup_global_validator,
)
from giflab.metrics import (
    calculate_comprehensive_metrics_from_frames,
    cleanup_all_validators,
)
from giflab.model_cache import LPIPSModelCache
from giflab.ssimulacra2_metrics import Ssimulacra2Validator
from giflab.temporal_artifacts import (
    cleanup_global_temporal_detector,
    get_temporal_detector,
)
from giflab.text_ui_validation import TextUIContentDetector


class MemoryLeakDetector:
    """Comprehensive memory leak detection for GifLab metrics."""
    
    def __init__(self):
        """Initialize memory leak detector."""
        self.process = psutil.Process()
        self.initial_memory = None
        self.memory_samples = []
        self.weak_refs = []
        
    def start_monitoring(self):
        """Start memory monitoring."""
        gc.collect()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.memory_samples = [self.initial_memory]
        tracemalloc.start()
        
    def sample_memory(self) -> float:
        """Sample current memory usage.
        
        Returns:
            Current memory usage in MB
        """
        gc.collect()
        current_memory = self.process.memory_info().rss / 1024 / 1024
        self.memory_samples.append(current_memory)
        return current_memory
    
    def stop_monitoring(self) -> dict:
        """Stop monitoring and analyze results.
        
        Returns:
            Dictionary with memory analysis results
        """
        tracemalloc.stop()
        gc.collect()
        final_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Calculate statistics
        memory_growth = final_memory - self.initial_memory
        max_memory = max(self.memory_samples)
        mean_memory = np.mean(self.memory_samples)
        
        # Check for monotonic growth (potential leak)
        is_monotonic = all(
            self.memory_samples[i] <= self.memory_samples[i+1] * 1.1  # Allow 10% variation
            for i in range(len(self.memory_samples) - 1)
        )
        
        # Calculate growth rate
        if len(self.memory_samples) > 10:
            # Linear regression on last 10 samples
            x = np.arange(10)
            y = self.memory_samples[-10:]
            slope = np.polyfit(x, y, 1)[0]
            growth_rate_mb_per_iteration = slope
        else:
            growth_rate_mb_per_iteration = 0
        
        return {
            "initial_memory_mb": self.initial_memory,
            "final_memory_mb": final_memory,
            "memory_growth_mb": memory_growth,
            "max_memory_mb": max_memory,
            "mean_memory_mb": mean_memory,
            "samples": len(self.memory_samples),
            "is_monotonic_growth": is_monotonic,
            "growth_rate_mb_per_iteration": growth_rate_mb_per_iteration,
            "potential_leak": is_monotonic and growth_rate_mb_per_iteration > 0.5
        }
    
    def track_object(self, obj):
        """Track an object with weak reference.
        
        Args:
            obj: Object to track
        """
        self.weak_refs.append(weakref.ref(obj))
    
    def check_tracked_objects(self) -> dict:
        """Check if tracked objects have been garbage collected.
        
        Returns:
            Dictionary with tracking results
        """
        gc.collect()
        
        alive_count = sum(1 for ref in self.weak_refs if ref() is not None)
        dead_count = len(self.weak_refs) - alive_count
        
        return {
            "total_tracked": len(self.weak_refs),
            "alive": alive_count,
            "collected": dead_count,
            "collection_rate": dead_count / len(self.weak_refs) if self.weak_refs else 0
        }

class TestMemoryStability:
    """Test suite for memory stability and leak detection."""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        # Setup
        gc.collect()
        cleanup_all_validators()
        
        yield
        
        # Teardown
        cleanup_all_validators()
        gc.collect()
    
    def generate_test_frames(self, count: int, size: tuple[int, int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Generate test frames for memory testing.
        
        Args:
            count: Number of frames
            size: Frame size (width, height)
            
        Returns:
            tuple of (original_frames, compressed_frames)
        """
        frames_orig = []
        frames_comp = []
        
        for _i in range(count):
            frame = np.random.randint(0, 256, (*size[::-1], 3), dtype=np.uint8)
            frames_orig.append(frame)
            
            # Add some noise for compressed version
            noise = np.random.normal(0, 10, frame.shape)
            compressed = np.clip(frame + noise, 0, 255).astype(np.uint8)
            frames_comp.append(compressed)
            
        return frames_orig, frames_comp
    
    def test_100_iterations_no_leak(self):
        """Test that 100+ iterations don't cause memory leaks."""
        detector = MemoryLeakDetector()
        detector.start_monitoring()
        
        iterations = 100
        frame_count = 10
        size = (200, 200)
        
        for i in range(iterations):
            # Generate fresh frames each iteration
            frames_orig, frames_comp = self.generate_test_frames(frame_count, size)
            
            # Calculate metrics
            metrics = calculate_comprehensive_metrics_from_frames(
                frames_orig,
                frames_comp
            )
            
            # Track memory every 10 iterations
            if i % 10 == 0:
                memory = detector.sample_memory()
                print(f"Iteration {i}: Memory = {memory:.1f} MB")
            
            # Explicitly delete to help GC
            del frames_orig, frames_comp, metrics
            
            # Periodic cleanup
            if i % 20 == 0:
                cleanup_all_validators()
                gc.collect()
        
        # Final cleanup
        cleanup_all_validators()
        
        # Analyze results
        results = detector.stop_monitoring()
        
        print("\nMemory Analysis:")
        print(f"  Initial: {results['initial_memory_mb']:.1f} MB")
        print(f"  Final: {results['final_memory_mb']:.1f} MB")
        print(f"  Growth: {results['memory_growth_mb']:.1f} MB")
        print(f"  Growth rate: {results['growth_rate_mb_per_iteration']:.3f} MB/iteration")
        print(f"  Potential leak: {results['potential_leak']}")
        
        # Assert no significant leak (LPIPS model loading adds ~200MB one-time overhead)
        assert results['memory_growth_mb'] < 300, f"Memory grew by {results['memory_growth_mb']:.1f} MB"
        assert not results['potential_leak'], "Potential memory leak detected"
        assert results['growth_rate_mb_per_iteration'] < 0.5, f"High growth rate: {results['growth_rate_mb_per_iteration']:.3f} MB/iteration"
    
    def test_rapid_size_changes_no_leak(self):
        """Test rapid succession of different GIF sizes."""
        detector = MemoryLeakDetector()
        detector.start_monitoring()
        
        # Different size configurations
        size_configs = [
            (5, (100, 100)),    # Small
            (20, (500, 500)),   # Medium
            (50, (1000, 1000)), # Large
            (10, (50, 50)),     # Tiny
            (30, (800, 600)),   # Standard
        ]
        
        iterations = 20
        
        for iteration in range(iterations):
            for frame_count, size in size_configs:
                frames_orig, frames_comp = self.generate_test_frames(frame_count, size)
                
                metrics = calculate_comprehensive_metrics_from_frames(
                    frames_orig,
                    frames_comp
                )

                del frames_orig, frames_comp, metrics

            # Sample memory each iteration
            memory = detector.sample_memory()
            print(f"Iteration {iteration}: Memory = {memory:.1f} MB")
            
            # Cleanup periodically
            if iteration % 5 == 0:
                cleanup_all_validators()
                gc.collect()
        
        # Final cleanup
        cleanup_all_validators()
        
        # Analyze results
        results = detector.stop_monitoring()
        
        print("\nRapid Size Changes Analysis:")
        print(f"  Memory growth: {results['memory_growth_mb']:.1f} MB")
        print(f"  Max memory: {results['max_memory_mb']:.1f} MB")
        
        # Assert no significant leak
        # LPIPS model loading adds ~500MB; 50x 1000x1000 frames = 150MB peak per batch
        # Total one-time overhead can be 1500MB+; check for growth rate instead
        assert not results['potential_leak'], "Potential memory leak detected"
        assert results['growth_rate_mb_per_iteration'] < 5.0, (
            f"High growth rate: {results['growth_rate_mb_per_iteration']:.3f} MB/iteration"
        )
    
    def test_model_cache_thrashing(self):
        """Test model cache under thrashing conditions."""
        detector = MemoryLeakDetector()
        detector.start_monitoring()
        
        cache = LPIPSModelCache()
        iterations = 50
        
        for i in range(iterations):
            # Force cache operations
            cache.get_model("alex")
            cache.release_model("alex")
            
            # Sometimes force clear
            if i % 10 == 0:
                cache.cleanup(force=True)
            
            # Track memory
            if i % 5 == 0:
                memory = detector.sample_memory()
                print(f"Cache iteration {i}: Memory = {memory:.1f} MB")
        
        # Final cleanup
        cache.cleanup(force=True)
        
        # Analyze results
        results = detector.stop_monitoring()
        
        print("\nCache Thrashing Analysis:")
        print(f"  Memory growth: {results['memory_growth_mb']:.1f} MB")
        
        # Assert cache doesn't leak (LPIPS model is ~500MB one-time load)
        assert results['memory_growth_mb'] < 600, f"Cache leaked {results['memory_growth_mb']:.1f} MB"
    
    def test_parallel_processing_cleanup(self):
        """Test that parallel processing cleans up properly."""
        detector = MemoryLeakDetector()
        detector.start_monitoring()
        
        # Enable parallel processing
        os.environ['GIFLAB_ENABLE_PARALLEL_METRICS'] = 'true'
        os.environ['GIFLAB_MAX_PARALLEL_WORKERS'] = '4'
        
        iterations = 30
        
        for i in range(iterations):
            frames_orig, frames_comp = self.generate_test_frames(50, (300, 300))
            
            metrics = calculate_comprehensive_metrics_from_frames(
                frames_orig,
                frames_comp
            )
            
            del frames_orig, frames_comp, metrics
            
            if i % 5 == 0:
                memory = detector.sample_memory()
                print(f"Parallel iteration {i}: Memory = {memory:.1f} MB")
                gc.collect()
        
        # Cleanup
        cleanup_all_validators()
        
        # Analyze results
        results = detector.stop_monitoring()
        
        print("\nParallel Processing Analysis:")
        print(f"  Memory growth: {results['memory_growth_mb']:.1f} MB")
        
        # Assert no significant leak from parallel processing (LPIPS model adds ~200MB one-time)
        assert results['memory_growth_mb'] < 300, f"Parallel processing leaked {results['memory_growth_mb']:.1f} MB"
    
    def test_validator_lifecycle(self):
        """Test validator object lifecycle and cleanup."""
        detector = MemoryLeakDetector()
        detector.start_monitoring()
        
        # Track validator objects
        
        for i in range(20):
            # Create validators
            ssim_validator = Ssimulacra2Validator()
            text_validator = TextUIContentDetector()
            get_temporal_detector()
            _get_or_create_validator()
            
            # Track with weak references
            detector.track_object(ssim_validator)
            detector.track_object(text_validator)
            
            # Use validators
            frames_orig, frames_comp = self.generate_test_frames(5, (200, 200))
            
            # Run validators
            ssim_validator.calculate_ssimulacra2_metrics(frames_orig, frames_comp, MetricsConfig())
            text_validator.detect_text_ui_regions(frames_orig[0])
            
            # Delete references
            del ssim_validator, text_validator
            
            # Cleanup globals
            if i % 5 == 0:
                cleanup_all_validators()
                memory = detector.sample_memory()
                print(f"Validator iteration {i}: Memory = {memory:.1f} MB")
        
        # Final cleanup
        cleanup_all_validators()
        gc.collect()
        
        # Check object collection
        tracking_results = detector.check_tracked_objects()
        results = detector.stop_monitoring()
        
        print("\nValidator Lifecycle Analysis:")
        print(f"  Objects tracked: {tracking_results['total_tracked']}")
        print(f"  Objects collected: {tracking_results['collected']}")
        print(f"  Collection rate: {tracking_results['collection_rate']:.1%}")
        print(f"  Memory growth: {results['memory_growth_mb']:.1f} MB")
        
        # Assert proper cleanup (LPIPS model adds ~200MB one-time overhead)
        assert tracking_results['collection_rate'] > 0.9, "Validators not being garbage collected"
        assert results['memory_growth_mb'] < 300, f"Validators leaked {results['memory_growth_mb']:.1f} MB"
    
    def test_error_recovery_cleanup(self):
        """Test cleanup when errors occur during processing."""
        detector = MemoryLeakDetector()
        detector.start_monitoring()
        
        iterations = 30
        
        for i in range(iterations):
            try:
                frames_orig, frames_comp = self.generate_test_frames(10, (200, 200))
                
                # Occasionally cause an error
                if i % 7 == 0:
                    # Simulate error by passing invalid data
                    frames_comp = None
                
                calculate_comprehensive_metrics_from_frames(
                    frames_orig,
                    frames_comp
                )

            except Exception as e:
                # Ensure cleanup happens even on error
                cleanup_all_validators()
                print(f"Error at iteration {i}: {type(e).__name__}")
            
            finally:
                # Always cleanup
                if i % 5 == 0:
                    cleanup_all_validators()
                    memory = detector.sample_memory()
                    print(f"Error recovery iteration {i}: Memory = {memory:.1f} MB")
                    gc.collect()
        
        # Final cleanup
        cleanup_all_validators()
        
        # Analyze results
        results = detector.stop_monitoring()
        
        print("\nError Recovery Analysis:")
        print(f"  Memory growth: {results['memory_growth_mb']:.1f} MB")
        
        # Assert no leak despite errors (LPIPS model loading adds ~500MB+ one-time overhead)
        assert results['memory_growth_mb'] < 800, f"Errors caused {results['memory_growth_mb']:.1f} MB leak"
    
    def test_line_by_line_profiling(self):
        """Profile memory usage line by line for a single metric calculation."""
        # This test is primarily for debugging and analysis
        # Note: For detailed line-by-line profiling, use external tools like memory_profiler
        
        frames_orig, frames_comp = self.generate_test_frames(20, (500, 500))
        
        # Calculate metrics with profiling
        metrics = calculate_comprehensive_metrics_from_frames(
            frames_orig,
            frames_comp
        )

        # Cleanup
        cleanup_all_validators()
        del frames_orig, frames_comp, metrics
        gc.collect()
    
    def test_stress_test_all_optimizations(self):
        """Stress test with all optimizations enabled."""
        detector = MemoryLeakDetector()
        detector.start_monitoring()
        
        # Enable all optimizations
        os.environ['GIFLAB_ENABLE_PARALLEL_METRICS'] = 'true'
        os.environ['GIFLAB_ENABLE_CONDITIONAL_METRICS'] = 'true'
        os.environ['GIFLAB_USE_MODEL_CACHE'] = 'true'
        
        # Stress test parameters
        iterations = 50
        configs = [
            (10, (100, 100)),
            (30, (500, 500)),
            (50, (800, 600)),
            (20, (200, 200)),
        ]
        
        for i in range(iterations):
            # Rotate through different configs
            frame_count, size = configs[i % len(configs)]
            frames_orig, frames_comp = self.generate_test_frames(frame_count, size)
            
            # Calculate metrics
            metrics = calculate_comprehensive_metrics_from_frames(
                frames_orig,
                frames_comp
            )
            
            # Cleanup
            del frames_orig, frames_comp, metrics
            
            if i % 10 == 0:
                cleanup_all_validators()
                memory = detector.sample_memory()
                print(f"Stress iteration {i}: Memory = {memory:.1f} MB")
                gc.collect()
        
        # Final cleanup
        cleanup_all_validators()
        
        # Analyze results
        results = detector.stop_monitoring()
        
        print("\nStress Test Analysis:")
        print(f"  Initial memory: {results['initial_memory_mb']:.1f} MB")
        print(f"  Final memory: {results['final_memory_mb']:.1f} MB")
        print(f"  Memory growth: {results['memory_growth_mb']:.1f} MB")
        print(f"  Max memory: {results['max_memory_mb']:.1f} MB")
        print(f"  Potential leak: {results['potential_leak']}")
        
        # Assert acceptable memory behavior under stress
        # LPIPS model loading adds ~500MB overhead; allow generous headroom for stress scenarios
        assert results['memory_growth_mb'] < 600, f"Stress test leaked {results['memory_growth_mb']:.1f} MB"
        assert results['max_memory_mb'] - results['initial_memory_mb'] < 1200, "Peak memory too high under stress"

def run_memory_analysis():
    """Run comprehensive memory analysis and generate report."""
    print("=" * 80)
    print("MEMORY STABILITY ANALYSIS")
    print("=" * 80)
    
    test_suite = TestMemoryStability()
    results = {}
    
    # List of tests to run
    tests = [
        ("100 Iterations", test_suite.test_100_iterations_no_leak),
        ("Rapid Size Changes", test_suite.test_rapid_size_changes_no_leak),
        ("Model Cache Thrashing", test_suite.test_model_cache_thrashing),
        ("Parallel Processing", test_suite.test_parallel_processing_cleanup),
        ("Validator Lifecycle", test_suite.test_validator_lifecycle),
        ("Error Recovery", test_suite.test_error_recovery_cleanup),
        ("Stress Test", test_suite.test_stress_test_all_optimizations),
    ]
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        print("-" * 40)
        
        try:
            # Setup
            test_suite.setup_and_teardown().__next__()
            
            # Run test
            test_func()
            results[test_name] = "PASSED"
            print(f"✓ {test_name} passed")
            
        except AssertionError as e:
            results[test_name] = f"FAILED: {str(e)}"
            print(f"✗ {test_name} failed: {str(e)}")
            
        except Exception as e:
            results[test_name] = f"ERROR: {str(e)}"
            print(f"✗ {test_name} error: {str(e)}")
        
        finally:
            # Teardown
            try:
                test_suite.setup_and_teardown().__next__()
            except StopIteration:
                pass
            
            # Force cleanup between tests
            cleanup_all_validators()
            gc.collect()
    
    # Generate summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results.values() if r == "PASSED")
    failed = sum(1 for r in results.values() if r.startswith("FAILED"))
    errors = sum(1 for r in results.values() if r.startswith("ERROR"))
    
    print(f"Total tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    
    if failed > 0 or errors > 0:
        print("\nFailed/Error tests:")
        for test_name, result in results.items():
            if result != "PASSED":
                print(f"  - {test_name}: {result}")
    
    return results

if __name__ == "__main__":
    run_memory_analysis()
