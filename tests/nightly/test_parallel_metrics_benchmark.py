#!/usr/bin/env python3
"""Performance benchmarks for parallel metrics processing.

This test suite validates the performance improvements from parallel processing
and ensures that results remain deterministic and accurate.
"""

import os
import time
import tempfile
from pathlib import Path
import numpy as np
import pytest
from unittest import mock
import multiprocessing as mp

# Import the modules we're testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from giflab.metrics import calculate_comprehensive_metrics_from_frames
from giflab.parallel_metrics import ParallelMetricsCalculator, ParallelConfig
from giflab.config import DEFAULT_METRICS_CONFIG


class TestParallelMetricsBenchmark:
    """Benchmark tests for parallel metrics calculation."""
    
    @classmethod
    def setup_class(cls):
        """Create synthetic test data for benchmarking."""
        # Generate different sizes of frame sets for testing
        cls.test_scenarios = {
            "small": cls._generate_frames(5, (100, 100)),
            "medium": cls._generate_frames(20, (256, 256)),
            "large": cls._generate_frames(50, (512, 512)),
            "xl": cls._generate_frames(100, (512, 512)),
        }
    
    @staticmethod
    def _generate_frames(num_frames: int, size: tuple[int, int]) -> list[np.ndarray]:
        """Generate synthetic frames for testing."""
        frames = []
        for i in range(num_frames):
            # Create frames with some variation to make metrics meaningful
            frame = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            # Add some pattern to make frames more realistic
            pattern = np.sin(np.linspace(0, np.pi * i, size[0]))[:, np.newaxis] * 50
            frame[:, :, 0] = np.clip(frame[:, :, 0] + pattern, 0, 255)
            frames.append(frame)
        return frames
    
    def test_parallel_vs_sequential_performance(self):
        """Compare performance of parallel vs sequential processing."""
        results = []
        
        for scenario_name, frames in self.test_scenarios.items():
            # Create slightly modified frames for compression comparison
            compressed_frames = [
                np.clip(f.astype(np.float32) * 0.95, 0, 255).astype(np.uint8)
                for f in frames
            ]
            
            # Test sequential processing
            start = time.perf_counter()
            with mock.patch.dict(os.environ, {"GIFLAB_ENABLE_PARALLEL_METRICS": "false"}):
                seq_metrics = calculate_comprehensive_metrics_from_frames(
                    frames, compressed_frames, config=DEFAULT_METRICS_CONFIG
                )
            seq_time = time.perf_counter() - start
            
            # Test parallel processing
            start = time.perf_counter()
            with mock.patch.dict(os.environ, {"GIFLAB_ENABLE_PARALLEL_METRICS": "true"}):
                par_metrics = calculate_comprehensive_metrics_from_frames(
                    frames, compressed_frames, config=DEFAULT_METRICS_CONFIG
                )
            par_time = time.perf_counter() - start
            
            speedup = seq_time / par_time if par_time > 0 else 1.0
            
            results.append({
                "scenario": scenario_name,
                "frames": len(frames),
                "sequential_time": seq_time,
                "parallel_time": par_time,
                "speedup": speedup,
                "seq_ssim": seq_metrics.get("ssim", 0),
                "par_ssim": par_metrics.get("ssim", 0),
            })
            
            print(f"\n{scenario_name}: {len(frames)} frames")
            print(f"  Sequential: {seq_time:.3f}s")
            print(f"  Parallel:   {par_time:.3f}s")
            print(f"  Speedup:    {speedup:.2f}x")
        
        # Verify that larger scenarios benefit more from parallelization
        large_speedup = next(r for r in results if r["scenario"] == "large")["speedup"]
        small_speedup = next(r for r in results if r["scenario"] == "small")["speedup"]
        
        # Large scenarios should have better speedup (or at least not worse)
        assert large_speedup >= small_speedup * 0.8, \
            f"Large scenario speedup ({large_speedup:.2f}x) should be >= small ({small_speedup:.2f}x)"
    
    def test_deterministic_results(self):
        """Ensure parallel processing produces deterministic results."""
        frames = self.test_scenarios["medium"]
        compressed_frames = [
            np.clip(f.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
            for f in frames
        ]
        
        # Run parallel processing multiple times
        results = []
        with mock.patch.dict(os.environ, {"GIFLAB_ENABLE_PARALLEL_METRICS": "true"}):
            for _ in range(3):
                metrics = calculate_comprehensive_metrics_from_frames(
                    frames, compressed_frames, config=DEFAULT_METRICS_CONFIG
                )
                results.append(metrics)
        
        # Verify all runs produce identical results
        key_metrics = ["ssim", "psnr", "mse", "fsim", "gmsd"]
        for metric in key_metrics:
            values = [r.get(metric, 0) for r in results]
            assert all(abs(v - values[0]) < 1e-6 for v in values), \
                f"Metric {metric} not deterministic: {values}"
    
    def test_worker_count_scaling(self):
        """Test performance with different worker counts."""
        frames = self.test_scenarios["large"]
        compressed_frames = [
            np.clip(f.astype(np.float32) * 0.95, 0, 255).astype(np.uint8)
            for f in frames
        ]
        
        worker_counts = [1, 2, 4, mp.cpu_count()]
        results = []
        
        for workers in worker_counts:
            start = time.perf_counter()
            with mock.patch.dict(os.environ, {
                "GIFLAB_ENABLE_PARALLEL_METRICS": "true",
                "GIFLAB_MAX_PARALLEL_WORKERS": str(workers)
            }):
                _ = calculate_comprehensive_metrics_from_frames(
                    frames, compressed_frames, config=DEFAULT_METRICS_CONFIG
                )
            elapsed = time.perf_counter() - start
            
            results.append({
                "workers": workers,
                "time": elapsed,
                "frames_per_second": len(frames) / elapsed
            })
            
            print(f"\nWorkers: {workers}")
            print(f"  Time: {elapsed:.3f}s")
            print(f"  FPS:  {len(frames) / elapsed:.1f}")
        
        # Verify that more workers generally improve performance
        single_worker_time = results[0]["time"]
        multi_worker_time = results[-1]["time"]
        assert multi_worker_time < single_worker_time, \
            "Multi-worker should be faster than single worker"
    
    def test_chunk_strategies(self):
        """Test different chunking strategies."""
        frames = self.test_scenarios["medium"]
        compressed_frames = [
            np.clip(f.astype(np.float32) * 0.95, 0, 255).astype(np.uint8)
            for f in frames
        ]
        
        strategies = ["adaptive", "fixed", "dynamic"]
        results = []
        
        for strategy in strategies:
            start = time.perf_counter()
            with mock.patch.dict(os.environ, {
                "GIFLAB_ENABLE_PARALLEL_METRICS": "true",
                "GIFLAB_CHUNK_STRATEGY": strategy
            }):
                metrics = calculate_comprehensive_metrics_from_frames(
                    frames, compressed_frames, config=DEFAULT_METRICS_CONFIG
                )
            elapsed = time.perf_counter() - start
            
            results.append({
                "strategy": strategy,
                "time": elapsed,
                "ssim": metrics.get("ssim", 0)
            })
            
            print(f"\nStrategy: {strategy}")
            print(f"  Time: {elapsed:.3f}s")
            print(f"  SSIM: {metrics.get('ssim', 0):.4f}")
        
        # All strategies should produce same results
        ssim_values = [r["ssim"] for r in results]
        assert all(abs(v - ssim_values[0]) < 1e-6 for v in ssim_values), \
            f"Different strategies produced different results: {ssim_values}"
    
    def test_memory_efficiency(self):
        """Test that parallel processing doesn't cause excessive memory usage."""
        import psutil
        process = psutil.Process()
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large dataset
        frames = self.test_scenarios["xl"]
        compressed_frames = [
            np.clip(f.astype(np.float32) * 0.95, 0, 255).astype(np.uint8)
            for f in frames
        ]
        
        with mock.patch.dict(os.environ, {"GIFLAB_ENABLE_PARALLEL_METRICS": "true"}):
            _ = calculate_comprehensive_metrics_from_frames(
                frames, compressed_frames, config=DEFAULT_METRICS_CONFIG
            )
        
        # Check memory after processing
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - baseline_memory
        
        print(f"\nMemory usage:")
        print(f"  Baseline: {baseline_memory:.1f} MB")
        print(f"  Peak:     {peak_memory:.1f} MB")
        print(f"  Increase: {memory_increase:.1f} MB")
        
        # Memory increase should be reasonable (less than 12x frame data size)
        # Note: LPIPS model loading, intermediate arrays, and metric computation
        # allocate significant temporary memory beyond the raw frame data
        frame_data_size = sum(f.nbytes for f in frames) / 1024 / 1024
        assert memory_increase < frame_data_size * 12, \
            f"Excessive memory usage: {memory_increase:.1f} MB for {frame_data_size:.1f} MB of frames"
    
    @pytest.mark.parametrize("error_type", ["import_error", "calculation_error"])
    def test_fallback_to_sequential(self, error_type):
        """Test graceful fallback to sequential processing on errors."""
        frames = self.test_scenarios["small"]
        compressed_frames = [
            np.clip(f.astype(np.float32) * 0.95, 0, 255).astype(np.uint8)
            for f in frames
        ]
        
        if error_type == "import_error":
            # Simulate import error
            with mock.patch("giflab.parallel_metrics.ParallelMetricsCalculator", side_effect=ImportError):
                metrics = calculate_comprehensive_metrics_from_frames(
                    frames, compressed_frames, config=DEFAULT_METRICS_CONFIG
                )
        else:
            # Simulate calculation error
            with mock.patch("giflab.parallel_metrics.ParallelMetricsCalculator.calculate_frame_metrics_parallel",
                          side_effect=RuntimeError("Test error")):
                metrics = calculate_comprehensive_metrics_from_frames(
                    frames, compressed_frames, config=DEFAULT_METRICS_CONFIG
                )
        
        # Should still produce valid results (via fallback)
        assert "ssim" in metrics
        assert 0 <= metrics["ssim"] <= 1
        assert "psnr" in metrics


class TestParallelMetricsAccuracy:
    """Test accuracy of parallel metrics calculation."""
    
    def test_metric_accuracy_vs_sequential(self):
        """Verify parallel results match sequential exactly."""
        # Create test frames
        frames = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        compressed_frames = [
            np.clip(f.astype(np.float32) * 0.9 + np.random.normal(0, 5, f.shape), 0, 255).astype(np.uint8)
            for f in frames
        ]
        
        # Calculate with sequential
        with mock.patch.dict(os.environ, {"GIFLAB_ENABLE_PARALLEL_METRICS": "false"}):
            seq_metrics = calculate_comprehensive_metrics_from_frames(
                frames, compressed_frames, config=DEFAULT_METRICS_CONFIG
            )
        
        # Calculate with parallel
        with mock.patch.dict(os.environ, {"GIFLAB_ENABLE_PARALLEL_METRICS": "true"}):
            par_metrics = calculate_comprehensive_metrics_from_frames(
                frames, compressed_frames, config=DEFAULT_METRICS_CONFIG
            )
        
        # Compare all frame-level metrics
        metrics_to_check = [
            "ssim", "ms_ssim", "psnr", "mse", "rmse",
            "fsim", "gmsd", "chist", "edge_similarity",
            "texture_similarity", "sharpness_similarity"
        ]
        
        for metric in metrics_to_check:
            seq_val = seq_metrics.get(metric, 0)
            par_val = par_metrics.get(metric, 0)
            
            # Values should be identical (or very close for floating point)
            assert abs(seq_val - par_val) < 1e-10, \
                f"Metric {metric} differs: sequential={seq_val}, parallel={par_val}"
            
            # Also check std, min, max variants
            for suffix in ["_std", "_min", "_max"]:
                key = f"{metric}{suffix}"
                if key in seq_metrics:
                    seq_val = seq_metrics[key]
                    par_val = par_metrics.get(key, 0)
                    assert abs(seq_val - par_val) < 1e-10, \
                        f"Metric {key} differs: sequential={seq_val}, parallel={par_val}"
    
    def test_edge_cases(self):
        """Test edge cases like single frame, empty frames, etc."""
        test_cases = [
            # Single frame
            ([np.zeros((10, 10, 3), dtype=np.uint8)],
             [np.ones((10, 10, 3), dtype=np.uint8) * 128]),
            
            # Identical frames
            ([np.ones((50, 50, 3), dtype=np.uint8) * 100] * 5,
             [np.ones((50, 50, 3), dtype=np.uint8) * 100] * 5),
            
            # Very small frames
            ([np.random.randint(0, 255, (2, 2, 3), dtype=np.uint8) for _ in range(3)],
             [np.random.randint(0, 255, (2, 2, 3), dtype=np.uint8) for _ in range(3)]),
        ]
        
        for orig_frames, comp_frames in test_cases:
            # Should not crash and produce valid results
            with mock.patch.dict(os.environ, {"GIFLAB_ENABLE_PARALLEL_METRICS": "true"}):
                metrics = calculate_comprehensive_metrics_from_frames(
                    orig_frames, comp_frames, config=DEFAULT_METRICS_CONFIG
                )
            
            assert "ssim" in metrics
            assert "composite_quality" in metrics
            assert metrics["frame_count"] == len(orig_frames)


if __name__ == "__main__":
    # Run benchmarks
    print("=" * 60)
    print("Running Parallel Metrics Benchmarks")
    print("=" * 60)
    
    benchmark = TestParallelMetricsBenchmark()
    benchmark.setup_class()
    
    print("\n1. Performance Comparison")
    benchmark.test_parallel_vs_sequential_performance()
    
    print("\n2. Worker Scaling")
    benchmark.test_worker_count_scaling()
    
    print("\n3. Chunk Strategies")
    benchmark.test_chunk_strategies()
    
    print("\n4. Memory Efficiency")
    benchmark.test_memory_efficiency()
    
    print("\n5. Deterministic Results")
    benchmark.test_deterministic_results()
    
    print("\n" + "=" * 60)
    print("All benchmarks completed successfully!")
    print("=" * 60)