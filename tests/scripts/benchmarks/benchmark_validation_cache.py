"""Performance benchmarks for ValidationCache."""

import time
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pytest

from giflab.caching.validation_cache import ValidationCache, reset_validation_cache
from giflab.caching.metrics_integration import (
    calculate_ms_ssim_cached,
    calculate_ssim_cached,
    calculate_lpips_cached,
)


class BenchmarkValidationCache:
    """Benchmark suite for ValidationCache performance."""
    
    @staticmethod
    def generate_test_frames(num_frames: int = 10, size: Tuple[int, int] = (100, 100)) -> List[np.ndarray]:
        """Generate random test frames for benchmarking."""
        np.random.seed(42)
        frames = []
        for i in range(num_frames):
            frame = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
            frames.append(frame)
        return frames
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset cache singleton before each test."""
        reset_validation_cache()
        yield
        reset_validation_cache()
    
    def benchmark_cache_put_get(self, temp_cache_dir: Path, num_operations: int = 1000):
        """Benchmark basic put/get operations."""
        cache = ValidationCache(
            memory_limit_mb=50,
            disk_path=temp_cache_dir / "benchmark.db",
            disk_limit_mb=200,
            enabled=True,
        )
        
        frames = self.generate_test_frames(20)
        
        # Benchmark PUT operations
        put_times = []
        for i in range(num_operations):
            frame1 = frames[i % len(frames)]
            frame2 = frames[(i + 1) % len(frames)]
            
            start = time.perf_counter()
            cache.put(
                frame1, frame2, 
                f"metric_{i % 5}",  # Rotate through 5 metric types
                float(i),
                config={"param": i % 10},
                frame_indices=(i, i + 1),
            )
            put_times.append(time.perf_counter() - start)
        
        # Benchmark GET operations (cache hits)
        get_hit_times = []
        for i in range(num_operations):
            frame1 = frames[i % len(frames)]
            frame2 = frames[(i + 1) % len(frames)]
            
            start = time.perf_counter()
            result = cache.get(
                frame1, frame2,
                f"metric_{i % 5}",
                config={"param": i % 10},
                frame_indices=(i, i + 1),
            )
            get_hit_times.append(time.perf_counter() - start)
        
        # Benchmark GET operations (cache misses)
        get_miss_times = []
        for i in range(100):
            frame1 = frames[0]
            frame2 = frames[1]
            
            start = time.perf_counter()
            result = cache.get(
                frame1, frame2,
                "nonexistent_metric",
                config={"nonexistent": i},
            )
            get_miss_times.append(time.perf_counter() - start)
        
        # Calculate statistics
        stats = cache.get_stats()
        
        return {
            "put_avg_ms": np.mean(put_times) * 1000,
            "put_median_ms": np.median(put_times) * 1000,
            "put_p95_ms": np.percentile(put_times, 95) * 1000,
            "get_hit_avg_ms": np.mean(get_hit_times) * 1000,
            "get_hit_median_ms": np.median(get_hit_times) * 1000,
            "get_hit_p95_ms": np.percentile(get_hit_times, 95) * 1000,
            "get_miss_avg_ms": np.mean(get_miss_times) * 1000,
            "get_miss_median_ms": np.median(get_miss_times) * 1000,
            "hit_rate": stats.hit_rate,
            "evictions": stats.evictions,
        }
    
    def benchmark_memory_eviction(self, temp_cache_dir: Path):
        """Benchmark LRU eviction performance."""
        # Small cache to force evictions
        cache = ValidationCache(
            memory_limit_mb=1,  # Very small cache
            disk_path=temp_cache_dir / "eviction.db",
            enabled=True,
        )
        
        frames = self.generate_test_frames(50, size=(50, 50))
        
        eviction_times = []
        for i in range(100):
            frame1 = frames[i % len(frames)]
            frame2 = frames[(i + 1) % len(frames)]
            
            start = time.perf_counter()
            cache.put(
                frame1, frame2,
                "metric",
                {"large_data": list(range(1000))},  # Large value to force eviction
            )
            eviction_times.append(time.perf_counter() - start)
        
        stats = cache.get_stats()
        
        return {
            "eviction_avg_ms": np.mean(eviction_times) * 1000,
            "eviction_median_ms": np.median(eviction_times) * 1000,
            "eviction_p95_ms": np.percentile(eviction_times, 95) * 1000,
            "total_evictions": stats.evictions,
        }
    
    def benchmark_disk_persistence(self, temp_cache_dir: Path):
        """Benchmark disk persistence performance."""
        cache_path = temp_cache_dir / "persist.db"
        
        # First run - populate cache
        cache1 = ValidationCache(
            memory_limit_mb=10,
            disk_path=cache_path,
            enabled=True,
        )
        
        frames = self.generate_test_frames(20)
        
        # Populate with data
        populate_times = []
        for i in range(100):
            frame1 = frames[i % len(frames)]
            frame2 = frames[(i + 1) % len(frames)]
            
            start = time.perf_counter()
            cache1.put(frame1, frame2, f"metric_{i}", float(i))
            populate_times.append(time.perf_counter() - start)
        
        # Clear memory to force disk reads
        cache1._memory_cache.clear()
        cache1._memory_bytes = 0
        
        # Benchmark disk reads
        disk_read_times = []
        for i in range(100):
            frame1 = frames[i % len(frames)]
            frame2 = frames[(i + 1) % len(frames)]
            
            start = time.perf_counter()
            result = cache1.get(frame1, frame2, f"metric_{i}")
            disk_read_times.append(time.perf_counter() - start)
        
        return {
            "populate_avg_ms": np.mean(populate_times) * 1000,
            "disk_read_avg_ms": np.mean(disk_read_times) * 1000,
            "disk_read_median_ms": np.median(disk_read_times) * 1000,
            "disk_read_p95_ms": np.percentile(disk_read_times, 95) * 1000,
        }
    
    def benchmark_metric_caching(self, temp_cache_dir: Path):
        """Benchmark actual metric calculation caching."""
        from unittest.mock import patch
        
        config = {
            "enabled": True,
            "memory_limit_mb": 50,
            "disk_path": temp_cache_dir / "metrics.db",
            "cache_ms_ssim": True,
            "cache_ssim": True,
            "cache_lpips": True,
        }
        
        frames = self.generate_test_frames(10, size=(200, 200))
        
        with patch("giflab.caching.metrics_integration.VALIDATION_CACHE", config):
            with patch("giflab.config.VALIDATION_CACHE", config):
                # Mock actual calculations with delays
                def mock_ms_ssim(f1, f2, scales=5):
                    time.sleep(0.01)  # Simulate 10ms calculation
                    return 0.95
                
                def mock_ssim(f1, f2, data_range=255):
                    time.sleep(0.005)  # Simulate 5ms calculation
                    return 0.88
                
                def mock_lpips(frames1, frames2, net="alex", version="0.1"):
                    time.sleep(0.02)  # Simulate 20ms calculation
                    return [0.12]
                
                with patch("giflab.metrics.calculate_ms_ssim", side_effect=mock_ms_ssim):
                    with patch("skimage.metrics.structural_similarity", side_effect=mock_ssim):
                        with patch("giflab.deep_perceptual_metrics.calculate_lpips_frames", side_effect=mock_lpips):
                            
                            # Reset cache
                            reset_validation_cache()
                            
                            # Benchmark MS-SSIM
                            ms_ssim_times = {"first": [], "cached": []}
                            for i in range(20):
                                frame1 = frames[i % len(frames)]
                                frame2 = frames[(i + 1) % len(frames)]
                                
                                # First calculation
                                start = time.perf_counter()
                                result1 = calculate_ms_ssim_cached(frame1, frame2)
                                ms_ssim_times["first"].append(time.perf_counter() - start)
                                
                                # Cached calculation
                                start = time.perf_counter()
                                result2 = calculate_ms_ssim_cached(frame1, frame2)
                                ms_ssim_times["cached"].append(time.perf_counter() - start)
                            
                            # Benchmark SSIM
                            ssim_times = {"first": [], "cached": []}
                            for i in range(20):
                                frame1 = frames[i % len(frames)]
                                frame2 = frames[(i + 2) % len(frames)]
                                
                                # First calculation
                                start = time.perf_counter()
                                result1 = calculate_ssim_cached(frame1, frame2)
                                ssim_times["first"].append(time.perf_counter() - start)
                                
                                # Cached calculation
                                start = time.perf_counter()
                                result2 = calculate_ssim_cached(frame1, frame2)
                                ssim_times["cached"].append(time.perf_counter() - start)
                            
                            # Benchmark LPIPS
                            lpips_times = {"first": [], "cached": []}
                            for i in range(20):
                                frame1 = frames[i % len(frames)]
                                frame2 = frames[(i + 3) % len(frames)]
                                
                                # First calculation
                                start = time.perf_counter()
                                result1 = calculate_lpips_cached(frame1, frame2)
                                lpips_times["first"].append(time.perf_counter() - start)
                                
                                # Cached calculation
                                start = time.perf_counter()
                                result2 = calculate_lpips_cached(frame1, frame2)
                                lpips_times["cached"].append(time.perf_counter() - start)
        
        return {
            "ms_ssim": {
                "first_avg_ms": np.mean(ms_ssim_times["first"]) * 1000,
                "cached_avg_ms": np.mean(ms_ssim_times["cached"]) * 1000,
                "speedup": np.mean(ms_ssim_times["first"]) / np.mean(ms_ssim_times["cached"]),
            },
            "ssim": {
                "first_avg_ms": np.mean(ssim_times["first"]) * 1000,
                "cached_avg_ms": np.mean(ssim_times["cached"]) * 1000,
                "speedup": np.mean(ssim_times["first"]) / np.mean(ssim_times["cached"]),
            },
            "lpips": {
                "first_avg_ms": np.mean(lpips_times["first"]) * 1000,
                "cached_avg_ms": np.mean(lpips_times["cached"]) * 1000,
                "speedup": np.mean(lpips_times["first"]) / np.mean(lpips_times["cached"]),
            },
        }
    
    def benchmark_concurrent_access(self, temp_cache_dir: Path):
        """Benchmark concurrent cache access."""
        import threading
        import queue
        
        cache = ValidationCache(
            memory_limit_mb=50,
            disk_path=temp_cache_dir / "concurrent.db",
            enabled=True,
        )
        
        frames = self.generate_test_frames(20)
        results_queue = queue.Queue()
        
        def worker(thread_id: int, num_ops: int):
            times = []
            for i in range(num_ops):
                frame1 = frames[(thread_id + i) % len(frames)]
                frame2 = frames[(thread_id + i + 1) % len(frames)]
                
                start = time.perf_counter()
                
                # Mix of puts and gets
                if i % 3 == 0:
                    cache.put(frame1, frame2, f"metric_{thread_id}_{i}", float(i))
                else:
                    cache.get(frame1, frame2, f"metric_{thread_id}_{i // 3}")
                
                times.append(time.perf_counter() - start)
            
            results_queue.put((thread_id, times))
        
        # Run with different thread counts
        thread_counts = [1, 2, 4, 8]
        results = {}
        
        for num_threads in thread_counts:
            threads = []
            ops_per_thread = 100 // num_threads
            
            start_time = time.perf_counter()
            
            for i in range(num_threads):
                t = threading.Thread(target=worker, args=(i, ops_per_thread))
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
            
            total_time = time.perf_counter() - start_time
            
            # Collect times from all threads
            all_times = []
            while not results_queue.empty():
                _, times = results_queue.get()
                all_times.extend(times)
            
            results[f"{num_threads}_threads"] = {
                "total_time_ms": total_time * 1000,
                "avg_op_time_ms": np.mean(all_times) * 1000 if all_times else 0,
                "throughput_ops_per_sec": (num_threads * ops_per_thread) / total_time,
            }
        
        return results
    
    def run_all_benchmarks(self, temp_cache_dir: Path) -> Dict[str, Any]:
        """Run all benchmarks and return results."""
        print("Running ValidationCache Performance Benchmarks...")
        print("=" * 60)
        
        results = {}
        
        # Basic operations
        print("\n1. Basic Put/Get Operations...")
        results["basic_operations"] = self.benchmark_cache_put_get(temp_cache_dir)
        print(f"   Put avg: {results['basic_operations']['put_avg_ms']:.2f}ms")
        print(f"   Get (hit) avg: {results['basic_operations']['get_hit_avg_ms']:.2f}ms")
        print(f"   Get (miss) avg: {results['basic_operations']['get_miss_avg_ms']:.2f}ms")
        print(f"   Hit rate: {results['basic_operations']['hit_rate']:.1%}")
        
        # Memory eviction
        print("\n2. Memory Eviction Performance...")
        results["eviction"] = self.benchmark_memory_eviction(temp_cache_dir)
        print(f"   Eviction avg: {results['eviction']['eviction_avg_ms']:.2f}ms")
        print(f"   Total evictions: {results['eviction']['total_evictions']}")
        
        # Disk persistence
        print("\n3. Disk Persistence Performance...")
        results["disk"] = self.benchmark_disk_persistence(temp_cache_dir)
        print(f"   Populate avg: {results['disk']['populate_avg_ms']:.2f}ms")
        print(f"   Disk read avg: {results['disk']['disk_read_avg_ms']:.2f}ms")
        
        # Metric caching
        print("\n4. Metric Calculation Caching...")
        results["metrics"] = self.benchmark_metric_caching(temp_cache_dir)
        for metric, data in results["metrics"].items():
            print(f"   {metric}:")
            print(f"      First: {data['first_avg_ms']:.2f}ms")
            print(f"      Cached: {data['cached_avg_ms']:.2f}ms")
            print(f"      Speedup: {data['speedup']:.1f}x")
        
        # Concurrent access
        print("\n5. Concurrent Access Performance...")
        results["concurrent"] = self.benchmark_concurrent_access(temp_cache_dir)
        for threads, data in results["concurrent"].items():
            print(f"   {threads}:")
            print(f"      Total time: {data['total_time_ms']:.2f}ms")
            print(f"      Throughput: {data['throughput_ops_per_sec']:.0f} ops/sec")
        
        print("\n" + "=" * 60)
        print("Benchmarks Complete!")
        
        return results


def test_validation_cache_performance():
    """Run performance benchmarks as a test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        benchmark = BenchmarkValidationCache()
        results = benchmark.run_all_benchmarks(Path(tmpdir))
        
        # Assert performance expectations
        assert results["basic_operations"]["get_hit_avg_ms"] < 1.0, "Cache hits should be < 1ms"
        assert results["basic_operations"]["hit_rate"] > 0.9, "Hit rate should be > 90%"
        
        # Assert speedups
        assert results["metrics"]["ms_ssim"]["speedup"] > 5, "MS-SSIM should have > 5x speedup"
        assert results["metrics"]["ssim"]["speedup"] > 3, "SSIM should have > 3x speedup"
        assert results["metrics"]["lpips"]["speedup"] > 10, "LPIPS should have > 10x speedup"


if __name__ == "__main__":
    # Run benchmarks directly
    with tempfile.TemporaryDirectory() as tmpdir:
        benchmark = BenchmarkValidationCache()
        results = benchmark.run_all_benchmarks(Path(tmpdir))
        
        # Save results to JSON
        import json
        with open("validation_cache_benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to validation_cache_benchmark_results.json")