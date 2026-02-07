"""Performance benchmarking for frame cache system."""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from giflab.caching import get_frame_cache, reset_frame_cache
from giflab.config import FRAME_CACHE
from giflab.metrics import extract_gif_frames


class FrameCacheBenchmark:
    """Benchmark suite for frame cache performance."""
    
    def __init__(self, output_dir: Path):
        """Initialize benchmark suite.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, any] = {}
    
    def create_test_gifs(self, test_dir: Path) -> Dict[str, Path]:
        """Create test GIF files of various sizes.
        
        Args:
            test_dir: Directory to create test files in
            
        Returns:
            Dictionary mapping test names to file paths
        """
        test_dir.mkdir(parents=True, exist_ok=True)
        test_files = {}
        
        # Small GIF (10 frames, 100x100)
        test_files["small"] = self._create_gif(
            test_dir / "small.gif",
            frames=10,
            size=(100, 100)
        )
        
        # Medium GIF (50 frames, 200x200)
        test_files["medium"] = self._create_gif(
            test_dir / "medium.gif",
            frames=50,
            size=(200, 200)
        )
        
        # Large GIF (100 frames, 400x400)
        test_files["large"] = self._create_gif(
            test_dir / "large.gif",
            frames=100,
            size=(400, 400)
        )
        
        # Extra large GIF (200 frames, 600x600)
        test_files["xlarge"] = self._create_gif(
            test_dir / "xlarge.gif",
            frames=200,
            size=(600, 600)
        )
        
        return test_files
    
    def _create_gif(self, path: Path, frames: int, size: tuple[int, int]) -> Path:
        """Create a test GIF file.
        
        Args:
            path: Path to save GIF
            frames: Number of frames
            size: Frame size (width, height)
            
        Returns:
            Path to created GIF
        """
        images = []
        for i in range(frames):
            # Create varied content for realistic testing
            arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            # Add some structure (not just noise)
            arr[i % size[0], :, :] = 255  # Horizontal line
            arr[:, i % size[1], :] = 0    # Vertical line
            
            img = Image.fromarray(arr)
            images.append(img)
        
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0
        )
        
        return path
    
    def benchmark_single_file(
        self,
        gif_path: Path,
        iterations: int = 10,
        max_frames: int | None = None
    ) -> Dict[str, float]:
        """Benchmark frame extraction for a single file.
        
        Args:
            gif_path: Path to GIF file
            iterations: Number of iterations to run
            max_frames: Maximum frames to extract
            
        Returns:
            Dictionary with timing statistics
        """
        # Warm up (ensure file is in OS cache)
        extract_gif_frames(gif_path, max_frames=max_frames)
        
        times_with_cache = []
        times_without_cache = []
        
        # Test with cache enabled
        FRAME_CACHE["enabled"] = True
        reset_frame_cache()
        cache = get_frame_cache()
        
        # First extraction (cache miss)
        start = time.perf_counter()
        extract_gif_frames(gif_path, max_frames=max_frames)
        first_time = time.perf_counter() - start
        
        # Subsequent extractions (cache hits)
        for _ in range(iterations):
            start = time.perf_counter()
            extract_gif_frames(gif_path, max_frames=max_frames)
            times_with_cache.append(time.perf_counter() - start)
        
        cache_stats = cache.get_stats()
        
        # Test without cache
        FRAME_CACHE["enabled"] = False
        reset_frame_cache()
        
        for _ in range(iterations):
            start = time.perf_counter()
            extract_gif_frames(gif_path, max_frames=max_frames)
            times_without_cache.append(time.perf_counter() - start)
        
        # Re-enable cache for next test
        FRAME_CACHE["enabled"] = True
        
        return {
            "file": str(gif_path.name),
            "file_size_mb": gif_path.stat().st_size / (1024 * 1024),
            "first_extraction_time": first_time,
            "cached_mean": statistics.mean(times_with_cache),
            "cached_median": statistics.median(times_with_cache),
            "cached_stdev": statistics.stdev(times_with_cache) if len(times_with_cache) > 1 else 0,
            "cached_min": min(times_with_cache),
            "cached_max": max(times_with_cache),
            "uncached_mean": statistics.mean(times_without_cache),
            "uncached_median": statistics.median(times_without_cache),
            "uncached_stdev": statistics.stdev(times_without_cache) if len(times_without_cache) > 1 else 0,
            "uncached_min": min(times_without_cache),
            "uncached_max": max(times_without_cache),
            "speedup": statistics.mean(times_without_cache) / statistics.mean(times_with_cache),
            "cache_hits": cache_stats.hits,
            "cache_misses": cache_stats.misses,
            "cache_hit_rate": cache_stats.hit_rate,
        }
    
    def benchmark_cache_size_impact(
        self,
        test_files: Dict[str, Path],
        memory_limits_mb: List[int] = [10, 50, 100, 500]
    ) -> List[Dict]:
        """Benchmark impact of different cache memory limits.
        
        Args:
            test_files: Dictionary of test files
            memory_limits_mb: List of memory limits to test
            
        Returns:
            List of benchmark results for each memory limit
        """
        results = []
        
        for limit_mb in memory_limits_mb:
            FRAME_CACHE["memory_limit_mb"] = limit_mb
            reset_frame_cache()
            
            limit_results = {
                "memory_limit_mb": limit_mb,
                "files": {}
            }
            
            # Test each file
            for name, path in test_files.items():
                result = self.benchmark_single_file(path, iterations=5)
                limit_results["files"][name] = result
            
            # Calculate aggregate metrics
            cache = get_frame_cache()
            final_stats = cache.get_stats()
            
            limit_results["total_evictions"] = final_stats.evictions
            limit_results["final_memory_mb"] = final_stats.memory_bytes / (1024 * 1024)
            
            results.append(limit_results)
        
        return results
    
    def benchmark_concurrent_access(
        self,
        gif_path: Path,
        num_threads: int = 10,
        iterations_per_thread: int = 5
    ) -> Dict:
        """Benchmark concurrent cache access.
        
        Args:
            gif_path: Path to GIF file
            num_threads: Number of concurrent threads
            iterations_per_thread: Iterations per thread
            
        Returns:
            Dictionary with concurrency benchmark results
        """
        import concurrent.futures
        import threading
        
        reset_frame_cache()
        thread_times = {}
        thread_lock = threading.Lock()
        
        def worker(thread_id: int):
            times = []
            for _ in range(iterations_per_thread):
                start = time.perf_counter()
                extract_gif_frames(gif_path, max_frames=30)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            
            with thread_lock:
                thread_times[thread_id] = times
        
        # Run concurrent extractions
        start_concurrent = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(worker, i)
                for i in range(num_threads)
            ]
            concurrent.futures.wait(futures)
        total_concurrent_time = time.perf_counter() - start_concurrent
        
        # Calculate statistics
        all_times = []
        for times in thread_times.values():
            all_times.extend(times)
        
        cache = get_frame_cache()
        cache_stats = cache.get_stats()
        
        # Compare with sequential execution
        reset_frame_cache()
        start_sequential = time.perf_counter()
        for _ in range(num_threads * iterations_per_thread):
            extract_gif_frames(gif_path, max_frames=30)
        total_sequential_time = time.perf_counter() - start_sequential
        
        return {
            "num_threads": num_threads,
            "iterations_per_thread": iterations_per_thread,
            "total_operations": num_threads * iterations_per_thread,
            "concurrent_total_time": total_concurrent_time,
            "sequential_total_time": total_sequential_time,
            "concurrency_speedup": total_sequential_time / total_concurrent_time,
            "concurrent_mean_time": statistics.mean(all_times),
            "concurrent_median_time": statistics.median(all_times),
            "cache_hits": cache_stats.hits,
            "cache_misses": cache_stats.misses,
            "cache_hit_rate": cache_stats.hit_rate,
        }
    
    def benchmark_cache_warming(
        self,
        test_files: List[Path],
        max_frames: int = 30
    ) -> Dict:
        """Benchmark cache warming performance.
        
        Args:
            test_files: List of GIF files to warm
            max_frames: Maximum frames per file
            
        Returns:
            Dictionary with warming benchmark results
        """
        reset_frame_cache()
        cache = get_frame_cache()
        
        # Time cache warming
        start_warm = time.perf_counter()
        cache.warm_cache(test_files, max_frames=max_frames)
        warm_time = time.perf_counter() - start_warm
        
        stats_after_warm = cache.get_stats()
        
        # Time subsequent accesses (should all be hits)
        hit_times = []
        for path in test_files:
            start = time.perf_counter()
            extract_gif_frames(path, max_frames=max_frames)
            hit_times.append(time.perf_counter() - start)
        
        stats_final = cache.get_stats()
        
        return {
            "num_files": len(test_files),
            "warm_time": warm_time,
            "warm_time_per_file": warm_time / len(test_files),
            "hit_times_mean": statistics.mean(hit_times),
            "hit_times_median": statistics.median(hit_times),
            "entries_after_warm": stats_after_warm.disk_entries,
            "memory_after_warm_mb": stats_after_warm.memory_bytes / (1024 * 1024),
            "final_hit_rate": stats_final.hit_rate,
        }
    
    def run_full_benchmark(self, temp_dir: Path) -> None:
        """Run complete benchmark suite.
        
        Args:
            temp_dir: Temporary directory for test files
        """
        print("🚀 Starting Frame Cache Performance Benchmark")
        print("=" * 50)
        
        # Create test files
        print("\n📁 Creating test files...")
        test_files = self.create_test_gifs(temp_dir / "test_gifs")
        
        # Benchmark individual files
        print("\n📊 Benchmarking individual files...")
        file_results = {}
        for name, path in test_files.items():
            print(f"  Testing {name}...")
            result = self.benchmark_single_file(path, iterations=10, max_frames=30)
            file_results[name] = result
            print(f"    Speedup: {result['speedup']:.2f}x")
            print(f"    Cached: {result['cached_mean']*1000:.2f}ms")
            print(f"    Uncached: {result['uncached_mean']*1000:.2f}ms")
        
        self.results["individual_files"] = file_results
        
        # Benchmark memory limits
        print("\n💾 Benchmarking memory limit impact...")
        memory_results = self.benchmark_cache_size_impact(
            test_files,
            memory_limits_mb=[10, 50, 100, 500]
        )
        self.results["memory_limits"] = memory_results
        
        for result in memory_results:
            print(f"  {result['memory_limit_mb']}MB limit:")
            print(f"    Evictions: {result['total_evictions']}")
            print(f"    Final memory: {result['final_memory_mb']:.1f}MB")
        
        # Benchmark concurrent access
        print("\n🔄 Benchmarking concurrent access...")
        concurrent_result = self.benchmark_concurrent_access(
            test_files["medium"],
            num_threads=10,
            iterations_per_thread=5
        )
        self.results["concurrent_access"] = concurrent_result
        print(f"  Concurrency speedup: {concurrent_result['concurrency_speedup']:.2f}x")
        print(f"  Cache hit rate: {concurrent_result['cache_hit_rate']:.1%}")
        
        # Benchmark cache warming
        print("\n🔥 Benchmarking cache warming...")
        warm_result = self.benchmark_cache_warming(
            list(test_files.values()),
            max_frames=30
        )
        self.results["cache_warming"] = warm_result
        print(f"  Warm time: {warm_result['warm_time']:.2f}s")
        print(f"  Per file: {warm_result['warm_time_per_file']:.3f}s")
        print(f"  Hit time: {warm_result['hit_times_mean']*1000:.2f}ms")
        
        # Save results
        results_file = self.output_dir / "frame_cache_benchmark.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✅ Benchmark complete! Results saved to {results_file}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 50)
        print("📈 BENCHMARK SUMMARY")
        print("=" * 50)
        
        if "individual_files" in self.results:
            print("\n🎯 Cache Performance Improvements:")
            for name, result in self.results["individual_files"].items():
                speedup = result["speedup"]
                cached_ms = result["cached_mean"] * 1000
                uncached_ms = result["uncached_mean"] * 1000
                print(f"  {name:10s}: {speedup:5.1f}x speedup "
                      f"({uncached_ms:6.1f}ms → {cached_ms:6.1f}ms)")
        
        if "concurrent_access" in self.results:
            concurrent = self.results["concurrent_access"]
            print(f"\n🔄 Concurrent Performance:")
            print(f"  Concurrency speedup: {concurrent['concurrency_speedup']:.2f}x")
            print(f"  Cache hit rate: {concurrent['cache_hit_rate']:.1%}")
        
        if "cache_warming" in self.results:
            warm = self.results["cache_warming"]
            print(f"\n🔥 Cache Warming:")
            print(f"  Files warmed: {warm['num_files']}")
            print(f"  Time per file: {warm['warm_time_per_file']*1000:.1f}ms")
            print(f"  Subsequent access: {warm['hit_times_mean']*1000:.1f}ms")
        
        # Overall assessment
        print("\n✨ Overall Assessment:")
        
        avg_speedup = statistics.mean([
            r["speedup"] for r in self.results.get("individual_files", {}).values()
        ])
        
        if avg_speedup > 50:
            print(f"  🚀 EXCELLENT: {avg_speedup:.1f}x average speedup!")
        elif avg_speedup > 20:
            print(f"  💚 GREAT: {avg_speedup:.1f}x average speedup!")
        elif avg_speedup > 10:
            print(f"  🟢 GOOD: {avg_speedup:.1f}x average speedup")
        elif avg_speedup > 5:
            print(f"  🟡 MODERATE: {avg_speedup:.1f}x average speedup")
        else:
            print(f"  🔴 LIMITED: {avg_speedup:.1f}x average speedup")
        
        print("\n  Frame caching provides significant performance improvements,")
        print("  especially for repeated validations and large GIF files.")


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark frame cache performance"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--temp",
        type=Path,
        default=Path("/tmp/giflab_benchmark"),
        help="Temporary directory for test files"
    )
    
    args = parser.parse_args()
    
    # Ensure cache is enabled
    FRAME_CACHE["enabled"] = True
    
    # Run benchmark
    benchmark = FrameCacheBenchmark(args.output)
    benchmark.run_full_benchmark(args.temp)


if __name__ == "__main__":
    main()