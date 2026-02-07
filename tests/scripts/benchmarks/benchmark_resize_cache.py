#!/usr/bin/env python
"""Performance benchmarks for ResizedFrameCache.

This script benchmarks the performance improvements from using the ResizedFrameCache
across various scenarios including different frame sizes, interpolation methods,
and usage patterns.

Usage:
    poetry run python tests/performance/benchmark_resize_cache.py
"""

import time
import statistics
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from giflab.caching.resized_frame_cache import (
    ResizedFrameCache,
    get_resize_cache,
    resize_frame_cached,
)
from giflab.metrics import calculate_ms_ssim, _resize_if_needed

console = Console()


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    uncached_time: float
    cached_time: float
    speedup: float
    cache_hit_rate: float
    memory_mb: float


class ResizeCacheBenchmark:
    """Benchmark suite for resize cache performance."""
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.results: List[BenchmarkResult] = []
        
    def setup(self):
        """Setup for benchmarks."""
        # Clear any existing cache
        cache = get_resize_cache()
        cache.clear()
        console.print("[yellow]Cache cleared for benchmarking[/yellow]")
    
    def generate_test_frames(self, size: Tuple[int, int], count: int = 10) -> List[np.ndarray]:
        """Generate test frames of specified size."""
        h, w = size
        return [
            np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            for _ in range(count)
        ]
    
    def benchmark_single_resize(self, name: str, frame_size: Tuple[int, int], 
                               target_size: Tuple[int, int], iterations: int = 100):
        """Benchmark single frame resize operation."""
        console.print(f"\n[bold]Benchmarking: {name}[/bold]")
        
        frame = self.generate_test_frames(frame_size, 1)[0]
        cache = get_resize_cache()
        cache.clear()
        
        # Benchmark without cache
        uncached_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            uncached_times.append(time.perf_counter() - start)
        
        uncached_avg = statistics.mean(uncached_times)
        
        # Prime the cache
        resize_frame_cached(frame, target_size, cv2.INTER_AREA, use_cache=True)
        
        # Benchmark with cache
        cached_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            resize_frame_cached(frame, target_size, cv2.INTER_AREA, use_cache=True)
            cached_times.append(time.perf_counter() - start)
        
        cached_avg = statistics.mean(cached_times)
        
        stats = cache.get_stats()
        
        result = BenchmarkResult(
            name=name,
            uncached_time=uncached_avg * 1000,  # Convert to ms
            cached_time=cached_avg * 1000,
            speedup=uncached_avg / cached_avg if cached_avg > 0 else 0,
            cache_hit_rate=stats["hit_rate"],
            memory_mb=stats["memory_mb"]
        )
        
        self.results.append(result)
        console.print(f"  Speedup: [green]{result.speedup:.1f}x[/green]")
    
    def benchmark_multiple_sizes(self):
        """Benchmark resizing to multiple target sizes."""
        console.print("\n[bold]Benchmarking: Multiple Target Sizes[/bold]")
        
        frame = self.generate_test_frames((800, 800), 1)[0]
        target_sizes = [(400, 400), (200, 200), (100, 100), (50, 50)]
        
        cache = get_resize_cache()
        cache.clear()
        
        # Benchmark without cache
        start = time.perf_counter()
        for _ in range(20):
            for size in target_sizes:
                cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        uncached_time = time.perf_counter() - start
        
        # Prime the cache
        for size in target_sizes:
            resize_frame_cached(frame, size, cv2.INTER_AREA, use_cache=True)
        
        # Benchmark with cache
        start = time.perf_counter()
        for _ in range(20):
            for size in target_sizes:
                resize_frame_cached(frame, size, cv2.INTER_AREA, use_cache=True)
        cached_time = time.perf_counter() - start
        
        stats = cache.get_stats()
        
        result = BenchmarkResult(
            name="Multiple Sizes (4 targets)",
            uncached_time=uncached_time * 1000,
            cached_time=cached_time * 1000,
            speedup=uncached_time / cached_time if cached_time > 0 else 0,
            cache_hit_rate=stats["hit_rate"],
            memory_mb=stats["memory_mb"]
        )
        
        self.results.append(result)
        console.print(f"  Speedup: [green]{result.speedup:.1f}x[/green]")
    
    def benchmark_ms_ssim(self):
        """Benchmark MS-SSIM calculation with resize cache."""
        console.print("\n[bold]Benchmarking: MS-SSIM with Resize Cache[/bold]")
        
        frame1 = self.generate_test_frames((512, 512), 1)[0]
        frame2 = self.generate_test_frames((512, 512), 1)[0]
        
        cache = get_resize_cache()
        cache.clear()
        
        # Benchmark without cache
        start = time.perf_counter()
        for _ in range(10):
            calculate_ms_ssim(frame1, frame2, scales=4, use_cache=False)
        uncached_time = time.perf_counter() - start
        
        cache.clear()
        
        # Benchmark with cache (first run primes it)
        start = time.perf_counter()
        for _ in range(10):
            calculate_ms_ssim(frame1, frame2, scales=4, use_cache=True)
        cached_time = time.perf_counter() - start
        
        stats = cache.get_stats()
        
        result = BenchmarkResult(
            name="MS-SSIM (4 scales)",
            uncached_time=uncached_time * 1000,
            cached_time=cached_time * 1000,
            speedup=uncached_time / cached_time if cached_time > 0 else 0,
            cache_hit_rate=stats["hit_rate"],
            memory_mb=stats["memory_mb"]
        )
        
        self.results.append(result)
        console.print(f"  Speedup: [green]{result.speedup:.1f}x[/green]")
    
    def benchmark_buffer_pooling(self):
        """Benchmark buffer pooling efficiency."""
        console.print("\n[bold]Benchmarking: Buffer Pooling[/bold]")
        
        # Test with pooling disabled
        cache_no_pool = ResizedFrameCache(memory_limit_mb=50, enable_pooling=False)
        frames = self.generate_test_frames((300, 300), 50)
        
        start = time.perf_counter()
        for frame in frames:
            cache_no_pool.get(frame, (150, 150), cv2.INTER_AREA)
        time_no_pool = time.perf_counter() - start
        
        # Test with pooling enabled
        cache_with_pool = ResizedFrameCache(memory_limit_mb=50, enable_pooling=True)
        
        start = time.perf_counter()
        for frame in frames:
            cache_with_pool.get(frame, (150, 150), cv2.INTER_AREA)
        time_with_pool = time.perf_counter() - start
        
        pool_stats = cache_with_pool._buffer_pool.get_stats() if cache_with_pool._buffer_pool else {}
        
        result = BenchmarkResult(
            name="Buffer Pooling (50 frames)",
            uncached_time=time_no_pool * 1000,
            cached_time=time_with_pool * 1000,
            speedup=time_no_pool / time_with_pool if time_with_pool > 0 else 0,
            cache_hit_rate=pool_stats.get("reuse_rate", 0),
            memory_mb=pool_stats.get("total_memory_mb", 0)
        )
        
        self.results.append(result)
        console.print(f"  Improvement: [green]{result.speedup:.1f}x[/green]")
        if pool_stats:
            console.print(f"  Buffer reuse rate: [cyan]{pool_stats['reuse_rate']:.1%}[/cyan]")
    
    def benchmark_interpolation_methods(self):
        """Benchmark different interpolation methods."""
        console.print("\n[bold]Benchmarking: Interpolation Methods[/bold]")
        
        frame = self.generate_test_frames((600, 600), 1)[0]
        target_size = (300, 300)
        methods = [
            (cv2.INTER_AREA, "AREA"),
            (cv2.INTER_LINEAR, "LINEAR"),
            (cv2.INTER_CUBIC, "CUBIC"),
            (cv2.INTER_LANCZOS4, "LANCZOS4"),
        ]
        
        for interp, name in methods:
            cache = get_resize_cache()
            cache.clear()
            
            # Uncached
            start = time.perf_counter()
            for _ in range(50):
                cv2.resize(frame, target_size, interpolation=interp)
            uncached_time = time.perf_counter() - start
            
            # Prime cache
            resize_frame_cached(frame, target_size, interp, use_cache=True)
            
            # Cached
            start = time.perf_counter()
            for _ in range(50):
                resize_frame_cached(frame, target_size, interp, use_cache=True)
            cached_time = time.perf_counter() - start
            
            stats = cache.get_stats()
            
            result = BenchmarkResult(
                name=f"Interpolation: {name}",
                uncached_time=uncached_time * 1000,
                cached_time=cached_time * 1000,
                speedup=uncached_time / cached_time if cached_time > 0 else 0,
                cache_hit_rate=stats["hit_rate"],
                memory_mb=stats["memory_mb"]
            )
            
            self.results.append(result)
    
    def print_results(self):
        """Print benchmark results in a formatted table."""
        console.print("\n[bold cyan]Resize Cache Performance Benchmark Results[/bold cyan]")
        console.print("=" * 80)
        
        table = Table(title="Performance Metrics")
        table.add_column("Benchmark", style="cyan")
        table.add_column("Uncached (ms)", justify="right", style="red")
        table.add_column("Cached (ms)", justify="right", style="green")
        table.add_column("Speedup", justify="right", style="bold green")
        table.add_column("Hit Rate", justify="right", style="yellow")
        table.add_column("Memory (MB)", justify="right", style="blue")
        
        for result in self.results:
            table.add_row(
                result.name,
                f"{result.uncached_time:.2f}",
                f"{result.cached_time:.2f}",
                f"{result.speedup:.1f}x",
                f"{result.cache_hit_rate:.1%}",
                f"{result.memory_mb:.1f}"
            )
        
        console.print(table)
        
        # Summary statistics
        avg_speedup = statistics.mean(r.speedup for r in self.results)
        max_speedup = max(r.speedup for r in self.results)
        min_speedup = min(r.speedup for r in self.results)
        
        console.print("\n[bold]Summary:[/bold]")
        console.print(f"  Average Speedup: [green]{avg_speedup:.1f}x[/green]")
        console.print(f"  Maximum Speedup: [green]{max_speedup:.1f}x[/green]")
        console.print(f"  Minimum Speedup: [yellow]{min_speedup:.1f}x[/yellow]")
        
        # Performance recommendations
        console.print("\n[bold]Performance Insights:[/bold]")
        if avg_speedup > 10:
            console.print("  ✅ [green]Excellent cache performance - significant speedup achieved[/green]")
        elif avg_speedup > 5:
            console.print("  ✅ [green]Good cache performance - notable speedup achieved[/green]")
        elif avg_speedup > 2:
            console.print("  🟡 [yellow]Moderate cache performance - some benefit from caching[/yellow]")
        else:
            console.print("  🔴 [red]Limited cache benefit - consider profiling usage patterns[/red]")
        
        # Memory efficiency
        total_memory = sum(r.memory_mb for r in self.results)
        if total_memory < 50:
            console.print("  ✅ [green]Efficient memory usage[/green]")
        elif total_memory < 200:
            console.print("  🟡 [yellow]Moderate memory usage[/yellow]")
        else:
            console.print("  🔴 [red]High memory usage - consider adjusting cache limits[/red]")


def main():
    """Run the benchmark suite."""
    console.print("[bold cyan]Starting Resize Cache Performance Benchmarks[/bold cyan]")
    console.print("This will take a few minutes to complete...\n")
    
    benchmark = ResizeCacheBenchmark()
    benchmark.setup()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Run benchmarks
        task = progress.add_task("[cyan]Running benchmarks...", total=7)
        
        benchmark.benchmark_single_resize("Small frames (200x200 → 100x100)", 
                                         (200, 200), (100, 100))
        progress.update(task, advance=1)
        
        benchmark.benchmark_single_resize("Medium frames (600x600 → 300x300)", 
                                         (600, 600), (300, 300))
        progress.update(task, advance=1)
        
        benchmark.benchmark_single_resize("Large frames (1200x1200 → 600x600)", 
                                         (1200, 1200), (600, 600))
        progress.update(task, advance=1)
        
        benchmark.benchmark_multiple_sizes()
        progress.update(task, advance=1)
        
        benchmark.benchmark_ms_ssim()
        progress.update(task, advance=1)
        
        benchmark.benchmark_buffer_pooling()
        progress.update(task, advance=1)
        
        benchmark.benchmark_interpolation_methods()
        progress.update(task, advance=1)
    
    # Print results
    benchmark.print_results()
    
    console.print("\n[bold green]Benchmark complete![/bold green]")


if __name__ == "__main__":
    main()