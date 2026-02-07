#!/usr/bin/env python3
"""Comprehensive Performance Benchmark Suite for Phase 3 Optimizations.

This module provides exhaustive performance testing covering:
1. Model caching efficiency (Phase 1)
2. Parallel processing scalability (Phase 2.1)
3. Conditional processing effectiveness (Phase 4)

Benchmark scenarios cover diverse GIF characteristics:
- Size: Small (10 frames), Medium (30-50 frames), Large (100+ frames)
- Quality: High (>35dB PSNR), Medium (25-35dB), Low (<25dB)
- Content: Text/UI, Gradients, Animations, Static
- Resolution: 100x100 to 1920x1080
"""

import gc
import json
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import psutil
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from giflab.metrics import calculate_comprehensive_metrics
from giflab.conditional_metrics import ConditionalMetricsCalculator
from giflab.parallel_metrics import ParallelMetricsCalculator
from giflab.model_cache import LPIPSModelCache
from giflab.ssimulacra2_metrics import Ssimulacra2Validator
from giflab.text_ui_validation import TextUIContentDetector


@dataclass
class BenchmarkScenario:
    """Definition of a benchmark test scenario."""
    name: str
    description: str
    frame_count: int
    resolution: Tuple[int, int]
    quality_level: str  # 'high', 'medium', 'low'
    content_type: str  # 'text', 'gradient', 'animation', 'static', 'mixed'
    expected_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    scenario: BenchmarkScenario
    execution_time: float
    memory_peak: float
    memory_delta: float
    metrics_calculated: List[str]
    metrics_skipped: List[str]
    cache_hits: int
    cache_misses: int
    parallel_speedup: Optional[float] = None
    conditional_speedup: Optional[float] = None
    errors: List[str] = field(default_factory=list)


class ComprehensiveBenchmarkSuite:
    """Comprehensive benchmark suite for Phase 3 optimizations."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize benchmark suite.
        
        Args:
            output_dir: Directory for benchmark results and test data
        """
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        
        self.scenarios = self._define_scenarios()
        self.results: List[BenchmarkResult] = []
        
    def _define_scenarios(self) -> List[BenchmarkScenario]:
        """Define comprehensive benchmark scenarios."""
        scenarios = []
        
        # Small GIF scenarios (test overhead)
        scenarios.extend([
            BenchmarkScenario(
                name="small_high_quality_static",
                description="Small static GIF with high quality - tests conditional skip efficiency",
                frame_count=10,
                resolution=(100, 100),
                quality_level="high",
                content_type="static",
                expected_metrics={"psnr": 40.0, "ssim": 0.98}
            ),
            BenchmarkScenario(
                name="small_text_ui",
                description="Small GIF with text/UI content - tests text detection",
                frame_count=10,
                resolution=(200, 200),
                quality_level="medium",
                content_type="text",
                expected_metrics={"edge_density": 0.15, "text_regions": 3}
            ),
            BenchmarkScenario(
                name="small_animated_low_quality",
                description="Small animated GIF with low quality - tests full metric suite",
                frame_count=15,
                resolution=(150, 150),
                quality_level="low",
                content_type="animation",
                expected_metrics={"psnr": 20.0, "temporal_change": 0.3}
            ),
        ])
        
        # Medium GIF scenarios (typical usage)
        scenarios.extend([
            BenchmarkScenario(
                name="medium_gradient_high_quality",
                description="Medium GIF with gradients - tests gradient detection",
                frame_count=30,
                resolution=(500, 500),
                quality_level="high",
                content_type="gradient",
                expected_metrics={"gradient_magnitude": 0.8, "psnr": 38.0}
            ),
            BenchmarkScenario(
                name="medium_mixed_content",
                description="Medium GIF with mixed content - tests balanced optimization",
                frame_count=40,
                resolution=(640, 480),
                quality_level="medium",
                content_type="mixed",
                expected_metrics={"complexity_score": 0.6}
            ),
            BenchmarkScenario(
                name="medium_animated_text",
                description="Medium animated text GIF - tests combined optimizations",
                frame_count=50,
                resolution=(800, 600),
                quality_level="medium",
                content_type="text",
                expected_metrics={"temporal_consistency": 0.85}
            ),
        ])
        
        # Large GIF scenarios (test scalability)
        scenarios.extend([
            BenchmarkScenario(
                name="large_hd_animation",
                description="Large HD animated GIF - tests parallel processing benefits",
                frame_count=100,
                resolution=(1920, 1080),
                quality_level="medium",
                content_type="animation",
                expected_metrics={"frame_differences": 0.25}
            ),
            BenchmarkScenario(
                name="large_4k_static",
                description="Large 4K static GIF - tests memory efficiency",
                frame_count=120,
                resolution=(3840, 2160),
                quality_level="high",
                content_type="static",
                expected_metrics={"psnr": 42.0}
            ),
            BenchmarkScenario(
                name="large_low_quality_mixed",
                description="Large low quality mixed content - stress test",
                frame_count=150,
                resolution=(1280, 720),
                quality_level="low",
                content_type="mixed",
                expected_metrics={"ssimulacra2": 30.0}
            ),
        ])
        
        # Edge cases
        scenarios.extend([
            BenchmarkScenario(
                name="edge_single_frame",
                description="Single frame GIF - tests minimum overhead",
                frame_count=1,
                resolution=(256, 256),
                quality_level="medium",
                content_type="static"
            ),
            BenchmarkScenario(
                name="edge_tiny_resolution",
                description="Tiny resolution GIF - tests small data handling",
                frame_count=20,
                resolution=(32, 32),
                quality_level="low",
                content_type="animation"
            ),
            BenchmarkScenario(
                name="edge_extreme_frames",
                description="Extreme frame count - tests scalability limits",
                frame_count=500,
                resolution=(320, 240),
                quality_level="medium",
                content_type="animation"
            ),
        ])
        
        return scenarios
    
    def _generate_test_gif(self, scenario: BenchmarkScenario) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Generate synthetic test GIF frames based on scenario.
        
        Args:
            scenario: Benchmark scenario definition
            
        Returns:
            Tuple of (original_frames, compressed_frames)
        """
        width, height = scenario.resolution
        frames_orig = []
        frames_comp = []
        
        np.random.seed(42)  # Deterministic generation
        
        for i in range(scenario.frame_count):
            # Generate original frame based on content type
            if scenario.content_type == "static":
                # Static content - same base image
                if i == 0:
                    base_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                frame = base_image.copy()
                
            elif scenario.content_type == "gradient":
                # Gradient patterns
                x_grad = np.linspace(0, 255, width)
                y_grad = np.linspace(0, 255, height)
                xx, yy = np.meshgrid(x_grad, y_grad)
                frame = np.stack([
                    (xx * np.sin(i * 0.1) % 256).astype(np.uint8),
                    (yy * np.cos(i * 0.1) % 256).astype(np.uint8),
                    ((xx + yy) * 0.5 % 256).astype(np.uint8)
                ], axis=-1)
                
            elif scenario.content_type == "text":
                # Text-like patterns with sharp edges
                frame = np.ones((height, width, 3), dtype=np.uint8) * 255
                # Add text-like rectangles
                for j in range(5):
                    y = 20 + j * 30
                    x = 20 + (i * 5) % 100
                    frame[y:y+15, x:x+80] = 0
                    
            elif scenario.content_type == "animation":
                # Animated content with motion
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                # Moving circle
                center_x = int(width * (0.5 + 0.3 * np.sin(i * 0.2)))
                center_y = int(height * (0.5 + 0.3 * np.cos(i * 0.2)))
                y, x = np.ogrid[:height, :width]
                mask = (x - center_x)**2 + (y - center_y)**2 <= (min(width, height) // 4)**2
                frame[mask] = [255, 100, 100]
                
            else:  # mixed
                # Combination of different elements
                frame = np.random.randint(100, 200, (height, width, 3), dtype=np.uint8)
                # Add some structure
                frame[height//4:3*height//4, width//4:3*width//4] = 255
                
            frames_orig.append(frame)
            
            # Generate compressed version based on quality level
            if scenario.quality_level == "high":
                # Minimal degradation
                noise = np.random.normal(0, 3, frame.shape)
                compressed = np.clip(frame + noise, 0, 255).astype(np.uint8)
                
            elif scenario.quality_level == "medium":
                # Moderate degradation
                noise = np.random.normal(0, 10, frame.shape)
                compressed = np.clip(frame + noise, 0, 255).astype(np.uint8)
                # Slight color quantization
                compressed = (compressed // 8) * 8
                
            else:  # low
                # Heavy degradation
                noise = np.random.normal(0, 25, frame.shape)
                compressed = np.clip(frame + noise, 0, 255).astype(np.uint8)
                # Heavy quantization
                compressed = (compressed // 32) * 32
                
            frames_comp.append(compressed)
            
        return frames_orig, frames_comp
    
    def run_baseline_benchmark(self, scenario: BenchmarkScenario) -> BenchmarkResult:
        """Run baseline benchmark without optimizations.
        
        Args:
            scenario: Benchmark scenario to run
            
        Returns:
            Benchmark results
        """
        # Generate test data
        frames_orig, frames_comp = self._generate_test_gif(scenario)
        
        # Force cleanup before benchmark
        gc.collect()
        
        # Start monitoring
        start_time = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        
        try:
            # Disable all optimizations
            os.environ['GIFLAB_ENABLE_PARALLEL_METRICS'] = 'false'
            os.environ['GIFLAB_ENABLE_CONDITIONAL_METRICS'] = 'false'
            os.environ['GIFLAB_USE_MODEL_CACHE'] = 'false'
            
            # Run metrics calculation
            metrics = calculate_comprehensive_metrics(
                original_frames=frames_orig,
                compressed_frames=frames_comp
            )
            
            metrics_calculated = list(metrics.keys())
            metrics_skipped = []
            
        except Exception as e:
            metrics_calculated = []
            metrics_skipped = []
            errors = [f"Baseline error: {str(e)}"]
        else:
            errors = []
            
        # Stop monitoring
        end_time = time.perf_counter()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        execution_time = end_time - start_time
        memory_delta = current_memory - start_memory
        
        return BenchmarkResult(
            scenario=scenario,
            execution_time=execution_time,
            memory_peak=peak_memory / 1024 / 1024,  # Convert to MB
            memory_delta=memory_delta / 1024 / 1024,
            metrics_calculated=metrics_calculated,
            metrics_skipped=metrics_skipped,
            cache_hits=0,
            cache_misses=0,
            errors=errors
        )
    
    def run_optimized_benchmark(self, scenario: BenchmarkScenario) -> BenchmarkResult:
        """Run benchmark with all optimizations enabled.
        
        Args:
            scenario: Benchmark scenario to run
            
        Returns:
            Benchmark results
        """
        # Generate test data
        frames_orig, frames_comp = self._generate_test_gif(scenario)
        
        # Force cleanup before benchmark
        gc.collect()
        
        # Clear model cache
        cache = LPIPSModelCache()
        cache.clear(force=True)
        
        # Start monitoring
        start_time = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        
        try:
            # Enable all optimizations
            os.environ['GIFLAB_ENABLE_PARALLEL_METRICS'] = 'true'
            os.environ['GIFLAB_ENABLE_CONDITIONAL_METRICS'] = 'true'
            os.environ['GIFLAB_USE_MODEL_CACHE'] = 'true'
            os.environ['GIFLAB_MAX_PARALLEL_WORKERS'] = str(psutil.cpu_count())
            
            # Run metrics calculation
            metrics = calculate_comprehensive_metrics(
                original_frames=frames_orig,
                compressed_frames=frames_comp
            )
            
            # Check what was calculated vs skipped
            conditional_calc = ConditionalMetricsCalculator()
            quality_tier = conditional_calc._assess_quality(frames_orig[:5], frames_comp[:5])
            
            if quality_tier == "HIGH":
                metrics_skipped = ["lpips", "ssimulacra2", "deep_features"]
                metrics_calculated = [k for k in metrics.keys() if k not in metrics_skipped]
            else:
                metrics_calculated = list(metrics.keys())
                metrics_skipped = []
                
            # Get cache statistics
            cache_info = cache.get_model_cache_info()
            cache_hits = cache_info.get("cache_hits", 0)
            cache_misses = cache_info.get("cache_misses", 0)
            
        except Exception as e:
            metrics_calculated = []
            metrics_skipped = []
            cache_hits = 0
            cache_misses = 0
            errors = [f"Optimized error: {str(e)}"]
        else:
            errors = []
            
        # Stop monitoring
        end_time = time.perf_counter()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        execution_time = end_time - start_time
        memory_delta = current_memory - start_memory
        
        return BenchmarkResult(
            scenario=scenario,
            execution_time=execution_time,
            memory_peak=peak_memory / 1024 / 1024,
            memory_delta=memory_delta / 1024 / 1024,
            metrics_calculated=metrics_calculated,
            metrics_skipped=metrics_skipped,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            errors=errors
        )
    
    def run_parallel_only_benchmark(self, scenario: BenchmarkScenario) -> BenchmarkResult:
        """Run benchmark with only parallel processing enabled.
        
        Args:
            scenario: Benchmark scenario to run
            
        Returns:
            Benchmark results
        """
        # Generate test data
        frames_orig, frames_comp = self._generate_test_gif(scenario)
        
        # Force cleanup
        gc.collect()
        
        # Start monitoring
        start_time = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        
        try:
            # Enable only parallel processing
            os.environ['GIFLAB_ENABLE_PARALLEL_METRICS'] = 'true'
            os.environ['GIFLAB_ENABLE_CONDITIONAL_METRICS'] = 'false'
            os.environ['GIFLAB_USE_MODEL_CACHE'] = 'true'  # Keep cache for fair comparison
            
            metrics = calculate_comprehensive_metrics(
                original_frames=frames_orig,
                compressed_frames=frames_comp
            )
            
            metrics_calculated = list(metrics.keys())
            metrics_skipped = []
            
        except Exception as e:
            metrics_calculated = []
            metrics_skipped = []
            errors = [f"Parallel-only error: {str(e)}"]
        else:
            errors = []
            
        # Stop monitoring
        end_time = time.perf_counter()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        execution_time = end_time - start_time
        memory_delta = current_memory - start_memory
        
        return BenchmarkResult(
            scenario=scenario,
            execution_time=execution_time,
            memory_peak=peak_memory / 1024 / 1024,
            memory_delta=memory_delta / 1024 / 1024,
            metrics_calculated=metrics_calculated,
            metrics_skipped=metrics_skipped,
            cache_hits=0,
            cache_misses=0,
            errors=errors
        )
    
    def run_conditional_only_benchmark(self, scenario: BenchmarkScenario) -> BenchmarkResult:
        """Run benchmark with only conditional processing enabled.
        
        Args:
            scenario: Benchmark scenario to run
            
        Returns:
            Benchmark results
        """
        # Generate test data
        frames_orig, frames_comp = self._generate_test_gif(scenario)
        
        # Force cleanup
        gc.collect()
        
        # Start monitoring
        start_time = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        
        try:
            # Enable only conditional processing
            os.environ['GIFLAB_ENABLE_PARALLEL_METRICS'] = 'false'
            os.environ['GIFLAB_ENABLE_CONDITIONAL_METRICS'] = 'true'
            os.environ['GIFLAB_USE_MODEL_CACHE'] = 'true'  # Keep cache for fair comparison
            
            metrics = calculate_comprehensive_metrics(
                original_frames=frames_orig,
                compressed_frames=frames_comp
            )
            
            # Track what was skipped
            conditional_calc = ConditionalMetricsCalculator()
            quality_tier = conditional_calc._assess_quality(frames_orig[:5], frames_comp[:5])
            
            if quality_tier == "HIGH":
                metrics_skipped = ["lpips", "ssimulacra2", "deep_features"]
                metrics_calculated = [k for k in metrics.keys() if k not in metrics_skipped]
            else:
                metrics_calculated = list(metrics.keys())
                metrics_skipped = []
                
        except Exception as e:
            metrics_calculated = []
            metrics_skipped = []
            errors = [f"Conditional-only error: {str(e)}"]
        else:
            errors = []
            
        # Stop monitoring
        end_time = time.perf_counter()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        execution_time = end_time - start_time
        memory_delta = current_memory - start_memory
        
        return BenchmarkResult(
            scenario=scenario,
            execution_time=execution_time,
            memory_peak=peak_memory / 1024 / 1024,
            memory_delta=memory_delta / 1024 / 1024,
            metrics_calculated=metrics_calculated,
            metrics_skipped=metrics_skipped,
            cache_hits=0,
            cache_misses=0,
            errors=errors
        )
    
    def run_comprehensive_benchmark(self) -> Dict:
        """Run complete benchmark suite across all scenarios.
        
        Returns:
            Dictionary containing all benchmark results and analysis
        """
        print("=" * 80)
        print("COMPREHENSIVE PERFORMANCE BENCHMARK SUITE")
        print("=" * 80)
        
        all_results = {
            "baseline": [],
            "parallel_only": [],
            "conditional_only": [],
            "fully_optimized": [],
            "comparisons": []
        }
        
        for i, scenario in enumerate(self.scenarios, 1):
            print(f"\n[{i}/{len(self.scenarios)}] Running scenario: {scenario.name}")
            print(f"  Description: {scenario.description}")
            print(f"  Frames: {scenario.frame_count}, Resolution: {scenario.resolution}")
            print(f"  Quality: {scenario.quality_level}, Content: {scenario.content_type}")
            
            # Run baseline
            print("  - Running baseline...")
            baseline_result = self.run_baseline_benchmark(scenario)
            all_results["baseline"].append(baseline_result)
            
            # Run parallel only
            print("  - Running parallel-only...")
            parallel_result = self.run_parallel_only_benchmark(scenario)
            all_results["parallel_only"].append(parallel_result)
            
            # Run conditional only
            print("  - Running conditional-only...")
            conditional_result = self.run_conditional_only_benchmark(scenario)
            all_results["conditional_only"].append(conditional_result)
            
            # Run fully optimized
            print("  - Running fully optimized...")
            optimized_result = self.run_optimized_benchmark(scenario)
            all_results["fully_optimized"].append(optimized_result)
            
            # Calculate speedups
            if baseline_result.execution_time > 0:
                parallel_speedup = baseline_result.execution_time / parallel_result.execution_time
                conditional_speedup = baseline_result.execution_time / conditional_result.execution_time
                full_speedup = baseline_result.execution_time / optimized_result.execution_time
                
                comparison = {
                    "scenario": scenario.name,
                    "baseline_time": baseline_result.execution_time,
                    "parallel_speedup": parallel_speedup,
                    "conditional_speedup": conditional_speedup,
                    "full_speedup": full_speedup,
                    "memory_reduction": (baseline_result.memory_peak - optimized_result.memory_peak) / baseline_result.memory_peak * 100,
                    "metrics_skipped": len(optimized_result.metrics_skipped)
                }
                all_results["comparisons"].append(comparison)
                
                print(f"  Results:")
                print(f"    Baseline: {baseline_result.execution_time:.3f}s")
                print(f"    Parallel speedup: {parallel_speedup:.2f}x")
                print(f"    Conditional speedup: {conditional_speedup:.2f}x")
                print(f"    Full optimization speedup: {full_speedup:.2f}x")
                print(f"    Memory reduction: {comparison['memory_reduction']:.1f}%")
                if optimized_result.metrics_skipped:
                    print(f"    Metrics skipped: {', '.join(optimized_result.metrics_skipped)}")
        
        # Generate summary statistics
        summary = self._generate_summary(all_results)
        all_results["summary"] = summary
        
        # Save results
        self._save_results(all_results)
        
        return all_results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate summary statistics from benchmark results.
        
        Args:
            results: All benchmark results
            
        Returns:
            Summary statistics
        """
        comparisons = results["comparisons"]
        
        if not comparisons:
            return {}
        
        # Calculate aggregate statistics
        speedups_parallel = [c["parallel_speedup"] for c in comparisons]
        speedups_conditional = [c["conditional_speedup"] for c in comparisons]
        speedups_full = [c["full_speedup"] for c in comparisons]
        memory_reductions = [c["memory_reduction"] for c in comparisons]
        
        summary = {
            "total_scenarios": len(comparisons),
            "parallel_optimization": {
                "mean_speedup": np.mean(speedups_parallel),
                "median_speedup": np.median(speedups_parallel),
                "min_speedup": np.min(speedups_parallel),
                "max_speedup": np.max(speedups_parallel),
                "std_speedup": np.std(speedups_parallel)
            },
            "conditional_optimization": {
                "mean_speedup": np.mean(speedups_conditional),
                "median_speedup": np.median(speedups_conditional),
                "min_speedup": np.min(speedups_conditional),
                "max_speedup": np.max(speedups_conditional),
                "std_speedup": np.std(speedups_conditional)
            },
            "full_optimization": {
                "mean_speedup": np.mean(speedups_full),
                "median_speedup": np.median(speedups_full),
                "min_speedup": np.min(speedups_full),
                "max_speedup": np.max(speedups_full),
                "std_speedup": np.std(speedups_full)
            },
            "memory_optimization": {
                "mean_reduction": np.mean(memory_reductions),
                "median_reduction": np.median(memory_reductions),
                "max_reduction": np.max(memory_reductions)
            },
            "best_scenarios": {
                "parallel": comparisons[np.argmax(speedups_parallel)]["scenario"],
                "conditional": comparisons[np.argmax(speedups_conditional)]["scenario"],
                "full": comparisons[np.argmax(speedups_full)]["scenario"]
            },
            "worst_scenarios": {
                "parallel": comparisons[np.argmin(speedups_parallel)]["scenario"],
                "conditional": comparisons[np.argmin(speedups_conditional)]["scenario"],
                "full": comparisons[np.argmin(speedups_full)]["scenario"]
            }
        }
        
        # Group by scenario characteristics
        by_quality = {"high": [], "medium": [], "low": []}
        by_size = {"small": [], "medium": [], "large": []}
        
        for scenario, comparison in zip(self.scenarios, comparisons):
            by_quality[scenario.quality_level].append(comparison["full_speedup"])
            
            if scenario.frame_count <= 15:
                by_size["small"].append(comparison["full_speedup"])
            elif scenario.frame_count <= 50:
                by_size["medium"].append(comparison["full_speedup"])
            else:
                by_size["large"].append(comparison["full_speedup"])
        
        summary["speedup_by_quality"] = {
            k: {"mean": np.mean(v), "count": len(v)} if v else {"mean": 0, "count": 0}
            for k, v in by_quality.items()
        }
        
        summary["speedup_by_size"] = {
            k: {"mean": np.mean(v), "count": len(v)} if v else {"mean": 0, "count": 0}
            for k, v in by_size.items()
        }
        
        return summary
    
    def _save_results(self, results: Dict):
        """Save benchmark results to file.
        
        Args:
            results: All benchmark results
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return convert_numpy(obj.__dict__)
            return obj
        
        serializable_results = convert_numpy(results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        # Also generate human-readable report
        report_file = self.output_dir / f"benchmark_report_{timestamp}.txt"
        self._generate_report(results, report_file)
        print(f"Report saved to: {report_file}")
    
    def _generate_report(self, results: Dict, output_file: Path):
        """Generate human-readable benchmark report.
        
        Args:
            results: All benchmark results
            output_file: Path to save report
        """
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE PERFORMANCE BENCHMARK REPORT\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Executive Summary
            summary = results.get("summary", {})
            if summary:
                f.write("EXECUTIVE SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Scenarios Tested: {summary['total_scenarios']}\n\n")
                
                f.write("Overall Performance Improvements:\n")
                full_opt = summary.get("full_optimization", {})
                f.write(f"  Mean Speedup: {full_opt.get('mean_speedup', 0):.2f}x\n")
                f.write(f"  Median Speedup: {full_opt.get('median_speedup', 0):.2f}x\n")
                f.write(f"  Best Case: {full_opt.get('max_speedup', 0):.2f}x\n")
                f.write(f"  Worst Case: {full_opt.get('min_speedup', 0):.2f}x\n\n")
                
                mem_opt = summary.get("memory_optimization", {})
                f.write("Memory Optimization:\n")
                f.write(f"  Mean Reduction: {mem_opt.get('mean_reduction', 0):.1f}%\n")
                f.write(f"  Max Reduction: {mem_opt.get('max_reduction', 0):.1f}%\n\n")
            
            # Detailed Results by Scenario
            f.write("DETAILED RESULTS BY SCENARIO\n")
            f.write("-" * 40 + "\n\n")
            
            for comparison in results.get("comparisons", []):
                f.write(f"Scenario: {comparison['scenario']}\n")
                f.write(f"  Baseline Time: {comparison['baseline_time']:.3f}s\n")
                f.write(f"  Speedups:\n")
                f.write(f"    Parallel Only: {comparison['parallel_speedup']:.2f}x\n")
                f.write(f"    Conditional Only: {comparison['conditional_speedup']:.2f}x\n")
                f.write(f"    Full Optimization: {comparison['full_speedup']:.2f}x\n")
                f.write(f"  Memory Reduction: {comparison['memory_reduction']:.1f}%\n")
                f.write(f"  Metrics Skipped: {comparison['metrics_skipped']}\n")
                f.write("\n")
            
            # Performance by Categories
            if summary:
                f.write("PERFORMANCE BY CATEGORY\n")
                f.write("-" * 40 + "\n\n")
                
                f.write("By Quality Level:\n")
                for level, stats in summary.get("speedup_by_quality", {}).items():
                    if stats["count"] > 0:
                        f.write(f"  {level.capitalize()}: {stats['mean']:.2f}x average speedup ({stats['count']} scenarios)\n")
                
                f.write("\nBy GIF Size:\n")
                for size, stats in summary.get("speedup_by_size", {}).items():
                    if stats["count"] > 0:
                        f.write(f"  {size.capitalize()}: {stats['mean']:.2f}x average speedup ({stats['count']} scenarios)\n")
            
            # Recommendations
            f.write("\n" + "=" * 80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            f.write(self._generate_recommendations(results))
    
    def _generate_recommendations(self, results: Dict) -> str:
        """Generate recommendations based on benchmark results.
        
        Args:
            results: Benchmark results
            
        Returns:
            Recommendations text
        """
        recommendations = []
        summary = results.get("summary", {})
        
        if not summary:
            return "Insufficient data for recommendations.\n"
        
        # Analyze parallel optimization effectiveness
        parallel_mean = summary["parallel_optimization"]["mean_speedup"]
        if parallel_mean < 1.1:
            recommendations.append(
                "• Parallel processing shows minimal benefit ({}x average). "
                "Consider disabling for small GIFs to reduce overhead.".format(f"{parallel_mean:.2f}")
            )
        elif parallel_mean > 1.5:
            recommendations.append(
                "• Parallel processing is highly effective ({}x average). "
                "Ensure it's enabled for production workloads.".format(f"{parallel_mean:.2f}")
            )
        
        # Analyze conditional optimization effectiveness
        conditional_mean = summary["conditional_optimization"]["mean_speedup"]
        if conditional_mean > 1.3:
            recommendations.append(
                "• Conditional processing is very effective ({}x average). "
                "This should be the primary optimization strategy.".format(f"{conditional_mean:.2f}")
            )
        
        # Quality-based recommendations
        quality_speedups = summary.get("speedup_by_quality", {})
        if quality_speedups.get("high", {}).get("mean", 0) > 1.5:
            recommendations.append(
                "• High-quality GIFs benefit significantly from optimization ({}x). "
                "Ensure quality assessment thresholds are properly tuned.".format(
                    f"{quality_speedups['high']['mean']:.2f}"
                )
            )
        
        # Size-based recommendations
        size_speedups = summary.get("speedup_by_size", {})
        if size_speedups.get("large", {}).get("mean", 0) > size_speedups.get("small", {}).get("mean", 0) * 1.5:
            recommendations.append(
                "• Large GIFs show better optimization gains than small ones. "
                "Consider adaptive thresholds based on frame count."
            )
        
        # Memory recommendations
        mem_reduction = summary["memory_optimization"]["mean_reduction"]
        if mem_reduction > 20:
            recommendations.append(
                f"• Memory optimization is effective ({mem_reduction:.1f}% average reduction). "
                "This validates the conditional processing approach."
            )
        
        # General recommendations
        recommendations.append(
            "\n• Continue monitoring performance metrics in production to validate these results."
        )
        recommendations.append(
            "• Consider implementing adaptive optimization strategies based on GIF characteristics."
        )
        recommendations.append(
            "• Regularly re-run benchmarks as the codebase evolves to detect regressions."
        )
        
        return "\n".join(recommendations) + "\n"


def main():
    """Run comprehensive benchmark suite."""
    suite = ComprehensiveBenchmarkSuite()
    results = suite.run_comprehensive_benchmark()
    
    # Print final summary
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    
    summary = results.get("summary", {})
    if summary:
        print(f"Mean Speedup (Full Optimization): {summary['full_optimization']['mean_speedup']:.2f}x")
        print(f"Best Case Speedup: {summary['full_optimization']['max_speedup']:.2f}x")
        print(f"Mean Memory Reduction: {summary['memory_optimization']['mean_reduction']:.1f}%")
    
    print("\nResults and report have been saved to the benchmark_results directory.")


if __name__ == "__main__":
    main()