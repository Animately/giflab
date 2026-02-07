#!/usr/bin/env python3
"""Profile metrics calculation to identify bottlenecks for parallelization."""

import time
from pathlib import Path
from contextlib import contextmanager
import numpy as np
from typing import Any

# Import metrics modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from giflab.metrics import (
    extract_gif_frames,
    resize_to_common_dimensions,
    align_frames,
    ssim,
    calculate_ms_ssim,
    calculate_safe_psnr,
    mse,
    rmse,
    fsim,
    gmsd,
    chist,
    edge_similarity,
    texture_similarity,
    sharpness_similarity,
    calculate_temporal_consistency,
    detect_disposal_artifacts,
)
import cv2


@contextmanager
def timer(name: str):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    print(f"[{name}] Starting...")
    yield
    elapsed = time.perf_counter() - start
    print(f"[{name}] Completed in {elapsed:.3f}s")


def profile_frame_metrics(aligned_pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    """Profile individual frame-level metric calculations."""
    timings = {}
    
    # Sample the first, middle, and last frame pairs for profiling
    sample_indices = [0, len(aligned_pairs) // 2, -1]
    sample_pairs = [aligned_pairs[i] for i in sample_indices if i < len(aligned_pairs)]
    
    # Profile each metric type
    for orig_frame, comp_frame in sample_pairs:
        # Convert to grayscale for SSIM
        if len(orig_frame.shape) == 3:
            orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_RGB2GRAY)
            comp_gray = cv2.cvtColor(comp_frame, cv2.COLOR_RGB2GRAY)
        else:
            orig_gray = orig_frame
            comp_gray = comp_frame
        
        # SSIM
        start = time.perf_counter()
        _ = ssim(orig_gray, comp_gray, data_range=255.0)
        timings.setdefault("ssim", []).append(time.perf_counter() - start)
        
        # MS-SSIM
        start = time.perf_counter()
        _ = calculate_ms_ssim(orig_frame, comp_frame)
        timings.setdefault("ms_ssim", []).append(time.perf_counter() - start)
        
        # PSNR
        start = time.perf_counter()
        _ = calculate_safe_psnr(orig_frame, comp_frame)
        timings.setdefault("psnr", []).append(time.perf_counter() - start)
        
        # MSE/RMSE
        start = time.perf_counter()
        _ = mse(orig_frame, comp_frame)
        _ = rmse(orig_frame, comp_frame)
        timings.setdefault("mse_rmse", []).append(time.perf_counter() - start)
        
        # FSIM
        start = time.perf_counter()
        _ = fsim(orig_frame, comp_frame)
        timings.setdefault("fsim", []).append(time.perf_counter() - start)
        
        # GMSD
        start = time.perf_counter()
        _ = gmsd(orig_frame, comp_frame)
        timings.setdefault("gmsd", []).append(time.perf_counter() - start)
        
        # Color histogram
        start = time.perf_counter()
        _ = chist(orig_frame, comp_frame)
        timings.setdefault("chist", []).append(time.perf_counter() - start)
        
        # Edge similarity
        start = time.perf_counter()
        _ = edge_similarity(orig_frame, comp_frame, 100, 200)
        timings.setdefault("edge_similarity", []).append(time.perf_counter() - start)
        
        # Texture similarity
        start = time.perf_counter()
        _ = texture_similarity(orig_frame, comp_frame)
        timings.setdefault("texture_similarity", []).append(time.perf_counter() - start)
        
        # Sharpness similarity
        start = time.perf_counter()
        _ = sharpness_similarity(orig_frame, comp_frame)
        timings.setdefault("sharpness_similarity", []).append(time.perf_counter() - start)
    
    # Calculate average timings
    avg_timings = {k: np.mean(v) for k, v in timings.items()}
    return avg_timings


def main():
    """Main profiling function."""
    # Use test fixtures
    test_dir = Path(__file__).parent.parent / "fixtures" / "test_gifs"
    original_path = test_dir / "solid_dot_32x32_10frames_100ms.gif"
    compressed_path = test_dir / "solid_dot_32x32_10frames_100ms.gif"  # Use same file for profiling
    
    if not original_path.exists():
        print(f"Test file not found: {original_path}")
        print("Creating synthetic test data...")
        # Create synthetic frames for profiling
        frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(30)]
        aligned_pairs = [(f, f) for f in frames]
    else:
        with timer("Extract frames"):
            original_result = extract_gif_frames(original_path, max_frames=100)
            compressed_result = extract_gif_frames(compressed_path, max_frames=100)
        
        with timer("Resize frames"):
            orig_resized, comp_resized = resize_to_common_dimensions(
                original_result.frames, compressed_result.frames
            )
        
        with timer("Align frames"):
            aligned_pairs = align_frames(orig_resized, comp_resized)
    
    print(f"\nProfiling {len(aligned_pairs)} frame pairs...")
    print("=" * 60)
    
    # Profile frame-level metrics
    print("\n📊 Frame-Level Metrics (Parallelizable):")
    print("-" * 60)
    avg_timings = profile_frame_metrics(aligned_pairs)
    
    total_sequential_time = 0
    for metric, avg_time in sorted(avg_timings.items(), key=lambda x: x[1], reverse=True):
        per_frame_time = avg_time * len(aligned_pairs)
        total_sequential_time += per_frame_time
        print(f"{metric:20s}: {avg_time*1000:7.3f}ms/frame | Total: {per_frame_time:.3f}s")
    
    print(f"\nTotal sequential time for frame metrics: {total_sequential_time:.3f}s")
    
    # Estimate parallel speedup
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    estimated_parallel_time = total_sequential_time / min(cpu_count, len(aligned_pairs))
    estimated_speedup = total_sequential_time / estimated_parallel_time
    
    print(f"\n🚀 Parallelization Potential:")
    print(f"  CPU cores available: {cpu_count}")
    print(f"  Estimated parallel time: {estimated_parallel_time:.3f}s")
    print(f"  Potential speedup: {estimated_speedup:.1f}x")
    
    # Profile global metrics (harder to parallelize)
    if len(aligned_pairs) > 0:
        print("\n📊 Global Metrics (Sequential):")
        print("-" * 60)
        
        orig_frames = [p[0] for p in aligned_pairs]
        comp_frames = [p[1] for p in aligned_pairs]
        
        with timer("Temporal consistency"):
            _ = calculate_temporal_consistency(orig_frames)
            _ = calculate_temporal_consistency(comp_frames)
        
        with timer("Disposal artifacts"):
            _ = detect_disposal_artifacts(orig_frames, False)
            _ = detect_disposal_artifacts(comp_frames, False)
    
    print("\n✅ Profiling complete!")


if __name__ == "__main__":
    main()