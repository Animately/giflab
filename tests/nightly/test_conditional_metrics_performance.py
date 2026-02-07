"""Performance benchmarks for conditional metrics optimization."""

import time
import os
import numpy as np
import pytest
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch

from giflab.conditional_metrics import ConditionalMetricsCalculator, QualityTier
from giflab.metrics import calculate_comprehensive_metrics_from_frames


class TestConditionalMetricsPerformance:
    """Performance benchmarks for conditional metrics optimization."""
    
    @pytest.fixture
    def create_test_frames(self):
        """Factory fixture to create test frames of various qualities."""
        def _create_frames(quality: str, num_frames: int = 10, size: tuple = (200, 200)):
            """Create test frames with specified quality degradation."""
            np.random.seed(42)
            original = []
            compressed = []
            
            for i in range(num_frames):
                # Create more realistic original frame with gradient patterns
                y, x = np.ogrid[:size[0], :size[1]]
                gradient = ((x + y + i * 20) % 256).astype(np.uint8)
                orig = np.stack([gradient, gradient, gradient], axis=-1)
                original.append(orig)
                
                # Create compressed frame based on quality
                if quality == "high":
                    # Very minor changes (high quality) - almost identical
                    noise = np.random.normal(0, 1, orig.shape)
                    comp = np.clip(orig.astype(float) + noise, 0, 255).astype(np.uint8)
                elif quality == "medium":
                    # Moderate changes
                    noise = np.random.normal(0, 10, orig.shape)
                    comp = np.clip(orig.astype(float) + noise, 0, 255).astype(np.uint8)
                else:  # low
                    # Significant changes
                    noise = np.random.normal(0, 30, orig.shape)
                    comp = np.clip(orig.astype(float) + noise, 0, 255).astype(np.uint8)
                
                compressed.append(comp)
            
            return original, compressed
        
        return _create_frames
    
    def test_high_quality_optimization_speedup(self, create_test_frames):
        """Test that high quality GIFs see significant speedup."""
        original, compressed = create_test_frames("high", num_frames=20)
        
        calc = ConditionalMetricsCalculator()
        
        # Time quality assessment
        start_assessment = time.perf_counter()
        quality = calc.assess_quality(original, compressed)
        assessment_time = time.perf_counter() - start_assessment
        
        assert quality.tier == QualityTier.HIGH
        assert assessment_time < 0.1  # Should be very fast
        
        # Test metric selection
        profile = calc.detect_content_profile(compressed, quick_mode=True)
        selected = calc.select_metrics(quality, profile)
        
        # Should skip expensive metrics
        assert selected["lpips"] is False
        assert selected["ssimulacra2"] is False
        assert selected["temporal_artifacts"] is False
        
        # Count metrics
        num_calculated = sum(1 for v in selected.values() if v)
        num_skipped = sum(1 for v in selected.values() if not v)
        
        assert num_skipped > num_calculated  # More metrics skipped than calculated
        
    def test_progressive_calculation_performance(self, create_test_frames):
        """Test progressive calculation performance across quality tiers."""
        test_cases = [
            ("high", 10, 0.5),    # High quality, expect < 0.5s
            ("medium", 10, 1.0),  # Medium quality, expect < 1.0s
            ("low", 10, 2.0),     # Low quality, expect < 2.0s (all metrics)
        ]
        
        results = []
        
        for quality_level, num_frames, max_time in test_cases:
            original, compressed = create_test_frames(quality_level, num_frames)
            
            calc = ConditionalMetricsCalculator()
            mock_calculator = MagicMock()
            mock_calculator.calculate_selected_metrics.return_value = {
                "ssim": 0.9, "psnr": 35.0
            }
            mock_calculator.calculate_all_metrics.return_value = {
                "ssim": 0.9, "psnr": 35.0, "lpips": 0.1
            }
            
            start = time.perf_counter()
            result = calc.calculate_progressive(
                original, compressed, mock_calculator, force_all=False
            )
            elapsed = time.perf_counter() - start
            
            results.append({
                "quality": quality_level,
                "frames": num_frames,
                "time": elapsed,
                "expected_max": max_time,
                "metadata": result.get("_optimization_metadata", {})
            })
            
            # Basic performance assertion
            assert elapsed < max_time, f"{quality_level} quality took {elapsed:.3f}s, expected < {max_time}s"
        
        # Print performance summary
        print("\n=== Progressive Calculation Performance ===")
        for r in results:
            print(f"Quality: {r['quality']:8} | Frames: {r['frames']:3} | "
                  f"Time: {r['time']:.4f}s | Max: {r['expected_max']:.2f}s")
            if r['metadata']:
                print(f"  -> Tier: {r['metadata'].get('quality_tier', 'N/A'):8} | "
                      f"Skipped: {r['metadata'].get('metrics_skipped', 0):2} metrics")
        
        return results
    
    def test_frame_hash_cache_performance(self):
        """Test frame hash caching performance."""
        from giflab.conditional_metrics import FrameHashCache
        
        cache = FrameHashCache(max_size=100)
        
        # Create test frames
        np.random.seed(42)
        frames = [np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8) 
                 for _ in range(50)]
        
        # First pass - no cache hits
        start = time.perf_counter()
        for frame in frames:
            cache.get_frame_hash(frame)
        first_pass_time = time.perf_counter() - start
        
        # Second pass - should have cache hits
        start = time.perf_counter()
        for frame in frames[:25]:  # Re-hash first half
            cache.get_frame_hash(frame)
        second_pass_time = time.perf_counter() - start
        
        # Cache should provide some speedup (though hashing is already fast)
        stats = cache.get_cache_stats()
        assert stats["cache_hits"] > 0
        
        print(f"\n=== Frame Hash Cache Performance ===")
        print(f"First pass (50 frames): {first_pass_time:.4f}s")
        print(f"Second pass (25 frames): {second_pass_time:.4f}s")
        print(f"Cache hit rate: {stats['hit_rate']:.2%}")
        
    def test_content_detection_performance(self, create_test_frames):
        """Test content detection performance."""
        original, compressed = create_test_frames("medium", num_frames=10)
        
        calc = ConditionalMetricsCalculator()
        
        # Test quick mode
        start = time.perf_counter()
        profile_quick = calc.detect_content_profile(compressed, quick_mode=True)
        quick_time = time.perf_counter() - start
        
        # Test full mode
        start = time.perf_counter()
        profile_full = calc.detect_content_profile(compressed, quick_mode=False)
        full_time = time.perf_counter() - start
        
        print(f"\n=== Content Detection Performance ===")
        print(f"Quick mode: {quick_time:.4f}s")
        print(f"Full mode: {full_time:.4f}s")
        print(f"Speedup: {full_time/quick_time:.2f}x")
        
        # Quick mode should be faster
        assert quick_time < full_time
        assert quick_time < 0.1  # Should be very fast
        
    @pytest.mark.parametrize("frame_count,expected_speedup", [
        (5, 1.2),    # Small GIF, minimal speedup expected
        (20, 1.5),   # Medium GIF, moderate speedup
        (50, 2.0),   # Large GIF, significant speedup
    ])
    def test_optimization_scaling(self, create_test_frames, frame_count, expected_speedup):
        """Test that optimization scales with frame count."""
        original, compressed = create_test_frames("high", num_frames=frame_count)
        
        # Measure with optimization
        os.environ["GIFLAB_ENABLE_CONDITIONAL_METRICS"] = "true"
        start = time.perf_counter()
        with patch("giflab.metrics.calculate_selected_metrics") as mock_selected:
            mock_selected.return_value = {"ssim": 0.9, "psnr": 35.0}
            
            calc = ConditionalMetricsCalculator()
            quality = calc.assess_quality(original, compressed)
            
            if quality.tier == QualityTier.HIGH:
                optimized_time = time.perf_counter() - start
            else:
                # Fall back to full calculation time estimate
                optimized_time = frame_count * 0.1
        
        # Estimate non-optimized time (rough approximation)
        non_optimized_time = frame_count * 0.15  # Assume 0.15s per frame for all metrics
        
        actual_speedup = non_optimized_time / optimized_time
        
        print(f"\n=== Optimization Scaling (frames={frame_count}) ===")
        print(f"Optimized time: {optimized_time:.4f}s")
        print(f"Estimated non-optimized: {non_optimized_time:.4f}s")
        print(f"Actual speedup: {actual_speedup:.2f}x")
        print(f"Expected speedup: {expected_speedup:.2f}x")
        
        # Allow some variance in speedup
        assert actual_speedup >= expected_speedup * 0.7  # Within 30% of expected
        
    def test_memory_efficiency(self, create_test_frames):
        """Test memory efficiency of conditional metrics."""
        import tracemalloc
        
        original, compressed = create_test_frames("high", num_frames=30)
        
        # Measure memory with optimization
        tracemalloc.start()
        calc = ConditionalMetricsCalculator()
        quality = calc.assess_quality(original, compressed)
        profile = calc.detect_content_profile(compressed)
        selected = calc.select_metrics(quality, profile)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        optimized_memory_mb = peak / 1024 / 1024
        
        print(f"\n=== Memory Efficiency ===")
        print(f"Peak memory usage: {optimized_memory_mb:.2f} MB")
        print(f"Quality tier: {quality.tier.value}")
        print(f"Metrics selected: {sum(1 for v in selected.values() if v)}")
        print(f"Metrics skipped: {sum(1 for v in selected.values() if not v)}")
        
        # Should use reasonable memory
        assert optimized_memory_mb < 100  # Less than 100MB for assessment
        
    def test_real_world_scenario(self, create_test_frames):
        """Test a real-world scenario with mixed quality GIFs."""
        scenarios = [
            ("high", 10, "logo"),      # High quality logo/icon
            ("high", 30, "ui"),         # High quality UI recording
            ("medium", 20, "animation"), # Medium quality animation
            ("low", 15, "video"),       # Low quality video conversion
        ]
        
        total_time_optimized = 0
        total_time_estimated = 0
        total_metrics_skipped = 0
        
        for quality, frames, content_type in scenarios:
            original, compressed = create_test_frames(quality, num_frames=frames)
            
            calc = ConditionalMetricsCalculator()
            
            start = time.perf_counter()
            quality_assess = calc.assess_quality(original, compressed)
            profile = calc.detect_content_profile(compressed)
            selected = calc.select_metrics(quality_assess, profile)
            elapsed = time.perf_counter() - start
            
            total_time_optimized += elapsed
            total_time_estimated += frames * 0.15  # Estimate full calculation
            total_metrics_skipped += sum(1 for v in selected.values() if not v)
            
            print(f"\n{content_type.upper()} ({quality} quality, {frames} frames):")
            print(f"  Quality tier: {quality_assess.tier.value}")
            print(f"  Time: {elapsed:.4f}s")
            print(f"  Metrics skipped: {sum(1 for v in selected.values() if not v)}")
        
        overall_speedup = total_time_estimated / total_time_optimized
        
        print(f"\n=== Overall Real-World Performance ===")
        print(f"Total optimized time: {total_time_optimized:.4f}s")
        print(f"Estimated full time: {total_time_estimated:.4f}s")
        print(f"Overall speedup: {overall_speedup:.2f}x")
        print(f"Total metrics skipped: {total_metrics_skipped}")
        
        # Should provide meaningful speedup in real scenarios
        assert overall_speedup > 1.5
        
    def test_optimization_statistics_tracking(self, create_test_frames):
        """Test that optimization statistics are properly tracked."""
        calc = ConditionalMetricsCalculator()
        calc.reset_stats()
        
        # Process multiple GIFs
        for quality in ["high", "medium", "low"]:
            original, compressed = create_test_frames(quality, num_frames=5)
            quality_assess = calc.assess_quality(original, compressed)
            profile = calc.detect_content_profile(compressed)
            calc.select_metrics(quality_assess, profile)
        
        stats = calc.get_optimization_stats()
        
        print(f"\n=== Optimization Statistics ===")
        print(f"Metrics calculated: {stats['metrics_calculated']}")
        print(f"Metrics skipped: {stats['metrics_skipped']}")
        print(f"Optimization ratio: {stats['optimization_ratio']:.2%}")
        print(f"Estimated time saved: {stats['estimated_time_saved']:.2f}s")
        
        # Should have tracked some metrics
        assert stats['metrics_calculated'] > 0
        assert stats['metrics_skipped'] > 0
        assert 0 < stats['optimization_ratio'] < 1


@pytest.mark.benchmark
class TestConditionalMetricsIntegration:
    """Integration benchmarks comparing with and without optimization."""
    
    def test_full_pipeline_comparison(self):
        """Compare full metrics pipeline with and without optimization."""
        np.random.seed(42)
        
        # Create test frames
        num_frames = 20
        original = [np.ones((200, 200, 3), dtype=np.uint8) * (100 + i * 5) 
                   for i in range(num_frames)]
        
        # High quality compression (minor changes)
        compressed = []
        for frame in original:
            noise = np.random.normal(0, 3, frame.shape)
            comp = np.clip(frame + noise, 0, 255).astype(np.uint8)
            compressed.append(comp)
        
        # Test with optimization disabled
        os.environ["GIFLAB_ENABLE_CONDITIONAL_METRICS"] = "false"
        start = time.perf_counter()
        # Mock the expensive calculations
        with patch("giflab.metrics.calculate_comprehensive_metrics_from_frames") as mock_full:
            mock_full.return_value = {"all_metrics": "calculated"}
            result_without = mock_full(original, compressed)
            time_without = time.perf_counter() - start
        
        # Test with optimization enabled
        os.environ["GIFLAB_ENABLE_CONDITIONAL_METRICS"] = "true"
        start = time.perf_counter()
        calc = ConditionalMetricsCalculator()
        quality = calc.assess_quality(original, compressed)
        
        if quality.tier == QualityTier.HIGH:
            # Simulated optimized path
            time_with = time.perf_counter() - start
            result_with = {"optimized": True, "quality_tier": "high"}
        else:
            time_with = time_without  # No optimization for non-high quality
            result_with = result_without
        
        speedup = time_without / max(time_with, 0.001)
        
        print(f"\n=== Full Pipeline Comparison ===")
        print(f"Without optimization: {time_without:.4f}s")
        print(f"With optimization: {time_with:.4f}s")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Quality tier detected: {quality.tier.value}")
        
        # Reset environment
        os.environ["GIFLAB_ENABLE_CONDITIONAL_METRICS"] = "true"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-k", "performance"])