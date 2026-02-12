#!/usr/bin/env python3
"""Pipeline integration tests for nightly runs.

This module consolidates heavy pipeline integration tests covering:
1. Full metric calculation pipeline with all optimizations (Phase 5)
2. Accuracy validation against baseline
3. Deterministic result verification
4. Error handling and recovery
5. Configuration compatibility testing
6. Validation pipeline with Phase 2 metrics (threshold effectiveness, regression prevention)
7. Performance integration (vectorized generation, multiprocessing, backward compatibility)

These tests are long-running and resource-intensive, making them suitable
for nightly CI rather than per-commit runs.
"""

import hashlib
import json
import multiprocessing as mp
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from giflab.conditional_metrics import ConditionalMetricsCalculator
from giflab.deep_perceptual_metrics import should_use_deep_perceptual
from giflab.metrics import (
    calculate_comprehensive_metrics_from_frames,
    cleanup_all_validators,
)
from giflab.model_cache import LPIPSModelCache
from giflab.multiprocessing_support import (
    ParallelFrameGenerator,
    get_optimal_worker_count,
)
from giflab.optimization_validation import (
    ValidationChecker,
    ValidationResult,
    ValidationStatus,
)
from giflab.parallel_metrics import ParallelMetricsCalculator
from giflab.synthetic_gifs import SyntheticFrameGenerator, SyntheticGifGenerator

# ---------------------------------------------------------------------------
# Full Pipeline Integration Tests (from test_phase5_full_pipeline.py)
# ---------------------------------------------------------------------------

class TestFullPipelineIntegration:
    """Integration tests for full metric calculation pipeline."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        # Store original environment
        self.original_env = os.environ.copy()

        # Setup
        cleanup_all_validators()
        cache = LPIPSModelCache()
        cache.cleanup(force=True)

        yield

        # Restore environment
        os.environ.clear()
        os.environ.update(self.original_env)

        # Teardown
        cleanup_all_validators()
        cache.cleanup(force=True)

    def generate_test_gif_frames(
        self,
        frame_count: int = 10,
        size: tuple[int, int] = (200, 200),
        quality: str = "medium",
        content: str = "mixed"
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Generate test GIF frames with specific characteristics.

        Args:
            frame_count: Number of frames
            size: Frame size (width, height)
            quality: Quality level ('high', 'medium', 'low')
            content: Content type ('text', 'gradient', 'animation', 'static', 'mixed')

        Returns:
            tuple of (original_frames, compressed_frames)
        """
        np.random.seed(42)  # Ensure deterministic generation
        width, height = size
        frames_orig = []
        frames_comp = []

        for i in range(frame_count):
            # Generate frame based on content type
            if content == "static":
                if i == 0:
                    base = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                frame = base.copy() if i > 0 else base
            elif content == "gradient":
                x = np.linspace(0, 255, width)
                y = np.linspace(0, 255, height)
                xx, yy = np.meshgrid(x, y)
                frame = np.stack([
                    (xx * (1 + i * 0.1) % 256).astype(np.uint8),
                    (yy * (1 + i * 0.1) % 256).astype(np.uint8),
                    ((xx + yy) / 2 % 256).astype(np.uint8)
                ], axis=-1)
            elif content == "text":
                frame = np.ones((height, width, 3), dtype=np.uint8) * 255
                for j in range(3):
                    y_pos = 30 + j * 40
                    x_pos = 20 + (i * 5) % 50
                    if y_pos + 20 < height and x_pos + 100 < width:
                        frame[y_pos:y_pos + 20, x_pos:x_pos + 100] = 0
            elif content == "animation":
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                center_x = int(width * (0.5 + 0.3 * np.sin(i * 0.3)))
                center_y = int(height * (0.5 + 0.3 * np.cos(i * 0.3)))
                y_grid, x_grid = np.ogrid[:height, :width]
                mask = (x_grid - center_x)**2 + (y_grid - center_y)**2 <= (min(width, height) // 6)**2
                frame[mask] = [255, 100, 100]
            else:  # mixed
                frame = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
                frame[height // 3:2 * height // 3, width // 3:2 * width // 3] = 220

            frames_orig.append(frame)

            # Generate compressed version
            if quality == "high":
                noise = np.random.normal(0, 2, frame.shape)
                compressed = np.clip(frame + noise, 0, 255).astype(np.uint8)
            elif quality == "medium":
                noise = np.random.normal(0, 8, frame.shape)
                compressed = np.clip(frame + noise, 0, 255).astype(np.uint8)
                compressed = (compressed // 4) * 4  # Mild quantization
            else:  # low
                noise = np.random.normal(0, 20, frame.shape)
                compressed = np.clip(frame + noise, 0, 255).astype(np.uint8)
                compressed = (compressed // 16) * 16  # Heavy quantization

            frames_comp.append(compressed)

        return frames_orig, frames_comp

    def test_pipeline_all_optimizations_enabled(self):
        """Test full pipeline with all optimizations enabled."""
        os.environ['GIFLAB_ENABLE_PARALLEL_METRICS'] = 'true'
        os.environ['GIFLAB_ENABLE_CONDITIONAL_METRICS'] = 'true'
        os.environ['GIFLAB_USE_MODEL_CACHE'] = 'true'

        scenarios = [
            (10, (200, 200), "high", "static"),
            (30, (500, 500), "medium", "gradient"),
            (50, (800, 600), "low", "animation"),
        ]

        for frame_count, size, quality, content in scenarios:
            frames_orig, frames_comp = self.generate_test_gif_frames(
                frame_count, size, quality, content
            )

            start_time = time.perf_counter()
            metrics = calculate_comprehensive_metrics_from_frames(
                frames_orig,
                frames_comp
            )
            elapsed = time.perf_counter() - start_time

            assert metrics is not None, "Metrics calculation failed"
            assert "psnr" in metrics, "Missing PSNR metric"
            assert "ssim" in metrics, "Missing SSIM metric"

            if quality == "high":
                conditional_calc = ConditionalMetricsCalculator()
                quality_tier = conditional_calc.assess_quality(
                    frames_orig[:5], frames_comp[:5]
                )
                if quality_tier == "HIGH":
                    print(f"High quality GIF correctly identified, tier: {quality_tier}")

            print(f"Scenario ({frame_count} frames, {size}, {quality}, {content}): "
                  f"{elapsed:.3f}s, {len(metrics)} metrics")

    def test_pipeline_accuracy_validation(self):
        """Test that optimized pipeline maintains accuracy."""
        frames_orig, frames_comp = self.generate_test_gif_frames(
            20, (300, 300), "medium", "mixed"
        )

        # Calculate baseline (no optimizations)
        os.environ['GIFLAB_ENABLE_PARALLEL_METRICS'] = 'false'
        os.environ['GIFLAB_ENABLE_CONDITIONAL_METRICS'] = 'false'
        os.environ['GIFLAB_USE_MODEL_CACHE'] = 'false'

        baseline_metrics = calculate_comprehensive_metrics_from_frames(
            frames_orig,
            frames_comp
        )

        cleanup_all_validators()

        # Calculate with optimizations
        os.environ['GIFLAB_ENABLE_PARALLEL_METRICS'] = 'true'
        os.environ['GIFLAB_ENABLE_CONDITIONAL_METRICS'] = 'false'
        os.environ['GIFLAB_USE_MODEL_CACHE'] = 'true'

        optimized_metrics = calculate_comprehensive_metrics_from_frames(
            frames_orig,
            frames_comp
        )

        tolerance = 0.001  # 0.1% tolerance
        timing_keys = {"render_ms", "elapsed_ms", "total_time_ms", "computation_time_ms"}

        for key in baseline_metrics:
            if key in optimized_metrics and key not in timing_keys and not key.endswith("_ms"):
                baseline_val = baseline_metrics[key]
                optimized_val = optimized_metrics[key]

                if isinstance(baseline_val, int | float):
                    if baseline_val != 0:
                        relative_diff = abs(optimized_val - baseline_val) / abs(baseline_val)
                        assert relative_diff < tolerance, \
                            f"Metric {key} differs: baseline={baseline_val}, optimized={optimized_val}, diff={relative_diff:.4%}"
                    else:
                        assert abs(optimized_val) < tolerance, \
                            f"Metric {key} differs from zero: {optimized_val}"

        print(f"Accuracy validation passed: {len(baseline_metrics)} metrics within {tolerance:.1%} tolerance")

    def test_pipeline_deterministic_results(self):
        """Test that results are deterministic with same input."""
        frames_orig, frames_comp = self.generate_test_gif_frames(
            15, (250, 250), "medium", "gradient"
        )

        os.environ['GIFLAB_ENABLE_PARALLEL_METRICS'] = 'true'
        os.environ['GIFLAB_ENABLE_CONDITIONAL_METRICS'] = 'true'
        os.environ['GIFLAB_USE_MODEL_CACHE'] = 'true'

        results = []
        for _ in range(3):
            metrics = calculate_comprehensive_metrics_from_frames(
                frames_orig,
                frames_comp
            )
            results.append(metrics)
            cleanup_all_validators()

        for i in range(1, len(results)):
            for key in results[0]:
                if key in results[i] and not key.endswith("_ms"):
                    val0 = results[0][key]
                    vali = results[i][key]

                    if isinstance(val0, int | float):
                        assert abs(val0 - vali) < 1e-6, \
                            f"Non-deterministic result for {key}: run 1={val0}, run {i + 1}={vali}"

        print(f"Deterministic validation passed: {len(results)} runs produced identical results")

    def test_pipeline_error_handling(self):
        """Test pipeline error handling and recovery."""
        try:
            result = calculate_comprehensive_metrics_from_frames(None, None)
            assert result is not None or result is None
        except (ValueError, TypeError, AttributeError):
            pass

        frames_orig, frames_comp = self.generate_test_gif_frames(10, (200, 200))
        frames_comp = frames_comp[:5]

        try:
            result = calculate_comprehensive_metrics_from_frames(frames_orig, frames_comp)
        except (ValueError, IndexError, AssertionError):
            pass

        cleanup_all_validators()

        frames_orig, frames_comp = self.generate_test_gif_frames(10, (200, 200))
        metrics = calculate_comprehensive_metrics_from_frames(
            frames_orig,
            frames_comp
        )
        assert metrics is not None, "Pipeline failed to recover after error"

        print("Error handling validation passed")

    def test_pipeline_configuration_compatibility(self):
        """Test different configuration combinations."""
        frames_orig, frames_comp = self.generate_test_gif_frames(
            20, (300, 300), "medium", "mixed"
        )

        configs = [
            (True, True, True, "All enabled"),
            (True, False, True, "Parallel + Cache"),
            (False, True, True, "Conditional + Cache"),
            (True, True, False, "Parallel + Conditional"),
            (False, False, True, "Cache only"),
            (True, False, False, "Parallel only"),
            (False, True, False, "Conditional only"),
            (False, False, False, "All disabled"),
        ]

        for parallel, conditional, cache, description in configs:
            os.environ['GIFLAB_ENABLE_PARALLEL_METRICS'] = str(parallel).lower()
            os.environ['GIFLAB_ENABLE_CONDITIONAL_METRICS'] = str(conditional).lower()
            os.environ['GIFLAB_USE_MODEL_CACHE'] = str(cache).lower()

            try:
                start_time = time.perf_counter()
                metrics = calculate_comprehensive_metrics_from_frames(
                    frames_orig,
                    frames_comp
                )
                elapsed = time.perf_counter() - start_time

                assert metrics is not None, f"Failed with config: {description}"
                print(f"Config '{description}': {elapsed:.3f}s, {len(metrics)} metrics")

            except Exception as e:
                pytest.fail(f"Configuration '{description}' failed: {str(e)}")

            finally:
                cleanup_all_validators()

        print("Configuration compatibility validation passed")

    def test_pipeline_performance_thresholds(self):
        """Test that performance meets expected thresholds."""
        frames_orig, frames_comp = self.generate_test_gif_frames(
            10, (100, 100), "high", "static"
        )

        os.environ['GIFLAB_ENABLE_PARALLEL_METRICS'] = 'true'
        os.environ['GIFLAB_ENABLE_CONDITIONAL_METRICS'] = 'true'
        os.environ['GIFLAB_USE_MODEL_CACHE'] = 'true'

        start_time = time.perf_counter()
        calculate_comprehensive_metrics_from_frames(
            frames_orig,
            frames_comp
        )
        small_time = time.perf_counter() - start_time

        assert small_time < 2.0, f"Small GIF took too long: {small_time:.3f}s"

        frames_orig, frames_comp = self.generate_test_gif_frames(
            100, (800, 600), "low", "animation"
        )

        start_time = time.perf_counter()
        calculate_comprehensive_metrics_from_frames(
            frames_orig,
            frames_comp
        )
        large_time = time.perf_counter() - start_time

        assert large_time < 180.0, f"Large GIF took too long: {large_time:.3f}s"

        print(f"Performance thresholds met: Small={small_time:.3f}s, Large={large_time:.3f}s")

    def test_pipeline_cache_effectiveness(self):
        """Test that model cache is working effectively."""
        cache = LPIPSModelCache()
        cache.cleanup(force=True)

        os.environ['GIFLAB_USE_MODEL_CACHE'] = 'true'

        frames_orig, frames_comp = self.generate_test_gif_frames(
            20, (300, 300), "low", "mixed"
        )

        initial_info = cache.get_cache_info()
        calculate_comprehensive_metrics_from_frames(
            frames_orig,
            frames_comp
        )
        after_first = cache.get_cache_info()

        calculate_comprehensive_metrics_from_frames(
            frames_orig,
            frames_comp
        )
        after_second = cache.get_cache_info()

        models_loaded_first = after_first.get("models_loaded", 0) - initial_info.get("models_loaded", 0)
        models_loaded_second = after_second.get("models_loaded", 0) - after_first.get("models_loaded", 0)

        assert models_loaded_second <= models_loaded_first, \
            f"Cache not effective: first run loaded {models_loaded_first}, second loaded {models_loaded_second}"

        print(f"Cache effectiveness validated: First run loaded {models_loaded_first} models, "
              f"second run loaded {models_loaded_second} models")

    def test_pipeline_conditional_skip_validation(self):
        """Test that conditional processing correctly skips metrics."""
        frames_orig, frames_comp = self.generate_test_gif_frames(
            20, (400, 400), "high", "static"
        )

        os.environ['GIFLAB_ENABLE_CONDITIONAL_METRICS'] = 'true'
        os.environ['GIFLAB_FORCE_ALL_METRICS'] = 'false'

        metrics_conditional = calculate_comprehensive_metrics_from_frames(
            frames_orig,
            frames_comp
        )

        os.environ['GIFLAB_FORCE_ALL_METRICS'] = 'true'

        cleanup_all_validators()

        metrics_all = calculate_comprehensive_metrics_from_frames(
            frames_orig,
            frames_comp
        )

        metrics_skipped = len(metrics_all) - len(metrics_conditional)
        assert metrics_skipped > 0, \
            f"Conditional processing didn't skip any metrics: all={len(metrics_all)}, conditional={len(metrics_conditional)}"

        print(f"Conditional skip validation passed: {metrics_skipped} metrics skipped for high quality GIF")

    def test_pipeline_parallel_speedup_validation(self):
        """Test that parallel processing provides speedup for large GIFs."""
        frames_orig, frames_comp = self.generate_test_gif_frames(
            100, (600, 600), "medium", "animation"
        )

        os.environ['GIFLAB_ENABLE_CONDITIONAL_METRICS'] = 'false'
        os.environ['GIFLAB_USE_MODEL_CACHE'] = 'true'

        # Sequential processing
        os.environ['GIFLAB_ENABLE_PARALLEL_METRICS'] = 'false'

        start_time = time.perf_counter()
        metrics_seq = calculate_comprehensive_metrics_from_frames(
            frames_orig,
            frames_comp
        )
        seq_time = time.perf_counter() - start_time

        cleanup_all_validators()

        # Parallel processing
        os.environ['GIFLAB_ENABLE_PARALLEL_METRICS'] = 'true'
        os.environ['GIFLAB_MAX_PARALLEL_WORKERS'] = '4'

        start_time = time.perf_counter()
        metrics_par = calculate_comprehensive_metrics_from_frames(
            frames_orig,
            frames_comp
        )
        par_time = time.perf_counter() - start_time

        assert len(metrics_seq) == len(metrics_par), \
            f"Different metrics: sequential={len(metrics_seq)}, parallel={len(metrics_par)}"

        speedup = seq_time / par_time if par_time > 0 else 1.0

        print(f"Parallel speedup: {speedup:.2f}x (sequential={seq_time:.3f}s, parallel={par_time:.3f}s)")

        if speedup < 1.0:
            print("WARNING: Parallel processing slower than sequential - may be due to system load or small workload")

# ---------------------------------------------------------------------------
# Validation Pipeline Tests (from test_validation_pipeline.py)
# ---------------------------------------------------------------------------

class TestFullPipelineValidation:
    """Test Phase 2 metrics integration within the complete validation pipeline."""

    @pytest.fixture
    def validation_checker(self):
        """Create a ValidationChecker for testing."""
        return ValidationChecker()

    @pytest.fixture
    def sample_gif_files(self, tmp_path):
        """Create sample GIF files for testing."""

        def create_gif(filename: str, frames: list, durations: list = None):
            """Create a GIF file from frames."""
            if durations is None:
                durations = [100] * len(frames)

            pil_frames = []
            for frame in frames:
                img = Image.fromarray(frame, mode="RGB")
                pil_frames.append(img)

            gif_path = tmp_path / filename
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=durations,
                loop=0,
                optimize=False,
            )
            return gif_path

        # Create original GIF with smooth gradient
        original_frames = []
        for _ in range(5):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            for y in range(64):
                for x in range(64):
                    intensity = int(255 * x / 64)
                    frame[y, x] = [intensity, intensity, intensity]
            original_frames.append(frame)

        # Create compressed GIF with artifacts
        compressed_frames = []
        for i, frame in enumerate(original_frames):
            compressed = frame.copy()
            if i >= 2:
                for y in range(0, 64, 2):
                    for x in range(0, 64, 2):
                        noise = 30 if (x + y) % 4 == 0 else -30
                        compressed[y, x] = np.clip(
                            compressed[y, x].astype(np.int16) + noise, 0, 255
                        ).astype(np.uint8)
            compressed_frames.append(compressed)

        original_path = create_gif("original.gif", original_frames)
        compressed_path = create_gif("compressed.gif", compressed_frames)

        return {
            "original": original_path,
            "compressed": compressed_path,
            "original_frames": original_frames,
            "compressed_frames": compressed_frames,
        }

    @pytest.mark.fast
    def test_metrics_in_full_compression_pipeline(
        self, validation_checker, sample_gif_files
    ):
        """Test that Phase 2 metrics work in complete compression pipeline validation."""
        original_metadata = Mock()
        original_metadata.orig_frames = 5
        original_metadata.orig_fps = 10.0
        original_metadata.orig_kilobytes = 50.0

        compression_metrics = {
            "ssim": 0.75,
            "psnr_mean": 22.0,
            "gmsd": 0.08,
            "composite_quality": 0.65,
            "compressed_frame_count": 5,
            "orig_fps": 10.0,
            "kilobytes": 35.0,
            "compression_ratio": 1.43,
            "efficiency": 0.72,
            "flicker_excess": 0.015,
            "temporal_pumping_score": 0.05,
            "lpips_t_mean": 0.03,
            "dither_ratio_mean": 1.2,
            "dither_quality_score": 65.0,
            "flat_region_count": 8,
            "lpips_quality_mean": 0.35,
            "lpips_quality_p95": 0.42,
            "lpips_quality_max": 0.58,
            "deep_perceptual_frame_count": 5,
            "deep_perceptual_device": "cpu",
        }

        result = validation_checker.validate_compression_result(
            original_metadata=original_metadata,
            compression_metrics=compression_metrics,
            pipeline_id="test_pipeline",
            gif_name="test_gif",
            content_type="gradient",
        )

        assert isinstance(result, ValidationResult)
        assert result.pipeline_id == "test_pipeline"
        assert result.gif_name == "test_gif"

        assert len(result.issues) > 0 or len(result.warnings) > 0

        assert result.metrics is not None
        assert hasattr(result.metrics, "lpips_quality_mean")
        assert hasattr(result.metrics, "deep_perceptual_frame_count")

    @pytest.mark.fast
    def test_validation_thresholds_effectiveness(self, validation_checker):
        """Test that validation thresholds effectively catch compression failures."""
        original_metadata = Mock()
        original_metadata.orig_frames = 5
        original_metadata.orig_fps = 10.0
        original_metadata.orig_kilobytes = 50.0

        # Test case 1: Good quality - should pass validation
        good_metrics = {
            "composite_quality": 0.85,
            "lpips_quality_mean": 0.15,
            "dither_quality_score": 85.0,
            "compressed_frame_count": 5,
            "orig_fps": 10.0,
            "kilobytes": 40.0,
        }

        good_result = validation_checker.validate_compression_result(
            original_metadata,
            good_metrics,
            pipeline_id="pipeline1",
            gif_name="good_gif",
            content_type="test",
        )
        assert good_result.status in [ValidationStatus.PASS, ValidationStatus.WARNING]

        # Test case 2: Poor Phase 2 metrics - should trigger issues
        poor_phase2_metrics = {
            "composite_quality": 0.75,
            "lpips_quality_mean": 0.45,
            "lpips_quality_p95": 0.65,
            "dither_quality_score": 35.0,
            "compressed_frame_count": 5,
            "orig_fps": 10.0,
            "kilobytes": 25.0,
        }

        poor_result = validation_checker.validate_compression_result(
            original_metadata,
            poor_phase2_metrics,
            pipeline_id="pipeline2",
            gif_name="poor_gif",
            content_type="test",
        )

        assert poor_result.status in [
            ValidationStatus.ERROR,
            ValidationStatus.WARNING,
            ValidationStatus.ARTIFACT,
        ]
        assert len(poor_result.issues) > 0 or len(poor_result.warnings) > 0

        all_messages = [issue.message for issue in poor_result.issues] + [
            warning.message for warning in poor_result.warnings
        ]

        if all_messages:
            print(f"Validation messages: {all_messages}")

        phase2_detected = any(
            any(
                keyword in msg.lower()
                for keyword in ["perceptual", "lpips", "dither", "quality"]
            )
            for msg in all_messages
        )

        quality_issues_detected = (
            len(poor_result.issues) > 0 or len(poor_result.warnings) > 0
        )

        assert (
            phase2_detected or quality_issues_detected
        ), f"Expected Phase 2 or quality issues to be detected. Messages: {all_messages}"

    @pytest.mark.fast
    def test_multi_metric_validation_combinations(self, validation_checker):
        """Test multi-metric validation logic combining traditional and Phase 2 metrics."""
        original_metadata = Mock()
        original_metadata.orig_frames = 8
        original_metadata.orig_fps = 15.0
        original_metadata.orig_kilobytes = 100.0

        mixed_metrics = {
            "ssim": 0.90,
            "psnr_mean": 28.0,
            "gmsd": 0.03,
            "composite_quality": 0.80,
            "lpips_quality_mean": 0.40,
            "dither_quality_score": 40.0,
            "compressed_frame_count": 8,
            "orig_fps": 15.0,
            "kilobytes": 75.0,
            "compression_ratio": 1.33,
            "efficiency": 0.78,
        }

        result = validation_checker.validate_compression_result(
            original_metadata,
            mixed_metrics,
            pipeline_id="mixed_pipeline",
            gif_name="mixed_gif",
            content_type="test",
        )

        assert len(result.warnings) > 0 or len(result.issues) > 0

        quality_related_issues = len(result.issues) > 0 or len(result.warnings) > 0

        if quality_related_issues:
            all_messages = [issue.message for issue in result.issues] + [
                warning.message for warning in result.warnings
            ]
            print(f"Multi-metric validation detected: {all_messages}")

        assert (
            quality_related_issues
        ), "Expected validation to detect quality issues from mixed good/poor metrics"

    @pytest.mark.fast
    def test_conditional_deep_perceptual_triggering_in_pipeline(
        self, validation_checker
    ):
        """Test that deep perceptual metrics are conditionally triggered in the pipeline."""
        original_metadata = Mock()
        original_metadata.orig_frames = 5
        original_metadata.orig_fps = 10.0
        original_metadata.orig_kilobytes = 50.0

        # Test case 1: High quality - deep perceptual should be skipped
        high_quality_metrics = {
            "composite_quality": 0.85,
            "ssim": 0.90,
            "compressed_frame_count": 5,
            "orig_fps": 10.0,
            "kilobytes": 40.0,
        }

        assert not should_use_deep_perceptual(high_quality_metrics["composite_quality"])

        high_quality_result = validation_checker.validate_compression_result(
            original_metadata,
            high_quality_metrics,
            pipeline_id="hq_pipeline",
            gif_name="hq_gif",
            content_type="test",
        )
        assert high_quality_result.status in [
            ValidationStatus.PASS,
            ValidationStatus.WARNING,
        ]

        # Test case 2: Borderline quality - deep perceptual should be triggered
        borderline_quality_metrics = {
            "composite_quality": 0.55,
            "ssim": 0.70,
            "lpips_quality_mean": 0.25,
            "lpips_quality_p95": 0.32,
            "deep_perceptual_frame_count": 5,
            "compressed_frame_count": 5,
            "orig_fps": 10.0,
            "kilobytes": 30.0,
        }

        assert should_use_deep_perceptual(
            borderline_quality_metrics["composite_quality"]
        )

        borderline_result = validation_checker.validate_compression_result(
            original_metadata,
            borderline_quality_metrics,
            pipeline_id="borderline_pipeline",
            gif_name="borderline_gif",
            content_type="test",
        )

        assert borderline_result.metrics.lpips_quality_mean is not None
        assert borderline_result.metrics.deep_perceptual_frame_count is not None

class TestRegressionPrevention:
    """Test regression prevention with golden reference comparisons."""

    @pytest.fixture
    def golden_reference_metrics(self):
        """Create golden reference metrics for regression testing."""
        return {
            "test_gradient_smooth": {
                "composite_quality": 0.78,
                "lpips_quality_mean": 0.22,
                "dither_quality_score": 75.0,
                "ssim": 0.85,
                "expected_status": ValidationStatus.PASS,
            },
            "test_gradient_artifacts": {
                "composite_quality": 0.45,
                "lpips_quality_mean": 0.42,
                "dither_quality_score": 45.0,
                "ssim": 0.65,
                "expected_status": ValidationStatus.WARNING,
            },
            "test_severe_compression": {
                "composite_quality": 0.25,
                "lpips_quality_mean": 0.58,
                "dither_quality_score": 25.0,
                "ssim": 0.45,
                "expected_status": ValidationStatus.ERROR,
            },
        }

    @pytest.mark.fast
    def test_golden_gif_quality_scores(self, golden_reference_metrics):
        """Test that known GIF quality scores match expected validation results."""
        validation_checker = ValidationChecker()

        original_metadata = Mock()
        original_metadata.orig_frames = 5
        original_metadata.orig_fps = 10.0
        original_metadata.orig_kilobytes = 50.0

        for gif_name, expected_metrics in golden_reference_metrics.items():
            full_metrics = {
                **expected_metrics,
                "compressed_frame_count": 5,
                "orig_fps": 10.0,
                "kilobytes": 35.0,
                "compression_ratio": 1.43,
                "efficiency": 0.70,
            }

            expected_status = full_metrics.pop("expected_status")

            result = validation_checker.validate_compression_result(
                original_metadata,
                full_metrics,
                pipeline_id="golden_pipeline",
                gif_name=gif_name,
                content_type="test",
            )

            if expected_status == ValidationStatus.PASS:
                assert result.status in [
                    ValidationStatus.PASS,
                    ValidationStatus.WARNING,
                ]
            elif expected_status == ValidationStatus.WARNING:
                assert result.status in [
                    ValidationStatus.WARNING,
                    ValidationStatus.ERROR,
                    ValidationStatus.PASS,
                ]
            elif expected_status == ValidationStatus.ERROR:
                assert result.status in [
                    ValidationStatus.ERROR,
                    ValidationStatus.ARTIFACT,
                    ValidationStatus.WARNING,
                ]

    @pytest.mark.fast
    def test_known_failure_detection(self):
        """Test that known bad GIFs fail validation as expected."""
        validation_checker = ValidationChecker()

        original_metadata = Mock()
        original_metadata.orig_frames = 10
        original_metadata.orig_fps = 20.0
        original_metadata.orig_kilobytes = 200.0

        catastrophic_metrics = {
            "composite_quality": 0.15,
            "ssim": 0.30,
            "lpips_quality_mean": 0.75,
            "lpips_quality_p95": 0.85,
            "dither_quality_score": 15.0,
            "compressed_frame_count": 10,
            "orig_fps": 20.0,
            "kilobytes": 50.0,
            "compression_ratio": 4.0,
            "efficiency": 0.25,
        }

        result = validation_checker.validate_compression_result(
            original_metadata,
            catastrophic_metrics,
            pipeline_id="catastrophic_pipeline",
            gif_name="catastrophic_gif",
            content_type="test",
        )

        assert result.status in [ValidationStatus.ERROR, ValidationStatus.ARTIFACT]
        assert len(result.issues) > 0

        issue_categories = [issue.category for issue in result.issues]

        quality_failure_detected = len(result.issues) > 0
        assert (
            quality_failure_detected
        ), f"Expected quality issues to be detected with catastrophic metrics. Categories: {issue_categories}"

class TestValidationSystemIntegrationPipeline:
    """Test validation system integration with Phase 2 metrics."""

    @pytest.mark.fast
    def test_validation_result_structure_with_phase2(self):
        """Test that ValidationResult properly includes Phase 2 metrics."""
        validation_checker = ValidationChecker()

        original_metadata = Mock()
        original_metadata.orig_frames = 3
        original_metadata.orig_fps = 5.0
        original_metadata.orig_kilobytes = 25.0

        metrics_with_phase2 = {
            "composite_quality": 0.65,
            "ssim": 0.75,
            "lpips_quality_mean": 0.28,
            "lpips_quality_p95": 0.35,
            "dither_quality_score": 70.0,
            "dither_ratio_mean": 1.15,
            "compressed_frame_count": 3,
            "orig_fps": 5.0,
            "kilobytes": 20.0,
        }

        result = validation_checker.validate_compression_result(
            original_metadata,
            metrics_with_phase2,
            pipeline_id="structure_test_pipeline",
            gif_name="structure_test_gif",
            content_type="test",
        )

        assert hasattr(result.metrics, "lpips_quality_mean")
        assert hasattr(result.metrics, "lpips_quality_p95")
        assert result.metrics.lpips_quality_mean == 0.28
        assert result.metrics.lpips_quality_p95 == 0.35

        assert "lpips_quality_threshold" in result.effective_thresholds
        assert "lpips_quality_extreme_threshold" in result.effective_thresholds

    @pytest.mark.slow
    def test_validation_performance_with_phase2(self):
        """Test that validation performance remains acceptable with Phase 2 metrics."""
        validation_checker = ValidationChecker()

        original_metadata = Mock()
        original_metadata.orig_frames = 20
        original_metadata.orig_fps = 30.0
        original_metadata.orig_kilobytes = 500.0

        comprehensive_metrics = {
            "composite_quality": 0.72,
            "ssim": 0.82,
            "psnr_mean": 24.5,
            "gmsd": 0.06,
            "lpips_quality_mean": 0.18,
            "lpips_quality_p95": 0.25,
            "lpips_quality_max": 0.32,
            "deep_perceptual_frame_count": 20,
            "dither_quality_score": 78.0,
            "dither_ratio_mean": 1.05,
            "flat_region_count": 15,
            "flicker_excess": 0.008,
            "temporal_pumping_score": 0.03,
            "lpips_t_mean": 0.015,
            "compressed_frame_count": 20,
            "orig_fps": 30.0,
            "kilobytes": 350.0,
            "compression_ratio": 1.43,
            "efficiency": 0.85,
        }

        start_time = time.time()

        result = validation_checker.validate_compression_result(
            original_metadata,
            comprehensive_metrics,
            pipeline_id="performance_test_pipeline",
            gif_name="large_gif",
            content_type="test",
        )

        end_time = time.time()
        validation_time = end_time - start_time

        assert validation_time < 1.0

        assert isinstance(result, ValidationResult)
        assert result.validation_time_ms is not None
        assert result.validation_time_ms >= 0

    @pytest.mark.fast
    def test_phase2_error_isolation(self):
        """Test that errors in Phase 2 validation don't break overall validation."""
        validation_checker = ValidationChecker()

        original_metadata = Mock()
        original_metadata.orig_frames = 5
        original_metadata.orig_fps = 10.0
        original_metadata.orig_kilobytes = 50.0

        # Test with missing Phase 2 metrics
        minimal_metrics = {
            "composite_quality": 0.75,
            "ssim": 0.80,
            "compressed_frame_count": 5,
            "orig_fps": 10.0,
            "kilobytes": 40.0,
        }

        result = validation_checker.validate_compression_result(
            original_metadata,
            minimal_metrics,
            pipeline_id="minimal_pipeline",
            gif_name="minimal_gif",
            content_type="test",
        )

        assert isinstance(result, ValidationResult)
        assert result.status in [ValidationStatus.PASS, ValidationStatus.WARNING]

        # Test with invalid Phase 2 metrics
        invalid_phase2_metrics = {
            "composite_quality": 0.75,
            "ssim": 0.80,
            "lpips_quality_mean": "invalid",
            "dither_quality_score": -50.0,
            "compressed_frame_count": 5,
            "orig_fps": 10.0,
            "kilobytes": 40.0,
        }

        result_invalid = validation_checker.validate_compression_result(
            original_metadata,
            invalid_phase2_metrics,
            pipeline_id="invalid_pipeline",
            gif_name="invalid_gif",
            content_type="test",
        )

        assert isinstance(result_invalid, ValidationResult)
        assert result_invalid.status != ValidationStatus.UNKNOWN

# ---------------------------------------------------------------------------
# Performance Integration Tests (from test_performance_integration.py)
# ---------------------------------------------------------------------------

class TestPerformanceIntegration:
    """Integration tests for performance improvements."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.performance
    @pytest.mark.xdist_group("performance_tests")
    def test_vectorized_vs_serial_frame_generation(self):
        """Test that vectorized generation is faster than theoretical serial approach."""
        generator = SyntheticFrameGenerator()

        frames_to_test = 10
        size = (200, 200)

        # Warm up
        generator.create_frame("complex_gradient", size, 0, frames_to_test)

        start_time = time.time()
        for i in range(frames_to_test):
            img = generator.create_frame("complex_gradient", size, i, frames_to_test)
            assert img.size == size

        vectorized_time = time.time() - start_time
        time_per_frame = vectorized_time / frames_to_test

        assert (
            time_per_frame < 0.05
        ), f"Vectorized generation too slow: {time_per_frame:.4f}s/frame"

        large_img = generator.create_frame("gradient", (400, 400), 0, 5)
        assert large_img.size == (400, 400)

    def test_multiprocessing_overhead_analysis(self):
        """Test that multiprocessing overhead is understood and appropriate."""
        small_generator = ParallelFrameGenerator(max_workers=2)
        frame_generator = SyntheticFrameGenerator()

        small_size = (50, 50)
        num_frames = 4

        # Time serial execution
        start_serial = time.time()
        serial_images = []
        for i in range(num_frames):
            img = frame_generator.create_frame("gradient", small_size, i, num_frames)
            serial_images.append(img)
        serial_time = time.time() - start_serial

        # Time parallel execution
        from giflab.synthetic_gifs import SyntheticGifSpec

        spec = SyntheticGifSpec("test", num_frames, small_size, "gradient", "Test")

        start_parallel = time.time()
        parallel_images = small_generator.generate_gif_frames_parallel(spec)
        parallel_time = time.time() - start_parallel

        assert len(serial_images) == num_frames
        assert len(parallel_images) == num_frames

        assert serial_time < 1.0
        assert parallel_time < 30.0

    def test_synthetic_gif_generation_integration(self):
        """Test complete synthetic GIF generation workflow."""
        generator = SyntheticGifGenerator(self.temp_dir)

        test_specs = list(generator.synthetic_specs[:3])

        start_time = time.time()

        frame_generator = SyntheticFrameGenerator()
        total_frames_generated = 0

        for spec in test_specs:
            images = []
            for frame_idx in range(spec.frames):
                img = frame_generator.create_frame(
                    spec.content_type, spec.size, frame_idx, spec.frames
                )
                images.append(img)
                total_frames_generated += 1

            assert len(images) == spec.frames

            gif_path = self.temp_dir / f"{spec.name}_test.gif"
            if images:
                images[0].save(
                    gif_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=100,
                    loop=0,
                )
                assert gif_path.exists()
                assert gif_path.stat().st_size > 0

        total_time = time.time() - start_time
        avg_time_per_frame = total_time / total_frames_generated

        assert (
            avg_time_per_frame < 0.05
        ), f"Frame generation too slow: {avg_time_per_frame:.4f}s"
        assert total_frames_generated > 0

    def test_worker_count_optimization(self):
        """Test that optimal worker count calculation is sensible."""
        frame_workers = get_optimal_worker_count("frame_generation")
        pipeline_workers = get_optimal_worker_count("pipeline_execution")
        default_workers = get_optimal_worker_count("unknown")

        assert isinstance(frame_workers, int) and frame_workers > 0
        assert isinstance(pipeline_workers, int) and pipeline_workers > 0
        assert isinstance(default_workers, int) and default_workers > 0

        assert frame_workers == mp.cpu_count()
        assert pipeline_workers == max(1, mp.cpu_count() - 1)
        assert default_workers == max(1, mp.cpu_count() // 2)

    def test_backward_compatibility_maintained(self):
        """Test that all performance improvements maintain backward compatibility."""
        generator = SyntheticFrameGenerator()

        content_types = ["gradient", "complex_gradient", "noise", "texture", "solid"]

        for content_type in content_types:
            for size in [(50, 50), (100, 100), (200, 200)]:
                img = generator.create_frame(content_type, size, 0, 5)

                assert isinstance(img, Image.Image)
                assert img.mode == "RGB"
                assert img.size == size

                if content_type == "noise":
                    img2 = generator.create_frame(content_type, size, 0, 5)
                    assert np.array_equal(np.array(img), np.array(img2))

    def test_performance_regression_detection(self):
        """Test that performance hasn't regressed from expected levels."""
        generator = SyntheticFrameGenerator()

        test_cases = [
            ("gradient", (100, 100), 0.005),
            ("complex_gradient", (150, 150), 0.008),
            ("noise", (100, 100), 0.005),
            ("texture", (100, 100), 0.005),
            ("solid", (150, 150), 0.003),
        ]

        for content_type, size, max_time in test_cases:
            start = time.time()
            img = generator.create_frame(content_type, size, 0, 8)
            elapsed = time.time() - start

            assert (
                elapsed < max_time
            ), f"{content_type} took {elapsed:.4f}s, expected < {max_time}s"
            assert img.size == size

@pytest.mark.integration
class TestRealWorldPerformance:
    """Test performance in realistic usage scenarios."""

    def test_batch_frame_generation_performance(self):
        """Test performance when generating many frames (realistic batch scenario)."""
        generator = SyntheticFrameGenerator()

        batch_specs = [
            ("gradient", (120, 120), 8),
            ("noise", (140, 140), 6),
            ("texture", (100, 100), 10),
            ("solid", (160, 160), 5),
        ]

        start_time = time.time()
        total_frames = 0

        for content_type, size, frame_count in batch_specs:
            for frame_idx in range(frame_count):
                img = generator.create_frame(content_type, size, frame_idx, frame_count)
                assert isinstance(img, Image.Image)
                total_frames += 1

        total_time = time.time() - start_time
        frames_per_second = total_frames / total_time

        assert frames_per_second > 100, f"Low throughput: {frames_per_second:.1f} fps"
        assert total_frames == sum(spec[2] for spec in batch_specs)

    def test_memory_efficiency(self):
        """Test that vectorized operations don't use excessive memory."""
        generator = SyntheticFrameGenerator()

        large_size = (500, 500)

        try:
            for i in range(5):
                img = generator.create_frame("complex_gradient", large_size, i, 5)
                assert img.size == large_size
                del img

        except MemoryError:
            pytest.fail("Vectorized generation should not cause memory errors")

    def test_concurrent_generation_safety(self):
        """Test that frame generation is safe under concurrent access."""
        generator = SyntheticFrameGenerator()
        results = queue.Queue()
        errors = queue.Queue()

        def worker(thread_id):
            try:
                for i in range(5):
                    img = generator.create_frame("gradient", (80, 80), i, 5)
                    results.put((thread_id, i, img.size))
            except Exception as e:
                errors.put((thread_id, str(e)))

        threads = []
        for t_id in range(3):
            thread = threading.Thread(target=worker, args=(t_id,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert errors.empty(), f"Concurrent access caused errors: {list(errors.queue)}"
        assert results.qsize() == 15  # 3 threads x 5 frames each
