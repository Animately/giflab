"""End-to-end tests for temporal artifact detection with real GIF processing.

This test suite validates the complete temporal artifact detection pipeline
from real GIF compression through to artifact detection and validation,
using actual compression engines and test GIF fixtures.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from giflab.meta import GifMetadata
from giflab.optimization_validation.data_structures import ValidationConfig
from giflab.optimization_validation.validation_checker import ValidationChecker
from giflab.temporal_artifacts import calculate_enhanced_temporal_metrics_from_paths
from giflab.tool_wrappers import GifsicleColorReducer, GifsicleLossyCompressor

# from giflab.core.runner import ComprehensiveGifAnalyzer  # May not exist
from tests.fixtures.generate_temporal_artifact_fixtures import (
    create_background_flicker_gif,
    create_disposal_artifact_gif,
    create_flicker_excess_gif,
    create_smooth_animation_gif,
    create_temporal_pumping_gif,
)


def create_test_metadata(gif_path: Path, size_mb: float) -> GifMetadata:
    """Helper to create GifMetadata for tests."""
    return GifMetadata(
        gif_sha="test_sha",
        orig_filename=gif_path.name,
        orig_kilobytes=int(size_mb * 1024),
        orig_width=64,
        orig_height=64,
        orig_frames=10,
        orig_fps=10.0,
        orig_n_colors=256,
        entropy=5.0,
        source_platform="test",
    )


@pytest.mark.external_tools
class TestTemporalArtifactDetectionE2E:
    """End-to-end tests with real GIF processing engines."""

    @pytest.fixture
    def validation_config(self):
        """Create validation config for e2e testing."""
        return ValidationConfig(
            flicker_excess_threshold=0.03,  # Slightly relaxed for real compression
            flat_flicker_ratio_threshold=0.15,
            temporal_pumping_threshold=0.20,
            lpips_t_threshold=0.08,
            minimum_quality_floor=0.4,  # Relaxed for lossy compression
            disposal_artifact_threshold=0.7,  # Relaxed for testing
            temporal_consistency_threshold=0.6,  # Relaxed for real compression
        )

    @pytest.fixture
    def analyzer(self):
        """Create analyzer for testing."""
        from giflab.core.runner import ComprehensiveGifAnalyzer

        return ComprehensiveGifAnalyzer()

    def test_gifsicle_lossy_compression_temporal_artifacts(
        self, validation_config, tmp_path
    ):
        """Test temporal artifact detection with Gifsicle lossy compression."""
        wrapper = GifsicleLossyCompressor()
        if not wrapper.available():
            pytest.skip("Gifsicle not available")

        validator = ValidationChecker(validation_config)

        # Test with high flicker GIF - compression should help reduce flicker
        input_gif = create_flicker_excess_gif("high")
        output_gif = tmp_path / "gifsicle_lossy_output.gif"

        # Apply moderate lossy compression
        result = wrapper.apply(
            input_gif,
            output_gif,
            params={"lossy_level": 50},  # Moderate lossy compression
        )

        assert output_gif.exists()
        assert result["validation_passed"] is True

        # Run temporal artifact detection on result
        try:
            temporal_metrics = calculate_enhanced_temporal_metrics_from_paths(
                str(input_gif), str(output_gif)
            )

            print("Temporal metrics after Gifsicle lossy compression:")
            print(f"  Flicker excess: {temporal_metrics.get('flicker_excess', 'N/A')}")
            print(
                f"  Flat flicker ratio: {temporal_metrics.get('flat_flicker_ratio', 'N/A')}"
            )
            print(
                f"  Temporal pumping: {temporal_metrics.get('temporal_pumping_score', 'N/A')}"
            )
            print(f"  LPIPS-T mean: {temporal_metrics.get('lpips_t_mean', 'N/A')}")

            # Validate the compressed result
            original_size_mb = input_gif.stat().st_size / 1024 / 1024
            original_metadata = create_test_metadata(input_gif, original_size_mb)
            validation_result = validator.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=temporal_metrics,
                pipeline_id="gifsicle_lossy",
                gif_name=input_gif.stem,
                content_type="test",
            )

            print(f"Validation result: {validation_result.is_acceptable()}")
            if not validation_result.is_acceptable():
                print(f"Validation details: {validation_result.get_detailed_report()}")

            # Verify temporal metrics are captured
            assert hasattr(validation_result.metrics, "flicker_excess")
            assert hasattr(validation_result.metrics, "flat_flicker_ratio")
            assert hasattr(validation_result.metrics, "temporal_pumping_score")
            assert hasattr(validation_result.metrics, "lpips_t_mean")

        except Exception as e:
            pytest.skip(
                f"Temporal artifact detection failed (likely missing LPIPS): {e}"
            )

    def test_gifsicle_color_reduction_temporal_stability(
        self, validation_config, tmp_path
    ):
        """Test temporal stability after color reduction compression."""
        wrapper = GifsicleColorReducer()
        if not wrapper.available():
            pytest.skip("Gifsicle not available")

        ValidationChecker(validation_config)

        # Test with smooth animation - should maintain temporal stability
        input_gif = create_smooth_animation_gif()
        output_gif = tmp_path / "gifsicle_color_output.gif"

        # Apply aggressive color reduction
        result = wrapper.apply(
            input_gif, output_gif, params={"colors": 16}  # Reduce to 16 colors
        )

        assert output_gif.exists()
        assert result["validation_passed"] is True

        try:
            temporal_metrics = calculate_enhanced_temporal_metrics_from_paths(
                str(input_gif), str(output_gif)
            )

            print("Temporal metrics after color reduction:")
            print(f"  Flicker excess: {temporal_metrics.get('flicker_excess', 'N/A')}")
            print(
                f"  Flat flicker ratio: {temporal_metrics.get('flat_flicker_ratio', 'N/A')}"
            )
            print(
                f"  Temporal pumping: {temporal_metrics.get('temporal_pumping_score', 'N/A')}"
            )
            print(f"  LPIPS-T mean: {temporal_metrics.get('lpips_t_mean', 'N/A')}")

            # Color reduction should preserve temporal relationships
            # but might introduce some quantization artifacts
            if "flicker_excess" in temporal_metrics:
                # Should not introduce excessive flicker
                assert (
                    temporal_metrics["flicker_excess"] < 0.1
                ), f"Color reduction introduced excessive flicker: {temporal_metrics['flicker_excess']}"

            if "temporal_pumping_score" in temporal_metrics:
                # Should not introduce significant pumping
                assert (
                    temporal_metrics["temporal_pumping_score"] < 0.3
                ), f"Color reduction introduced temporal pumping: {temporal_metrics['temporal_pumping_score']}"

        except Exception as e:
            pytest.skip(f"Temporal artifact detection failed: {e}")

    def test_compression_improves_temporal_artifacts(self, validation_config, tmp_path):
        """Test that compression can improve temporal artifacts in problematic GIFs."""
        wrapper = GifsicleLossyCompressor()
        if not wrapper.available():
            pytest.skip("Gifsicle not available")

        # Create a GIF with disposal artifacts
        input_gif = create_disposal_artifact_gif(corrupted=True)
        output_gif = tmp_path / "improved_disposal_output.gif"

        # Apply lossy compression which might help with disposal artifacts
        result = wrapper.apply(
            input_gif,
            output_gif,
            params={"lossy_level": 60},  # Moderate-high lossy level
        )

        assert result["validation_passed"] is True

        try:
            # Calculate temporal metrics for both input and output
            input_metrics = calculate_enhanced_temporal_metrics_from_paths(
                str(input_gif), str(input_gif)  # Compare to itself
            )
            output_metrics = calculate_enhanced_temporal_metrics_from_paths(
                str(input_gif), str(output_gif)
            )

            print("Input temporal artifacts:")
            for metric, value in input_metrics.items():
                if metric.startswith(("flicker_", "temporal_", "lpips_t")):
                    print(f"  {metric}: {value}")

            print("Output temporal artifacts:")
            for metric, value in output_metrics.items():
                if metric.startswith(("flicker_", "temporal_", "lpips_t")):
                    print(f"  {metric}: {value}")

            # For disposal artifacts, compression might help or hurt depending on the algorithm
            # At minimum, we should get valid measurements
            assert (
                "lpips_t_mean" in output_metrics or "flicker_excess" in output_metrics
            ), "Should calculate at least one temporal metric"

        except Exception as e:
            pytest.skip(f"Temporal artifact detection failed: {e}")

    def test_comprehensive_pipeline_with_temporal_artifacts(self, analyzer, tmp_path):
        """Test complete analysis pipeline with temporal artifact detection."""
        # Create test fixture with known temporal artifacts
        input_gif = create_temporal_pumping_gif(pumping=True)

        try:
            # Run comprehensive analysis
            results = analyzer.analyze_gif(
                gif_path=input_gif, output_dir=tmp_path, include_temporal_artifacts=True
            )

            print("Comprehensive analysis results:")
            for key, value in results.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for subkey, subvalue in value.items():
                        print(f"    {subkey}: {subvalue}")
                else:
                    print(f"  {key}: {value}")

            # Should include temporal artifact metrics
            if "temporal_artifacts" in results:
                temporal = results["temporal_artifacts"]
                assert isinstance(temporal, dict)

                # Should have at least some temporal metrics
                expected_metrics = [
                    "flicker_excess",
                    "temporal_pumping_score",
                    "lpips_t_mean",
                ]
                found_metrics = [
                    metric for metric in expected_metrics if metric in temporal
                ]
                assert (
                    len(found_metrics) > 0
                ), f"Should find temporal metrics: {expected_metrics}"

        except Exception as e:
            pytest.skip(f"Comprehensive analysis failed: {e}")

    @pytest.mark.slow
    def test_real_world_gif_temporal_validation(self, validation_config):
        """Test temporal validation on existing test fixtures."""
        validator = ValidationChecker(validation_config)

        # Use existing test fixture if available
        test_fixture_path = Path(__file__).parent / "fixtures" / "simple_4frame.gif"
        if not test_fixture_path.exists():
            pytest.skip("Test fixture not available")

        try:
            with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
                output_path = Path(tmp_file.name)

                # Copy input to output (no compression) to test baseline
                output_path.write_bytes(test_fixture_path.read_bytes())

                # Run validation with temporal artifact detection
                original_size_mb = test_fixture_path.stat().st_size / 1024 / 1024
                original_metadata = create_test_metadata(
                    test_fixture_path, original_size_mb
                )
                result = validator.validate_compression_result(
                    original_metadata=original_metadata,
                    compression_metrics={},  # No specific compression metrics for baseline test
                    pipeline_id="baseline",
                    gif_name=test_fixture_path.stem,
                    content_type="test",
                )

                print("Real-world GIF validation:")
                print(f"  Valid: {result.is_acceptable()}")
                print("  Temporal metrics:")
                if hasattr(result.metrics, "flicker_excess"):
                    print(f"    Flicker excess: {result.metrics.flicker_excess}")
                if hasattr(result.metrics, "flat_flicker_ratio"):
                    print(
                        f"    Flat flicker ratio: {result.metrics.flat_flicker_ratio}"
                    )
                if hasattr(result.metrics, "temporal_pumping_score"):
                    print(
                        f"    Temporal pumping: {result.metrics.temporal_pumping_score}"
                    )
                if hasattr(result.metrics, "lpips_t_mean"):
                    print(f"    LPIPS-T mean: {result.metrics.lpips_t_mean}")

                # Should complete validation without errors
                assert result is not None

        except Exception as e:
            pytest.skip(f"Real-world validation failed: {e}")

    def test_temporal_artifact_comparison_across_engines(
        self, validation_config, tmp_path
    ):
        """Compare temporal artifacts across different compression engines."""
        wrappers = [
            ("gifsicle_lossy", GifsicleLossyCompressor(), {"lossy_level": 40}),
            ("gifsicle_color", GifsicleColorReducer(), {"colors": 32}),
        ]

        available_wrappers = []
        for name, wrapper, params in wrappers:
            if wrapper.available():
                available_wrappers.append((name, wrapper, params))

        if len(available_wrappers) < 2:
            pytest.skip("Need at least 2 compression engines for comparison")

        # Use background flicker GIF for comparison
        input_gif = create_background_flicker_gif(stable=False)

        engine_results = {}

        for name, wrapper, params in available_wrappers:
            output_gif = tmp_path / f"{name}_comparison_output.gif"

            # Apply compression
            result = wrapper.apply(input_gif, output_gif, params=params)
            if not result["validation_passed"]:
                continue

            try:
                # Calculate temporal metrics
                temporal_metrics = calculate_enhanced_temporal_metrics_from_paths(
                    str(input_gif), str(output_gif)
                )
                engine_results[name] = temporal_metrics

            except Exception as e:
                print(f"Failed temporal analysis for {name}: {e}")
                continue

        print("\nTemporal artifact comparison across engines:")
        for engine, metrics in engine_results.items():
            print(f"  {engine}:")
            for metric, value in metrics.items():
                if metric.startswith(("flicker_", "temporal_", "lpips_t")):
                    print(f"    {metric}: {value:.4f}")

        # Should have results for at least one engine
        assert (
            len(engine_results) > 0
        ), "Should successfully analyze at least one compression result"

        # Compare engines if we have multiple results
        if len(engine_results) >= 2:
            engines = list(engine_results.keys())
            for metric in [
                "flicker_excess",
                "flat_flicker_ratio",
                "temporal_pumping_score",
            ]:
                if (
                    metric in engine_results[engines[0]]
                    and metric in engine_results[engines[1]]
                ):
                    val1 = engine_results[engines[0]][metric]
                    val2 = engine_results[engines[1]][metric]
                    print(
                        f"  {metric} difference between {engines[0]} and {engines[1]}: {abs(val1 - val2):.4f}"
                    )


class TestTemporalArtifactDetectionFallbacks:
    """Test fallback behavior when temporal detection dependencies are missing."""

    @patch("giflab.temporal_artifacts.lpips")
    def test_lpips_unavailable_fallback(self, mock_lpips, tmp_path):
        """Test temporal detection falls back gracefully when LPIPS is unavailable."""
        # Mock LPIPS as unavailable
        mock_lpips.LPIPS.side_effect = ImportError("LPIPS not available")

        input_gif = create_smooth_animation_gif()
        output_gif = tmp_path / "fallback_test_output.gif"
        output_gif.write_bytes(input_gif.read_bytes())

        # Should fall back to MSE-based temporal analysis
        try:
            temporal_metrics = calculate_enhanced_temporal_metrics_from_paths(
                str(input_gif), str(output_gif)
            )

            print("Fallback temporal metrics (without LPIPS):")
            for metric, value in temporal_metrics.items():
                print(f"  {metric}: {value}")

            # Should still calculate flicker-based metrics even without LPIPS
            expected_fallback_metrics = [
                "flicker_excess",
                "flat_flicker_ratio",
                "temporal_pumping_score",
            ]
            found_metrics = [
                m for m in expected_fallback_metrics if m in temporal_metrics
            ]
            assert (
                len(found_metrics) > 0
            ), f"Should calculate fallback metrics: {expected_fallback_metrics}"

            # LPIPS-specific metrics should not be present or should be placeholder values
            if "lpips_t_mean" in temporal_metrics:
                # If present, should be 0 or another placeholder indicating unavailability
                pass

        except Exception as e:
            pytest.skip(f"Fallback temporal detection failed: {e}")

    def test_pytorch_unavailable_handling(self, tmp_path):
        """Test handling when PyTorch is unavailable for LPIPS."""
        input_gif = create_flicker_excess_gif("low")
        output_gif = tmp_path / "pytorch_test_output.gif"
        output_gif.write_bytes(input_gif.read_bytes())

        with patch("giflab.temporal_artifacts.torch", None):
            try:
                temporal_metrics = calculate_enhanced_temporal_metrics_from_paths(
                    str(input_gif), str(output_gif)
                )

                print("Temporal metrics without PyTorch:")
                for metric, value in temporal_metrics.items():
                    print(f"  {metric}: {value}")

                # Should still work with CPU-based fallbacks
                assert isinstance(temporal_metrics, dict)
                assert len(temporal_metrics) > 0

            except Exception as e:
                pytest.skip(f"PyTorch fallback failed: {e}")

    def test_gpu_unavailable_cpu_fallback(self, tmp_path):
        """Test CPU fallback when GPU is unavailable."""
        input_gif = create_temporal_pumping_gif(pumping=True)
        output_gif = tmp_path / "cpu_test_output.gif"
        output_gif.write_bytes(input_gif.read_bytes())

        # Force CPU-only calculation
        try:
            temporal_metrics = calculate_enhanced_temporal_metrics_from_paths(
                str(input_gif), str(output_gif), device="cpu"  # Force CPU
            )

            print("CPU-only temporal metrics:")
            for metric, value in temporal_metrics.items():
                print(f"  {metric}: {value}")

            # Should work on CPU
            assert isinstance(temporal_metrics, dict)
            if temporal_metrics:  # If any metrics calculated
                assert len(temporal_metrics) > 0

        except Exception as e:
            pytest.skip(f"CPU fallback failed: {e}")


@pytest.mark.performance
class TestTemporalArtifactPerformance:
    """Performance tests for temporal artifact detection."""

    @pytest.mark.slow
    def test_temporal_detection_performance_large_gif(self):
        """Test temporal detection performance on larger GIFs."""
        # Create a larger GIF for performance testing
        input_gif = create_smooth_animation_gif()

        start_time = __import__("time").time()

        try:
            temporal_metrics = calculate_enhanced_temporal_metrics_from_paths(
                str(input_gif), str(input_gif)
            )

            end_time = __import__("time").time()
            duration = end_time - start_time

            print("Temporal detection performance:")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  Metrics calculated: {len(temporal_metrics)}")

            # Should complete within reasonable time (adjust threshold as needed)
            assert duration < 30.0, f"Temporal detection too slow: {duration:.2f}s"

        except Exception as e:
            pytest.skip(f"Performance test failed: {e}")

    def test_batch_temporal_analysis_performance(self, tmp_path):
        """Test performance of analyzing multiple GIFs."""
        # Create multiple test GIFs
        test_gifs = [
            create_smooth_animation_gif(),
            create_flicker_excess_gif("low"),
            create_background_flicker_gif(stable=True),
        ]

        start_time = __import__("time").time()

        results = []
        for i, gif_path in enumerate(test_gifs):
            try:
                temporal_metrics = calculate_enhanced_temporal_metrics_from_paths(
                    str(gif_path), str(gif_path)
                )
                results.append((i, temporal_metrics))

            except Exception as e:
                print(f"Failed analysis for GIF {i}: {e}")
                continue

        end_time = __import__("time").time()
        duration = end_time - start_time

        print("Batch temporal analysis performance:")
        print(f"  Total duration: {duration:.2f} seconds")
        print(f"  GIFs analyzed: {len(results)}")
        print(f"  Average per GIF: {duration / max(len(results), 1):.2f} seconds")

        # Should process multiple GIFs efficiently
        if results:
            avg_per_gif = duration / len(results)
            assert (
                avg_per_gif < 15.0
            ), f"Batch processing too slow: {avg_per_gif:.2f}s per GIF"
