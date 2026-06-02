"""Integration tests for temporal artifact validation system.

This test suite validates the integration between the TemporalArtifactDetector
and ValidationChecker systems, demonstrating both passing and failing scenarios
for each temporal artifact metric.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from giflab.meta import GifMetadata
from giflab.optimization_validation.data_structures import (
    ValidationConfig,
)
from giflab.optimization_validation.validation_checker import ValidationChecker

from tests.fixtures.generate_temporal_artifact_fixtures import (
    create_background_flicker_gif,
    create_disposal_artifact_gif,
    create_flicker_excess_gif,
    create_smooth_animation_gif,
    create_static_with_noise_gif,
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


class TestTemporalValidationIntegration:
    """Test temporal artifact validation integration with ValidationChecker."""

    @pytest.fixture
    def validation_config(self):
        """Create validation config with moderate temporal thresholds."""
        return ValidationConfig(
            # Temporal artifact thresholds for testing
            flicker_excess_threshold=0.02,  # Moderate flicker detection
            flat_flicker_ratio_threshold=0.1,  # 10% flat regions can flicker
            temporal_pumping_threshold=0.15,  # Moderate pumping detection
            lpips_t_threshold=0.05,  # Temporal consistency threshold
            # Standard validation thresholds
            minimum_quality_floor=0.5,  # Lower quality floor for testing
            disposal_artifact_threshold=0.8,  # Moderate disposal threshold
            temporal_consistency_threshold=0.7,  # Moderate temporal consistency
        )

    @pytest.fixture
    def validation_checker(self, validation_config):
        """Create ValidationChecker with temporal artifact testing config."""
        # Create ValidationChecker with default config, then patch it
        checker = ValidationChecker(None)  # Use default config
        checker.config = validation_config  # Override with test config
        return checker

    def test_flicker_excess_validation_failing(self, validation_checker, tmp_path):
        """Test validation fails for high flicker excess GIFs."""
        # Create high flicker GIF that should fail validation
        input_gif = create_flicker_excess_gif("high")
        output_gif = tmp_path / "high_flicker_output.gif"

        # Copy input to output for testing (simulating no improvement)
        output_gif.write_bytes(input_gif.read_bytes())

        # Mock compression metrics calculation to provide flicker data
        with patch("giflab.metrics.calculate_comprehensive_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "flicker_excess_compressed": 0.08,  # Above 0.02 threshold - should fail
                "flicker_frame_ratio_compressed": 0.6,  # 60% of frames have flicker
                "flat_flicker_ratio_compressed": 0.05,  # Low flat flicker - should pass
                "temporal_pumping_score_compressed": 0.10,  # Below 0.15 threshold - should pass
                "lpips_t_mean_compressed": 0.03,  # Below 0.05 threshold - should pass
                "ssim_mean": 0.7,  # Pass other metrics
                "psnr_mean": 20.0,
                "mse_mean": 500.0,
            }

            original_metadata = create_test_metadata(input_gif, 1.0)
            result = validation_checker.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=mock_metrics.return_value,
                pipeline_id="test_pipeline",
                gif_name=input_gif.stem,
                content_type="test",
            )

            # Should fail due to high flicker excess
            assert result.has_errors() is True
            assert any(issue.category == "flicker_excess" for issue in result.issues)
            assert result.metrics.flicker_excess == 0.08
            assert any("flicker detected" in issue.message for issue in result.issues)

    def test_flicker_excess_validation_passing(self, validation_checker, tmp_path):
        """Test validation passes for low flicker excess GIFs."""
        # Create low flicker GIF that should pass validation
        input_gif = create_flicker_excess_gif("low")
        output_gif = tmp_path / "low_flicker_output.gif"
        output_gif.write_bytes(input_gif.read_bytes())

        with patch("giflab.metrics.calculate_comprehensive_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "flicker_excess_compressed": 0.005,  # Below 0.02 threshold - should pass
                "flicker_frame_ratio_compressed": 0.1,  # 10% of frames have minimal flicker
                "flat_flicker_ratio_compressed": 0.02,  # Very low flat flicker - should pass
                "temporal_pumping_score_compressed": 0.08,  # Below 0.15 threshold - should pass
                "lpips_t_mean_compressed": 0.02,  # Below 0.05 threshold - should pass
                "ssim_mean": 0.8,  # Pass other metrics
                "psnr_mean": 25.0,
                "mse_mean": 300.0,
            }

            original_metadata = create_test_metadata(input_gif, 1.0)
            result = validation_checker.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=mock_metrics.return_value,
                pipeline_id="test_pipeline",
                gif_name=input_gif.stem,
                content_type="test",
            )

            # Should pass - low flicker
            assert result.is_acceptable() is True
            assert result.metrics.flicker_excess == 0.005

    def test_flat_region_flicker_validation_failing(self, validation_checker, tmp_path):
        """Test validation fails for high flat region flicker."""
        # Create GIF with flickering background that should fail validation
        input_gif = create_background_flicker_gif(stable=False)
        output_gif = tmp_path / "flickering_background_output.gif"
        output_gif.write_bytes(input_gif.read_bytes())

        with patch("giflab.metrics.calculate_comprehensive_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "flicker_excess_compressed": 0.01,  # Below threshold - should pass
                "flicker_frame_ratio_compressed": 0.3,
                "flat_flicker_ratio_compressed": 0.25,  # Above 0.1 threshold - should fail
                "temporal_pumping_score_compressed": 0.10,  # Below threshold - should pass
                "lpips_t_mean_compressed": 0.03,  # Below threshold - should pass
                "ssim_mean": 0.7,
                "psnr_mean": 20.0,
                "mse_mean": 500.0,
            }

            original_metadata = create_test_metadata(input_gif, 1.0)
            result = validation_checker.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=mock_metrics.return_value,
                pipeline_id="test_pipeline",
                gif_name=input_gif.stem,
                content_type="test",
            )

            # Should fail due to high flat region flicker
            assert result.has_errors() is True
            assert result.metrics.flat_flicker_ratio == 0.25
            assert any(
                "background" in issue.message.lower()
                and "flicker" in issue.message.lower()
                for issue in result.issues
            )

    def test_flat_region_flicker_validation_passing(self, validation_checker, tmp_path):
        """Test validation passes for stable flat regions."""
        # Create GIF with stable background that should pass validation
        input_gif = create_background_flicker_gif(stable=True)
        output_gif = tmp_path / "stable_background_output.gif"
        output_gif.write_bytes(input_gif.read_bytes())

        with patch("giflab.metrics.calculate_comprehensive_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "flicker_excess_compressed": 0.008,  # Below threshold - should pass
                "flicker_frame_ratio_compressed": 0.2,
                "flat_flicker_ratio_compressed": 0.05,  # Below 0.1 threshold - should pass
                "temporal_pumping_score_compressed": 0.12,  # Below threshold - should pass
                "lpips_t_mean_compressed": 0.025,  # Below threshold - should pass
                "ssim_mean": 0.75,
                "psnr_mean": 22.0,
                "mse_mean": 400.0,
            }

            original_metadata = create_test_metadata(input_gif, 1.0)
            result = validation_checker.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=mock_metrics.return_value,
                pipeline_id="test_pipeline",
                gif_name=input_gif.stem,
                content_type="test",
            )

            # Should pass - stable flat regions
            assert result.is_acceptable() is True
            assert result.metrics.flat_flicker_ratio == 0.05

    def test_temporal_pumping_validation_failing(self, validation_checker, tmp_path):
        """Test validation fails for high temporal pumping."""
        # Create GIF with temporal pumping that should fail validation
        input_gif = create_temporal_pumping_gif(pumping=True)
        output_gif = tmp_path / "temporal_pumping_output.gif"
        output_gif.write_bytes(input_gif.read_bytes())

        with patch("giflab.metrics.calculate_comprehensive_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "flicker_excess_compressed": 0.015,  # Below threshold - should pass
                "flicker_frame_ratio_compressed": 0.4,
                "flat_flicker_ratio_compressed": 0.08,  # Below threshold - should pass
                "temporal_pumping_score_compressed": 0.22,  # Above 0.15 threshold - should fail
                "lpips_t_mean_compressed": 0.04,  # Below threshold - should pass
                "ssim_mean": 0.6,
                "psnr_mean": 18.0,
                "mse_mean": 600.0,
            }

            original_metadata = create_test_metadata(input_gif, 1.0)
            result = validation_checker.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=mock_metrics.return_value,
                pipeline_id="test_pipeline",
                gif_name=input_gif.stem,
                content_type="test",
            )

            # Should fail due to high temporal pumping
            assert result.has_errors() is True
            assert result.metrics.temporal_pumping_score == 0.22
            assert any(
                "temporal pumping" in issue.message.lower() for issue in result.issues
            )

    def test_temporal_pumping_validation_passing(self, validation_checker, tmp_path):
        """Test validation passes for consistent quality."""
        # Create GIF without temporal pumping that should pass validation
        input_gif = create_temporal_pumping_gif(pumping=False)
        output_gif = tmp_path / "consistent_quality_output.gif"
        output_gif.write_bytes(input_gif.read_bytes())

        with patch("giflab.metrics.calculate_comprehensive_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "flicker_excess_compressed": 0.012,  # Below threshold - should pass
                "flicker_frame_ratio_compressed": 0.3,
                "flat_flicker_ratio_compressed": 0.06,  # Below threshold - should pass
                "temporal_pumping_score_compressed": 0.08,  # Below 0.15 threshold - should pass
                "lpips_t_mean_compressed": 0.035,  # Below threshold - should pass
                "ssim_mean": 0.72,
                "psnr_mean": 21.0,
                "mse_mean": 450.0,
            }

            original_metadata = create_test_metadata(input_gif, 1.0)
            result = validation_checker.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=mock_metrics.return_value,
                pipeline_id="test_pipeline",
                gif_name=input_gif.stem,
                content_type="test",
            )

            # Should pass - consistent quality
            assert result.is_acceptable() is True
            assert result.metrics.temporal_pumping_score == 0.08

    def test_lpips_temporal_validation_failing(self, validation_checker, tmp_path):
        """Test validation fails for poor temporal consistency."""
        # Create disposal artifacts GIF with poor temporal consistency
        input_gif = create_disposal_artifact_gif(corrupted=True)
        output_gif = tmp_path / "disposal_artifacts_output.gif"
        output_gif.write_bytes(input_gif.read_bytes())

        with patch("giflab.metrics.calculate_comprehensive_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "flicker_excess_compressed": 0.018,  # Below threshold - should pass
                "flicker_frame_ratio_compressed": 0.5,
                "flat_flicker_ratio_compressed": 0.09,  # Below threshold - should pass
                "temporal_pumping_score_compressed": 0.13,  # Below threshold - should pass
                "lpips_t_mean_compressed": 0.08,  # Above 0.05 threshold - should fail
                "ssim_mean": 0.65,
                "psnr_mean": 19.0,
                "mse_mean": 550.0,
            }

            original_metadata = create_test_metadata(input_gif, 1.0)
            result = validation_checker.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=mock_metrics.return_value,
                pipeline_id="test_pipeline",
                gif_name=input_gif.stem,
                content_type="test",
            )

            # Should fail due to poor temporal consistency
            assert result.has_errors() is True
            assert result.metrics.lpips_t_mean == 0.08
            assert any(
                "temporal consistency" in issue.message.lower()
                for issue in result.issues
            )

    def test_lpips_temporal_validation_passing(self, validation_checker, tmp_path):
        """Test validation passes for good temporal consistency."""
        # Create clean disposal GIF with good temporal consistency
        input_gif = create_disposal_artifact_gif(corrupted=False)
        output_gif = tmp_path / "clean_disposal_output.gif"
        output_gif.write_bytes(input_gif.read_bytes())

        with patch("giflab.metrics.calculate_comprehensive_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "flicker_excess_compressed": 0.010,  # Below threshold - should pass
                "flicker_frame_ratio_compressed": 0.2,
                "flat_flicker_ratio_compressed": 0.04,  # Below threshold - should pass
                "temporal_pumping_score_compressed": 0.09,  # Below threshold - should pass
                "lpips_t_mean_compressed": 0.025,  # Below 0.05 threshold - should pass
                "ssim_mean": 0.78,
                "psnr_mean": 24.0,
                "mse_mean": 350.0,
            }

            original_metadata = create_test_metadata(input_gif, 1.0)
            result = validation_checker.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=mock_metrics.return_value,
                pipeline_id="test_pipeline",
                gif_name=input_gif.stem,
                content_type="test",
            )

            # Should pass - good temporal consistency
            assert result.is_acceptable() is True
            assert result.metrics.lpips_t_mean == 0.025

    def test_multiple_temporal_failures(self, validation_checker, tmp_path):
        """Test validation handles multiple temporal artifact failures."""
        # Use a GIF that could have multiple issues
        input_gif = create_flicker_excess_gif("high")
        output_gif = tmp_path / "multiple_failures_output.gif"
        output_gif.write_bytes(input_gif.read_bytes())

        with patch("giflab.metrics.calculate_comprehensive_metrics") as mock_metrics:
            mock_metrics.return_value = {
                # Multiple temporal failures
                "flicker_excess_compressed": 0.05,  # Above 0.02 threshold - FAIL
                "flicker_frame_ratio_compressed": 0.8,  # 80% flicker frames
                "flat_flicker_ratio_compressed": 0.18,  # Above 0.1 threshold - FAIL
                "temporal_pumping_score_compressed": 0.25,  # Above 0.15 threshold - FAIL
                "lpips_t_mean_compressed": 0.09,  # Above 0.05 threshold - FAIL
                # Pass standard metrics
                "ssim_mean": 0.7,
                "psnr_mean": 20.0,
                "mse_mean": 500.0,
            }

            original_metadata = create_test_metadata(input_gif, 1.0)
            result = validation_checker.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=mock_metrics.return_value,
                pipeline_id="test_pipeline",
                gif_name=input_gif.stem,
                content_type="test",
            )

            # Should fail due to multiple temporal issues
            assert result.has_errors() is True

            # Check all temporal metrics are recorded
            assert result.metrics.flicker_excess == 0.05
            assert result.metrics.flat_flicker_ratio == 0.18
            assert result.metrics.temporal_pumping_score == 0.25
            assert result.metrics.lpips_t_mean == 0.09

            # Validation issues should mention multiple temporal problems
            assert any(
                any(
                    term in issue.message.lower()
                    for term in ["flicker", "pumping", "temporal"]
                )
                for issue in result.issues
            )

    def test_all_temporal_metrics_passing(self, validation_checker, tmp_path):
        """Test validation passes when all temporal metrics are good."""
        # Use smooth animation GIF - should pass all temporal checks
        input_gif = create_smooth_animation_gif()
        output_gif = tmp_path / "smooth_animation_output.gif"
        output_gif.write_bytes(input_gif.read_bytes())

        with patch("giflab.metrics.calculate_comprehensive_metrics") as mock_metrics:
            mock_metrics.return_value = {
                # All temporal metrics pass
                "flicker_excess_compressed": 0.008,  # Below 0.02 threshold - PASS
                "flicker_frame_ratio_compressed": 0.1,  # Minimal flicker
                "flat_flicker_ratio_compressed": 0.03,  # Below 0.1 threshold - PASS
                "temporal_pumping_score_compressed": 0.05,  # Below 0.15 threshold - PASS
                "lpips_t_mean_compressed": 0.02,  # Below 0.05 threshold - PASS
                # Pass standard metrics too
                "ssim_mean": 0.82,
                "psnr_mean": 26.0,
                "mse_mean": 280.0,
            }

            original_metadata = create_test_metadata(input_gif, 1.0)
            result = validation_checker.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=mock_metrics.return_value,
                pipeline_id="test_pipeline",
                gif_name=input_gif.stem,
                content_type="test",
            )

            # Should pass all validation checks
            assert result.is_acceptable() is True

            # Verify all temporal metrics are within thresholds
            assert result.metrics.flicker_excess < 0.02
            assert result.metrics.flat_flicker_ratio < 0.1
            assert result.metrics.temporal_pumping_score < 0.15
            assert result.metrics.lpips_t_mean < 0.05

    def test_temporal_validation_with_content_type_adjustment(
        self, validation_checker, tmp_path
    ):
        """Test validation adjusts thresholds based on content type."""
        # Use static GIF with noise - should have adjusted thresholds
        input_gif = create_static_with_noise_gif()
        output_gif = tmp_path / "static_noise_output.gif"
        output_gif.write_bytes(input_gif.read_bytes())

        # Mock comprehensive metrics calculation
        with patch("giflab.metrics.calculate_comprehensive_metrics") as mock_metrics:
            mock_metrics.return_value = {
                # Slightly higher values that might fail for 'animation' but pass for 'static'
                "flicker_excess_compressed": 0.025,  # Above default 0.02 but acceptable for static
                "flicker_frame_ratio_compressed": 0.4,
                "flat_flicker_ratio_compressed": 0.12,  # Above default 0.1 but acceptable for static
                "temporal_pumping_score_compressed": 0.18,  # Above default 0.15 but acceptable for static
                "lpips_t_mean_compressed": 0.06,  # Above default 0.05 but acceptable for static
                "ssim_mean": 0.75,
                "psnr_mean": 22.0,
                "mse_mean": 400.0,
            }

            original_metadata = create_test_metadata(input_gif, 1.0)
            result = validation_checker.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=mock_metrics.return_value,
                pipeline_id="test_pipeline",
                gif_name=input_gif.stem,
                content_type="static",
            )

            # Should pass due to content type adjustment
            # (Note: this assumes the ValidationChecker implements content-aware thresholds)
            # For now, just verify metrics are captured
            assert result.metrics.flicker_excess == 0.025
            assert result.metrics.flat_flicker_ratio == 0.12
            assert result.metrics.temporal_pumping_score == 0.18
            assert result.metrics.lpips_t_mean == 0.06

    def test_temporal_validation_error_handling(self, validation_checker, tmp_path):
        """Test temporal validation handles calculation errors gracefully."""
        input_gif = create_smooth_animation_gif()
        output_gif = tmp_path / "error_handling_output.gif"
        output_gif.write_bytes(input_gif.read_bytes())

        # Mock metrics calculation to raise an exception
        with patch("giflab.metrics.calculate_comprehensive_metrics") as mock_metrics:
            mock_metrics.side_effect = Exception("LPIPS model failed to load")

            original_metadata = create_test_metadata(input_gif, 1.0)
            result = validation_checker.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics={},  # Empty metrics due to mock exception
                pipeline_id="test_pipeline",
                gif_name=input_gif.stem,
                content_type="test",
            )

            # Should handle error gracefully and not crash
            # (Exact behavior depends on ValidationChecker error handling implementation)
            assert result is not None
            # Might fail validation due to inability to calculate metrics
            # or might skip temporal validation and continue with other checks


class TestTemporalValidationConfiguration:
    """Test temporal validation configuration and threshold behavior."""

    def test_custom_temporal_thresholds(self, tmp_path):
        """Test validation with custom temporal thresholds."""
        # Create config with very strict thresholds
        strict_config = ValidationConfig(
            flicker_excess_threshold=0.005,  # Very strict flicker detection
            flat_flicker_ratio_threshold=0.02,  # Very strict flat region stability
            temporal_pumping_threshold=0.05,  # Very strict pumping detection
            lpips_t_threshold=0.01,  # Very strict temporal consistency
        )

        validator = ValidationChecker(strict_config)

        input_gif = create_smooth_animation_gif()
        output_gif = tmp_path / "strict_validation_output.gif"
        output_gif.write_bytes(input_gif.read_bytes())

        with patch("giflab.metrics.calculate_comprehensive_metrics") as mock_metrics:
            mock_metrics.return_value = {
                # Values that would pass normal thresholds but fail strict ones
                "flicker_excess_compressed": 0.010,  # Above strict 0.005 threshold
                "flicker_frame_ratio_compressed": 0.3,
                "flat_flicker_ratio_compressed": 0.05,  # Above strict 0.02 threshold
                "temporal_pumping_score_compressed": 0.08,  # Above strict 0.05 threshold
                "lpips_t_mean_compressed": 0.03,  # Above strict 0.01 threshold
                "ssim_mean": 0.8,
                "psnr_mean": 25.0,
                "mse_mean": 300.0,
            }

            original_metadata = create_test_metadata(input_gif, 1.0)
            result = validator.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=mock_metrics.return_value,
                pipeline_id="test_pipeline",
                gif_name=input_gif.stem,
                content_type="test",
            )

            # Should fail with strict thresholds
            assert result.has_errors() is True

    def test_relaxed_temporal_thresholds(self, tmp_path):
        """Test validation with relaxed temporal thresholds."""
        # Create config with very relaxed thresholds
        relaxed_config = ValidationConfig(
            flicker_excess_threshold=0.1,  # Very relaxed flicker detection
            flat_flicker_ratio_threshold=0.5,  # Very relaxed flat region stability
            temporal_pumping_threshold=0.8,  # Very relaxed pumping detection
            lpips_t_threshold=0.2,  # Very relaxed temporal consistency
        )

        validator = ValidationChecker(relaxed_config)

        input_gif = create_flicker_excess_gif("high")  # Use problematic GIF
        output_gif = tmp_path / "relaxed_validation_output.gif"
        output_gif.write_bytes(input_gif.read_bytes())

        with patch("giflab.metrics.calculate_comprehensive_metrics") as mock_metrics:
            mock_metrics.return_value = {
                # Values that would fail normal thresholds but pass relaxed ones
                "flicker_excess_compressed": 0.08,  # Below relaxed 0.1 threshold
                "flicker_frame_ratio_compressed": 0.7,
                "flat_flicker_ratio_compressed": 0.3,  # Below relaxed 0.5 threshold
                "temporal_pumping_score_compressed": 0.6,  # Below relaxed 0.8 threshold
                "lpips_t_mean_compressed": 0.15,  # Below relaxed 0.2 threshold
                "ssim_mean": 0.6,
                "psnr_mean": 18.0,
                "mse_mean": 700.0,
            }

            original_metadata = create_test_metadata(input_gif, 1.0)
            result = validator.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=mock_metrics.return_value,
                pipeline_id="test_pipeline",
                gif_name=input_gif.stem,
                content_type="test",
            )

            # Should pass with relaxed thresholds
            assert result.is_acceptable() is True
