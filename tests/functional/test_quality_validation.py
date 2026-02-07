"""Tests for quality validation system.

This test suite validates that the quality validation system correctly
integrates with the existing metrics system to detect catastrophic quality
failures while allowing normal compression degradation.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from giflab.config import DEFAULT_METRICS_CONFIG
from giflab.wrapper_validation import WrapperOutputValidator
from giflab.wrapper_validation.quality_validation import QualityThresholdValidator


class TestQualityThresholdValidator:
    """Test quality threshold validation functionality."""

    def test_quality_validator_initialization(self):
        """Test quality validator initializes properly."""
        validator = QualityThresholdValidator()
        assert validator.metrics_config == DEFAULT_METRICS_CONFIG
        assert "min_composite_quality" in validator.catastrophic_thresholds
        assert validator.catastrophic_thresholds["min_composite_quality"] == 0.1

    def test_quality_validator_custom_config(self):
        """Test quality validator with custom metrics config."""
        custom_config = Mock()
        custom_config.USE_ENHANCED_COMPOSITE_QUALITY = False

        validator = QualityThresholdValidator(custom_config)
        assert validator.metrics_config == custom_config

    @patch(
        "giflab.wrapper_validation.quality_validation.calculate_comprehensive_metrics"
    )
    @patch("giflab.enhanced_metrics.calculate_composite_quality")
    def test_validate_quality_degradation_acceptable(
        self, mock_enhanced_quality, mock_comprehensive_metrics
    ):
        """Test quality validation with acceptable degradation."""
        validator = QualityThresholdValidator()

        # Mock good quality metrics
        mock_comprehensive_metrics.return_value = {
            "ssim_mean": 0.85,
            "psnr_mean": 25.0,
            "mse_mean": 150.0,
            "temporal_consistency": 0.9,
        }
        mock_enhanced_quality.return_value = 0.75  # Well above 0.1 threshold

        result = validator.validate_quality_degradation(
            Path("input.gif"),
            Path("output.gif"),
            {"engine": "test"},
            "lossy_compression",
        )

        assert result.is_valid
        assert result.validation_type == "quality_degradation"
        assert (
            abs(result.details["composite_quality"] - 0.634) < 0.01
        )  # Approximate expected value
        assert result.details["operation_type"] == "lossy_compression"

    @patch(
        "giflab.wrapper_validation.quality_validation.calculate_comprehensive_metrics"
    )
    @patch("giflab.enhanced_metrics.calculate_composite_quality")
    def test_validate_quality_degradation_catastrophic_failure(
        self, mock_enhanced_quality, mock_comprehensive_metrics
    ):
        """Test quality validation with catastrophic quality failure."""
        validator = QualityThresholdValidator()

        # Mock catastrophically bad quality metrics
        mock_comprehensive_metrics.return_value = {
            "ssim_mean": 0.05,  # Very poor structural similarity
            "psnr_mean": 8.0,  # Below 10dB threshold
            "mse_mean": 15000.0,  # Above 10000 threshold
            "temporal_consistency": 0.05,
        }
        mock_enhanced_quality.return_value = 0.02  # Well below 0.1 threshold

        result = validator.validate_quality_degradation(
            Path("input.gif"), Path("output.gif"), {"engine": "test"}, "color_reduction"
        )

        assert result.is_valid is False
        # Error message should contain outlier information
        assert "Metric outliers detected" in result.error_message
        assert (
            abs(result.details["composite_quality"] - 0.114) < 0.01
        )  # Approximate expected value

    @patch(
        "giflab.wrapper_validation.quality_validation.calculate_comprehensive_metrics"
    )
    def test_validate_quality_degradation_legacy_mode(self, mock_comprehensive_metrics):
        """Test quality validation in legacy 4-metric mode."""
        # Create config with legacy mode
        config = Mock()
        config.USE_ENHANCED_COMPOSITE_QUALITY = False
        config.SSIM_WEIGHT = 0.3
        config.MS_SSIM_WEIGHT = 0.35
        config.PSNR_WEIGHT = 0.25
        config.TEMPORAL_WEIGHT = 0.1
        config.PSNR_MAX_DB = 50.0

        validator = QualityThresholdValidator(config)

        # Mock reasonable quality metrics
        mock_comprehensive_metrics.return_value = {
            "ssim_mean": 0.7,
            "ms_ssim_mean": 0.75,
            "psnr_mean": 20.0,
            "temporal_consistency": 0.8,
        }

        result = validator.validate_quality_degradation(
            Path("input.gif"), Path("output.gif"), {"engine": "test"}, "frame_reduction"
        )

        assert result.is_valid
        assert result.details["quality_type"] == "legacy_composite"
        # Legacy composite should be calculated from weighted average
        # (0.7 * 0.3) + (0.75 * 0.35) + (0.4 * 0.25) + (0.8 * 0.1) = 0.655
        expected_quality = 0.655
        assert abs(result.details["composite_quality"] - expected_quality) < 0.01

    def test_check_metric_outliers(self):
        """Test outlier detection in individual metrics."""
        validator = QualityThresholdValidator()

        # Test with mixed metrics - some good, some bad
        metrics = {
            "ssim_mean": 0.3,  # Above threshold (0.2)
            "mse_mean": 8000.0,  # Below threshold (10000)
            "psnr_mean": 5.0,  # Below threshold (10.0)
            "temporal_consistency": 0.15,  # Above threshold (0.1)
        }

        outlier_checks = validator._check_metric_outliers(metrics)

        assert outlier_checks["ssim"]["acceptable"] is True
        assert outlier_checks["mse"]["acceptable"] is True
        assert outlier_checks["psnr"]["acceptable"] is False  # 5.0 < 10.0
        assert outlier_checks["temporal"]["acceptable"] is True

    @patch(
        "giflab.wrapper_validation.quality_validation.calculate_comprehensive_metrics"
    )
    def test_validate_quality_variance(self, mock_comprehensive_metrics):
        """Test quality variance validation."""
        validator = QualityThresholdValidator()

        # Mock metrics with positional variance data
        mock_comprehensive_metrics.return_value = {
            "ssim_positional_variance": 0.1,  # Low variance
            "mse_positional_variance": 0.2,  # Low variance
            "psnr_positional_variance": 0.95,  # High variance (bad)
        }

        result = validator.validate_quality_variance(
            Path("input.gif"), Path("output.gif"), {"engine": "test"}
        )

        assert result.is_valid is False  # Due to high PSNR variance
        assert result.validation_type == "quality_variance"
        assert "High quality variance detected" in result.error_message

        # Check individual variance indicators
        variance_indicators = result.details["variance_indicators"]
        assert len(variance_indicators) == 3

        psnr_indicator = next(i for i in variance_indicators if i["metric"] == "psnr")
        assert psnr_indicator["acceptable"] is False
        assert psnr_indicator["variance"] == 0.95


class TestQualityValidationIntegration:
    """Test integration of quality validation with main validator."""

    def test_main_validator_has_quality_validator(self):
        """Test main validator initializes with quality validator."""
        validator = WrapperOutputValidator()
        assert hasattr(validator, "quality_validator")
        assert isinstance(validator.quality_validator, QualityThresholdValidator)

    @patch.object(QualityThresholdValidator, "validate_quality_degradation")
    def test_quality_validation_in_wrapper_output(self, mock_quality_validation):
        """Test quality validation is called during wrapper output validation."""
        validator = WrapperOutputValidator()

        # Mock quality validation result
        mock_quality_result = Mock()
        mock_quality_result.is_valid = True
        mock_quality_validation.return_value = mock_quality_result

        # Mock other validation methods to avoid external dependencies
        validator.validate_file_integrity = Mock(return_value=Mock(is_valid=True))
        validator.validate_timing_preservation = Mock(return_value=Mock(is_valid=True))

        # Run wrapper output validation
        results = validator.validate_wrapper_output(
            input_path=Path("input.gif"),
            output_path=Path("output.gif"),
            wrapper_params={"lossy_level": 40},
            wrapper_metadata={"engine": "test"},
            wrapper_type="lossy_compression",
        )

        # Quality validation should have been called
        mock_quality_validation.assert_called_once_with(
            Path("input.gif"),
            Path("output.gif"),
            {"engine": "test"},
            "lossy_compression",
        )

        # Quality result should be included in results
        assert mock_quality_result in results

    @patch.object(QualityThresholdValidator, "validate_quality_degradation")
    def test_quality_validation_error_handling(self, mock_quality_validation):
        """Test quality validation errors don't break overall validation."""
        validator = WrapperOutputValidator()

        # Mock quality validation to raise exception
        mock_quality_validation.side_effect = Exception("Quality calculation failed")

        # Mock other validation methods
        validator.validate_file_integrity = Mock(return_value=Mock(is_valid=True))
        validator.validate_timing_preservation = Mock(return_value=Mock(is_valid=True))

        # Run wrapper output validation
        results = validator.validate_wrapper_output(
            input_path=Path("input.gif"),
            output_path=Path("output.gif"),
            wrapper_params={"colors": 32},
            wrapper_metadata={"engine": "test"},
            wrapper_type="color_reduction",
        )

        # Should still return results from other validations
        assert len(results) >= 2  # At least file integrity and timing
        # Quality validation failure should be handled gracefully
        assert all(isinstance(r.is_valid, bool) for r in results)


@pytest.mark.external_tools
class TestQualityValidationRealScenarios:
    """Test quality validation with real compression scenarios."""

    @pytest.fixture
    def test_gif(self):
        """Path to test GIF fixture."""
        return Path(__file__).parent / "fixtures" / "test_4_frames.gif"

    def test_quality_validation_with_gifsicle(self, test_gif):
        """Test quality validation with actual Gifsicle compression."""
        from giflab.tool_wrappers import GifsicleLossyCompressor

        wrapper = GifsicleLossyCompressor()
        if not wrapper.available():
            pytest.skip("Gifsicle not available")

        WrapperOutputValidator()

        with tempfile.NamedTemporaryFile(suffix=".gif") as tmp_file:
            output_path = Path(tmp_file.name)

            # Apply moderate lossy compression
            result = wrapper.apply(test_gif, output_path, params={"lossy_level": 30})

            # Should have validation results including quality
            assert "validations" in result

            # Look for quality validation
            quality_validations = [
                v
                for v in result["validations"]
                if v["validation_type"] == "quality_degradation"
            ]

            if len(quality_validations) > 0:
                quality_validation = quality_validations[0]
                print(f"Quality validation result: {quality_validation['is_valid']}")

                if not quality_validation["is_valid"]:
                    print(f"Quality failure: {quality_validation['error_message']}")
                    print(
                        f"Composite quality: {quality_validation['actual'].get('composite_quality')}"
                    )
                else:
                    print(
                        f"Quality acceptable: composite={quality_validation['actual'].get('composite_quality')}"
                    )
            else:
                print(
                    "Quality validation not found - may have been skipped due to missing dependencies"
                )

    def test_quality_thresholds_realistic(self):
        """Test that quality thresholds are realistic for normal compression."""
        validator = QualityThresholdValidator()

        # Check that thresholds are set to catch only severe issues
        thresholds = validator.catastrophic_thresholds

        assert (
            thresholds["min_composite_quality"] <= 0.2
        )  # Should allow significant degradation
        assert (
            thresholds["min_ssim_mean"] <= 0.3
        )  # Should allow noticeable quality loss
        assert (
            thresholds["min_psnr_mean"] <= 15.0
        )  # Should allow visible compression artifacts
        assert thresholds["max_mse_mean"] >= 5000.0  # Should allow significant MSE

        print(f"Quality thresholds: {thresholds}")
        print("✅ Thresholds are set to catch catastrophic failures only")
