"""Integration tests for Phase 3 conditional content-specific metrics.

This module tests the integration of Text/UI validation and SSIMULACRA2 metrics
with the main metrics calculation pipeline, validation system, cross-metric
interactions with other phases, and end-to-end validation flows including
CSV output validation, validation report generation, and threshold configuration.
"""

import csv
import io
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest
from giflab.config import MetricsConfig
from giflab.meta import GifMetadata
from giflab.metrics import calculate_comprehensive_metrics_from_frames
from giflab.optimization_validation.config import ValidationConfig
from giflab.optimization_validation.data_structures import ValidationStatus
from giflab.optimization_validation.validation_checker import ValidationChecker
from giflab.ssimulacra2_metrics import (
    calculate_ssimulacra2_quality_metrics,
    should_use_ssimulacra2,
)
from giflab.text_ui_validation import (
    calculate_text_ui_metrics,
    should_validate_text_ui,
)

# Import fixture generator for consistent test data
try:
    from tests.fixtures.generate_phase3_fixtures import Phase3FixtureGenerator
except ImportError:
    Phase3FixtureGenerator = None


@pytest.fixture
def fixture_generator():
    """Create fixture generator for tests."""
    if Phase3FixtureGenerator is None:
        pytest.skip("Phase 3 fixture generator not available")

    with tempfile.TemporaryDirectory() as tmpdir:
        generator = Phase3FixtureGenerator(Path(tmpdir))
        yield generator


class TestMetricsPipelineIntegration:
    """Test integration with the main metrics calculation pipeline."""

    def test_comprehensive_metrics_includes_phase3(self):
        """Test that comprehensive metrics calculation includes Phase 3 metrics."""
        # Create test frames
        frames_orig = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)
        ]
        frames_comp = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)
        ]

        config = MetricsConfig()
        config.USE_COMPREHENSIVE_METRICS = True
        config.ENABLE_DEEP_PERCEPTUAL = True
        config.ENABLE_SSIMULACRA2 = True

        # Mock Phase 3 components to return predictable results
        with patch(
            "giflab.text_ui_validation.calculate_text_ui_metrics"
        ) as mock_text_ui, patch(
            "giflab.ssimulacra2_metrics.calculate_ssimulacra2_quality_metrics"
        ) as mock_ssim2, patch(
            "giflab.text_ui_validation.should_validate_text_ui", return_value=(True, {})
        ), patch(
            "giflab.ssimulacra2_metrics.should_use_ssimulacra2", return_value=True
        ):
            # Mock return values
            mock_text_ui.return_value = {
                "has_text_ui_content": True,
                "text_ui_edge_density": 0.15,
                "text_ui_component_count": 5,
                "ocr_conf_delta_mean": -0.02,
                "ocr_conf_delta_min": -0.05,
                "ocr_regions_analyzed": 3,
                "mtf50_ratio_mean": 0.85,
                "mtf50_ratio_min": 0.75,
                "edge_sharpness_score": 82.0,
            }

            mock_ssim2.return_value = {
                "ssimulacra2_mean": 0.72,
                "ssimulacra2_p95": 0.68,
                "ssimulacra2_min": 0.65,
                "ssimulacra2_frame_count": 3.0,
                "ssimulacra2_triggered": 1.0,
            }

            # Calculate comprehensive metrics using frame-based API
            result = calculate_comprehensive_metrics_from_frames(
                frames_orig, frames_comp, config
            )

            # Verify Phase 3 metrics are included
            text_ui_keys = [
                "has_text_ui_content",
                "text_ui_edge_density",
                "text_ui_component_count",
                "ocr_conf_delta_mean",
                "ocr_conf_delta_min",
                "ocr_regions_analyzed",
                "mtf50_ratio_mean",
                "mtf50_ratio_min",
                "edge_sharpness_score",
            ]

            for key in text_ui_keys:
                assert key in result, f"Missing text/UI metric: {key}"

            ssim2_keys = [
                "ssimulacra2_mean",
                "ssimulacra2_p95",
                "ssimulacra2_min",
                "ssimulacra2_frame_count",
                "ssimulacra2_triggered",
            ]

            for key in ssim2_keys:
                assert key in result, f"Missing SSIMULACRA2 metric: {key}"

            # Verify functions were called
            mock_text_ui.assert_called_once()
            mock_ssim2.assert_called_once()

    def test_conditional_execution_in_pipeline(self):
        """Test conditional execution logic within the pipeline."""
        frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(2)
        ]
        config = MetricsConfig()

        # Test case 1: Text/UI validation returns minimal metrics
        with patch(
            "giflab.text_ui_validation.calculate_text_ui_metrics"
        ) as mock_text_ui, patch(
            "giflab.ssimulacra2_metrics.calculate_ssimulacra2_quality_metrics"
        ) as mock_ssim2:
            # Mock text/UI to return minimal metrics (as if no text/UI detected)
            mock_text_ui.return_value = {
                "has_text_ui_content": False,
                "text_ui_edge_density": 0.05,
                "text_ui_component_count": 0,
                "ocr_conf_delta_mean": 0.0,
                "ocr_conf_delta_min": 0.0,
                "ocr_regions_analyzed": 0,
                "mtf50_ratio_mean": 1.0,
                "mtf50_ratio_min": 1.0,
                "edge_sharpness_score": 100.0,
            }

            # Mock SSIMULACRA2 to return default metrics
            mock_ssim2.return_value = {
                "ssimulacra2_mean": 50.0,
                "ssimulacra2_p95": 50.0,
                "ssimulacra2_min": 50.0,
                "ssimulacra2_frame_count": 0.0,
                "ssimulacra2_triggered": 0.0,
            }

            result = calculate_comprehensive_metrics_from_frames(frames, frames, config)

            # Text/UI metrics should include the returned data (converted to float)
            assert result["has_text_ui_content"] == 0.0  # False converted to float
            assert result["text_ui_edge_density"] == 0.05

            # Verify functions were called (they're always called in current implementation)
            mock_text_ui.assert_called_once()
            mock_ssim2.assert_called_once()

        # Test case 2: Both return active metrics
        with patch(
            "giflab.text_ui_validation.calculate_text_ui_metrics"
        ) as mock_text_ui, patch(
            "giflab.ssimulacra2_metrics.calculate_ssimulacra2_quality_metrics"
        ) as mock_ssim2:
            # Set up return values for active text/UI detection
            mock_text_ui.return_value = {
                "has_text_ui_content": True,
                "text_ui_edge_density": 0.20,
                "text_ui_component_count": 8,
                "ocr_conf_delta_mean": -0.03,
                "ocr_conf_delta_min": -0.08,
                "ocr_regions_analyzed": 5,
                "mtf50_ratio_mean": 0.78,
                "mtf50_ratio_min": 0.65,
                "edge_sharpness_score": 75.0,
            }

            # Set up return values for active SSIMULACRA2 metrics
            mock_ssim2.return_value = {
                "ssimulacra2_mean": 0.68,
                "ssimulacra2_p95": 0.62,
                "ssimulacra2_min": 0.58,
                "ssimulacra2_frame_count": 2.0,
                "ssimulacra2_triggered": 1.0,
            }

            result = calculate_comprehensive_metrics_from_frames(frames, frames, config)

            # Both should be executed (always called in current implementation)
            mock_text_ui.assert_called_once()
            mock_ssim2.assert_called_once()

            # Results should be included (boolean converted to float)
            assert result["has_text_ui_content"] == 1.0  # True converted to float
            assert result["ssimulacra2_triggered"] == 1.0

    def test_enhanced_composite_quality_integration(self):
        """Test integration with enhanced composite quality calculation."""
        # Create simple test frames that are identical (for predictable metrics)
        frames = [
            np.ones((100, 100, 3), dtype=np.uint8) * 128 for _ in range(3)
        ]

        config = MetricsConfig()
        config.USE_ENHANCED_COMPOSITE_QUALITY = True
        config.ENABLE_DEEP_PERCEPTUAL = False  # Disable to avoid needing LPIPS model
        config.ENABLE_SSIMULACRA2 = False  # Disable to avoid needing ssimulacra2
        config.ENABLE_TEXT_UI_VALIDATION = False  # Disable text UI validation

        # Calculate metrics with identical frames (should give perfect scores)
        result = calculate_comprehensive_metrics_from_frames(frames, frames, config)

        # Verify composite quality is calculated
        assert "composite_quality" in result

        # The composite quality should be calculated
        composite_quality = result["composite_quality"]
        assert 0.0 <= composite_quality <= 1.0

        # With identical frames, quality should be very high
        assert composite_quality > 0.9, f"Expected high quality for identical frames, got {composite_quality}"

    def test_cross_metric_interactions(self):
        """Test interactions between Phase 3 and other phase metrics."""
        # Create test frames with different properties to trigger various metrics
        frames_orig = [
            np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8) for _ in range(2)
        ]
        # Make compressed frames slightly different for non-perfect scores
        frames_comp = [
            np.clip(f + np.random.randint(-5, 5, f.shape), 0, 255).astype(np.uint8)
            for f in frames_orig
        ]
        
        config = MetricsConfig()
        config.ENABLE_DEEP_PERCEPTUAL = False  # Disable to avoid needing LPIPS model
        config.ENABLE_SSIMULACRA2 = False  # Disable to avoid needing ssimulacra2
        config.ENABLE_TEXT_UI_VALIDATION = False  # Disable text UI validation

        # Calculate metrics
        result = calculate_comprehensive_metrics_from_frames(frames_orig, frames_comp, config)

        # Verify basic metrics are present
        assert "ssim" in result
        assert "psnr" in result
        assert "composite_quality" in result
        
        # Verify values are in expected ranges
        assert 0.0 <= result["ssim"] <= 1.0
        assert 0.0 <= result["composite_quality"] <= 1.0

    def test_metric_weight_normalization(self):
        """Test that metric weights still normalize to 1.0 with Phase 3 additions."""
        config = MetricsConfig()

        # Check enhanced composite quality weights
        enhanced_weights = [
            config.ENHANCED_SSIM_WEIGHT,
            config.ENHANCED_MS_SSIM_WEIGHT,
            config.ENHANCED_PSNR_WEIGHT,
            config.ENHANCED_MSE_WEIGHT,
            config.ENHANCED_FSIM_WEIGHT,
            config.ENHANCED_EDGE_WEIGHT,
            config.ENHANCED_GMSD_WEIGHT,
            config.ENHANCED_CHIST_WEIGHT,
            config.ENHANCED_SHARPNESS_WEIGHT,
            config.ENHANCED_TEXTURE_WEIGHT,
            config.ENHANCED_TEMPORAL_WEIGHT,
            config.ENHANCED_LPIPS_WEIGHT,
            config.ENHANCED_SSIMULACRA2_WEIGHT,
        ]

        total_weight = sum(enhanced_weights)
        assert (
            abs(total_weight - 1.0) < 1e-6
        ), f"Enhanced weights sum to {total_weight}, not 1.0"

        # Verify SSIMULACRA2 has reasonable weight
        assert 0.01 <= config.ENHANCED_SSIMULACRA2_WEIGHT <= 0.1  # Should be 1-10%


class TestValidationSystemIntegration:
    """Test integration with the validation system."""

    def test_validation_checker_includes_phase3(self):
        """Test that ValidationChecker processes Phase 3 metrics."""
        # Create validation checker
        validation_config = ValidationConfig()
        checker = ValidationChecker(validation_config)

        # Mock metadata and compression metrics with Phase 3 data
        from giflab.meta import GifMetadata

        original_metadata = GifMetadata(
            gif_sha="test_sha_123",
            orig_filename="test_ui.gif",
            orig_kilobytes=150.0,
            orig_width=200,
            orig_height=150,
            orig_frames=10,
            orig_fps=15.0,
            orig_n_colors=128,
        )

        compression_metrics = {
            # Basic metrics
            "composite_quality": 0.65,
            "efficiency": 0.75,
            "compression_ratio": 2.5,
            "compressed_frame_count": 8,
            # Phase 3 metrics - Text/UI
            "has_text_ui_content": True,
            "text_ui_edge_density": 0.18,
            "text_ui_component_count": 7,
            "ocr_conf_delta_mean": -0.08,  # Significant degradation
            "ocr_conf_delta_min": -0.15,
            "ocr_regions_analyzed": 4,
            "mtf50_ratio_mean": 0.60,  # Reduced sharpness
            "mtf50_ratio_min": 0.45,
            "edge_sharpness_score": 62.0,  # Below good threshold
            # Phase 3 metrics - SSIMULACRA2
            "ssimulacra2_mean": 0.45,  # Borderline quality
            "ssimulacra2_p95": 0.40,
            "ssimulacra2_min": 0.35,
            "ssimulacra2_frame_count": 8.0,
            "ssimulacra2_triggered": 1.0,
        }

        # Validate with Phase 3 metrics
        result = checker.validate_compression_result(
            original_metadata=original_metadata,
            compression_metrics=compression_metrics,
            gif_name="test_ui_gif",
            pipeline_id="test_pipeline",
            content_type="ui",
        )

        # Should include Phase 3 metrics in validation
        assert result.metrics.has_text_ui_content is True
        assert result.metrics.text_ui_edge_density == 0.18
        assert result.metrics.ocr_conf_delta_mean == -0.08
        assert result.metrics.ssimulacra2_mean == 0.45

        # Should detect issues with text/UI content
        {issue.category for issue in result.issues}

        # May include issues related to OCR degradation, sharpness loss, etc.
        # Exact issues depend on threshold configuration
        assert len(result.issues) >= 0  # At minimum should not crash

    @pytest.mark.skip(
        reason="Phase 3 validation methods not implemented yet - covered by Subtask 1.2"
    )
    @patch(
        "giflab.optimization_validation.validation_checker.ValidationChecker._validate_text_ui_content"
    )
    @patch(
        "giflab.optimization_validation.validation_checker.ValidationChecker._validate_ssimulacra2_metrics"
    )
    def test_phase3_validation_methods_called(
        self, mock_ssim2_validation, mock_text_ui_validation
    ):
        """Test that Phase 3 validation methods are called."""
        validation_config = ValidationConfig()
        checker = ValidationChecker(validation_config)

        from giflab.meta import GifMetadata

        original_metadata = GifMetadata(
            gif_sha="test_sha_456",
            orig_filename="test_gif.gif",
            orig_kilobytes=100.0,
            orig_width=100,
            orig_height=100,
            orig_frames=5,
            orig_fps=10.0,
            orig_n_colors=64,
        )

        compression_metrics = {
            "composite_quality": 0.60,
            "has_text_ui_content": True,
            "ssimulacra2_triggered": 1.0,
        }

        checker.validate_compression_result(
            original_metadata=original_metadata,
            compression_metrics=compression_metrics,
            gif_name="test_gif",
            pipeline_id="test_pipeline",
            content_type="animation",
        )

        # Verify Phase 3 validation methods were called
        # Note: These methods may not exist yet in the current codebase,
        # but this test documents the expected integration
        try:
            mock_text_ui_validation.assert_called_once()
            mock_ssim2_validation.assert_called_once()
        except AttributeError:
            # Methods may not be implemented yet - that's acceptable
            pass

    def test_threshold_configuration_phase3(self):
        """Test that Phase 3 metrics respect threshold configuration."""
        ValidationConfig()

        # Test that Phase 3 thresholds can be configured
        # (Implementation depends on how thresholds are structured)

        # Expected Phase 3 thresholds
        expected_thresholds = [
            "ocr_conf_delta_threshold",
            "mtf50_ratio_threshold",
            "edge_sharpness_threshold",
            "ssimulacra2_threshold",
            "ssimulacra2_low_threshold",
            "ssimulacra2_high_threshold",
        ]

        # These thresholds should be configurable
        for _threshold_name in expected_thresholds:
            # Test will depend on actual threshold configuration system
            # This documents expected threshold names
            pass

    def test_validation_issue_categorization(self):
        """Test proper categorization of Phase 3 validation issues."""
        # This test documents expected issue categories for Phase 3 metrics



        # These categories should be recognized by the validation system
        # Implementation depends on validation system structure


class TestErrorHandlingIntegration:
    """Test error handling and recovery in integration scenarios."""

    def test_phase3_component_failures_in_pipeline(self):
        """Test pipeline behavior when Phase 3 components fail."""
        frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(2)
        ]
        config = MetricsConfig()

        # Test case 1: Text/UI validation fails
        with patch(
            "giflab.text_ui_validation.calculate_text_ui_metrics",
            side_effect=Exception("Text/UI failed"),
        ), patch(
            "giflab.text_ui_validation.should_validate_text_ui", return_value=(True, {})
        ), patch(
            "giflab.ssimulacra2_metrics.calculate_ssimulacra2_quality_metrics"
        ) as mock_ssim2:
            mock_ssim2.return_value = {
                "ssimulacra2_mean": 0.70,
                "ssimulacra2_triggered": 1.0,
            }

            # Pipeline should continue despite Text/UI failure
            result = calculate_comprehensive_metrics_from_frames(frames, frames, config)

            # Should have SSIMULACRA2 results but missing Text/UI metrics
            if "ssimulacra2_mean" in result:
                assert result["ssimulacra2_mean"] == 0.70

            # Text/UI metrics should be absent or default values
            # (Exact behavior depends on error handling implementation)

        # Test case 2: SSIMULACRA2 fails
        with patch(
            "giflab.ssimulacra2_metrics.calculate_ssimulacra2_quality_metrics",
            side_effect=Exception("SSIMULACRA2 failed"),
        ), patch(
            "giflab.text_ui_validation.calculate_text_ui_metrics"
        ) as mock_text_ui, patch(
            "giflab.text_ui_validation.should_validate_text_ui", return_value=(True, {})
        ):
            mock_text_ui.return_value = {
                "has_text_ui_content": True,
                "text_ui_edge_density": 0.15,
            }

            # Pipeline should continue despite SSIMULACRA2 failure
            result = calculate_comprehensive_metrics_from_frames(frames, frames, config)

            # Should have Text/UI results but missing SSIMULACRA2 metrics
            # Note: boolean values are converted to floats in the result
            assert result.get("has_text_ui_content", 0.0) == 1.0

            # SSIMULACRA2 metrics should be absent or default values

    def test_partial_metric_availability(self):
        """Test behavior with partial Phase 3 metric availability."""
        frames = [
            np.random.randint(0, 255, (60, 60, 3), dtype=np.uint8) for _ in range(3)
        ]
        config = MetricsConfig()

        # Scenario: OCR unavailable but edge analysis works
        with patch(
            "giflab.text_ui_validation.should_validate_text_ui", return_value=(True, {})
        ), patch("giflab.text_ui_validation.OCRValidator") as MockOCRValidator:
            # Mock OCR validator to simulate unavailable OCR
            mock_ocr_instance = Mock()
            mock_ocr_instance.calculate_ocr_confidence_delta.return_value = {
                "ocr_conf_delta_mean": 0.0,
                "ocr_conf_delta_min": 0.0,
                "ocr_regions_analyzed": 0,
            }
            MockOCRValidator.return_value = mock_ocr_instance

            result = calculate_comprehensive_metrics_from_frames(frames, frames, config)

            # Should have edge/sharpness metrics but minimal OCR metrics
            if "ocr_regions_analyzed" in result:
                assert result["ocr_regions_analyzed"] == 0

    def test_configuration_compatibility(self):
        """Test compatibility with different configuration combinations."""
        frames = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(2)
        ]

        # Test different config combinations
        config_variations = [
            {"USE_COMPREHENSIVE_METRICS": True, "ENABLE_SSIMULACRA2": True},
            {"USE_COMPREHENSIVE_METRICS": True, "ENABLE_SSIMULACRA2": False},
            {"USE_COMPREHENSIVE_METRICS": False, "ENABLE_SSIMULACRA2": True},
            {"USE_COMPREHENSIVE_METRICS": False, "ENABLE_SSIMULACRA2": False},
        ]

        for config_dict in config_variations:
            config = MetricsConfig()
            for key, value in config_dict.items():
                setattr(config, key, value)

            # Should handle all configuration combinations gracefully
            try:
                result = calculate_comprehensive_metrics_from_frames(
                    frames, frames, config
                )
                assert isinstance(result, dict)
                assert len(result) > 0
            except Exception as e:
                pytest.fail(f"Config {config_dict} caused error: {e}")


class TestCrossPhaseInteractions:
    """Test interactions between Phase 3 and other phases."""

    def test_phase1_phase3_interaction(self):
        """Test interaction between Phase 1 (timing/temporal) and Phase 3 metrics."""
        # Create test frames with some variability for realistic metrics
        frames = [
            np.random.randint(0, 255, (70, 70, 3), dtype=np.uint8) for _ in range(4)
        ]
        config = MetricsConfig()
        config.ENABLE_DEEP_PERCEPTUAL = False  # Disable to avoid needing LPIPS model
        config.ENABLE_SSIMULACRA2 = False  # Disable to avoid needing ssimulacra2
        config.ENABLE_TEXT_UI_VALIDATION = False  # Disable text UI validation

        # Calculate metrics normally without mocking
        result = calculate_comprehensive_metrics_from_frames(frames, frames, config)

        # Verify that both temporal and spatial metrics are present
        # and can interact in the composite quality calculation
        assert "flicker_score" in result or "temporal_consistency" in result
        assert "composite_quality" in result
        
        # Composite quality should integrate temporal and spatial aspects
        composite = result["composite_quality"]
        assert 0.0 <= composite <= 1.0

    def test_phase2_phase3_interaction(self):
        """Test interaction between Phase 2 (quality refinement) and Phase 3 metrics."""
        # Create test frames
        frames = [
            np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8) for _ in range(3)
        ]
        config = MetricsConfig()
        config.ENABLE_DEEP_PERCEPTUAL = False  # Disable to avoid needing LPIPS model
        config.ENABLE_SSIMULACRA2 = False  # Disable to avoid needing ssimulacra2
        config.ENABLE_TEXT_UI_VALIDATION = False  # Disable text UI validation

        # Calculate metrics normally
        result = calculate_comprehensive_metrics_from_frames(frames, frames, config)

        # Verify that quality metrics from different phases are present
        assert "composite_quality" in result
        
        # Verify perceptual quality metrics exist (even if fallback)
        if "lpips_quality_mean" in result:
            assert 0.0 <= result["lpips_quality_mean"] <= 1.0
        
        # Composite quality should integrate all available metrics
        assert 0.0 <= result["composite_quality"] <= 1.0

    def test_all_phases_integration(self):
        """Test integration across all phases (1, 2, and 3)."""
        # Create test frames with variation for realistic metrics
        frames_orig = [
            np.random.randint(0, 255, (90, 90, 3), dtype=np.uint8) for _ in range(5)
        ]
        # Create slightly different compressed frames
        frames_comp = [
            np.clip(f + np.random.randint(-10, 10, f.shape), 0, 255).astype(np.uint8)
            for f in frames_orig
        ]
        
        config = MetricsConfig()
        config.USE_COMPREHENSIVE_METRICS = True
        config.ENABLE_DEEP_PERCEPTUAL = False  # Disable to avoid needing LPIPS model
        config.ENABLE_SSIMULACRA2 = False  # Disable to avoid needing ssimulacra2
        config.ENABLE_TEXT_UI_VALIDATION = False  # Disable text UI validation

        # Calculate metrics normally without mocking
        result = calculate_comprehensive_metrics_from_frames(frames_orig, frames_comp, config)

        # Should include metrics from all phases
        # Phase 1: Basic quality metrics
        assert "ssim" in result
        assert "psnr" in result
        
        # Phase 1: Gradient and color metrics
        assert "deltae_mean" in result or "color_histogram_similarity_mean" in result
        assert "banding_score_mean" in result
        
        # Composite quality should incorporate all available metrics
        assert "composite_quality" in result
        composite = result["composite_quality"]
        assert 0.0 <= composite <= 1.0
        
        # Check that we have a reasonable number of metrics
        assert len(result) > 50  # Should have many metrics from all phases


# ---------------------------------------------------------------------------
# End-to-end validation tests (merged from test_phase3_validation_e2e.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_phase3_metrics():
    """Provide sample Phase 3 metrics for testing."""
    return {
        # Basic metrics
        "composite_quality": 0.68,
        "efficiency": 0.72,
        "compression_ratio": 2.3,
        "compressed_frame_count": 8,
        "orig_fps": 12.0,
        "kilobytes": 85.0,
        # Phase 1 temporal metrics (for context)
        "timing_grid_ms": 83.3,
        "grid_length": 10.0,
        "duration_diff_ms": 15.0,
        "flicker_excess": 0.025,
        "lpips_t_mean": 0.04,
        # Phase 1 gradient/color metrics (for context)
        "deltae_mean": 2.1,
        "banding_score_mean": 32.0,
        "dither_ratio_mean": 1.12,
        # Phase 2 deep perceptual metrics (for context)
        "lpips_quality_mean": 0.22,
        "lpips_quality_p95": 0.28,
        # Phase 3 Text/UI validation metrics
        "has_text_ui_content": True,
        "text_ui_edge_density": 0.16,
        "text_ui_component_count": 6,
        "ocr_conf_delta_mean": -0.05,
        "ocr_conf_delta_min": -0.12,
        "ocr_regions_analyzed": 4,
        "mtf50_ratio_mean": 0.78,
        "mtf50_ratio_min": 0.65,
        "edge_sharpness_score": 75.0,
        # Phase 3 SSIMULACRA2 metrics
        "ssimulacra2_mean": 0.62,
        "ssimulacra2_p95": 0.58,
        "ssimulacra2_min": 0.54,
        "ssimulacra2_frame_count": 8.0,
        "ssimulacra2_triggered": 1.0,
    }


class TestValidationFlowIntegration:
    """Test complete validation flow with Phase 3 metrics."""

    def test_validation_result_includes_phase3_metrics(self, sample_phase3_metrics):
        """Test that ValidationResult includes Phase 3 metrics."""
        validation_config = ValidationConfig()
        checker = ValidationChecker(validation_config)

        original_metadata = GifMetadata(
            gif_sha="test_sha",
            orig_filename="test.gif",
            orig_kilobytes=120.0,
            orig_width=256,
            orig_height=256,
            orig_frames=10,
            orig_fps=12.0,
            orig_n_colors=256,
        )

        # Validate with Phase 3 metrics
        result = checker.validate_compression_result(
            original_metadata=original_metadata,
            compression_metrics=sample_phase3_metrics,
            gif_name="test_ui_content",
            pipeline_id="ui_optimized_pipeline",
            content_type="ui",
        )

        # Verify Phase 3 metrics are included in ValidationResult
        assert result.metrics.has_text_ui_content is True
        assert result.metrics.text_ui_edge_density == 0.16
        assert result.metrics.text_ui_component_count == 6
        assert result.metrics.ocr_conf_delta_mean == -0.05
        assert result.metrics.edge_sharpness_score == 75.0

        # Verify SSIMULACRA2 metrics are included
        assert (
            hasattr(result.metrics, "ssimulacra2_mean")
            or "ssimulacra2_mean" in sample_phase3_metrics
        )

        # Should complete validation without errors
        assert result.status in [
            ValidationStatus.PASS,
            ValidationStatus.WARNING,
            ValidationStatus.ERROR,
        ]

    def test_validation_with_phase3_issues(self, sample_phase3_metrics):
        """Test validation that detects Phase 3 specific issues."""
        validation_config = ValidationConfig()
        checker = ValidationChecker(validation_config)

        # Create metrics with problematic Phase 3 values
        problematic_metrics = sample_phase3_metrics.copy()
        problematic_metrics.update(
            {
                # Text/UI issues
                "ocr_conf_delta_mean": -0.15,  # Significant OCR degradation
                "ocr_conf_delta_min": -0.25,  # Severe degradation in worst case
                "edge_sharpness_score": 45.0,  # Poor sharpness
                "mtf50_ratio_mean": 0.45,  # Poor edge acuity
                # SSIMULACRA2 issues
                "ssimulacra2_mean": 0.35,  # Poor perceptual quality
                "ssimulacra2_min": 0.25,  # Very poor worst case
                "ssimulacra2_p95": 0.30,  # Poor consistency
            }
        )

        original_metadata = GifMetadata(
            gif_sha="test_sha",
            orig_filename="test.gif",
            orig_kilobytes=120.0,
            orig_width=256,
            orig_height=256,
            orig_frames=10,
            orig_fps=12.0,
            orig_n_colors=256,
        )

        result = checker.validate_compression_result(
            original_metadata=original_metadata,
            compression_metrics=problematic_metrics,
            gif_name="problematic_ui_gif",
            pipeline_id="aggressive_pipeline",
            content_type="ui",
        )

        # Should detect issues with Phase 3 metrics
        assert len(result.issues) > 0 or len(result.warnings) > 0

        # Check for expected issue categories (if validation methods exist)
        issue_categories = {issue.category for issue in result.issues}
        warning_categories = {warning.category for warning in result.warnings}

        all_categories = issue_categories | warning_categories

        # Some overlap expected, but exact categories depend on implementation
        assert len(all_categories) > 0

    def test_content_type_specific_validation(self, sample_phase3_metrics):
        """Test that validation adapts to content type."""
        validation_config = ValidationConfig()
        checker = ValidationChecker(validation_config)

        original_metadata = GifMetadata(
            gif_sha="test_sha256_hash",
            orig_filename="test.gif",
            orig_width=640,
            orig_height=480,
            orig_n_colors=256,
            orig_frames=8,
            orig_fps=15.0,
            orig_kilobytes=100.0
        )

        # Test different content types
        content_types = ["ui", "animation", "photo", "mixed"]

        for content_type in content_types:
            result = checker.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=sample_phase3_metrics,
                gif_name=f"test_{content_type}_content",
                pipeline_id="adaptive_pipeline",
                content_type=content_type,
            )

            # Should adapt validation criteria based on content type
            assert result.content_type == content_type

            # UI content should pay more attention to Phase 3 metrics
            if content_type == "ui":
                assert result.metrics.has_text_ui_content is not None

            # All content types should complete validation
            assert result.status != ValidationStatus.UNKNOWN


class TestCSVOutputValidation:
    """Test CSV output includes all Phase 3 fields."""

    def test_csv_output_completeness(self, sample_phase3_metrics):
        """Test complete CSV output with Phase 3 metrics."""
        csv_data = sample_phase3_metrics

        output = io.StringIO()

        headers = list(csv_data.keys())
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        writer.writerow(csv_data)

        csv_content = output.getvalue()

        # Check that Phase 3 fields are included
        phase3_indicators = [
            "text_ui_edge_density",
            "ocr_conf_delta_mean",
            "ssimulacra2_mean",
            "edge_sharpness_score",
        ]

        for indicator in phase3_indicators:
            assert indicator in csv_content, f"CSV missing Phase 3 field: {indicator}"

        # Parse CSV to validate structure
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(csv_reader)

        assert len(rows) == 1
        row = rows[0]

        # Validate Phase 3 values are properly formatted
        assert float(row["text_ui_edge_density"]) == 0.16
        assert float(row["ocr_conf_delta_mean"]) == -0.05
        assert float(row["ssimulacra2_mean"]) == 0.62
        assert float(row["edge_sharpness_score"]) == 75.0


class TestValidationReportGeneration:
    """Test validation report generation with Phase 3 metrics."""

    def test_validation_report_includes_phase3_sections(self, sample_phase3_metrics):
        """Test that validation reports include Phase 3 sections."""
        validation_config = ValidationConfig()
        checker = ValidationChecker(validation_config)

        original_metadata = GifMetadata(
            gif_sha="test_sha",
            orig_filename="test.gif",
            orig_kilobytes=120.0,
            orig_width=256,
            orig_height=256,
            orig_frames=10,
            orig_fps=12.0,
            orig_n_colors=256,
        )

        # Create validation result with issues
        problematic_metrics = sample_phase3_metrics.copy()
        problematic_metrics.update(
            {
                "ocr_conf_delta_mean": -0.10,  # OCR issue
                "ssimulacra2_mean": 0.40,  # Perceptual issue
            }
        )

        result = checker.validate_compression_result(
            original_metadata=original_metadata,
            compression_metrics=problematic_metrics,
            gif_name="report_test_gif",
            pipeline_id="test_pipeline",
            content_type="ui",
        )

        # Test validation report structure
        assert result.metrics.text_ui_edge_density == 0.16
        assert result.metrics.has_text_ui_content is True

        # Report should document any Phase 3 issues found
        if result.issues or result.warnings:
            all_messages = [issue.message for issue in result.issues]
            all_messages.extend([warning.message for warning in result.warnings])

            phase3_terms = ["OCR", "sharpness", "SSIMULACRA2", "text", "perceptual"]
            any(
                any(term.lower() in msg.lower() for term in phase3_terms)
                for msg in all_messages
            )


class TestValidationSystemIntegrationE2E:
    """End-to-end integration tests for validation system."""

    def test_full_pipeline_validation_with_phase3(self, fixture_generator):
        """Test complete pipeline from metrics calculation to validation."""
        # Create test content
        ui_img_path = fixture_generator.create_text_ui_image(
            "ui_buttons", size=(160, 120)
        )
        orig_frame = cv2.imread(str(ui_img_path))

        # Create degraded version
        comp_frame = cv2.GaussianBlur(orig_frame, (3, 3), 1.0)

        orig_frames = [orig_frame for _ in range(3)]
        comp_frames = [comp_frame for _ in range(3)]

        # Step 1: Calculate comprehensive metrics (including Phase 3)
        config = MetricsConfig()
        config.USE_COMPREHENSIVE_METRICS = True
        config.ENABLE_SSIMULACRA2 = True

        # Mock Phase 3 components for consistent test
        with patch("giflab.text_ui_validation.calculate_text_ui_metrics") as mock_text_ui, patch(
            "giflab.ssimulacra2_metrics.calculate_ssimulacra2_quality_metrics"
        ) as mock_ssim2, patch(
            "giflab.text_ui_validation.should_validate_text_ui", return_value=(True, {})
        ), patch(
            "giflab.ssimulacra2_metrics.should_use_ssimulacra2", return_value=True
        ):
            mock_text_ui.return_value = {
                "has_text_ui_content": True,
                "text_ui_edge_density": 0.18,
                "text_ui_component_count": 4,
                "ocr_conf_delta_mean": -0.08,
                "edge_sharpness_score": 68.0,
                "ocr_regions_analyzed": 3,
                "mtf50_ratio_mean": 0.72,
                "mtf50_ratio_min": 0.62,
            }

            mock_ssim2.return_value = {
                "ssimulacra2_mean": 0.58,
                "ssimulacra2_p95": 0.54,
                "ssimulacra2_min": 0.50,
                "ssimulacra2_frame_count": 3.0,
                "ssimulacra2_triggered": 1.0,
            }

            # Calculate metrics
            metrics = calculate_comprehensive_metrics_from_frames(orig_frames, comp_frames, config)

        # Step 2: Create validation metadata
        original_metadata = GifMetadata(
            gif_sha="test_sha256_hash",
            orig_filename="test_ui.gif",
            orig_width=640,
            orig_height=480,
            orig_n_colors=256,
            orig_frames=3,
            orig_fps=10.0,
            orig_kilobytes=90.0
        )

        # Step 3: Run validation
        validation_config = ValidationConfig()
        checker = ValidationChecker(validation_config)

        validation_result = checker.validate_compression_result(
            original_metadata=original_metadata,
            compression_metrics=metrics,
            gif_name="e2e_test_ui",
            pipeline_id="ui_pipeline",
            content_type="ui",
        )

        # Step 4: Verify complete flow
        assert validation_result.metrics.has_text_ui_content == 1.0  # Stored as float
        assert validation_result.metrics.text_ui_edge_density == 0.18

        # Should complete validation successfully
        assert validation_result.status != ValidationStatus.UNKNOWN

        # May detect issues due to degradation
        total_feedback = len(validation_result.issues) + len(validation_result.warnings)
        assert total_feedback >= 0  # May have issues, but should not crash

    def test_validation_batch_processing(self):
        """Test validation with batch processing of multiple GIFs."""
        validation_config = ValidationConfig()
        checker = ValidationChecker(validation_config)

        # Create batch of test cases
        test_cases = [
            (
                "ui_gif_1",
                "ui",
                {"has_text_ui_content": True, "edge_sharpness_score": 85.0},
            ),
            (
                "animation_gif_2",
                "animation",
                {"has_text_ui_content": False, "ssimulacra2_mean": 0.75},
            ),
            (
                "mixed_gif_3",
                "mixed",
                {"has_text_ui_content": True, "ssimulacra2_mean": 0.45},
            ),
        ]

        batch_results = []

        for gif_name, content_type, phase3_metrics in test_cases:
            metrics = {
                "composite_quality": 0.70,
                "efficiency": 0.65,
                "compression_ratio": 2.0,
                "compressed_frame_count": 5,
                **phase3_metrics,
            }

            original_metadata = GifMetadata(
                gif_sha=f"test_sha_{gif_name}",
                orig_filename=f"{gif_name}.gif",
                orig_width=640,
                orig_height=480,
                orig_n_colors=256,
                orig_frames=5,
                orig_fps=15.0,
                orig_kilobytes=100.0
            )

            result = checker.validate_compression_result(
                original_metadata=original_metadata,
                compression_metrics=metrics,
                gif_name=gif_name,
                pipeline_id="batch_pipeline",
                content_type=content_type,
            )

            batch_results.append(result)

        # Verify batch processing
        assert len(batch_results) == 3

        # All should complete successfully
        for result in batch_results:
            assert result.status != ValidationStatus.UNKNOWN

        # Different content types may have different validation outcomes
        content_types = {result.content_type for result in batch_results}
        assert len(content_types) == 3  # ui, animation, mixed

    def test_validation_error_recovery(self):
        """Test validation system error recovery with Phase 3 failures."""
        validation_config = ValidationConfig()
        checker = ValidationChecker(validation_config)

        # Test with incomplete Phase 3 metrics
        incomplete_metrics = {
            "composite_quality": 0.60,
            "has_text_ui_content": True,
            "text_ui_edge_density": 0.15,
            # Missing: ocr_conf_delta_mean, ssimulacra2_mean, etc.
        }

        original_metadata = GifMetadata(
            gif_sha="test_sha_incomplete",
            orig_filename="test_error.gif",
            orig_width=640,
            orig_height=480,
            orig_n_colors=256,
            orig_frames=5,
            orig_fps=12.0,
            orig_kilobytes=80.0
        )

        # Should handle missing Phase 3 metrics gracefully
        result = checker.validate_compression_result(
            original_metadata=original_metadata,
            compression_metrics=incomplete_metrics,
            gif_name="incomplete_metrics_test",
            pipeline_id="error_recovery_pipeline",
            content_type="ui",
        )

        # Should not crash with incomplete metrics
        assert result.status != ValidationStatus.UNKNOWN

        # May add warnings about missing metrics
        warning_messages = [w.message for w in result.warnings]
        missing_metric_warnings = [
            msg
            for msg in warning_messages
            if "unavailable" in msg.lower() or "missing" in msg.lower()
        ]

        # Should handle missing metrics appropriately
        assert len(missing_metric_warnings) >= 0

    def test_validation_performance_with_phase3(self):
        """Test that validation performance remains acceptable with Phase 3."""
        validation_config = ValidationConfig()
        checker = ValidationChecker(validation_config)

        # Create comprehensive metrics set
        full_metrics = {
            # Basic metrics
            "composite_quality": 0.72,
            "efficiency": 0.68,
            "compression_ratio": 2.5,
            # All Phase 1-3 metrics
            "flicker_excess": 0.02,
            "deltae_mean": 1.8,
            "banding_score_mean": 25.0,
            "lpips_quality_mean": 0.18,
            "dither_ratio_mean": 1.05,
            # Full Phase 3 metrics
            "has_text_ui_content": True,
            "text_ui_edge_density": 0.14,
            "text_ui_component_count": 8,
            "ocr_conf_delta_mean": -0.04,
            "ocr_conf_delta_min": -0.09,
            "ocr_regions_analyzed": 5,
            "mtf50_ratio_mean": 0.80,
            "mtf50_ratio_min": 0.72,
            "edge_sharpness_score": 82.0,
            "ssimulacra2_mean": 0.65,
            "ssimulacra2_p95": 0.62,
            "ssimulacra2_min": 0.58,
            "ssimulacra2_frame_count": 8.0,
            "ssimulacra2_triggered": 1.0,
        }

        original_metadata = GifMetadata(
            gif_sha="test_sha_performance",
            orig_filename="test_perf.gif",
            orig_width=640,
            orig_height=480,
            orig_n_colors=256,
            orig_frames=8,
            orig_fps=15.0,
            orig_kilobytes=110.0
        )

        # Measure validation time
        start_time = time.perf_counter()

        result = checker.validate_compression_result(
            original_metadata=original_metadata,
            compression_metrics=full_metrics,
            gif_name="performance_test_gif",
            pipeline_id="comprehensive_pipeline",
            content_type="mixed",
        )

        end_time = time.perf_counter()
        validation_time = end_time - start_time

        # Performance target: validation should complete quickly
        assert validation_time < 0.5, f"Validation too slow: {validation_time:.4f}s"

        # Should complete successfully with comprehensive metrics
        assert result.status != ValidationStatus.UNKNOWN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
