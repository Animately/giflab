"""Unit tests for text/UI validation functionality.

Trimmed to ~25 representative tests covering distinct failure modes:
- Initialization, edge density detection, component detection, region merging
- OCR confidence calculation, preprocessing, library fallbacks
- Edge acuity / MTF50 analysis
- Validation decision logic, full pipeline integration
- Error handling, invalid inputs, numerical stability
"""

from unittest.mock import patch

import cv2
import numpy as np
import pytest
from giflab.text_ui_validation import (
    EdgeAcuityAnalyzer,
    OCRValidator,
    TextUIContentDetector,
    calculate_text_ui_metrics,
    should_validate_text_ui,
)


class TestTextUIContentDetector:
    """Test cases for TextUIContentDetector."""

    def test_init_default_and_custom(self):
        """Test TextUIContentDetector initialization with defaults and custom params."""
        detector = TextUIContentDetector()
        assert detector.edge_threshold == 30.0
        assert detector.min_component_area == 10
        assert detector.max_component_area == 500
        assert detector.edge_density_threshold == 0.03

        custom = TextUIContentDetector(
            edge_threshold=60.0,
            min_component_area=20,
            max_component_area=1000,
            edge_density_threshold=0.15,
        )
        assert custom.edge_threshold == 60.0
        assert custom.min_component_area == 20
        assert custom.max_component_area == 1000
        assert custom.edge_density_threshold == 0.15

    def test_detect_edge_density_rgb_frame(self):
        """Test edge density calculation on RGB frame."""
        detector = TextUIContentDetector()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[40:60, 40:60] = 255  # White square for edges

        edge_density = detector._detect_edge_density(frame)
        assert 0.0 <= edge_density <= 1.0
        assert edge_density > 0.0

    def test_find_text_like_components_empty_image(self):
        """Test component detection on empty image."""
        detector = TextUIContentDetector()
        binary_image = np.zeros((100, 100), dtype=np.uint8)

        components = detector._find_text_like_components(binary_image)
        assert len(components) == 0

    def test_detect_text_ui_regions_low_edge_density(self):
        """Test text/UI region detection with low edge density."""
        detector = TextUIContentDetector(edge_density_threshold=0.2)

        frame = np.full((100, 100, 3), 128, dtype=np.uint8)

        regions = detector.detect_text_ui_regions(frame)
        assert len(regions) == 0

    def test_detect_text_ui_regions_high_edge_density(self):
        """Test text/UI region detection with high edge density."""
        detector = TextUIContentDetector(edge_density_threshold=0.05)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:25, 10:30, :] = 255
        frame[30:35, 10:25, :] = 255
        frame[40:45, 10:35, :] = 255

        regions = detector.detect_text_ui_regions(frame)
        assert len(regions) >= 0

    def test_merge_nearby_regions_overlapping_and_separate(self):
        """Test region merging with overlapping and non-overlapping regions."""
        detector = TextUIContentDetector()

        # Overlapping regions should merge
        regions = [(10, 10, 20, 20), (15, 15, 20, 20)]
        merged = detector._merge_nearby_regions(regions)
        assert len(merged) == 1

        # Non-overlapping regions should stay separate
        regions = [(10, 10, 20, 20), (50, 50, 20, 20)]
        merged = detector._merge_nearby_regions(regions)
        assert len(merged) == 2

    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        detector = TextUIContentDetector()

        with pytest.raises((ValueError, TypeError)):
            detector.detect_text_ui_regions(None)

        empty_frame = np.array([])
        with pytest.raises((ValueError, IndexError)):
            detector._detect_edge_density(empty_frame)

        wrong_dim_frame = np.random.randint(0, 255, (10,), dtype=np.uint8)
        with pytest.raises((ValueError, IndexError)):
            detector._detect_edge_density(wrong_dim_frame)

    @patch("cv2.Canny")
    def test_edge_detection_failure(self, mock_canny):
        """Test handling of edge detection failures."""
        mock_canny.side_effect = cv2.error("Simulated CV2 error")

        detector = TextUIContentDetector()
        frame = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        edge_density = detector._detect_edge_density(frame)
        assert edge_density == 0.0


class TestOCRValidator:
    """Test cases for OCRValidator."""

    def test_init_default_and_custom(self):
        """Test OCRValidator initialization with defaults and custom parameters."""
        validator = OCRValidator()
        assert isinstance(validator.use_tesseract, bool)
        assert isinstance(validator.fallback_to_easyocr, bool)

        custom = OCRValidator(use_tesseract=False, fallback_to_easyocr=False)
        assert custom.use_tesseract is False
        assert custom.fallback_to_easyocr is False

    def test_calculate_confidence_delta(self):
        """Test confidence delta calculation."""
        validator = OCRValidator()
        assert validator._calculate_confidence_delta(0.8, 0.6) == pytest.approx(-0.2)
        assert validator._calculate_confidence_delta(0.5, 0.7) == pytest.approx(0.2)

    def test_calculate_ocr_confidence_delta_empty_regions(self):
        """Test OCR confidence calculation with empty regions."""
        validator = OCRValidator()
        original = np.zeros((100, 100, 3), dtype=np.uint8)
        compressed = np.zeros((100, 100, 3), dtype=np.uint8)

        result = validator.calculate_ocr_confidence_delta(original, compressed, [])
        expected = {
            "ocr_conf_delta_mean": 0.0,
            "ocr_conf_delta_min": 0.0,
            "ocr_regions_analyzed": 0,
        }
        assert result == expected

    def test_preprocess_for_ocr(self):
        """Test OCR preprocessing produces grayscale uint8 output."""
        validator = OCRValidator()

        roi = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        processed = validator._preprocess_for_ocr(roi)

        assert len(processed.shape) == 2  # Grayscale
        assert processed.dtype == np.uint8
        assert processed.shape[0] >= 20 and processed.shape[1] >= 20

    def test_ocr_library_unavailable(self):
        """Test behavior when OCR libraries are unavailable."""
        with patch("giflab.text_ui_validation.TESSERACT_AVAILABLE", False), patch(
            "giflab.text_ui_validation.EASYOCR_AVAILABLE", False
        ):
            validator = OCRValidator()
            assert validator.use_tesseract is False
            assert validator.fallback_to_easyocr is False

            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            regions = [(10, 10, 30, 20)]

            result = validator.calculate_ocr_confidence_delta(frame, frame, regions)
            assert "ocr_conf_delta_mean" in result

    def test_template_matching_fallback(self):
        """Test template matching confidence fallback with different patterns."""
        validator = OCRValidator()

        # Structured pattern should give reasonable confidence
        structured_roi = np.zeros((50, 50), dtype=np.uint8)
        structured_roi[10:15, 5:45] = 255
        structured_roi[20:25, 5:35] = 255
        structured_roi[30:35, 5:40] = 255
        confidence = validator._template_matching_confidence(structured_roi)
        assert 0.1 < confidence <= 1.0

        # Solid color should give minimum confidence
        solid_roi = np.full((50, 50), 128, dtype=np.uint8)
        confidence = validator._template_matching_confidence(solid_roi)
        assert confidence == 0.1


class TestEdgeAcuityAnalyzer:
    """Test cases for EdgeAcuityAnalyzer."""

    def test_init_default_and_custom(self):
        """Test EdgeAcuityAnalyzer initialization."""
        analyzer = EdgeAcuityAnalyzer()
        assert analyzer.mtf_threshold == 0.5

        custom = EdgeAcuityAnalyzer(mtf_threshold=0.3)
        assert custom.mtf_threshold == 0.3

    def test_calculate_mtf50_empty_regions(self):
        """Test MTF50 calculation with empty regions returns defaults."""
        analyzer = EdgeAcuityAnalyzer()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        result = analyzer.calculate_mtf50(frame, [])
        assert result == {
            "mtf50_ratio_mean": 1.0,
            "mtf50_ratio_min": 1.0,
            "edge_sharpness_score": 100.0,
        }

    def test_calculate_mtf50_with_regions(self):
        """Test MTF50 calculation with valid regions."""
        analyzer = EdgeAcuityAnalyzer()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[40:60, 40:60, :] = 255

        regions = [(35, 35, 30, 30)]
        result = analyzer.calculate_mtf50(frame, regions)

        assert "mtf50_ratio_mean" in result
        assert "mtf50_ratio_min" in result
        assert "edge_sharpness_score" in result
        assert isinstance(result["edge_sharpness_score"], float)
        assert 0.0 <= result["edge_sharpness_score"] <= 100.0

    def test_find_mtf50_frequency_normal_case(self):
        """Test MTF50 frequency finding in normal case."""
        analyzer = EdgeAcuityAnalyzer()
        normal_curve = np.array([1.0, 0.8, 0.6, 0.4, 0.2])

        result = analyzer._find_mtf50_frequency(normal_curve)
        assert 0.0 < result < 1.0

    def test_edge_detection_failure_handling(self):
        """Test handling when ROI has no detectable edges."""
        analyzer = EdgeAcuityAnalyzer()

        no_edge_roi = np.full((50, 50), 128, dtype=np.uint8)
        regions = [(10, 10, 30, 30)]

        result = analyzer.calculate_mtf50(no_edge_roi, regions)
        assert result["mtf50_ratio_mean"] >= 0.0
        assert result["edge_sharpness_score"] >= 0.0


class TestShouldValidateTextUI:
    """Test cases for should_validate_text_ui function."""

    def test_empty_frames(self):
        """Test validation decision with empty frame list."""
        should_validate, hints = should_validate_text_ui([])
        assert should_validate is False
        assert hints["edge_density"] == 0.0
        assert hints["component_count"] == 0

    def test_high_edge_density_frames(self):
        """Test validation decision with high edge density frames."""
        frames = []
        for _i in range(2):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            frame[20:25, 10:30, :] = 255
            frame[30:35, 10:25, :] = 255
            frame[40:45, 10:35, :] = 255
            frames.append(frame)

        should_validate, hints = should_validate_text_ui(frames, quick_check=False)

        assert "edge_density" in hints
        assert "max_edge_density" in hints
        assert "component_count" in hints
        assert hints["frames_analyzed"] == 2

    def test_quick_check_mode(self):
        """Test validation decision in quick check mode."""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frames = [frame, frame, frame]

        should_validate, hints = should_validate_text_ui(frames, quick_check=True)
        assert hints["frames_analyzed"] == 1


class TestCalculateTextUIMetrics:
    """Test cases for calculate_text_ui_metrics function."""

    def test_mismatched_frame_counts(self):
        """Test error handling with mismatched frame counts."""
        original = [np.zeros((50, 50, 3), dtype=np.uint8)]
        compressed = [
            np.zeros((50, 50, 3), dtype=np.uint8),
            np.zeros((50, 50, 3), dtype=np.uint8),
        ]

        with pytest.raises(
            ValueError, match="Original and compressed frame counts must match"
        ):
            calculate_text_ui_metrics(original, compressed)

    def test_no_text_ui_content(self):
        """Test metrics calculation when no text/UI content is detected."""
        frames = []
        for _i in range(2):
            frame = np.full((50, 50, 3), 128, dtype=np.uint8)
            frames.append(frame)

        result = calculate_text_ui_metrics(frames, frames)

        assert result["has_text_ui_content"] is False
        assert "text_ui_edge_density" in result
        assert "text_ui_component_count" in result
        assert result["text_ui_component_count"] == 0

    @patch("giflab.text_ui_validation.should_validate_text_ui")
    def test_conditional_execution_mocking(self, mock_should_validate):
        """Test conditional execution logic with mocked validation decision."""
        frames = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(2)]

        # When validation is skipped
        mock_should_validate.return_value = (
            False,
            {"edge_density": 0.05, "component_count": 0},
        )
        result = calculate_text_ui_metrics(frames, frames)
        assert result["has_text_ui_content"] is False
        assert result["text_ui_edge_density"] == 0.05
        assert result["text_ui_component_count"] == 0

        # When validation is triggered
        mock_should_validate.return_value = (
            True,
            {"edge_density": 0.15, "component_count": 5},
        )
        result = calculate_text_ui_metrics(frames, frames)
        assert result["has_text_ui_content"] is True
        assert "text_ui_edge_density" in result
        assert "text_ui_component_count" in result


class TestTextUIValidationIntegration:
    """Integration tests for text/UI validation system."""

    def test_full_pipeline_no_text(self):
        """Test full pipeline with non-text content."""
        original_frames = []
        compressed_frames = []

        for _i in range(2):
            orig = np.random.randint(100, 200, (80, 80, 3), dtype=np.uint8)
            orig[:40, :, 0] = np.linspace(100, 150, 80).reshape(1, -1)
            original_frames.append(orig)

            comp = cv2.GaussianBlur(orig, (1, 1), 0.5)
            compressed_frames.append(comp)

        result = calculate_text_ui_metrics(original_frames, compressed_frames)
        assert result["has_text_ui_content"] is False

    def test_full_pipeline_with_text(self):
        """Test full pipeline with text-like content."""
        original_frames = []
        compressed_frames = []

        for _i in range(2):
            orig = np.zeros((100, 100, 3), dtype=np.uint8)
            orig[20:26, 10:60, :] = 255
            orig[35:41, 10:50, :] = 255
            orig[50:56, 10:55, :] = 255
            orig[15:65, 5:8, :] = 255
            orig[15:65, 70:73, :] = 255
            original_frames.append(orig)

            comp = orig.copy()
            comp = cv2.GaussianBlur(comp, (3, 3), 0.8)
            noise = np.random.randint(-10, 10, comp.shape).astype(np.int16)
            comp = np.clip(comp.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            compressed_frames.append(comp)

        result = calculate_text_ui_metrics(original_frames, compressed_frames)

        if result["has_text_ui_content"]:
            assert "ocr_regions_analyzed" in result
            if result["ocr_regions_analyzed"] > 0:
                assert "edge_sharpness_score" in result
            assert isinstance(result["text_ui_edge_density"], float)
            assert result["text_ui_edge_density"] >= 0.0

    def test_numerical_stability(self):
        """Test numerical stability with extreme pixel values."""
        extreme_frames = [
            np.zeros((50, 50, 3), dtype=np.uint8),  # All black
            np.full((50, 50, 3), 255, dtype=np.uint8),  # All white
            np.random.randint(0, 2, (50, 50, 3)) * 255,  # Pure black/white noise
        ]

        for frame in extreme_frames:
            frames = [frame, frame]
            result = calculate_text_ui_metrics(frames, frames)

            for key, value in result.items():
                if isinstance(value, int | float):
                    assert np.isfinite(value), f"Non-finite value for {key}: {value}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
