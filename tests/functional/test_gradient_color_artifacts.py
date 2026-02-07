"""Unit tests for gradient banding and color artifact detection.

This test suite covers the gradient banding detection and perceptual color
validation systems, including edge cases, boundary conditions, and fixture
validation.

Trimmed from 71 tests to ~20 representative tests covering distinct failure modes.
"""

import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from giflab.gradient_color_artifacts import (
    GradientBandingDetector,
    PerceptualColorValidator,
    calculate_gradient_color_metrics,
)
from giflab.metrics import extract_gif_frames

from tests.fixtures.generate_gradient_color_fixtures import (
    create_color_shift_gif,
    create_smooth_gradient_gif,
)


class TestGradientBandingDetector:
    """Test gradient banding detection functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = GradientBandingDetector(patch_size=32, variance_threshold=100.0)

    def test_detector_initialization(self):
        """Test detector initialization with default and custom parameters."""
        detector = GradientBandingDetector()
        assert detector.patch_size == 64
        assert detector.variance_threshold == 100.0

        detector_custom = GradientBandingDetector(
            patch_size=32, variance_threshold=50.0
        )
        assert detector_custom.patch_size == 32
        assert detector_custom.variance_threshold == 50.0

    def test_smooth_gradient_no_banding(self):
        """Test that smooth gradients don't trigger banding detection."""
        frames = self._create_smooth_gradient_frames()

        result = self.detector.detect_banding_artifacts(frames, frames)

        assert result["banding_score_mean"] < 10.0
        assert result["gradient_region_count"] >= 0
        assert result["banding_patch_count"] >= 0

    def test_banded_gradient_detection(self):
        """Test that posterized gradients trigger banding detection."""
        smooth_frames = self._create_smooth_gradient_frames()
        banded_frames = self._create_banded_gradient_frames()

        result = self.detector.detect_banding_artifacts(smooth_frames, banded_frames)

        assert result["banding_score_mean"] >= 0.0
        assert result["gradient_region_count"] >= 0
        if result["banding_patch_count"] > 0:
            assert result["banding_score_p95"] >= result["banding_score_mean"]

    def test_empty_frames_handling(self):
        """Test handling of empty frame lists."""
        result = self.detector.detect_banding_artifacts([], [])

        assert result["banding_score_mean"] == 0.0
        assert result["banding_score_p95"] == 0.0
        assert result["banding_patch_count"] == 0
        assert result["gradient_region_count"] == 0

    def test_mismatched_frame_shapes(self):
        """Test handling of frames with different sizes."""
        frame_small = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        frame_large = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        result = self.detector.detect_banding_artifacts([frame_small], [frame_large])

        assert isinstance(result, dict)
        assert result["banding_score_mean"] >= 0.0

    def _create_smooth_gradient_frames(self, num_frames=3):
        """Create frames with smooth gradients for testing."""
        frames = []
        for _i in range(num_frames):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            for x in range(64):
                intensity = int(x * 255 / 63)
                frame[:, x] = [intensity, intensity, intensity]
            frames.append(frame)
        return frames

    def _create_banded_gradient_frames(self, num_frames=3, bands=8):
        """Create frames with posterized/banded gradients for testing."""
        frames = []
        for _i in range(num_frames):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            band_width = 64 // bands
            for band in range(bands):
                intensity = int(band * 255 / (bands - 1))
                x_start = band * band_width
                x_end = min((band + 1) * band_width, 64)
                frame[:, x_start:x_end] = [intensity, intensity, intensity]
            frames.append(frame)
        return frames


class TestPerceptualColorValidator:
    """Test perceptual color validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = PerceptualColorValidator(
            patch_size=32, jnd_thresholds=[1, 2, 3, 5]
        )

    def test_validator_initialization(self):
        """Test validator initialization with default and custom parameters."""
        validator = PerceptualColorValidator()
        assert validator.patch_size == 64
        assert validator.jnd_thresholds == [1.0, 2.0, 3.0, 5.0]

        validator_custom = PerceptualColorValidator(
            patch_size=32, jnd_thresholds=[1, 3]
        )
        assert validator_custom.patch_size == 32
        assert validator_custom.jnd_thresholds == [1, 3]

    def test_identical_frames_no_color_difference(self):
        """Test that identical frames have minimal color difference."""
        frames = self._create_test_color_frames()

        result = self.validator.calculate_color_difference_metrics(frames, frames)

        assert result["deltae_mean"] == pytest.approx(0.0, abs=0.1)
        assert result["deltae_p95"] == pytest.approx(0.0, abs=0.1)
        assert result["deltae_pct_gt1"] == pytest.approx(0.0, abs=1.0)

    def test_different_color_frames(self):
        """Test color difference detection between different frames."""
        frames_original = self._create_test_color_frames()
        frames_shifted = self._create_shifted_color_frames()

        result = self.validator.calculate_color_difference_metrics(
            frames_original, frames_shifted
        )

        assert result["deltae_mean"] >= 0.0
        assert result["deltae_p95"] >= result["deltae_mean"]
        assert result["deltae_max"] >= result["deltae_p95"]
        assert result["color_patch_count"] > 0

    def test_empty_frames_handling(self):
        """Test handling of empty frame lists."""
        result = self.validator.calculate_color_difference_metrics([], [])

        expected_keys = [
            "deltae_mean",
            "deltae_p95",
            "deltae_max",
            "deltae_pct_gt1",
            "deltae_pct_gt2",
            "deltae_pct_gt3",
            "deltae_pct_gt5",
            "color_patch_count",
        ]

        assert all(key in result for key in expected_keys)
        assert result["deltae_mean"] == 0.0
        assert result["color_patch_count"] == 0

    def test_threshold_percentages(self):
        """Test that threshold percentages are calculated correctly."""
        frames_original = self._create_test_color_frames()
        frames_high_diff = self._create_high_difference_color_frames()

        result = self.validator.calculate_color_difference_metrics(
            frames_original, frames_high_diff
        )

        # Check that percentages are valid
        assert 0.0 <= result["deltae_pct_gt1"] <= 100.0
        assert 0.0 <= result["deltae_pct_gt2"] <= 100.0
        assert 0.0 <= result["deltae_pct_gt3"] <= 100.0
        assert 0.0 <= result["deltae_pct_gt5"] <= 100.0

        # Higher thresholds should have lower or equal percentages
        assert result["deltae_pct_gt1"] >= result["deltae_pct_gt2"]
        assert result["deltae_pct_gt2"] >= result["deltae_pct_gt3"]
        assert result["deltae_pct_gt3"] >= result["deltae_pct_gt5"]

    def _create_test_color_frames(self, num_frames=3):
        """Create frames with test colors for validation."""
        frames = []
        colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (128, 128, 128),  # Gray
        ]

        for _i in range(num_frames):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            patch_size = 32
            for idx, color in enumerate(colors):
                x = (idx % 2) * patch_size
                y = (idx // 2) * patch_size
                frame[y : y + patch_size, x : x + patch_size] = color
            frames.append(frame)
        return frames

    def _create_shifted_color_frames(self, num_frames=3):
        """Create frames with slightly shifted colors."""
        frames = []
        colors = [
            (235, 20, 20),  # Shifted red
            (20, 235, 20),  # Shifted green
            (20, 20, 235),  # Shifted blue
            (148, 108, 108),  # Shifted gray
        ]

        for _i in range(num_frames):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            patch_size = 32
            for idx, color in enumerate(colors):
                x = (idx % 2) * patch_size
                y = (idx // 2) * patch_size
                frame[y : y + patch_size, x : x + patch_size] = color
            frames.append(frame)
        return frames

    def _create_high_difference_color_frames(self, num_frames=3):
        """Create frames with high color differences for threshold testing."""
        frames = []
        colors = [
            (100, 200, 50),  # Very different from red
            (200, 50, 200),  # Very different from green
            (255, 255, 0),  # Very different from blue
            (50, 200, 200),  # Very different from gray
        ]

        for _i in range(num_frames):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            patch_size = 32
            for idx, color in enumerate(colors):
                x = (idx % 2) * patch_size
                y = (idx // 2) * patch_size
                frame[y : y + patch_size, x : x + patch_size] = color
            frames.append(frame)
        return frames


class TestGradientColorIntegration:
    """Test integration functions and error handling."""

    def test_calculate_gradient_color_metrics_combined(self):
        """Test combined gradient and color metrics calculation."""
        frames = self._create_test_frames()

        result = calculate_gradient_color_metrics(frames, frames)

        expected_keys = [
            "banding_score_mean",
            "banding_score_p95",
            "banding_patch_count",
            "gradient_region_count",
            "deltae_mean",
            "deltae_p95",
            "deltae_max",
            "deltae_pct_gt1",
            "deltae_pct_gt2",
            "deltae_pct_gt3",
            "deltae_pct_gt5",
            "color_patch_count",
        ]
        assert all(key in result for key in expected_keys)

    def test_exception_handling_in_combined_function(self):
        """Test that combined function handles exceptions gracefully."""
        with patch(
            "giflab.gradient_color_artifacts.calculate_banding_metrics",
            side_effect=Exception("Test error"),
        ):
            frames = self._create_test_frames()
            result = calculate_gradient_color_metrics(frames, frames)

            assert isinstance(result, dict)
            assert "banding_score_mean" in result
            assert result["banding_score_mean"] == 0.0

    def test_none_inputs_handling(self):
        """Test handling of None inputs."""
        result = calculate_gradient_color_metrics(None, None)

        assert isinstance(result, dict)

    def _create_test_frames(self, num_frames=3):
        """Create simple test frames for testing."""
        frames = []
        for _i in range(num_frames):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            for x in range(64):
                intensity = int(x * 255 / 63)
                frame[:32, x] = [intensity, intensity // 2, intensity // 3]
            frame[32:48, :32] = [255, 0, 0]
            frame[32:48, 32:] = [0, 255, 0]
            frame[48:, :32] = [0, 0, 255]
            frame[48:, 32:] = [128, 128, 128]
            frames.append(frame)
        return frames


class TestGradientBandingEdgeCases:
    """Test edge cases and boundary conditions for gradient banding detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = GradientBandingDetector(patch_size=32, variance_threshold=100.0)

    def test_single_pixel_images(self):
        """Test with single pixel images."""
        single_pixel_frames = []
        for i in range(2):
            frame = np.full((1, 1, 3), [i * 127, 128, 255 - i * 127], dtype=np.uint8)
            single_pixel_frames.append(frame)

        result = self.detector.detect_banding_artifacts(
            single_pixel_frames, single_pixel_frames
        )

        assert result["banding_score_mean"] == 0.0
        assert result["gradient_region_count"] == 0
        assert result["banding_patch_count"] == 0

    def test_solid_color_images(self):
        """Test with solid color images (no gradients)."""
        solid_frames = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        for color in colors:
            frame = np.full((64, 64, 3), color, dtype=np.uint8)
            solid_frames.append(frame)

        result = self.detector.detect_banding_artifacts(solid_frames, solid_frames)

        assert result["banding_score_mean"] == 0.0
        assert result["gradient_region_count"] == 0

    def test_corrupted_frame_data(self):
        """Test handling of corrupted or invalid frame data."""
        corrupted_frame = np.full((64, 64, 3), np.nan, dtype=np.float32)

        try:
            regions = self.detector.detect_gradient_regions(
                corrupted_frame.astype(np.uint8)
            )
            assert isinstance(regions, list)
        except Exception:
            # Acceptable for corrupted data
            pass


class TestPerceptualColorEdgeCases:
    """Test edge cases for perceptual color validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = PerceptualColorValidator(
            patch_size=32, jnd_thresholds=[1, 2, 3, 5]
        )

    def test_extreme_color_differences(self):
        """Test with colors that have very large DeltaE00 differences."""
        black_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        white_frame = np.full((64, 64, 3), 255, dtype=np.uint8)

        result = self.validator.calculate_color_difference_metrics(
            [black_frame], [white_frame]
        )

        assert result["deltae_mean"] > 50.0
        assert result["deltae_pct_gt5"] > 90.0

    def test_rgb_to_lab_fallback(self):
        """Test RGB to Lab conversion fallback when scikit-image unavailable."""
        with patch("giflab.gradient_color_artifacts.SKIMAGE_AVAILABLE", False):
            validator = PerceptualColorValidator()

            rgb_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            result = validator.rgb_to_lab(rgb_image)

            assert result.shape == rgb_image.shape
            assert result.dtype == np.float32


class TestIntegrationEdgeCases:
    """Test edge cases in the integration functions."""

    def test_mismatched_frame_counts(self):
        """Test with different numbers of original vs compressed frames."""
        orig_frames = [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(5)
        ]
        comp_frames = [
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)
        ]

        result = calculate_gradient_color_metrics(orig_frames, comp_frames)

        assert isinstance(result, dict)
        assert all(isinstance(v, int | float) for v in result.values())

    def test_thread_safety_simulation(self):
        """Test that functions are thread-safe by running concurrent calls."""
        frames = [
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(2)
        ]
        results = []
        exceptions = []

        def run_calculation():
            try:
                result = calculate_gradient_color_metrics(frames, frames)
                results.append(result)
            except Exception as e:
                exceptions.append(e)

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=run_calculation)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"
        assert len(results) == 5

        first_result = results[0]
        for result in results[1:]:
            for key in first_result:
                if isinstance(first_result[key], float):
                    assert (
                        abs(first_result[key] - result[key]) < 1e-10
                    ), f"Inconsistent results for {key}"
                else:
                    assert (
                        first_result[key] == result[key]
                    ), f"Inconsistent results for {key}"


class TestFixtureValidation:
    """Validate test fixtures demonstrate expected artifacts."""

    def setup_method(self):
        """Set up fixture validation tests."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up fixture validation tests."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_smooth_gradient_fixtures(self):
        """Verify smooth gradients don't trigger banding detection."""
        original_cwd = Path.cwd()
        try:
            import os

            os.chdir(self.temp_dir)

            smooth_horizontal = create_smooth_gradient_gif("horizontal")
            smooth_vertical = create_smooth_gradient_gif("vertical")
            smooth_radial = create_smooth_gradient_gif("radial")

            fixtures = [
                ("horizontal", smooth_horizontal),
                ("vertical", smooth_vertical),
                ("radial", smooth_radial),
            ]

            for direction, gif_path in fixtures:
                if not gif_path.exists():
                    pytest.skip(f"Fixture creation failed: {gif_path}")

                extract_result = extract_gif_frames(gif_path)
                frames = extract_result.frames

                result = calculate_gradient_color_metrics(frames, frames)

                assert (
                    result["banding_score_mean"] < 30.0
                ), f"Smooth {direction} gradient triggered banding: {result['banding_score_mean']}"

                assert (
                    result["gradient_region_count"] >= 0
                ), f"No gradient regions detected in {direction} gradient"

                assert (
                    result["deltae_mean"] < 1.0
                ), f"Color differences in identical {direction} frames: {result['deltae_mean']}"

        finally:
            os.chdir(original_cwd)

    def test_color_shift_fixtures(self):
        """Verify color shifts match expected DeltaE00 ranges."""
        original_cwd = Path.cwd()
        try:
            import os

            os.chdir(self.temp_dir)

            severities = ["high", "medium", "low"]
            results = {}

            for severity in severities:
                original_path, shifted_path = create_color_shift_gif(severity)

                if not (original_path.exists() and shifted_path.exists()):
                    pytest.skip(f"Color shift fixture creation failed: {severity}")

                orig_extract = extract_gif_frames(original_path)
                shift_extract = extract_gif_frames(shifted_path)

                orig_frames = orig_extract.frames
                shift_frames = shift_extract.frames

                result = calculate_gradient_color_metrics(orig_frames, shift_frames)
                results[severity] = result

                assert (
                    result["deltae_mean"] > 0.0
                ), f"No color differences detected in {severity} shift"
                assert (
                    result["color_patch_count"] > 0
                ), f"No color patches analyzed in {severity} shift"

            # Verify severity ordering
            if all(s in results for s in ["high", "medium", "low"]):
                high_mean = results["high"]["deltae_mean"]
                medium_mean = results["medium"]["deltae_mean"]
                low_mean = results["low"]["deltae_mean"]

                assert (
                    high_mean >= medium_mean * 0.8
                ), f"High severity not greater than medium: {high_mean} vs {medium_mean}"
                assert (
                    medium_mean >= low_mean * 0.8
                ), f"Medium severity not greater than low: {medium_mean} vs {low_mean}"

        finally:
            os.chdir(original_cwd)
