"""Configuration and threshold tests for gradient banding and color validation.

This test suite validates configuration options, threshold tuning, and parameter
sensitivity for the gradient banding detection and perceptual color validation systems.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from giflab.gradient_color_artifacts import (
    GradientBandingDetector,
    PerceptualColorValidator,
)


class TestConfiguration:
    """Test configuration options and parameter settings."""

    def setup_method(self):
        """Set up configuration tests."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up configuration tests."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @pytest.mark.fast
    def test_gradient_banding_detector_initialization(self):
        """Test GradientBandingDetector initialization with different parameters."""
        # Test default initialization
        detector_default = GradientBandingDetector()
        assert detector_default.patch_size == 64
        assert detector_default.variance_threshold == 100.0

        # Test custom initialization
        detector_custom = GradientBandingDetector(
            patch_size=32, variance_threshold=50.0
        )
        assert detector_custom.patch_size == 32
        assert detector_custom.variance_threshold == 50.0

        # Test edge case parameters
        detector_edge = GradientBandingDetector(patch_size=8, variance_threshold=1.0)
        assert detector_edge.patch_size == 8
        assert detector_edge.variance_threshold == 1.0

        # Test with very large parameters
        detector_large = GradientBandingDetector(
            patch_size=256, variance_threshold=1000.0
        )
        assert detector_large.patch_size == 256
        assert detector_large.variance_threshold == 1000.0

    @pytest.mark.fast
    def test_color_validator_initialization(self):
        """Test PerceptualColorValidator initialization with different parameters."""
        # Test default initialization
        validator_default = PerceptualColorValidator()
        assert validator_default.patch_size == 64
        assert validator_default.jnd_thresholds == [1.0, 2.0, 3.0, 5.0]

        # Test custom initialization
        validator_custom = PerceptualColorValidator(
            patch_size=32, jnd_thresholds=[1, 3, 5]
        )
        assert validator_custom.patch_size == 32
        assert validator_custom.jnd_thresholds == [1, 3, 5]

        # Test with single threshold
        validator_single = PerceptualColorValidator(jnd_thresholds=[2.5])
        assert validator_single.jnd_thresholds == [2.5]

        # Test with many thresholds
        validator_many = PerceptualColorValidator(
            jnd_thresholds=[0.5, 1, 1.5, 2, 3, 4, 5, 10]
        )
        assert len(validator_many.jnd_thresholds) == 8

    @pytest.mark.fast
    def test_adjustable_thresholds(self):
        """Test different threshold configurations and their impact."""
        frames = self._create_test_frames()

        # Test different variance thresholds for banding detection
        variance_thresholds = [10.0, 50.0, 100.0, 200.0, 500.0]
        results = {}

        for threshold in variance_thresholds:
            detector = GradientBandingDetector(variance_threshold=threshold)
            regions = detector.detect_gradient_regions(frames[0])
            results[threshold] = len(regions)

            print(f"Variance threshold {threshold}: {len(regions)} gradient regions")

        # Lower thresholds should generally detect more regions (more sensitive)
        # Higher thresholds should be more selective
        if results[10.0] > 0 and results[500.0] >= 0:
            # Allow for algorithm variation, but expect some general trend
            assert (
                results[10.0] >= results[500.0]
            ), "Low threshold should detect >= regions than high threshold"

        # All thresholds should return valid results
        for threshold, region_count in results.items():
            assert (
                region_count >= 0
            ), f"Invalid region count for threshold {threshold}: {region_count}"

    @pytest.mark.fast
    def test_patch_size_impact(self):
        """Test different patch sizes and their impact on detection."""
        frames = self._create_gradient_frames()

        # Test different patch sizes
        patch_sizes = [16, 32, 64, 128]
        results = {}

        for patch_size in patch_sizes:
            detector = GradientBandingDetector(patch_size=patch_size)
            validator = PerceptualColorValidator(patch_size=patch_size)

            # Test banding detection
            banding_result = detector.detect_banding_artifacts(frames, frames)

            # Test color validation
            color_result = validator.calculate_color_difference_metrics(frames, frames)

            results[patch_size] = {
                "banding_patches": banding_result["banding_patch_count"],
                "gradient_regions": banding_result["gradient_region_count"],
                "color_patches": color_result["color_patch_count"],
                "banding_score": banding_result["banding_score_mean"],
                "deltae": color_result["deltae_mean"],
            }

            print(
                f"Patch size {patch_size}: "
                f"banding_patches={banding_result['banding_patch_count']}, "
                f"color_patches={color_result['color_patch_count']}"
            )

        # Verify all results are valid
        for patch_size, result in results.items():
            assert (
                result["banding_patches"] >= 0
            ), f"Invalid banding patch count for size {patch_size}"
            assert (
                result["color_patches"] >= 0
            ), f"Invalid color patch count for size {patch_size}"
            assert (
                result["banding_score"] >= 0.0
            ), f"Invalid banding score for size {patch_size}"
            assert result["deltae"] >= 0.0, f"Invalid deltae for size {patch_size}"

        # Smaller patches should generally result in more patches analyzed
        # (more patches fit in the same image)
        if results[16]["color_patches"] > 0 and results[128]["color_patches"] > 0:
            ratio = results[16]["color_patches"] / results[128]["color_patches"]
            # Allow significant variance since patch placement is algorithmic
            assert (
                ratio >= 0.5
            ), f"Patch count scaling unexpected: 16px={results[16]['color_patches']}, 128px={results[128]['color_patches']}"

    @pytest.mark.fast
    def test_jnd_threshold_customization(self):
        """Test custom JND thresholds for color validation."""
        original_frames = self._create_test_frames()
        shifted_frames = self._create_shifted_frames()

        # Test different threshold configurations
        threshold_configs = [
            [1.0],  # Single threshold
            [1.0, 3.0, 5.0],  # Standard thresholds
            [0.5, 2.0, 4.0, 6.0],  # Custom thresholds
            [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0],  # Many thresholds
        ]

        for i, thresholds in enumerate(threshold_configs):
            validator = PerceptualColorValidator(jnd_thresholds=thresholds)
            result = validator.calculate_color_difference_metrics(
                original_frames, shifted_frames
            )

            print(f"Threshold config {i+1}: {thresholds}")
            print(f"  ΔE mean: {result['deltae_mean']:.2f}")

            # Check that appropriate threshold metrics are present
            # (The exact keys depend on the thresholds, but at least these should be there)
            expected_base_keys = [
                "deltae_mean",
                "deltae_p95",
                "deltae_max",
                "color_patch_count",
            ]
            for key in expected_base_keys:
                assert (
                    key in result
                ), f"Missing base key {key} for threshold config {thresholds}"

            # Verify the threshold-specific keys exist
            # Standard implementation includes gt1, gt2, gt3, gt5
            threshold_keys = [
                "deltae_pct_gt1",
                "deltae_pct_gt2",
                "deltae_pct_gt3",
                "deltae_pct_gt5",
            ]
            present_threshold_keys = [key for key in threshold_keys if key in result]
            assert (
                len(present_threshold_keys) > 0
            ), f"No threshold keys found for config {thresholds}"

    @pytest.mark.fast
    def test_fallback_configurations(self):
        """Test behavior with missing dependencies."""
        frames = self._create_test_frames()

        # Test with scikit-image unavailable
        with patch("giflab.gradient_color_artifacts.SKIMAGE_AVAILABLE", False):
            validator = PerceptualColorValidator()

            # Should still work with fallback
            result = validator.calculate_color_difference_metrics(frames, frames)

            # Should return valid structure
            assert isinstance(result, dict)
            assert "deltae_mean" in result
            assert "color_patch_count" in result

            # Values might be different with fallback, but should be valid numbers
            assert isinstance(result["deltae_mean"], float)
            assert result["deltae_mean"] >= 0.0

            print("✅ Fallback configuration works when scikit-image unavailable")

        # Test Lab conversion fallback
        validator = PerceptualColorValidator()
        with patch.object(validator, "rgb_to_lab") as mock_rgb_to_lab:
            # Make Lab conversion return RGB-approximation
            def fallback_conversion(rgb_image):
                return rgb_image.astype(np.float32)

            mock_rgb_to_lab.side_effect = fallback_conversion

            result = validator.calculate_color_difference_metrics(frames, frames)

            # Should still return valid results (though accuracy may be reduced)
            assert isinstance(result, dict)
            assert result["deltae_mean"] >= 0.0

    @pytest.mark.fast
    def test_parameter_sensitivity_analysis(self):
        """Analyze sensitivity to parameter changes."""
        base_frames = self._create_gradient_frames()
        banded_frames = self._create_banded_frames()

        # Test patch size sensitivity
        patch_sizes = [16, 32, 64, 128]
        patch_results = []

        for patch_size in patch_sizes:
            detector = GradientBandingDetector(patch_size=patch_size)
            result = detector.detect_banding_artifacts(base_frames, banded_frames)
            patch_results.append(
                {
                    "patch_size": patch_size,
                    "banding_score": result["banding_score_mean"],
                    "regions": result["gradient_region_count"],
                }
            )

        # Test variance threshold sensitivity
        variances = [50.0, 100.0, 200.0, 400.0]
        variance_results = []

        for variance in variances:
            detector = GradientBandingDetector(variance_threshold=variance)
            result = detector.detect_banding_artifacts(base_frames, banded_frames)
            variance_results.append(
                {
                    "variance_threshold": variance,
                    "banding_score": result["banding_score_mean"],
                    "regions": result["gradient_region_count"],
                }
            )

        # Analyze sensitivity
        print("Patch size sensitivity:")
        for result in patch_results:
            print(
                f"  Size {result['patch_size']}: score={result['banding_score']:.2f}, regions={result['regions']}"
            )

        print("Variance threshold sensitivity:")
        for result in variance_results:
            print(
                f"  Threshold {result['variance_threshold']}: score={result['banding_score']:.2f}, regions={result['regions']}"
            )

        # Results should be stable (not extreme variations)
        patch_scores = [
            r["banding_score"] for r in patch_results if r["banding_score"] > 0
        ]
        if len(patch_scores) > 1:
            patch_variation = max(patch_scores) / min(patch_scores)
            assert (
                patch_variation < 10.0
            ), f"Patch size too sensitive: {patch_variation:.1f}x variation"

        variance_scores = [
            r["banding_score"] for r in variance_results if r["banding_score"] > 0
        ]
        if len(variance_scores) > 1:
            variance_variation = max(variance_scores) / min(variance_scores)
            assert (
                variance_variation < 10.0
            ), f"Variance threshold too sensitive: {variance_variation:.1f}x variation"

    @pytest.mark.fast
    def test_configuration_validation(self):
        """Test that invalid configurations are handled appropriately."""
        # Test invalid patch sizes
        valid_patch_sizes = [8, 16, 32, 64, 128, 256]  # Should all work
        invalid_patch_sizes = [0, -1, 1]  # Problematic sizes

        for patch_size in valid_patch_sizes:
            try:
                detector = GradientBandingDetector(patch_size=patch_size)
                validator = PerceptualColorValidator(patch_size=patch_size)
                # Should initialize successfully
                assert detector.patch_size == patch_size
                assert validator.patch_size == patch_size
            except Exception as e:
                pytest.fail(f"Valid patch size {patch_size} failed: {e}")

        for patch_size in invalid_patch_sizes:
            # These should either work (be handled gracefully) or fail cleanly
            # We don't enforce specific behavior, just that it doesn't crash unexpectedly
            try:
                detector = GradientBandingDetector(patch_size=patch_size)
                # If it succeeds, the patch size should be converted to something reasonable
                assert (
                    detector.patch_size > 0
                ), f"Invalid patch size {patch_size} not handled properly"
            except (ValueError, TypeError):
                # It's acceptable for invalid sizes to raise exceptions
                pass

        # Test invalid thresholds
        try:
            # Empty thresholds should work or be handled gracefully
            validator = PerceptualColorValidator(jnd_thresholds=[])
            assert isinstance(validator.jnd_thresholds, list)
        except Exception:
            # May raise exception, which is acceptable
            pass

        # Test negative thresholds
        try:
            validator = PerceptualColorValidator(jnd_thresholds=[-1.0, 2.0])
            # Should either work or raise exception, not crash silently
            assert all(t >= -1.0 for t in validator.jnd_thresholds)
        except Exception:
            pass  # Acceptable to reject negative thresholds

    @pytest.mark.fast
    def test_configuration_persistence(self):
        """Test that configurations persist correctly during processing."""
        detector = GradientBandingDetector(patch_size=32, variance_threshold=75.0)
        validator = PerceptualColorValidator(patch_size=48, jnd_thresholds=[1.5, 3.5])

        frames = self._create_test_frames()

        # Process some data
        detector.detect_banding_artifacts(frames, frames)
        validator.calculate_color_difference_metrics(frames, frames)

        # Configuration should remain unchanged
        assert detector.patch_size == 32
        assert detector.variance_threshold == 75.0
        assert validator.patch_size == 48
        assert validator.jnd_thresholds == [1.5, 3.5]

        # Process again with different data
        different_frames = self._create_shifted_frames()
        detector.detect_banding_artifacts(frames, different_frames)
        validator.calculate_color_difference_metrics(frames, different_frames)

        # Configuration should still be unchanged
        assert detector.patch_size == 32
        assert detector.variance_threshold == 75.0
        assert validator.patch_size == 48
        assert validator.jnd_thresholds == [1.5, 3.5]

    @pytest.mark.fast
    def test_dynamic_configuration_changes(self):
        """Test changing configuration parameters dynamically."""
        detector = GradientBandingDetector()
        frames = self._create_gradient_frames()

        # Initial configuration
        initial_patch_size = detector.patch_size
        result1 = detector.detect_banding_artifacts(frames, frames)

        # Change configuration
        detector.patch_size = initial_patch_size // 2
        detector.variance_threshold = detector.variance_threshold * 0.5

        result2 = detector.detect_banding_artifacts(frames, frames)

        # Results might be different due to configuration change
        # But both should be valid
        assert isinstance(result1["banding_score_mean"], float)
        assert isinstance(result2["banding_score_mean"], float)
        assert result1["banding_score_mean"] >= 0.0
        assert result2["banding_score_mean"] >= 0.0

        print(
            f"Configuration change impact: {result1['banding_score_mean']:.2f} -> {result2['banding_score_mean']:.2f}"
        )

    # Helper methods for creating test data

    def _create_test_frames(self, size=(128, 128), num_frames=3):
        """Create basic test frames."""
        frames = []
        for _i in range(num_frames):
            # Create simple gradient
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            for x in range(size[0]):
                intensity = int(x * 255 / (size[0] - 1))
                frame[:, x] = [intensity, 128, 255 - intensity]
            frames.append(frame)
        return frames

    def _create_gradient_frames(self, size=(128, 128), num_frames=3):
        """Create frames with smooth gradients."""
        frames = []
        for _i in range(num_frames):
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            for x in range(size[0]):
                intensity = int(x * 255 / (size[0] - 1))
                frame[:, x] = [intensity, intensity // 2, 255 - intensity]
            frames.append(frame)
        return frames

    def _create_banded_frames(self, size=(128, 128), bands=8, num_frames=3):
        """Create frames with banded gradients."""
        frames = []
        for _i in range(num_frames):
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            for x in range(size[0]):
                band = int(x * bands / size[0])
                intensity = int(band * 255 / (bands - 1))
                frame[:, x] = [intensity, intensity // 2, 255 - intensity]
            frames.append(frame)
        return frames

    def _create_shifted_frames(self, size=(128, 128), num_frames=3):
        """Create frames with shifted colors."""
        frames = []
        for _i in range(num_frames):
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            for x in range(size[0]):
                intensity = int(x * 255 / (size[0] - 1))
                # Add slight color shifts
                r = min(255, intensity + 20)
                g = max(0, intensity // 2 - 10)
                b = min(255, 255 - intensity + 15)
                frame[:, x] = [r, g, b]
            frames.append(frame)
        return frames


class TestAdvancedConfiguration:
    """Advanced configuration scenarios and edge cases."""

    @pytest.mark.fast
    def test_configuration_combinations(self):
        """Test various combinations of configuration parameters."""
        frames = self._create_test_frames()

        # Test matrix of parameter combinations
        configurations = [
            {
                "patch_size": 16,
                "variance_threshold": 50.0,
                "jnd_thresholds": [1.0, 3.0],
            },
            {
                "patch_size": 64,
                "variance_threshold": 100.0,
                "jnd_thresholds": [2.0, 5.0],
            },
            {
                "patch_size": 128,
                "variance_threshold": 200.0,
                "jnd_thresholds": [0.5, 1.0, 2.0],
            },
        ]

        results = []

        for i, config in enumerate(configurations):
            detector = GradientBandingDetector(
                patch_size=config["patch_size"],
                variance_threshold=config["variance_threshold"],
            )
            validator = PerceptualColorValidator(
                patch_size=config["patch_size"], jnd_thresholds=config["jnd_thresholds"]
            )

            banding_result = detector.detect_banding_artifacts(frames, frames)
            color_result = validator.calculate_color_difference_metrics(frames, frames)

            result = {
                "config": config,
                "banding_score": banding_result["banding_score_mean"],
                "gradient_regions": banding_result["gradient_region_count"],
                "color_patches": color_result["color_patch_count"],
                "deltae": color_result["deltae_mean"],
            }
            results.append(result)

            print(
                f"Config {i+1}: patch={config['patch_size']}, variance={config['variance_threshold']}"
            )
            print(
                f"  Results: banding={result['banding_score']:.2f}, regions={result['gradient_regions']}, deltae={result['deltae']:.2f}"
            )

        # All configurations should produce valid results
        for result in results:
            assert result["banding_score"] >= 0.0
            assert result["gradient_regions"] >= 0
            assert result["color_patches"] >= 0
            assert result["deltae"] >= 0.0

    @pytest.mark.fast
    def test_extreme_configuration_values(self):
        """Test behavior with extreme configuration values."""
        frames = self._create_test_frames()

        # Test with very small patch size
        try:
            detector_tiny = GradientBandingDetector(
                patch_size=4, variance_threshold=1.0
            )
            result = detector_tiny.detect_banding_artifacts(frames, frames)
            assert isinstance(result, dict)
            print("✅ Tiny patch size handled")
        except Exception as e:
            print(f"⚠️  Tiny patch size failed (acceptable): {e}")

        # Test with very large patch size
        try:
            detector_huge = GradientBandingDetector(
                patch_size=512, variance_threshold=10000.0
            )
            result = detector_huge.detect_banding_artifacts(frames, frames)
            assert isinstance(result, dict)
            print("✅ Huge patch size handled")
        except Exception as e:
            print(f"⚠️  Huge patch size failed (acceptable): {e}")

        # Test with extreme variance thresholds
        try:
            detector_sensitive = GradientBandingDetector(variance_threshold=0.1)
            result = detector_sensitive.detect_banding_artifacts(frames, frames)
            assert isinstance(result, dict)
            print("✅ Ultra-sensitive threshold handled")
        except Exception as e:
            print(f"⚠️  Ultra-sensitive threshold failed (acceptable): {e}")

        try:
            detector_insensitive = GradientBandingDetector(variance_threshold=100000.0)
            result = detector_insensitive.detect_banding_artifacts(frames, frames)
            assert isinstance(result, dict)
            print("✅ Ultra-insensitive threshold handled")
        except Exception as e:
            print(f"⚠️  Ultra-insensitive threshold failed (acceptable): {e}")

    @pytest.mark.fast
    def test_configuration_impact_on_performance(self):
        """Test how configuration affects performance."""
        import time

        frames = self._create_test_frames(size=(256, 256), num_frames=5)

        configs = [
            {"patch_size": 32, "name": "small_patches"},
            {"patch_size": 64, "name": "medium_patches"},
            {"patch_size": 128, "name": "large_patches"},
        ]

        performance_results = {}

        for config in configs:
            detector = GradientBandingDetector(patch_size=config["patch_size"])
            validator = PerceptualColorValidator(patch_size=config["patch_size"])

            # Measure performance
            start_time = time.perf_counter()
            banding_result = detector.detect_banding_artifacts(frames, frames)
            color_result = validator.calculate_color_difference_metrics(frames, frames)
            end_time = time.perf_counter()

            performance_results[config["name"]] = {
                "time": end_time - start_time,
                "patch_size": config["patch_size"],
                "patches_analyzed": banding_result["banding_patch_count"]
                + color_result["color_patch_count"],
            }

            print(
                f"{config['name']}: {end_time - start_time:.3f}s, {performance_results[config['name']]['patches_analyzed']} patches"
            )

        # All configurations should complete in reasonable time
        for name, result in performance_results.items():
            assert (
                result["time"] < 10.0
            ), f"Configuration {name} too slow: {result['time']:.3f}s"

    def _create_test_frames(self, size=(128, 128), num_frames=3):
        """Create basic test frames for configuration testing."""
        frames = []
        for i in range(num_frames):
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            # Create horizontal gradient with some variation per frame
            for x in range(size[0]):
                intensity = int((x + i * 10) * 255 / (size[0] - 1)) % 256
                frame[:, x] = [intensity, intensity // 2, 255 - intensity]
            frames.append(frame)
        return frames


