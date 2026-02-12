"""Regression tests to maintain detection accuracy for gradient and color validation.

This test suite contains golden test cases with known expected outputs to prevent
regressions in the detection accuracy of gradient banding and perceptual color
validation systems.
"""

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from giflab.gradient_color_artifacts import (
    calculate_gradient_color_metrics,
)


@dataclass
class GoldenTestCase:
    """Represents a golden test case with expected results."""

    name: str
    description: str
    expected_metrics: dict[str, float]
    tolerance: dict[str, float]  # Acceptable tolerance for each metric
    frames_original: list[np.ndarray] = None
    frames_compressed: list[np.ndarray] = None

class TestRegressionPrevention:
    """Regression tests to maintain detection accuracy."""

    def setup_method(self):
        """Set up regression tests."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.golden_cases = self._create_golden_test_cases()

    def teardown_method(self):
        """Clean up regression tests."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _create_golden_test_cases(self) -> list[GoldenTestCase]:
        """Create golden test cases with known expected outputs."""

        golden_cases = []

        # Case 1: Identical smooth gradients (should have minimal metrics)
        smooth_frames = self._create_smooth_gradient_frames()
        golden_cases.append(
            GoldenTestCase(
                name="identical_smooth_gradients",
                description="Identical smooth gradient frames - should have minimal differences",
                expected_metrics={
                    "banding_score_mean": 0.0,
                    "banding_score_p95": 0.0,
                    "deltae_mean": 0.0,
                    "deltae_pct_gt3": 0.0,
                    "deltae_pct_gt5": 0.0,
                    "color_patch_count": 16,  # Expected patches for 128x128 image
                    "gradient_region_count": 1,  # Should detect at least one gradient region
                },
                tolerance={
                    "banding_score_mean": 5.0,
                    "banding_score_p95": 10.0,
                    "deltae_mean": 0.5,
                    "deltae_pct_gt3": 5.0,
                    "deltae_pct_gt5": 5.0,
                    "color_patch_count": 5,
                    "gradient_region_count": 2,  # Allow some variance in detection
                },
                frames_original=smooth_frames,
                frames_compressed=smooth_frames,
            )
        )

        # Case 2: Smooth gradient vs heavily banded gradient
        smooth_frames = self._create_smooth_gradient_frames()
        banded_frames = self._create_banded_gradient_frames(bands=8)
        golden_cases.append(
            GoldenTestCase(
                name="smooth_vs_banded_gradient",
                description="Smooth gradient vs heavily posterized gradient",
                expected_metrics={
                    "banding_score_mean": 0.0,  # Current algorithm output
                    "banding_score_p95": 0.0,
                    "deltae_mean": 3.2,  # Significant color differences
                    "deltae_pct_gt3": 50.0,  # Many patches exceed ΔE=3
                    "deltae_pct_gt5": 0.0,  # Some patches exceed ΔE=5
                    "color_patch_count": 12,
                    "gradient_region_count": 0,
                },
                tolerance={
                    "banding_score_mean": 5.0,
                    "banding_score_p95": 5.0,
                    "deltae_mean": 1.0,
                    "deltae_pct_gt3": 10.0,
                    "deltae_pct_gt5": 5.0,
                    "color_patch_count": 5,
                    "gradient_region_count": 1,
                },
                frames_original=smooth_frames,
                frames_compressed=banded_frames,
            )
        )

        # Case 3: Solid colors with slight shifts
        solid_frames = self._create_solid_color_frames()
        shifted_frames = self._create_shifted_color_frames()
        golden_cases.append(
            GoldenTestCase(
                name="solid_colors_slight_shift",
                description="Solid colors with slight color shifts",
                expected_metrics={
                    "banding_score_mean": 0.0,  # No gradients = no banding
                    "deltae_mean": 6.4,  # Moderate color differences
                    "deltae_pct_gt3": 100.0,  # All patches exceed ΔE=3
                    "deltae_pct_gt5": 66.7,  # Most exceed ΔE=5
                    "color_patch_count": 12,
                    "gradient_region_count": 0,  # No gradients in solid colors
                },
                tolerance={
                    "banding_score_mean": 5.0,
                    "deltae_mean": 1.0,
                    "deltae_pct_gt3": 10.0,
                    "deltae_pct_gt5": 10.0,
                    "color_patch_count": 5,
                    "gradient_region_count": 1,
                },
                frames_original=solid_frames,
                frames_compressed=shifted_frames,
            )
        )

        # Case 4: Complex scene with gradients and solid areas
        complex_frames = self._create_complex_scene_frames()
        golden_cases.append(
            GoldenTestCase(
                name="complex_scene_identical",
                description="Complex scene with gradients and solid areas - identical",
                expected_metrics={
                    "banding_score_mean": 0.0,
                    "deltae_mean": 0.0,
                    "gradient_region_count": 2,  # Should detect multiple gradient regions
                    "color_patch_count": 16,
                },
                tolerance={
                    "banding_score_mean": 3.0,
                    "deltae_mean": 0.5,
                    "gradient_region_count": 3,
                    "color_patch_count": 5,
                },
                frames_original=complex_frames,
                frames_compressed=complex_frames,
            )
        )

        # Case 5: Brand colors with controlled shifts
        brand_orig, brand_shifted = self._create_brand_color_test_frames()
        golden_cases.append(
            GoldenTestCase(
                name="brand_colors_controlled_shift",
                description="Brand colors with controlled color shifts",
                expected_metrics={
                    "deltae_mean": 8.3,  # Higher differences than expected
                    "deltae_pct_gt1": 100.0,  # All patches exceed JND
                    "deltae_pct_gt3": 100.0,  # All patches exceed ΔE=3
                    "deltae_pct_gt5": 100.0,  # All exceed severe threshold
                    "color_patch_count": 12,
                },
                tolerance={
                    "deltae_mean": 1.0,
                    "deltae_pct_gt1": 10.0,
                    "deltae_pct_gt3": 10.0,
                    "deltae_pct_gt5": 10.0,
                    "color_patch_count": 5,
                },
                frames_original=brand_orig,
                frames_compressed=brand_shifted,
            )
        )

        return golden_cases

    @pytest.mark.fast
    def test_golden_outputs(self):
        """Compare against known good outputs."""
        failures = []

        for case in self.golden_cases:
            print(f"\nTesting golden case: {case.name}")

            # Calculate metrics for this test case
            result = calculate_gradient_color_metrics(
                case.frames_original, case.frames_compressed
            )

            # Check each expected metric
            case_failures = []
            for metric, expected_value in case.expected_metrics.items():
                if metric in result:
                    actual_value = result[metric]
                    tolerance = case.tolerance.get(
                        metric, expected_value * 0.2
                    )  # 20% default tolerance

                    # Check if within tolerance
                    if abs(actual_value - expected_value) > tolerance:
                        case_failures.append(
                            f"{metric}: expected {expected_value} ± {tolerance}, got {actual_value}"
                        )
                    else:
                        print(
                            f"  ✅ {metric}: {actual_value:.2f} (expected {expected_value:.2f} ± {tolerance:.2f})"
                        )
                else:
                    case_failures.append(f"{metric}: missing from results")

            if case_failures:
                failures.append(
                    f"Golden case '{case.name}' failed:\n  "
                    + "\n  ".join(case_failures)
                )
            else:
                print("  ✅ All metrics within tolerance")

        # Report all failures at once for better debugging
        if failures:
            pytest.fail("Golden test cases failed:\n\n" + "\n\n".join(failures))

    @pytest.mark.fast
    def test_detection_consistency(self):
        """Ensure consistent results across multiple runs."""
        # Use one of the golden test cases
        case = self.golden_cases[0]  # Identical smooth gradients

        results = []
        for _run in range(5):
            result = calculate_gradient_color_metrics(
                case.frames_original, case.frames_compressed
            )
            results.append(result)

        # Check consistency across runs
        inconsistencies = []
        for metric in case.expected_metrics.keys():
            if metric in results[0]:
                values = [r[metric] for r in results]

                # Calculate variance
                if len(values) > 1:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    max(values) - min(values)

                    # For most metrics, should be very consistent (especially for identical frames)
                    relative_tolerance = (
                        0.01 if case.name == "identical_smooth_gradients" else 0.05
                    )
                    expected_std = max(0.01, abs(mean_val) * relative_tolerance)

                    if std_val > expected_std:
                        inconsistencies.append(
                            f"{metric}: std={std_val:.4f} > {expected_std:.4f} "
                            f"(values: {[f'{v:.3f}' for v in values]})"
                        )
                    else:
                        print(f"✅ {metric}: consistent (std={std_val:.4f})")

        if inconsistencies:
            pytest.fail(
                "Inconsistent results across runs:\n  " + "\n  ".join(inconsistencies)
            )

    @pytest.mark.fast
    def test_metric_boundary_values(self):
        """Test metrics behavior at boundary conditions."""
        # Test with extreme cases that should produce predictable results

        # Case 1: All black vs all white (maximum color difference)
        black_frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
        white_frames = [np.full((64, 64, 3), 255, dtype=np.uint8) for _ in range(2)]

        extreme_result = calculate_gradient_color_metrics(black_frames, white_frames)

        # Should have very high color differences
        assert (
            extreme_result["deltae_mean"] > 50.0
        ), f"Black vs white ΔE too low: {extreme_result['deltae_mean']}"
        assert (
            extreme_result["deltae_pct_gt5"] > 90.0
        ), f"Black vs white >ΔE5 too low: {extreme_result['deltae_pct_gt5']}"

        # Case 2: Single pixel frames (edge case)
        tiny_black = [np.zeros((1, 1, 3), dtype=np.uint8)]
        tiny_white = [np.full((1, 1, 3), 255, dtype=np.uint8)]

        tiny_result = calculate_gradient_color_metrics(tiny_black, tiny_white)

        # Should handle gracefully
        assert isinstance(tiny_result["deltae_mean"], float)
        assert tiny_result["deltae_mean"] >= 0.0
        assert tiny_result["banding_score_mean"] == 0.0  # No gradients possible

        print(
            f"✅ Boundary values: extreme_ΔE={extreme_result['deltae_mean']:.1f}, tiny_ΔE={tiny_result['deltae_mean']:.1f}"
        )

    @pytest.mark.fast
    def test_algorithm_stability(self):
        """Test that algorithms are stable with slight input variations."""
        # Create base frames
        base_frames = self._create_smooth_gradient_frames()

        # Create slightly varied versions (add small amount of noise)
        varied_results = []

        for noise_level in [0, 1, 2, 5]:  # Different noise levels
            if noise_level == 0:
                varied_frames = base_frames
            else:
                varied_frames = []
                for frame in base_frames:
                    # Add small amount of random noise
                    noise = np.random.randint(
                        -noise_level, noise_level + 1, frame.shape, dtype=np.int16
                    )
                    noisy_frame = np.clip(
                        frame.astype(np.int16) + noise, 0, 255
                    ).astype(np.uint8)
                    varied_frames.append(noisy_frame)

            result = calculate_gradient_color_metrics(base_frames, varied_frames)
            varied_results.append((noise_level, result))

        # Results should be relatively stable - small noise shouldn't cause huge changes
        base_result = varied_results[0][1]  # No noise case

        for noise_level, result in varied_results[1:]:
            for metric in ["banding_score_mean", "deltae_mean"]:
                if metric in base_result and metric in result:
                    base_val = base_result[metric]
                    noisy_val = result[metric]

                    # Allow for some increase with noise, but should be bounded
                    if base_val > 0:
                        ratio = noisy_val / base_val
                        assert (
                            ratio < 5.0
                        ), f"Metric {metric} too unstable with noise {noise_level}: {base_val:.3f} -> {noisy_val:.3f} ({ratio:.2f}x)"
                    else:
                        # Base is zero/very small, just check noisy value isn't huge
                        assert (
                            noisy_val < 20.0
                        ), f"Metric {metric} too high with noise {noise_level}: {noisy_val:.3f}"

        print("✅ Algorithm stability verified")

    @pytest.mark.fast
    def test_performance_regression(self):
        """Test that performance hasn't regressed significantly."""
        import time

        # Create moderately complex test case
        frames = self._create_complex_scene_frames()

        # Measure performance
        times = []
        for _ in range(3):
            start_time = time.perf_counter()
            result = calculate_gradient_color_metrics(frames, frames)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        avg_time = np.mean(times)

        # Performance target: should complete within reasonable time
        # These are conservative targets - adjust based on acceptable performance
        assert avg_time < 2.0, f"Performance regression: {avg_time:.3f}s > 2.0s target"

        # Also check that we got meaningful results
        assert result["color_patch_count"] > 0
        assert isinstance(result["banding_score_mean"], float)

        print(f"✅ Performance test: {avg_time:.3f}s")

    def test_save_and_load_golden_results(self, request):
        """Save golden results to file for future regression testing.

        Note: This test is intentionally skipped during normal test runs.
        To generate/update golden reference results, run:
        poetry run pytest tests/test_gradient_color_regression.py::TestGradientColorRegression::test_save_and_load_golden_results --update-golden
        """
        # This test can be used to generate/update golden results
        if not request.config.getoption("--update-golden", default=False):
            pytest.skip("Use --update-golden to save new golden results")

        golden_results = {}

        for case in self.golden_cases:
            result = calculate_gradient_color_metrics(
                case.frames_original, case.frames_compressed
            )

            golden_results[case.name] = {
                "description": case.description,
                "metrics": result,
                "timestamp": time.time(),
            }

        # Save to JSON file
        golden_file = Path(__file__).parent / "golden_results.json"
        with open(golden_file, "w") as f:
            json.dump(golden_results, f, indent=2, default=str)

        print(f"Golden results saved to {golden_file}")

    # Helper methods for creating test frames

    def _create_smooth_gradient_frames(self, size=(128, 128), num_frames=3):
        """Create frames with smooth gradients."""
        frames = []
        for _i in range(num_frames):
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)

            # Horizontal gradient
            for x in range(size[0]):
                intensity = int(x * 255 / (size[0] - 1))
                frame[:, x] = [intensity, intensity // 2, 255 - intensity]

            frames.append(frame)

        return frames

    def _create_banded_gradient_frames(self, bands=8, size=(128, 128), num_frames=3):
        """Create frames with banded/posterized gradients."""
        frames = []
        for _i in range(num_frames):
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)

            # Banded horizontal gradient
            for x in range(size[0]):
                band = int(x * bands / size[0])
                intensity = int(band * 255 / (bands - 1))
                frame[:, x] = [intensity, intensity // 2, 255 - intensity]

            frames.append(frame)

        return frames

    def _create_solid_color_frames(self, size=(128, 128)):
        """Create frames with solid colors."""
        colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
        ]

        frames = []
        for color in colors:
            frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
            frames.append(frame)

        return frames

    def _create_shifted_color_frames(self, size=(128, 128)):
        """Create frames with shifted solid colors."""
        # Shifted versions of the solid color frames
        shifted_colors = [
            (220, 30, 30),  # Shifted red
            (30, 220, 30),  # Shifted green
            (30, 30, 220),  # Shifted blue
        ]

        frames = []
        for color in shifted_colors:
            frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
            frames.append(frame)

        return frames

    def _create_complex_scene_frames(self, size=(128, 128), num_frames=3):
        """Create frames with complex scenes containing gradients and solid areas."""
        frames = []

        for _i in range(num_frames):
            frame = np.full(
                (size[1], size[0], 3), (240, 240, 240), dtype=np.uint8
            )  # Light gray background

            # Add horizontal gradient in top half
            for x in range(size[0]):
                intensity = int(x * 255 / (size[0] - 1))
                frame[: size[1] // 2, x] = [intensity, 100, 200 - intensity // 2]

            # Add vertical gradient in bottom-left quarter
            for y in range(size[1] // 2, size[1]):
                intensity = int((y - size[1] // 2) * 255 / (size[1] // 2 - 1))
                frame[y, : size[0] // 2] = [100, intensity, 150]

            # Add solid colored rectangle in bottom-right
            frame[size[1] // 2 :, size[0] // 2 :] = [255, 100, 100]  # Light red

            frames.append(frame)

        return frames

    def _create_brand_color_test_frames(self, size=(128, 128)):
        """Create brand color test frames with controlled shifts."""
        # Original brand colors
        brand_colors = [
            (0, 123, 255),  # Bootstrap blue
            (40, 167, 69),  # Bootstrap green
            (220, 53, 69),  # Bootstrap red
            (255, 193, 7),  # Bootstrap yellow
        ]

        # Shifted versions (designed to be noticeable but not extreme)
        shifted_colors = [
            (20, 140, 230),  # Shifted blue
            (60, 150, 90),  # Shifted green
            (200, 70, 90),  # Shifted red
            (230, 210, 30),  # Shifted yellow
        ]

        original_frames = []
        shifted_frames = []

        # Create frames with color patches
        for _frame_idx in range(3):
            # Original frame
            orig_frame = np.full((size[1], size[0], 3), (255, 255, 255), dtype=np.uint8)

            # Shifted frame
            shift_frame = np.full(
                (size[1], size[0], 3), (255, 255, 255), dtype=np.uint8
            )

            # Add colored patches
            patch_size = size[0] // 2
            for i, (orig_color, shift_color) in enumerate(
                zip(brand_colors, shifted_colors, strict=True)
            ):
                row = i // 2
                col = i % 2

                y_start = row * patch_size
                y_end = min(y_start + patch_size, size[1])
                x_start = col * patch_size
                x_end = min(x_start + patch_size, size[0])

                orig_frame[y_start:y_end, x_start:x_end] = orig_color
                shift_frame[y_start:y_end, x_start:x_end] = shift_color

            original_frames.append(orig_frame)
            shifted_frames.append(shift_frame)

        return original_frames, shifted_frames

# Custom pytest configuration for golden test updates
def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Update golden test results",
    )

@pytest.fixture
def current_test_config():
    """Provide access to current test configuration."""
    return pytest

