"""Integration tests for efficiency calculation with real experiment workflows."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from giflab.enhanced_metrics import (
    calculate_efficiency_metric,
    process_metrics_with_enhanced_quality,
)
from giflab.metrics import calculate_comprehensive_metrics
from PIL import Image


class TestEfficiencyIntegrationWorkflows:
    """Integration tests for efficiency calculation in real workflows."""

    @pytest.fixture
    def temp_gif_pair(self):
        """Create a pair of test GIFs for comparison."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create original GIF
            original_frames = []
            for i in range(10):
                # High quality gradient
                img = Image.new("RGB", (100, 100))
                pixels = []
                for y in range(100):
                    for x in range(100):
                        r = int(255 * x / 99)
                        g = int(255 * y / 99)
                        b = int(255 * (x + y + i) / (198 + 9))
                        pixels.append((r, g, b))
                img.putdata(pixels)
                original_frames.append(img)

            original_path = tmp_path / "original.gif"
            original_frames[0].save(
                original_path,
                save_all=True,
                append_images=original_frames[1:],
                duration=100,
                optimize=False,  # High quality
            )

            # Create compressed version (lower quality, smaller size)
            compressed_frames = []
            for i in range(5):  # Fewer frames
                # Lower quality, reduced colors
                img = original_frames[i * 2].quantize(colors=64)  # Reduce colors
                compressed_frames.append(img.convert("RGB"))

            compressed_path = tmp_path / "compressed.gif"
            compressed_frames[0].save(
                compressed_path,
                save_all=True,
                append_images=compressed_frames[1:],
                duration=200,  # Longer duration per frame
                optimize=True,  # Enable compression
            )

            yield original_path, compressed_path

    def test_end_to_end_efficiency_calculation(self, temp_gif_pair):
        """Test complete efficiency calculation from GIF files to final score."""
        original_path, compressed_path = temp_gif_pair

        # Calculate comprehensive metrics (includes enhanced composite quality)
        metrics = calculate_comprehensive_metrics(original_path, compressed_path)

        # Verify required metrics are present
        assert "composite_quality" in metrics
        assert "compression_ratio" in metrics
        assert "efficiency" in metrics

        # Verify efficiency is in valid range
        efficiency = metrics["efficiency"]
        assert 0.0 <= efficiency <= 1.0

        # Verify efficiency calculation is consistent
        expected_efficiency = calculate_efficiency_metric(
            metrics["compression_ratio"], metrics["composite_quality"]
        )
        assert abs(efficiency - expected_efficiency) < 0.001

    def test_efficiency_with_different_quality_levels(self):
        """Test efficiency calculation with various quality/compression combinations."""
        test_cases = [
            # (compression_ratio, quality, expected_efficiency_range)
            (20.0, 1.0, (0.95, 1.0)),  # Excellent: perfect quality + max compression
            (10.0, 0.9, (0.80, 0.95)),  # Very good: high quality + good compression
            (5.0, 0.8, (0.65, 0.80)),  # Good: moderate quality + moderate compression
            (2.0, 0.7, (0.45, 0.65)),  # Fair: lower quality + poor compression
            (1.1, 0.5, (0.20, 0.45)),  # Poor: poor quality + minimal compression
        ]

        for compression_ratio, quality, (min_eff, max_eff) in test_cases:
            efficiency = calculate_efficiency_metric(compression_ratio, quality)

            assert min_eff <= efficiency <= max_eff, (
                f"Efficiency {efficiency} not in expected range [{min_eff}, {max_eff}] "
                f"for compression={compression_ratio}, quality={quality}"
            )

    def test_efficiency_ranking_consistency(self):
        """Test that efficiency rankings are consistent and meaningful."""
        # Define algorithm performance profiles
        algorithm_profiles = {
            "excellent_balanced": (12.0, 0.95),  # High compression + high quality
            "quality_focused": (3.0, 0.98),  # Low compression + very high quality
            "compression_focused": (
                18.0,
                0.75,
            ),  # Very high compression + moderate quality
            "poor_balanced": (2.0, 0.6),  # Low compression + low quality
            "extreme_compression": (50.0, 0.4),  # Extreme compression + poor quality
        }

        efficiencies = {}
        for name, (compression, quality) in algorithm_profiles.items():
            efficiencies[name] = calculate_efficiency_metric(compression, quality)

        # Sort by efficiency
        ranked = sorted(efficiencies.items(), key=lambda x: x[1], reverse=True)

        # Verify reasonable ranking
        assert (
            ranked[0][0] == "excellent_balanced"
        ), "Balanced algorithm should rank highest"
        assert ranked[-1][0] in [
            "poor_balanced",
            "extreme_compression",
        ], "Poor algorithms should rank lowest"

        # Verify efficiency values are distinct
        efficiency_values = [eff for _, eff in ranked]
        for i in range(len(efficiency_values) - 1):
            assert (
                efficiency_values[i] > efficiency_values[i + 1]
            ), "Rankings should be distinct"

    def test_50_50_weighting_impact(self):
        """Test the impact of 50/50 weighting vs other weightings."""
        compression_ratio = 8.0
        composite_quality = 0.75

        # Calculate with current 50/50 weighting
        current_efficiency = calculate_efficiency_metric(
            compression_ratio, composite_quality
        )

        # Manually calculate what 60/40 and 40/60 would give
        normalized_compression = min(
            np.log(1 + compression_ratio) / np.log(1 + 20.0), 1.0
        )

        efficiency_60_40 = (composite_quality**0.6) * (normalized_compression**0.4)
        efficiency_40_60 = (composite_quality**0.4) * (normalized_compression**0.6)

        # Current 50/50 should be between the other two weightings
        if composite_quality > normalized_compression:
            # Quality is better, so 60/40 (quality-favored) should be higher
            assert efficiency_60_40 > current_efficiency > efficiency_40_60
        else:
            # Compression is better, so 40/60 (compression-favored) should be higher
            assert efficiency_40_60 > current_efficiency > efficiency_60_40

    def test_batch_efficiency_calculation(self):
        """Test efficiency calculation on batches of results."""
        # Simulate a batch of experiment results with raw metrics
        # Note: Must provide raw metrics (ssim_mean, ms_ssim_mean, etc.) not pre-calculated
        # enhanced_composite_quality, as process_metrics_with_enhanced_quality() calculates
        # enhanced composite quality from raw metrics
        batch_data = [
            {
                "compression_ratio": 5.0,
                "ssim_mean": 0.92,
                "ms_ssim_mean": 0.90,
                "algorithm": "A",
            },
            {
                "compression_ratio": 8.0,
                "ssim_mean": 0.85,
                "ms_ssim_mean": 0.82,
                "algorithm": "B",
            },
            {
                "compression_ratio": 12.0,
                "ssim_mean": 0.88,
                "ms_ssim_mean": 0.85,
                "algorithm": "C",
            },
            {
                "compression_ratio": 3.0,
                "ssim_mean": 0.96,
                "ms_ssim_mean": 0.94,
                "algorithm": "D",
            },
            {
                "compression_ratio": 15.0,
                "ssim_mean": 0.75,
                "ms_ssim_mean": 0.72,
                "algorithm": "E",
            },
        ]

        # Process each result
        processed_results = []
        for data in batch_data:
            processed = process_metrics_with_enhanced_quality(data)
            processed_results.append(processed)

        # Verify all have efficiency calculated
        for result in processed_results:
            assert "efficiency" in result
            assert 0.0 <= result["efficiency"] <= 1.0

        # Convert to DataFrame for analysis
        df = pd.DataFrame(processed_results)

        # Verify efficiency ranking makes sense
        df_sorted = df.sort_values("efficiency", ascending=False)

        # Top algorithms should have good balance of quality and compression
        top_algorithm = df_sorted.iloc[0]
        assert top_algorithm["composite_quality"] > 0.7
        assert top_algorithm["compression_ratio"] > 3.0

    def test_efficiency_with_edge_case_inputs(self):
        """Test efficiency calculation with edge case inputs."""
        edge_cases = [
            # (compression_ratio, quality, description)
            (0.0, 0.8, "zero compression"),
            (5.0, 0.0, "zero quality"),
            (0.0, 0.0, "both zero"),
            (1.0, 1.0, "minimal compression, perfect quality"),
            (100.0, 1.0, "extreme compression, perfect quality"),
            (1.0, 0.1, "minimal compression, poor quality"),
            (20.0, 0.5, "max practical compression, moderate quality"),
        ]

        for compression_ratio, quality, description in edge_cases:
            efficiency = calculate_efficiency_metric(compression_ratio, quality)

            # All should return valid efficiency values
            assert (
                0.0 <= efficiency <= 1.0
            ), f"Invalid efficiency for case: {description}"

            # Specific expectations for certain cases
            if compression_ratio <= 0.0 or quality <= 0.0:
                assert (
                    efficiency == 0.0
                ), f"Should be zero efficiency for case: {description}"

    def test_metrics_processing_integration(self):
        """Test integration of enhanced metrics processing."""
        # Simulate raw metrics from a real experiment
        raw_metrics = {
            "compression_ratio": 7.5,
            "ssim_mean": 0.85,
            "ms_ssim_mean": 0.82,
            "psnr_mean": 0.78,
            "temporal_consistency_compressed": 0.88,
            "mse_mean": 45.2,
            "fsim_mean": 0.81,
            "edge_similarity_mean": 0.79,
            "gmsd_mean": 0.12,
            "chist_mean": 0.76,
            "sharpness_similarity_mean": 0.74,
            "texture_similarity_mean": 0.83,
        }

        # Process with enhanced metrics
        processed = process_metrics_with_enhanced_quality(raw_metrics)

        # Verify all expected fields are added
        expected_additions = [
            "composite_quality",
            "efficiency",
        ]

        for field in expected_additions:
            assert field in processed, f"Missing expected field: {field}"

        # Verify composite quality is reasonable
        composite_quality = processed["composite_quality"]
        assert 0.0 <= composite_quality <= 1.0
        assert composite_quality > 0.5  # Should be decent with these metrics

        # Verify efficiency is reasonable
        efficiency = processed["efficiency"]
        assert 0.0 <= efficiency <= 1.0
        assert efficiency > 0.6  # Should be good with decent quality and compression

    def test_performance_overhead_efficiency(self):
        """Test that efficiency calculation doesn't add significant overhead."""
        import time

        # Test data
        test_cases = [
            (np.random.uniform(1, 20), np.random.uniform(0.3, 1.0)) for _ in range(1000)
        ]

        # Time the efficiency calculations
        start_time = time.time()
        for compression_ratio, quality in test_cases:
            calculate_efficiency_metric(compression_ratio, quality)
        end_time = time.time()

        total_time = end_time - start_time
        time_per_calculation = total_time / len(test_cases)

        # Should be very fast (less than 1ms per calculation)
        assert (
            time_per_calculation < 0.001
        ), f"Efficiency calculation too slow: {time_per_calculation:.6f}s per call"

    @pytest.mark.parametrize(
        "compression_ratio,quality",
        [
            (2.0, 0.5),
            (5.0, 0.8),
            (10.0, 0.9),
            (15.0, 0.7),
            (20.0, 1.0),
        ],
    )
    def test_efficiency_parametrized(self, compression_ratio, quality):
        """Parametrized test for efficiency calculation with various inputs."""
        efficiency = calculate_efficiency_metric(compression_ratio, quality)

        # Basic validation
        assert 0.0 <= efficiency <= 1.0

        # Monotonicity check: better quality should give better efficiency
        better_quality = min(quality + 0.1, 1.0)
        better_efficiency = calculate_efficiency_metric(
            compression_ratio, better_quality
        )
        assert better_efficiency >= efficiency

        # Monotonicity check: better compression should give better efficiency
        better_compression = compression_ratio * 1.2
        better_compression_efficiency = calculate_efficiency_metric(
            better_compression, quality
        )
        assert better_compression_efficiency >= efficiency

    def test_csv_output_integration(self):
        """Test that efficiency appears correctly in CSV output format."""
        # Simulate experiment results that would go into CSV
        experiment_results = [
            {
                "pipeline_id": "imagemagick-frame",
                "compression_ratio": 9.8,
                "enhanced_composite_quality": 1.0,
                "ssim_mean": 1.0,
            },
            {
                "pipeline_id": "gifsicle-frame",
                "compression_ratio": 9.5,
                "enhanced_composite_quality": 0.932,
                "ssim_mean": 0.95,
            },
            {
                "pipeline_id": "ffmpeg-frame",
                "compression_ratio": 1.9,
                "enhanced_composite_quality": 0.947,
                "ssim_mean": 0.98,
            },
        ]

        # Process each result to add efficiency
        for result in experiment_results:
            processed = process_metrics_with_enhanced_quality(result)
            result.update(processed)

        # Convert to DataFrame as would happen in real workflow
        df = pd.DataFrame(experiment_results)

        # Verify efficiency column exists and is reasonable
        assert "efficiency" in df.columns
        assert all(0.0 <= eff <= 1.0 for eff in df["efficiency"])

        # Verify ranking matches expected (imagemagick > gifsicle > ffmpeg)
        df_sorted = df.sort_values("efficiency", ascending=False)
        assert df_sorted.iloc[0]["pipeline_id"] == "imagemagick-frame"
        assert df_sorted.iloc[-1]["pipeline_id"] == "ffmpeg-frame"
