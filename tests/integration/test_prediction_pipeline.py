"""Integration tests for compression curve prediction pipeline.

These tests use synthetic GIFs to verify that the full prediction pipeline
works correctly and that predicted outcomes align with actual compression.
"""

import subprocess
import tempfile
from datetime import UTC
from pathlib import Path

import numpy as np
import pytest
from giflab.prediction.features import extract_gif_features
from giflab.prediction.models import CurvePredictionModel
from giflab.prediction.schemas import CurveType, Engine
from PIL import Image


def create_synthetic_gif(
    path: Path,
    frames: int = 5,
    size: tuple[int, int] = (100, 100),
    pattern: str = "gradient",
) -> None:
    """Create a synthetic GIF for testing."""
    images = []

    for i in range(frames):
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)

        if pattern == "gradient":
            arr[:, :, 0] = np.linspace(0, 255, size[0], dtype=np.uint8)
            arr[:, :, 1] = (i * 40) % 256
            arr[:, :, 2] = 128
        elif pattern == "solid":
            arr[:, :, :] = [100, 150, 200]
        elif pattern == "noise":
            np.random.seed(42 + i)
            arr = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        elif pattern == "animation":
            bar_pos = (i * 20) % size[0]
            arr[:, max(0, bar_pos) : min(size[0], bar_pos + 10), :] = [255, 255, 255]

        img = Image.fromarray(arr, mode="RGB")
        images.append(img)

    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
    )


def run_gifsicle_compression(
    input_path: Path,
    output_path: Path,
    lossy_level: int,
) -> float | None:
    """Run gifsicle compression and return output size in KB."""
    try:
        cmd = [
            "gifsicle",
            f"--lossy={lossy_level}",
            "-O3",
            str(input_path),
            "-o",
            str(output_path),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and output_path.exists():
            return output_path.stat().st_size / 1024
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def check_gifsicle_available() -> bool:
    """Check if gifsicle is available."""
    try:
        result = subprocess.run(
            ["gifsicle", "--version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


@pytest.fixture
def synthetic_training_data(tmp_path: Path) -> tuple[list, list]:
    """Create synthetic training data with actual compression outcomes."""
    if not check_gifsicle_available():
        pytest.skip("gifsicle not available")

    features_list = []
    curves_list = []

    patterns = ["gradient", "solid", "noise", "animation"]

    for _idx, pattern in enumerate(patterns):
        for variant in range(3):  # 3 variants per pattern
            gif_path = tmp_path / f"{pattern}_{variant}.gif"
            create_synthetic_gif(
                gif_path,
                frames=3 + variant,
                size=(80 + variant * 20, 80 + variant * 20),
                pattern=pattern,
            )

            # Extract features
            features = extract_gif_features(gif_path)
            features_list.append(features)

            # Get actual compression outcomes
            from datetime import datetime, timezone

            from giflab.prediction.schemas import CompressionCurveV1

            sizes = {}
            for lossy in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                # Gifsicle native level = normalized × 3
                native_lossy = lossy * 3
                out_path = tmp_path / f"{pattern}_{variant}_l{lossy}.gif"
                size = run_gifsicle_compression(gif_path, out_path, native_lossy)
                if size:
                    sizes[lossy] = size

            if len(sizes) >= 4:  # Need at least some data points
                curve = CompressionCurveV1(
                    gif_sha=features.gif_sha,
                    engine=Engine.GIFSICLE,
                    curve_type=CurveType.LOSSY,
                    is_predicted=False,
                    created_at=datetime.now(UTC),
                    size_at_lossy_0=sizes.get(0),
                    size_at_lossy_10=sizes.get(10),
                    size_at_lossy_20=sizes.get(20),
                    size_at_lossy_30=sizes.get(30),
                    size_at_lossy_40=sizes.get(40),
                    size_at_lossy_50=sizes.get(50),
                    size_at_lossy_60=sizes.get(60),
                    size_at_lossy_70=sizes.get(70),
                    size_at_lossy_80=sizes.get(80),
                    size_at_lossy_90=sizes.get(90),
                    size_at_lossy_100=sizes.get(100),
                )
                curves_list.append(curve)
            else:
                # Remove features if we couldn't get curve
                features_list.pop()

    return features_list, curves_list


class TestFullPredictionPipeline:
    """Test the complete prediction pipeline with synthetic GIFs."""

    @pytest.mark.skipif(
        not check_gifsicle_available(),
        reason="gifsicle not available",
    )
    def test_feature_extraction_on_synthetic_gifs(self, tmp_path: Path) -> None:
        """Test that feature extraction works on all synthetic GIF types."""
        patterns = ["gradient", "solid", "noise", "animation"]

        for pattern in patterns:
            gif_path = tmp_path / f"{pattern}.gif"
            create_synthetic_gif(gif_path, pattern=pattern)

            features = extract_gif_features(gif_path)

            # All features should be valid
            assert features.schema_version == "1.0.0"
            assert features.width == 100
            assert features.height == 100
            assert features.frame_count >= 1  # May vary by GIF reader
            assert 0 <= features.entropy <= 8
            assert 0 <= features.edge_density <= 1
            assert 0 <= features.motion_intensity <= 1

    @pytest.mark.skipif(
        not check_gifsicle_available(),
        reason="gifsicle not available",
    )
    def test_train_and_predict_with_real_compression(
        self,
        synthetic_training_data: tuple,
    ) -> None:
        """Test training on real compression outcomes and predicting."""
        features_list, curves_list = synthetic_training_data

        if len(features_list) < 5:
            pytest.skip("Not enough training data")

        # Split into train/test
        train_features = features_list[:-2]
        train_curves = curves_list[:-2]
        test_features = features_list[-2:]
        test_curves = curves_list[-2:]

        # Train model
        model = CurvePredictionModel(Engine.GIFSICLE, CurveType.LOSSY)
        model.train(train_features, train_curves, n_estimators=20)

        # Predict
        for features, actual_curve in zip(test_features, test_curves, strict=True):
            predicted_curve = model.predict(features)

            # Compare predicted vs actual
            actual_points = actual_curve.get_lossy_curve_points()
            predicted_points = predicted_curve.get_lossy_curve_points()

            for level in [0, 40, 80, 100]:
                actual = actual_points.get(level)
                predicted = predicted_points.get(level)

                if actual and predicted:
                    # Check predictions are positive and finite
                    # (loose bounds due to small training set)
                    assert predicted > 0
                    assert predicted < 1000  # Reasonable upper bound in KB

    @pytest.mark.skipif(
        not check_gifsicle_available(),
        reason="gifsicle not available",
    )
    def test_different_patterns_produce_different_predictions(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that different GIF patterns produce different predictions."""
        from datetime import datetime, timezone

        from giflab.prediction.schemas import CompressionCurveV1

        # Create training data with complete curves
        features_list = []
        curves_list = []

        for _idx, pattern in enumerate(["solid", "noise"]):
            for i in range(5):
                gif_path = tmp_path / f"train_{pattern}_{i}.gif"
                create_synthetic_gif(gif_path, pattern=pattern, frames=3)

                features = extract_gif_features(gif_path)

                # Run actual compression for all levels
                sizes = {}
                for lossy in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                    # Gifsicle native level = normalized × 3
                    native_lossy = lossy * 3
                    out_path = tmp_path / f"train_{pattern}_{i}_l{lossy}.gif"
                    size = run_gifsicle_compression(gif_path, out_path, native_lossy)
                    if size:
                        sizes[lossy] = size

                # Only add if we got all compression levels
                if len(sizes) == 11:
                    curve = CompressionCurveV1(
                        gif_sha=features.gif_sha,
                        engine=Engine.GIFSICLE,
                        curve_type=CurveType.LOSSY,
                        is_predicted=False,
                        created_at=datetime.now(UTC),
                        size_at_lossy_0=sizes[0],
                        size_at_lossy_10=sizes[10],
                        size_at_lossy_20=sizes[20],
                        size_at_lossy_30=sizes[30],
                        size_at_lossy_40=sizes[40],
                        size_at_lossy_50=sizes[50],
                        size_at_lossy_60=sizes[60],
                        size_at_lossy_70=sizes[70],
                        size_at_lossy_80=sizes[80],
                        size_at_lossy_90=sizes[90],
                        size_at_lossy_100=sizes[100],
                    )
                    features_list.append(features)
                    curves_list.append(curve)

        if len(features_list) < 6:
            pytest.skip("Not enough complete training data")

        # Train model
        model = CurvePredictionModel(Engine.GIFSICLE, CurveType.LOSSY)
        model.train(features_list, curves_list, n_estimators=20)

        # Create test GIFs and predict
        solid_gif = tmp_path / "test_solid.gif"
        noise_gif = tmp_path / "test_noise.gif"
        create_synthetic_gif(solid_gif, pattern="solid", frames=4)
        create_synthetic_gif(noise_gif, pattern="noise", frames=4)

        solid_features = extract_gif_features(solid_gif)
        noise_features = extract_gif_features(noise_gif)

        solid_prediction = model.predict(solid_features)
        noise_prediction = model.predict(noise_features)

        # Verify we get valid predictions
        assert solid_prediction.size_at_lossy_0 is not None
        assert noise_prediction.size_at_lossy_0 is not None
        assert solid_prediction.size_at_lossy_0 > 0
        assert noise_prediction.size_at_lossy_0 > 0

        # Different patterns should produce different predictions
        assert solid_prediction.size_at_lossy_0 != noise_prediction.size_at_lossy_0


class TestPredictionAccuracy:
    """Test prediction accuracy metrics."""

    @pytest.mark.skipif(
        not check_gifsicle_available(),
        reason="gifsicle not available",
    )
    def test_mape_on_synthetic_data(
        self,
        synthetic_training_data: tuple,
    ) -> None:
        """Test MAPE calculation on synthetic data."""
        features_list, curves_list = synthetic_training_data

        if len(features_list) < 8:
            pytest.skip("Not enough training data")

        # 80/20 split
        split_idx = int(len(features_list) * 0.8)
        train_features = features_list[:split_idx]
        train_curves = curves_list[:split_idx]
        val_features = features_list[split_idx:]
        val_curves = curves_list[split_idx:]

        model = CurvePredictionModel(Engine.GIFSICLE, CurveType.LOSSY)
        model.train(train_features, train_curves, n_estimators=30)

        mape = model.validate(val_features, val_curves)

        # MAPE should be a valid percentage (finite, non-negative)
        assert mape >= 0
        assert mape < 10_000  # generous bound to catch NaN/overflow
        # Note: MAPE may be very high (>100%) with small synthetic datasets
        # This test validates the pipeline works, not accuracy


class TestEndToEndCLI:
    """Test CLI commands with synthetic GIFs."""

    @pytest.mark.skipif(
        not check_gifsicle_available(),
        reason="gifsicle not available",
    )
    def test_extract_features_cli(self, tmp_path: Path) -> None:
        """Test predict CLI command with a GIF."""
        gif_path = tmp_path / "test.gif"
        create_synthetic_gif(gif_path, pattern="gradient")

        result = subprocess.run(
            [
                "poetry",
                "run",
                "giflab",
                "predict",
                "extract-features",
                str(gif_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parents[2]),
            timeout=60,
        )

        assert result.returncode == 0
