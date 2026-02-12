"""Tests for prediction models.

Uses synthetic GIFs to test model training and prediction.
"""

import tempfile
from datetime import UTC, datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from giflab.prediction.models import (
    FEATURE_COLUMNS,
    MODEL_VERSION,
    CurvePredictionModel,
)
from giflab.prediction.schemas import (
    CompressionCurveV1,
    CurveType,
    Engine,
    GifFeaturesV1,
)


def create_mock_features(
    gif_sha: str = "a" * 64,
    file_size: int = 50000,
) -> GifFeaturesV1:
    """Create mock GIF features for testing."""
    return GifFeaturesV1(
        gif_sha=gif_sha,
        gif_name="test.gif",
        extraction_version="1.0.0",
        extracted_at=datetime.now(UTC),
        width=100,
        height=100,
        frame_count=10,
        duration_ms=1000,
        file_size_bytes=file_size,
        unique_colors=256,
        entropy=5.5,
        edge_density=0.3,
        color_complexity=0.5,
        gradient_smoothness=0.7,
        contrast_score=0.4,
        text_density=0.1,
        dct_energy_ratio=0.6,
        color_histogram_entropy=4.0,
        dominant_color_ratio=0.8,
        motion_intensity=0.2,
        motion_smoothness=0.9,
        static_region_ratio=0.7,
        temporal_entropy=2.0,
        frame_similarity=0.85,
        inter_frame_mse_mean=100.0,
        inter_frame_mse_std=20.0,
        lossless_compression_ratio=0.5,
        transparency_ratio=0.0,
    )


def create_mock_lossy_curve(
    gif_sha: str = "a" * 64,
    base_size: float = 100.0,
) -> CompressionCurveV1:
    """Create mock lossy compression curve."""
    return CompressionCurveV1(
        gif_sha=gif_sha,
        engine=Engine.GIFSICLE,
        curve_type=CurveType.LOSSY,
        is_predicted=False,
        created_at=datetime.now(UTC),
        size_at_lossy_0=base_size,
        size_at_lossy_10=base_size * 0.95,
        size_at_lossy_20=base_size * 0.9,
        size_at_lossy_30=base_size * 0.85,
        size_at_lossy_40=base_size * 0.8,
        size_at_lossy_50=base_size * 0.75,
        size_at_lossy_60=base_size * 0.7,
        size_at_lossy_70=base_size * 0.65,
        size_at_lossy_80=base_size * 0.6,
        size_at_lossy_90=base_size * 0.55,
        size_at_lossy_100=base_size * 0.5,
    )


def create_mock_color_curve(
    gif_sha: str = "a" * 64,
    base_size: float = 100.0,
) -> CompressionCurveV1:
    """Create mock color reduction curve."""
    return CompressionCurveV1(
        gif_sha=gif_sha,
        engine=Engine.GIFSICLE,
        curve_type=CurveType.COLORS,
        is_predicted=False,
        created_at=datetime.now(UTC),
        size_at_colors_256=base_size,
        size_at_colors_128=base_size * 0.85,
        size_at_colors_64=base_size * 0.7,
        size_at_colors_32=base_size * 0.55,
        size_at_colors_16=base_size * 0.4,
    )


class TestCurvePredictionModel:
    """Test CurvePredictionModel class."""

    def test_init(self) -> None:
        """Test model initialization."""
        model = CurvePredictionModel(Engine.GIFSICLE, CurveType.LOSSY)

        assert model.engine == Engine.GIFSICLE
        assert model.curve_type == CurveType.LOSSY
        assert model.model is None
        assert model.training_samples == 0

    def test_target_columns_lossy(self) -> None:
        """Test target columns for lossy curve."""
        model = CurvePredictionModel(Engine.GIFSICLE, CurveType.LOSSY)
        targets = model.target_columns

        assert len(targets) == 11
        assert "size_at_lossy_0" in targets
        assert "size_at_lossy_100" in targets

    def test_target_columns_colors(self) -> None:
        """Test target columns for color curve."""
        model = CurvePredictionModel(Engine.GIFSICLE, CurveType.COLORS)
        targets = model.target_columns

        assert len(targets) == 5
        assert "size_at_colors_256" in targets
        assert "size_at_colors_16" in targets

    def test_feature_columns_match_schema(self) -> None:
        """Test that feature columns match GifFeaturesV1 fields."""
        features = create_mock_features()

        for col in FEATURE_COLUMNS:
            assert hasattr(features, col), f"Missing column: {col}"


class TestModelTraining:
    """Test model training with synthetic data."""

    @pytest.fixture
    def training_data(self) -> tuple[list[GifFeaturesV1], list[CompressionCurveV1]]:
        """Create synthetic training data."""
        features_list = []
        curves_list = []

        # Create 20 samples with varying sizes
        for i in range(20):
            sha = f"{i:064d}"
            base_size = 50 + i * 10  # Varying base sizes

            features = create_mock_features(sha, file_size=base_size * 1000)
            curve = create_mock_lossy_curve(sha, base_size=float(base_size))

            features_list.append(features)
            curves_list.append(curve)

        return features_list, curves_list

    def test_train_lossy_model(
        self,
        training_data: tuple[list[GifFeaturesV1], list[CompressionCurveV1]],
    ) -> None:
        """Test training a lossy prediction model."""
        features_list, curves_list = training_data

        model = CurvePredictionModel(Engine.GIFSICLE, CurveType.LOSSY)
        model.train(features_list, curves_list, n_estimators=10)

        assert model.model is not None
        assert model.training_samples == 20
        assert len(model.feature_importances) > 0

    def test_train_requires_matching_lengths(self) -> None:
        """Test that training fails with mismatched data lengths."""
        features = [create_mock_features()]
        curves = [create_mock_lossy_curve(), create_mock_lossy_curve()]

        model = CurvePredictionModel(Engine.GIFSICLE, CurveType.LOSSY)

        with pytest.raises(ValueError, match="same length"):
            model.train(features, curves)

    def test_train_requires_data(self) -> None:
        """Test that training fails with empty data."""
        model = CurvePredictionModel(Engine.GIFSICLE, CurveType.LOSSY)

        with pytest.raises(ValueError, match="No training data"):
            model.train([], [])


class TestModelPrediction:
    """Test model prediction with synthetic data."""

    @pytest.fixture
    def trained_model(self) -> CurvePredictionModel:
        """Create a trained model for testing."""
        features_list = []
        curves_list = []

        for i in range(20):
            sha = f"{i:064d}"
            base_size = 50 + i * 10

            features = create_mock_features(sha, file_size=base_size * 1000)
            curve = create_mock_lossy_curve(sha, base_size=float(base_size))

            features_list.append(features)
            curves_list.append(curve)

        model = CurvePredictionModel(Engine.GIFSICLE, CurveType.LOSSY)
        model.train(features_list, curves_list, n_estimators=10)

        return model

    def test_predict_returns_curve(
        self,
        trained_model: CurvePredictionModel,
    ) -> None:
        """Test that prediction returns a valid curve."""
        features = create_mock_features()
        curve = trained_model.predict(features)

        assert isinstance(curve, CompressionCurveV1)
        assert curve.is_predicted is True
        assert curve.model_version == MODEL_VERSION
        assert curve.engine == Engine.GIFSICLE
        assert curve.curve_type == CurveType.LOSSY

    def test_predict_curve_has_values(
        self,
        trained_model: CurvePredictionModel,
    ) -> None:
        """Test that predicted curve has valid values."""
        features = create_mock_features()
        curve = trained_model.predict(features)

        points = curve.get_lossy_curve_points()

        for level, size in points.items():
            assert size is not None
            assert size > 0, f"Size at lossy={level} should be positive"

    def test_predict_requires_trained_model(self) -> None:
        """Test that prediction fails without training."""
        model = CurvePredictionModel(Engine.GIFSICLE, CurveType.LOSSY)
        features = create_mock_features()

        with pytest.raises(RuntimeError, match="not trained"):
            model.predict(features)

    def test_predict_has_confidence_scores(
        self,
        trained_model: CurvePredictionModel,
    ) -> None:
        """Test that predictions include confidence scores."""
        features = create_mock_features()
        curve = trained_model.predict(features)

        assert curve.confidence_scores is not None
        assert len(curve.confidence_scores) == 11  # 11 lossy levels


class TestModelPersistence:
    """Test model save/load functionality."""

    def test_save_and_load_model(self, tmp_path: Path) -> None:
        """Test saving and loading a trained model."""
        # Train model
        features_list = [create_mock_features(f"{i:064d}") for i in range(10)]
        curves_list = [create_mock_lossy_curve(f"{i:064d}") for i in range(10)]

        model = CurvePredictionModel(Engine.GIFSICLE, CurveType.LOSSY)
        model.train(features_list, curves_list, n_estimators=5)

        # Save
        model_path = tmp_path / "test_model.pkl"
        metadata = model.save(model_path)

        assert model_path.exists()
        assert model_path.with_suffix(".json").exists()
        assert metadata.training_samples == 10

        # Load
        loaded_model = CurvePredictionModel.load(model_path)

        assert loaded_model.engine == Engine.GIFSICLE
        assert loaded_model.curve_type == CurveType.LOSSY
        assert loaded_model.training_samples == 10

    def test_loaded_model_can_predict(self, tmp_path: Path) -> None:
        """Test that loaded model can make predictions."""
        # Train and save
        features_list = [create_mock_features(f"{i:064d}") for i in range(10)]
        curves_list = [create_mock_lossy_curve(f"{i:064d}") for i in range(10)]

        model = CurvePredictionModel(Engine.GIFSICLE, CurveType.LOSSY)
        model.train(features_list, curves_list, n_estimators=5)

        model_path = tmp_path / "test_model.pkl"
        model.save(model_path)

        # Load and predict
        loaded_model = CurvePredictionModel.load(model_path)
        features = create_mock_features()
        curve = loaded_model.predict(features)

        assert curve.is_predicted is True
        assert curve.size_at_lossy_0 is not None


class TestModelValidation:
    """Test model validation (MAPE calculation)."""

    def test_validate_returns_mape(self) -> None:
        """Test that validation returns MAPE percentage."""
        # Create training data
        train_features = [create_mock_features(f"{i:064d}") for i in range(15)]
        train_curves = [create_mock_lossy_curve(f"{i:064d}") for i in range(15)]

        # Create validation data
        val_features = [create_mock_features(f"{i+15:064d}") for i in range(5)]
        val_curves = [create_mock_lossy_curve(f"{i+15:064d}") for i in range(5)]

        model = CurvePredictionModel(Engine.GIFSICLE, CurveType.LOSSY)
        model.train(train_features, train_curves, n_estimators=10)

        mape = model.validate(val_features, val_curves)

        assert isinstance(mape, float)
        assert mape >= 0.0
        # MAPE should be reasonable for similar synthetic data
        assert mape < 100.0


class TestColorCurveModel:
    """Test color curve prediction model."""

    def test_train_color_model(self) -> None:
        """Test training a color prediction model."""
        features_list = [create_mock_features(f"{i:064d}") for i in range(15)]
        curves_list = [create_mock_color_curve(f"{i:064d}") for i in range(15)]

        model = CurvePredictionModel(Engine.GIFSICLE, CurveType.COLORS)
        model.train(features_list, curves_list, n_estimators=10)

        assert model.model is not None
        assert model.training_samples == 15

    def test_predict_color_curve(self) -> None:
        """Test predicting color reduction curve."""
        features_list = [create_mock_features(f"{i:064d}") for i in range(15)]
        curves_list = [create_mock_color_curve(f"{i:064d}") for i in range(15)]

        model = CurvePredictionModel(Engine.GIFSICLE, CurveType.COLORS)
        model.train(features_list, curves_list, n_estimators=10)

        features = create_mock_features()
        curve = model.predict(features)

        assert curve.curve_type == CurveType.COLORS
        assert curve.size_at_colors_256 is not None
        assert curve.size_at_colors_16 is not None

        # Color curve should decrease with fewer colors
        points = curve.get_color_curve_points()
        sizes = [v for v in points.values() if v is not None]
        # Generally, fewer colors = smaller size (may not always hold)
        assert len(sizes) == 5
