"""Prediction models for compression curve prediction.

This module provides:
- Training of gradient boosting models for compression curves
- Prediction of lossy and color curves from GIF features
- Model serialization and versioning

Constitution Compliance:
- Principle II (ML-Ready Data): Versioned models, schema outputs
- Principle IV (Test-Driven Quality): Accuracy validation
"""

import logging
import pickle
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

import giflab

from giflab.prediction.schemas import (
    CompressionCurveV1,
    CurveType,
    Engine,
    GifFeaturesV1,
    PredictionModelMetadataV1,
)

logger = logging.getLogger(__name__)


def _get_git_commit() -> str:
    """Get short git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


# Model version
MODEL_VERSION = "1.0.0"

# Feature columns used for prediction (must match GifFeaturesV1 fields)
FEATURE_COLUMNS = [
    "width",
    "height",
    "frame_count",
    "duration_ms",
    "file_size_bytes",
    "unique_colors",
    "entropy",
    "edge_density",
    "color_complexity",
    "gradient_smoothness",
    "contrast_score",
    "text_density",
    "dct_energy_ratio",
    "color_histogram_entropy",
    "dominant_color_ratio",
    "motion_intensity",
    "motion_smoothness",
    "static_region_ratio",
    "temporal_entropy",
    "frame_similarity",
    "inter_frame_mse_mean",
    "inter_frame_mse_std",
    "lossless_compression_ratio",
    "transparency_ratio",
]

# Lossy level targets
LOSSY_LEVELS = [0, 20, 40, 60, 80, 100, 120]

# Color count targets
COLOR_COUNTS = [256, 128, 64, 32, 16]


class CurvePredictionModel:
    """Model for predicting compression curves."""

    def __init__(
        self,
        engine: Engine,
        curve_type: CurveType,
        model: MultiOutputRegressor | None = None,
    ) -> None:
        """Initialize the prediction model.

        Args:
            engine: Target compression engine.
            curve_type: Type of curve to predict (lossy or colors).
            model: Pre-trained model (optional).
        """
        self.engine = engine
        self.curve_type = curve_type
        self.model = model
        self.model_id = str(uuid.uuid4())
        self.training_samples = 0
        self.validation_mape = 0.0
        self.feature_importances: dict[str, float] = {}
        self.created_at = datetime.now(timezone.utc)

    @property
    def target_columns(self) -> list[str]:
        """Get target column names based on curve type."""
        if self.curve_type == CurveType.LOSSY:
            return [f"size_at_lossy_{level}" for level in LOSSY_LEVELS]
        else:
            return [f"size_at_colors_{count}" for count in COLOR_COUNTS]

    def train(
        self,
        features_list: list[GifFeaturesV1],
        curves_list: list[CompressionCurveV1],
        n_estimators: int = 100,
        max_depth: int = 5,
        random_state: int = 42,
    ) -> None:
        """Train the prediction model.

        Args:
            features_list: List of GIF features.
            curves_list: List of corresponding compression curves.
            n_estimators: Number of boosting stages.
            max_depth: Maximum tree depth.
            random_state: Random seed for reproducibility.
        """
        if len(features_list) != len(curves_list):
            raise ValueError("Features and curves must have same length")

        if len(features_list) == 0:
            raise ValueError("No training data provided")

        # Extract feature matrix
        X = self._features_to_matrix(features_list)

        # Extract target matrix
        y = self._curves_to_matrix(curves_list)

        # Filter out rows with missing targets
        valid_mask = ~np.isnan(y).any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) == 0:
            raise ValueError("No valid training samples after filtering")

        self.training_samples = len(X)

        # Train model
        base_model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        self.model = MultiOutputRegressor(base_model)
        self.model.fit(X, y)

        # Calculate feature importances (average across outputs)
        importances = np.zeros(len(FEATURE_COLUMNS))
        for estimator in self.model.estimators_:
            importances += estimator.feature_importances_
        importances /= len(self.model.estimators_)

        self.feature_importances = {
            col: float(imp)
            for col, imp in zip(FEATURE_COLUMNS, importances, strict=True)
        }

        logger.info(
            f"Trained {self.engine.value} {self.curve_type.value} model "
            f"on {self.training_samples} samples"
        )

    def predict(self, features: GifFeaturesV1) -> CompressionCurveV1:
        """Predict compression curve for given features.

        Args:
            features: Visual features of a GIF.

        Returns:
            Predicted compression curve.
        """
        if self.model is None:
            raise RuntimeError("Model not trained")

        # Convert features to matrix
        X = self._features_to_matrix([features])

        # Predict
        y_pred = self.model.predict(X)[0]

        # Calculate confidence scores (based on training data coverage)
        confidence_scores = self._calculate_confidence(features)

        # Build curve object
        return self._predictions_to_curve(
            features.gif_sha,
            y_pred,
            confidence_scores,
        )

    def validate(
        self,
        features_list: list[GifFeaturesV1],
        curves_list: list[CompressionCurveV1],
    ) -> float:
        """Validate model and return MAPE.

        Args:
            features_list: Validation features.
            curves_list: Actual curves.

        Returns:
            Mean Absolute Percentage Error.
        """
        if self.model is None:
            raise RuntimeError("Model not trained")

        X = self._features_to_matrix(features_list)
        y_true = self._curves_to_matrix(curves_list)

        # Filter valid rows
        valid_mask = ~np.isnan(y_true).any(axis=1)
        X = X[valid_mask]
        y_true = y_true[valid_mask]

        if len(X) == 0:
            return float("inf")

        y_pred = self.model.predict(X)

        # Calculate MAPE (absolute value in denominator for proper formula)
        epsilon = 1e-10
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon)))
        mape = float(mape * 100)
        self.validation_mape = float(mape)

        return self.validation_mape

    def save(self, path: Path) -> PredictionModelMetadataV1:
        """Save model to disk.

        Args:
            path: Path to save the model.

        Returns:
            Model metadata.
        """
        if self.model is None:
            raise RuntimeError("Model not trained")

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.model, f)

        metadata = PredictionModelMetadataV1(
            model_id=self.model_id,
            model_version=MODEL_VERSION,
            engine=self.engine,
            curve_type=self.curve_type,
            training_dataset_version="1.0.0",
            training_samples=self.training_samples,
            validation_mape=self.validation_mape,
            feature_importances=self.feature_importances,
            model_path=str(path),
            created_at=self.created_at,
            giflab_version=giflab.__version__,
            code_commit=_get_git_commit(),
        )

        # Save metadata alongside model
        metadata_path = path.with_suffix(".json")
        metadata_path.write_text(metadata.model_dump_json(indent=2))

        logger.info(f"Saved model to {path}")
        return metadata

    @classmethod
    def load(cls, path: Path) -> "CurvePredictionModel":
        """Load model from disk.

        Args:
            path: Path to the saved model.

        Returns:
            Loaded prediction model.
        """
        # Load metadata
        metadata_path = path.with_suffix(".json")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        metadata = PredictionModelMetadataV1.model_validate_json(
            metadata_path.read_text()
        )

        # Load model
        with open(path, "rb") as f:
            model = pickle.load(f)

        instance = cls(
            engine=metadata.engine,
            curve_type=metadata.curve_type,
            model=model,
        )
        instance.model_id = metadata.model_id
        instance.training_samples = metadata.training_samples
        instance.validation_mape = metadata.validation_mape
        instance.feature_importances = metadata.feature_importances
        instance.created_at = metadata.created_at

        return instance

    def _features_to_matrix(
        self,
        features_list: list[GifFeaturesV1],
    ) -> np.ndarray:
        """Convert features list to numpy matrix."""
        rows = []
        for features in features_list:
            row = [getattr(features, col) for col in FEATURE_COLUMNS]
            rows.append(row)
        return np.array(rows, dtype=np.float64)

    def _curves_to_matrix(
        self,
        curves_list: list[CompressionCurveV1],
    ) -> np.ndarray:
        """Convert curves list to numpy target matrix."""
        rows = []
        for curve in curves_list:
            if self.curve_type == CurveType.LOSSY:
                row = [
                    curve.size_at_lossy_0,
                    curve.size_at_lossy_20,
                    curve.size_at_lossy_40,
                    curve.size_at_lossy_60,
                    curve.size_at_lossy_80,
                    curve.size_at_lossy_100,
                    curve.size_at_lossy_120,
                ]
            else:
                row = [
                    curve.size_at_colors_256,
                    curve.size_at_colors_128,
                    curve.size_at_colors_64,
                    curve.size_at_colors_32,
                    curve.size_at_colors_16,
                ]
            # Replace None with NaN
            row = [v if v is not None else np.nan for v in row]
            rows.append(row)
        return np.array(rows, dtype=np.float64)

    def _predictions_to_curve(
        self,
        gif_sha: str,
        predictions: np.ndarray,
        confidence_scores: list[float],
    ) -> CompressionCurveV1:
        """Convert predictions to CompressionCurveV1."""
        clamped = [
            i for i, v in enumerate(predictions)
            if float(v) <= 0.1
        ]
        if clamped:
            logger.warning(
                "Prediction clamped to 0.1 KB for %s at "
                "indices %s (raw values: %s)",
                gif_sha,
                clamped,
                [float(predictions[i]) for i in clamped],
            )
        if self.curve_type == CurveType.LOSSY:
            return CompressionCurveV1(
                gif_sha=gif_sha,
                engine=self.engine,
                curve_type=self.curve_type,
                is_predicted=True,
                model_version=MODEL_VERSION,
                confidence_scores=confidence_scores,
                created_at=datetime.now(timezone.utc),
                size_at_lossy_0=max(0.1, float(predictions[0])),
                size_at_lossy_20=max(0.1, float(predictions[1])),
                size_at_lossy_40=max(0.1, float(predictions[2])),
                size_at_lossy_60=max(0.1, float(predictions[3])),
                size_at_lossy_80=max(0.1, float(predictions[4])),
                size_at_lossy_100=max(0.1, float(predictions[5])),
                size_at_lossy_120=max(0.1, float(predictions[6])),
            )
        else:
            return CompressionCurveV1(
                gif_sha=gif_sha,
                engine=self.engine,
                curve_type=self.curve_type,
                is_predicted=True,
                model_version=MODEL_VERSION,
                confidence_scores=confidence_scores,
                created_at=datetime.now(timezone.utc),
                size_at_colors_256=max(0.1, float(predictions[0])),
                size_at_colors_128=max(0.1, float(predictions[1])),
                size_at_colors_64=max(0.1, float(predictions[2])),
                size_at_colors_32=max(0.1, float(predictions[3])),
                size_at_colors_16=max(0.1, float(predictions[4])),
            )

    def _calculate_confidence(self, features: GifFeaturesV1) -> list[float]:
        """Calculate confidence scores for predictions.

        Simple heuristic: higher confidence for features within training range.
        """
        num_points = 7 if self.curve_type == CurveType.LOSSY else 5
        # Placeholder: return 0.8 for all points
        # TODO: Implement proper confidence based on training data distribution
        return [0.8] * num_points


def predict_lossy_curve(
    features: GifFeaturesV1,
    engine: Engine = Engine.GIFSICLE,
    model_dir: Path | None = None,
) -> CompressionCurveV1:
    """Predict lossy compression curve for a GIF.

    Args:
        features: Visual features of the GIF.
        engine: Target compression engine.
        model_dir: Directory containing trained models.

    Returns:
        Predicted lossy compression curve.
    """
    if model_dir is None:
        model_dir = Path("data/models")

    model_path = model_dir / f"{engine.value}_lossy_v1.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Run `giflab predict train` first."
        )

    model = CurvePredictionModel.load(model_path)
    return model.predict(features)


def predict_color_curve(
    features: GifFeaturesV1,
    engine: Engine = Engine.GIFSICLE,
    model_dir: Path | None = None,
) -> CompressionCurveV1:
    """Predict color reduction curve for a GIF.

    Args:
        features: Visual features of the GIF.
        engine: Target compression engine.
        model_dir: Directory containing trained models.

    Returns:
        Predicted color reduction curve.
    """
    if model_dir is None:
        model_dir = Path("data/models")

    model_path = model_dir / f"{engine.value}_color_v1.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Run `giflab predict train` first."
        )

    model = CurvePredictionModel.load(model_path)
    return model.predict(features)
