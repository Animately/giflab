"""Compression Curve Prediction module for GifLab.

This module provides functionality to:
- Extract visual features from GIFs for ML training
- Predict compression curves (file size vs lossy level)
- Predict color reduction curves (file size vs color count)
- Build training datasets for prediction models

Constitution Compliance:
- Principle II (ML-Ready Data): All outputs validated against Pydantic schemas
- Principle III (Poetry-First): Use `poetry run` for all commands
- Principle VI (LLM-Optimized): Explicit patterns, type hints, docstrings
"""

from giflab.prediction.schemas import (
    CompressionCurveV1,
    CurveType,
    DatasetSplit,
    Engine,
    GifFeaturesV1,
    PredictionModelMetadataV1,
    TrainingRecordV1,
)

__all__ = [
    # Enums
    "Engine",
    "CurveType",
    "DatasetSplit",
    # Schemas
    "GifFeaturesV1",
    "CompressionCurveV1",
    "TrainingRecordV1",
    "PredictionModelMetadataV1",
    # Functions
    "extract_gif_features",
    "predict_lossy_curve",
    "predict_color_curve",
    # Version
    "FEATURE_EXTRACTOR_VERSION",
]

# Feature extractor version - increment on feature extraction changes
FEATURE_EXTRACTOR_VERSION = "1.0.0"

# Lazy imports to avoid circular dependencies
def extract_gif_features(gif_path):
    """Extract visual features from a GIF."""
    from giflab.prediction.features import extract_gif_features as _extract
    return _extract(gif_path)


def predict_lossy_curve(features, engine=None, model_dir=None):
    """Predict lossy compression curve."""
    from giflab.prediction.models import predict_lossy_curve as _predict
    from giflab.prediction.schemas import Engine
    if engine is None:
        engine = Engine.GIFSICLE
    return _predict(features, engine, model_dir)


def predict_color_curve(features, engine=None, model_dir=None):
    """Predict color reduction curve."""
    from giflab.prediction.models import predict_color_curve as _predict
    from giflab.prediction.schemas import Engine
    if engine is None:
        engine = Engine.GIFSICLE
    return _predict(features, engine, model_dir)
