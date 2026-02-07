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

from __future__ import annotations

from pathlib import Path

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


def extract_gif_features(gif_path: Path | str) -> GifFeaturesV1:
    """Extract visual features from a GIF."""
    from giflab.prediction.features import extract_gif_features as _extract
    return _extract(Path(gif_path) if isinstance(gif_path, str) else gif_path)


def predict_lossy_curve(
    features: GifFeaturesV1,
    engine: Engine | None = None,
    model_dir: Path | None = None,
) -> CompressionCurveV1:
    """Predict lossy compression curve."""
    from giflab.prediction.models import predict_lossy_curve as _predict
    if engine is None:
        engine = Engine.GIFSICLE
    return _predict(features, engine, model_dir)


def predict_color_curve(
    features: GifFeaturesV1,
    engine: Engine | None = None,
    model_dir: Path | None = None,
) -> CompressionCurveV1:
    """Predict color reduction curve."""
    from giflab.prediction.models import predict_color_curve as _predict
    if engine is None:
        engine = Engine.GIFSICLE
    return _predict(features, engine, model_dir)
