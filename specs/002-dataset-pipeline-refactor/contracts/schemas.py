"""Updated Pydantic schemas for 002-dataset-pipeline-refactor.

This file defines the TARGET state of the schemas after refactoring.
Changes from the current schemas in src/giflab/prediction/schemas.py:

1. Engine enum expanded from 2 to 7 values
2. TrainingRecordV2 replaces hardcoded engine fields with flexible dict
3. Migration utility from V1 → V2
4. All other schemas (GifFeaturesV1, CompressionCurveV1, PredictionModelMetadataV1)
   remain unchanged.
"""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ─── Engine Enum (CHANGED: 2 → 7 values) ───────────────────────────────────


class Engine(str, Enum):
    """The 7 lossy compression engines.

    CHANGED from V1: Was only GIFSICLE and ANIMATELY.
    Now includes all 7 distinct compression algorithms.
    """

    GIFSICLE = "gifsicle"
    ANIMATELY_STANDARD = "animately-standard"
    ANIMATELY_ADVANCED = "animately-advanced"
    ANIMATELY_HARD = "animately-hard"
    IMAGEMAGICK = "imagemagick"
    FFMPEG = "ffmpeg"
    GIFSKI = "gifski"


# ─── Unchanged Enums ────────────────────────────────────────────────────────


class CurveType(str, Enum):
    """Types of compression curves."""

    LOSSY = "lossy"
    COLORS = "colors"


class DatasetSplit(str, Enum):
    """Training dataset splits."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


# ─── GifFeaturesV1 (UNCHANGED) ──────────────────────────────────────────────

# GifFeaturesV1 remains exactly as-is in src/giflab/prediction/schemas.py
# 25+ features: spatial, temporal, compressibility, transparency
# No changes needed for the refactoring.


# ─── CompressionCurveV1 (UNCHANGED structure, expanded Engine enum) ──────────

# CompressionCurveV1 remains structurally unchanged.
# The `engine` field now accepts all 7 Engine values instead of just 2.
# This is backward compatible since Engine is expanded, not contracted.


# ─── TrainingRecordV2 (NEW: replaces TrainingRecordV1) ──────────────────────


class TrainingRecordV2(BaseModel):
    """Paired features and outcomes for model training.

    Schema version: 2.0.0

    CHANGED from V1:
    - Replaced hardcoded `lossy_curve_gifsicle`, `lossy_curve_animately`,
      `color_curve_gifsicle`, `color_curve_animately` fields with flexible
      `lossy_curves` and `color_curves` dicts keyed by engine name.
    - Supports all 7 engines without schema changes per engine.
    """

    schema_version: Literal["2.0.0"] = "2.0.0"

    record_id: str = Field(..., description="Unique record identifier (UUID)")
    gif_sha: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="Reference to GIF",
    )
    dataset_version: str = Field(
        ...,
        pattern=r"^\d+\.\d+\.\d+$",
        description="Training dataset version",
    )
    split: DatasetSplit = Field(..., description="Dataset split")

    # Features (unchanged)
    features: "GifFeaturesV1" = Field(
        ...,
        description="Extracted visual features",
    )

    # Flexible engine curves (NEW in V2)
    # Keys are engine names from the Engine enum (e.g., "gifsicle", "animately-standard")
    lossy_curves: dict[str, "CompressionCurveV1"] = Field(
        default_factory=dict,
        description="Lossy compression curves per engine",
    )
    color_curves: dict[str, "CompressionCurveV1"] = Field(
        default_factory=dict,
        description="Color reduction curves per engine",
    )

    created_at: datetime = Field(..., description="Record creation timestamp")

    @field_validator("lossy_curves", "color_curves")
    @classmethod
    def validate_curve_engine_names(
        cls,
        v: dict[str, "CompressionCurveV1"],
    ) -> dict[str, "CompressionCurveV1"]:
        """Validate that curve keys are valid engine names."""
        valid_engines = {e.value for e in Engine}
        for key in v:
            if key not in valid_engines:
                raise ValueError(
                    f"Invalid engine name '{key}'. "
                    f"Valid engines: {sorted(valid_engines)}"
                )
        return v


# ─── Migration Utility ──────────────────────────────────────────────────────


def migrate_training_record_v1_to_v2(
    v1: "TrainingRecordV1",
) -> TrainingRecordV2:
    """Convert a TrainingRecordV1 to TrainingRecordV2.

    Maps:
    - lossy_curve_gifsicle → lossy_curves["gifsicle"]
    - lossy_curve_animately → lossy_curves["animately-standard"]
    - color_curve_gifsicle → color_curves["gifsicle"]
    - color_curve_animately → color_curves["animately-standard"]
    """
    lossy_curves: dict[str, "CompressionCurveV1"] = {}
    color_curves: dict[str, "CompressionCurveV1"] = {}

    # Map gifsicle curves
    lossy_curves["gifsicle"] = v1.lossy_curve_gifsicle
    color_curves["gifsicle"] = v1.color_curve_gifsicle

    # Map animately curves (V1 "animately" → V2 "animately-standard")
    if v1.lossy_curve_animately is not None:
        lossy_curves["animately-standard"] = v1.lossy_curve_animately
    if v1.color_curve_animately is not None:
        color_curves["animately-standard"] = v1.color_curve_animately

    return TrainingRecordV2(
        record_id=v1.record_id,
        gif_sha=v1.gif_sha,
        dataset_version=v1.dataset_version,
        split=v1.split,
        features=v1.features,
        lossy_curves=lossy_curves,
        color_curves=color_curves,
        created_at=v1.created_at,
    )


# ─── PredictionModelMetadataV1 (UNCHANGED, uses expanded Engine) ────────────

# PredictionModelMetadataV1 remains exactly as-is.
# The `engine` field now accepts all 7 Engine values.


# ─── Forward References ─────────────────────────────────────────────────────
# These are resolved at import time when used with the actual schemas module.
# In the contracts file, they serve as documentation of the interface.

# Note: GifFeaturesV1, CompressionCurveV1, TrainingRecordV1, and
# PredictionModelMetadataV1 are defined in src/giflab/prediction/schemas.py
# and are NOT duplicated here. Only new/changed schemas are defined above.
