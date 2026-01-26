"""Pydantic schemas for Compression Curve Prediction feature.

These schemas define the data contracts for feature extraction, prediction,
and training data. All pipeline outputs MUST validate against these schemas
per Constitution Principle II (ML-Ready Data).
"""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Engine(str, Enum):
    """Supported compression engines."""
    GIFSICLE = "gifsicle"
    ANIMATELY = "animately"


class CurveType(str, Enum):
    """Types of compression curves."""
    LOSSY = "lossy"
    COLORS = "colors"


class DatasetSplit(str, Enum):
    """Training dataset splits."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class GifFeaturesV1(BaseModel):
    """Visual characteristics extracted from a GIF for compression prediction.
    
    Schema version: 1.0.0
    """
    schema_version: Literal["1.0.0"] = "1.0.0"
    
    # Identity
    gif_sha: str = Field(..., min_length=64, max_length=64, description="SHA256 hash of GIF file")
    gif_name: str = Field(..., min_length=1, description="Original filename")
    extraction_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$", description="Feature extractor version")
    extracted_at: datetime = Field(..., description="Extraction timestamp")
    
    # Metadata
    width: int = Field(..., gt=0, description="Frame width in pixels")
    height: int = Field(..., gt=0, description="Frame height in pixels")
    frame_count: int = Field(..., ge=1, description="Total frames in GIF")
    duration_ms: int = Field(..., ge=0, description="Total animation duration")
    file_size_bytes: int = Field(..., gt=0, description="Original file size")
    unique_colors: int = Field(..., ge=1, le=256, description="Unique colors in palette")
    
    # Spatial Features
    entropy: float = Field(..., ge=0.0, le=8.0, description="Image entropy")
    edge_density: float = Field(..., ge=0.0, le=1.0, description="Edge pixel ratio")
    color_complexity: float = Field(..., ge=0.0, le=1.0, description="Color distribution complexity")
    gradient_smoothness: float = Field(..., ge=0.0, le=1.0, description="Gradient transition smoothness")
    contrast_score: float = Field(..., ge=0.0, le=1.0, description="Image contrast level")
    text_density: float = Field(..., ge=0.0, le=1.0, description="Text/UI element density")
    dct_energy_ratio: float = Field(..., ge=0.0, le=1.0, description="High-freq / low-freq DCT energy")
    color_histogram_entropy: float = Field(..., ge=0.0, le=8.0, description="Color usage distribution")
    dominant_color_ratio: float = Field(..., ge=0.0, le=1.0, description="Top-10 colors as % of pixels")
    
    # Temporal Features
    motion_intensity: float = Field(..., ge=0.0, le=1.0, description="Average frame-to-frame change")
    motion_smoothness: float = Field(..., ge=0.0, le=1.0, description="Motion consistency")
    static_region_ratio: float = Field(..., ge=0.0, le=1.0, description="Unchanging pixel ratio")
    temporal_entropy: float = Field(..., ge=0.0, le=8.0, description="Temporal complexity")
    frame_similarity: float = Field(..., ge=0.0, le=1.0, description="Average inter-frame similarity")
    inter_frame_mse_mean: float = Field(..., ge=0.0, description="Mean squared error between frames")
    inter_frame_mse_std: float = Field(..., ge=0.0, description="MSE standard deviation")
    
    # Compressibility
    lossless_compression_ratio: float = Field(..., ge=0.0, le=1.0, description="Size after lossless / original")

    @field_validator("gif_sha")
    @classmethod
    def validate_sha_hex(cls, v: str) -> str:
        if not all(c in "0123456789abcdef" for c in v.lower()):
            raise ValueError("gif_sha must be hexadecimal")
        return v.lower()


class CompressionCurveV1(BaseModel):
    """Predicted or actual file sizes at various compression parameter values.
    
    Schema version: 1.0.0
    """
    schema_version: Literal["1.0.0"] = "1.0.0"
    
    gif_sha: str = Field(..., min_length=64, max_length=64, description="Reference to GIF")
    engine: Engine = Field(..., description="Compression engine")
    curve_type: CurveType = Field(..., description="Parameter being varied")
    is_predicted: bool = Field(..., description="True if predicted, False if actual")
    model_version: str | None = Field(None, description="Prediction model version (if predicted)")
    confidence_scores: list[float] | None = Field(None, description="Confidence per point (if predicted)")
    created_at: datetime = Field(..., description="Generation timestamp")
    
    # Lossy curve points (populated when curve_type=lossy)
    size_at_lossy_0: float | None = Field(None, gt=0, description="File size KB at lossy=0")
    size_at_lossy_20: float | None = Field(None, gt=0, description="File size KB at lossy=20")
    size_at_lossy_40: float | None = Field(None, gt=0, description="File size KB at lossy=40")
    size_at_lossy_60: float | None = Field(None, gt=0, description="File size KB at lossy=60")
    size_at_lossy_80: float | None = Field(None, gt=0, description="File size KB at lossy=80")
    size_at_lossy_100: float | None = Field(None, gt=0, description="File size KB at lossy=100")
    size_at_lossy_120: float | None = Field(None, gt=0, description="File size KB at lossy=120")
    
    # Color curve points (populated when curve_type=colors)
    size_at_colors_256: float | None = Field(None, gt=0, description="File size KB at 256 colors")
    size_at_colors_128: float | None = Field(None, gt=0, description="File size KB at 128 colors")
    size_at_colors_64: float | None = Field(None, gt=0, description="File size KB at 64 colors")
    size_at_colors_32: float | None = Field(None, gt=0, description="File size KB at 32 colors")
    size_at_colors_16: float | None = Field(None, gt=0, description="File size KB at 16 colors")

    @field_validator("confidence_scores")
    @classmethod
    def validate_confidence_range(cls, v: list[float] | None) -> list[float] | None:
        if v is not None:
            for score in v:
                if not 0.0 <= score <= 1.0:
                    raise ValueError("confidence_scores must be between 0.0 and 1.0")
        return v


class TrainingRecordV1(BaseModel):
    """Paired features and outcomes for model training.
    
    Schema version: 1.0.0
    """
    schema_version: Literal["1.0.0"] = "1.0.0"
    
    record_id: str = Field(..., description="Unique record identifier (UUID)")
    gif_sha: str = Field(..., min_length=64, max_length=64, description="Reference to GIF")
    dataset_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$", description="Training dataset version")
    split: DatasetSplit = Field(..., description="Dataset split")
    features: GifFeaturesV1 = Field(..., description="Extracted visual features")
    lossy_curve_gifsicle: CompressionCurveV1 = Field(..., description="Actual gifsicle lossy curve")
    lossy_curve_animately: CompressionCurveV1 = Field(..., description="Actual animately lossy curve")
    color_curve_gifsicle: CompressionCurveV1 = Field(..., description="Actual gifsicle color curve")
    color_curve_animately: CompressionCurveV1 = Field(..., description="Actual animately color curve")
    created_at: datetime = Field(..., description="Record creation timestamp")


class PredictionModelMetadataV1(BaseModel):
    """Trained model metadata.
    
    Schema version: 1.0.0
    """
    schema_version: Literal["1.0.0"] = "1.0.0"
    
    model_id: str = Field(..., description="Unique model identifier (UUID)")
    model_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$", description="Model version")
    engine: Engine = Field(..., description="Target engine")
    curve_type: CurveType = Field(..., description="Curve being predicted")
    training_dataset_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$", description="Dataset used for training")
    training_samples: int = Field(..., gt=0, description="Number of training samples")
    validation_mape: float = Field(..., ge=0.0, description="Validation set MAPE")
    feature_importances: dict[str, float] = Field(..., description="Feature importance scores")
    model_path: str = Field(..., description="Path to pickled model file")
    created_at: datetime = Field(..., description="Training timestamp")
    giflab_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$", description="GifLab version at training")
    code_commit: str = Field(..., min_length=7, description="Git commit hash")
