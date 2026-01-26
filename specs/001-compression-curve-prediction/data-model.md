# Data Model: Compression Curve Prediction

**Feature**: 001-compression-curve-prediction  
**Date**: 2025-01-26

## Entities

### GifFeatures

Visual characteristics extracted from a GIF for compression prediction.

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `gif_sha` | string | SHA256 hash of GIF file (primary key) | Required, 64 hex chars |
| `gif_name` | string | Original filename | Required |
| `extraction_version` | string | Feature extractor version | Semver format |
| `extracted_at` | datetime | Extraction timestamp | ISO 8601 |
| **Metadata** ||||
| `width` | int | Frame width in pixels | > 0 |
| `height` | int | Frame height in pixels | > 0 |
| `frame_count` | int | Total frames in GIF | >= 1 |
| `duration_ms` | int | Total animation duration | >= 0 |
| `file_size_bytes` | int | Original file size | > 0 |
| `unique_colors` | int | Unique colors in palette | 1-256 |
| **Spatial Features** ||||
| `entropy` | float | Image entropy (complexity) | 0.0-8.0 |
| `edge_density` | float | Edge pixel ratio | 0.0-1.0 |
| `color_complexity` | float | Color distribution complexity | 0.0-1.0 |
| `gradient_smoothness` | float | Gradient transition smoothness | 0.0-1.0 |
| `contrast_score` | float | Image contrast level | 0.0-1.0 |
| `text_density` | float | Text/UI element density | 0.0-1.0 |
| `dct_energy_ratio` | float | High-freq / low-freq DCT energy | 0.0-1.0 |
| `color_histogram_entropy` | float | Color usage distribution | 0.0-8.0 |
| `dominant_color_ratio` | float | Top-10 colors as % of pixels | 0.0-1.0 |
| **Temporal Features** ||||
| `motion_intensity` | float | Average frame-to-frame change | 0.0-1.0 |
| `motion_smoothness` | float | Motion consistency | 0.0-1.0 |
| `static_region_ratio` | float | Unchanging pixel ratio | 0.0-1.0 |
| `temporal_entropy` | float | Temporal complexity | 0.0-8.0 |
| `frame_similarity` | float | Average inter-frame similarity | 0.0-1.0 |
| `inter_frame_mse_mean` | float | Mean squared error between frames | >= 0 |
| `inter_frame_mse_std` | float | MSE standard deviation | >= 0 |
| **Compressibility** ||||
| `lossless_compression_ratio` | float | Size after lossless / original | 0.0-1.0 |

---

### CompressionCurve

Predicted or actual file sizes at various compression parameter values.

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `gif_sha` | string | Reference to GIF | Required |
| `engine` | enum | Compression engine | gifsicle, animately |
| `curve_type` | enum | Parameter being varied | lossy, colors |
| `is_predicted` | bool | True if predicted, False if actual | Required |
| `model_version` | string | Prediction model version (if predicted) | Semver or null |
| `confidence_scores` | list[float] | Confidence per point (if predicted) | 0.0-1.0 each |
| `created_at` | datetime | Generation timestamp | ISO 8601 |
| **Lossy Curve Points** (if curve_type=lossy) ||||
| `size_at_lossy_0` | float | File size KB at lossy=0 | > 0 |
| `size_at_lossy_20` | float | File size KB at lossy=20 | > 0 |
| `size_at_lossy_40` | float | File size KB at lossy=40 | > 0 |
| `size_at_lossy_60` | float | File size KB at lossy=60 | > 0 |
| `size_at_lossy_80` | float | File size KB at lossy=80 | > 0 |
| `size_at_lossy_100` | float | File size KB at lossy=100 | > 0 |
| `size_at_lossy_120` | float | File size KB at lossy=120 | > 0 |
| **Color Curve Points** (if curve_type=colors) ||||
| `size_at_colors_256` | float | File size KB at 256 colors | > 0 |
| `size_at_colors_128` | float | File size KB at 128 colors | > 0 |
| `size_at_colors_64` | float | File size KB at 64 colors | > 0 |
| `size_at_colors_32` | float | File size KB at 32 colors | > 0 |
| `size_at_colors_16` | float | File size KB at 16 colors | > 0 |

---

### TrainingRecord

Paired features and outcomes for model training.

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `record_id` | string | Unique record identifier | UUID |
| `gif_sha` | string | Reference to GIF | Required |
| `dataset_version` | string | Training dataset version | Semver |
| `split` | enum | Dataset split | train, val, test |
| `features` | GifFeatures | Extracted visual features | Valid GifFeatures |
| `lossy_curve_gifsicle` | CompressionCurve | Actual gifsicle lossy curve | Valid, is_predicted=False |
| `lossy_curve_animately` | CompressionCurve | Actual animately lossy curve | Valid, is_predicted=False |
| `color_curve_gifsicle` | CompressionCurve | Actual gifsicle color curve | Valid, is_predicted=False |
| `color_curve_animately` | CompressionCurve | Actual animately color curve | Valid, is_predicted=False |
| `created_at` | datetime | Record creation timestamp | ISO 8601 |

---

### PredictionModel

Trained model metadata and artifacts.

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `model_id` | string | Unique model identifier | UUID |
| `model_version` | string | Model version | Semver |
| `engine` | enum | Target engine | gifsicle, animately |
| `curve_type` | enum | Curve being predicted | lossy, colors |
| `training_dataset_version` | string | Dataset used for training | Semver |
| `training_samples` | int | Number of training samples | > 0 |
| `validation_mape` | float | Validation set MAPE | >= 0 |
| `feature_importances` | dict[str, float] | Feature importance scores | Sum to 1.0 |
| `model_path` | string | Path to pickled model file | Valid path |
| `created_at` | datetime | Training timestamp | ISO 8601 |
| `giflab_version` | string | GifLab version at training | Semver |
| `code_commit` | string | Git commit hash | 7+ hex chars |

---

## Relationships

```
GifFeatures (1) ──── (1) TrainingRecord
                           │
                           ├── (1) CompressionCurve [lossy_gifsicle]
                           ├── (1) CompressionCurve [lossy_animately]
                           ├── (1) CompressionCurve [color_gifsicle]
                           └── (1) CompressionCurve [color_animately]

PredictionModel ──predicts──> CompressionCurve (is_predicted=True)
```

## State Transitions

### GifFeatures Lifecycle

```
[GIF File] ──extract──> [GifFeatures] ──validate──> [Stored/Cached]
                              │
                              └── deterministic: same GIF always produces same features
```

### TrainingRecord Lifecycle

```
[GifFeatures] + [Compression Runs] ──join──> [TrainingRecord]
                                                    │
                                    ┌───────────────┼───────────────┐
                                    ▼               ▼               ▼
                                 [train]         [val]          [test]
                                    │               │               │
                                    └───────────────┴───────────────┘
                                                    │
                                              [Model Training]
```

### PredictionModel Lifecycle

```
[TrainingRecords] ──train──> [PredictionModel v1.0.0]
                                      │
                    [New Data] ──retrain──> [PredictionModel v1.1.0]
```
