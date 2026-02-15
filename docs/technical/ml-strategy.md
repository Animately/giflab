# ML Strategy: Compression Curve Prediction

## Vision

GifLab builds prediction datasets and trains gradient boosting models that predict
compression curves from visual features alone. Given a new GIF, the trained model
estimates the output file size at every lossy level and color count -- without
running any compression engine -- enabling fast, intelligent pipeline selection.

---

## Current Implementation

### Feature Extraction (`prediction/features.py`)

Each GIF is analysed to produce 24 numeric features grouped into four categories:

| Category | Features |
|---|---|
| **Spatial** (9) | entropy, edge_density, color_complexity, gradient_smoothness, contrast_score, text_density, dct_energy_ratio, color_histogram_entropy, dominant_color_ratio |
| **Temporal** (7) | motion_intensity, motion_smoothness, static_region_ratio, temporal_entropy, frame_similarity, inter_frame_mse_mean, inter_frame_mse_std |
| **Compressibility** (1) | lossless_compression_ratio |
| **Transparency** (1) | transparency_ratio |

Six additional metadata columns are used as model inputs: `width`, `height`,
`frame_count`, `duration_ms`, `file_size_bytes`, `unique_colors`.

Optional CLIP-based content classification scores (screen_capture, vector_art,
photography, hand_drawn, 3d_rendered, pixel_art) can be extracted when CLIP is
available, but are not part of the core feature set used by the prediction model.

### Prediction Model (`prediction/models.py`)

`CurvePredictionModel` wraps scikit-learn's `GradientBoostingRegressor` inside a
`MultiOutputRegressor`. One model instance is trained per (engine, curve_type) pair.

- **Lossy curves**: 11 points at lossy levels 0, 10, 20, ... 100
- **Color curves**: 5 points at color counts 256, 128, 64, 32, 16

Key methods:

| Method | Purpose |
|---|---|
| `train()` | Fit gradient boosting on features + curves |
| `predict()` | Return a `CompressionCurveV1` with confidence scores |
| `validate()` | Evaluate on held-out data |
| `save()` / `load()` | Persist and restore trained models |

Standalone helpers `predict_lossy_curve()` and `predict_color_curve()` provide a
simplified interface for single predictions.

Model version: `1.0.0`.

### Schemas (`prediction/schemas.py`)

All data flows through versioned Pydantic schemas:

- **`GifFeaturesV1`** -- 24 visual features plus identity and metadata fields, with SHA256 validation
- **`CompressionCurveV1`** -- predicted or actual file sizes at each parameter level, with per-point confidence scores
- **`TrainingRecordV1`** -- paired features + curve for training
- **`PredictionModelMetadataV1`** -- model provenance (engine, curve type, version, sample count)
- **`Engine`** enum -- the 7 supported engines: gifsicle, animately-standard, animately-advanced, animately-hard, imagemagick, ffmpeg, gifski
- **`CurveType`** enum -- lossy, colors
- **`DatasetSplit`** enum -- train, val, test

### Storage (`storage.py`)

`GifLabStorage` manages a SQLite database with tables for `gif_features`,
`compression_runs`, `pipelines`, and `param_presets`.

The `get_training_data()` method exports paired features and compression results as
DataFrame-ready dicts, with configurable train/val/test splits (default 80/10/10).

### CLI (`cli/__init__.py`)

```bash
# Run compression pipelines and collect data
poetry run python -m giflab run --preset frame-focus

# Train prediction models from collected data
poetry run python -m giflab train --engine gifsicle --output data/models

# Export features and compression results
poetry run python -m giflab export

# View dataset statistics
poetry run python -m giflab stats
```

---

## How It Works End-to-End

1. **Data collection** -- `giflab run` processes GIFs through compression engine
   pipelines at multiple parameter levels, storing features and output sizes in
   SQLite.

2. **Feature extraction** -- Each GIF's visual properties are computed once and
   cached. The 24 features capture spatial complexity, temporal dynamics,
   compressibility, and transparency.

3. **Training** -- `giflab train` loads paired (features, curve) records, splits
   into train/val/test, and fits a `GradientBoostingRegressor` per engine and
   curve type.

4. **Prediction** -- Given a new GIF's features, the model predicts the full
   compression curve (file size at each lossy level or color count) along with
   per-point confidence scores.

5. **Pipeline selection** -- Predicted curves across all 7 engines allow
   selecting the best engine and parameter for a given size/quality target
   without running every compression.

---

## Future Research Directions

The following ideas are not yet implemented. They represent potential avenues for
extending the prediction system beyond gradient boosting.

- **Deep learning encoders** -- Replace hand-crafted features with ResNet or
  Vision Transformer embeddings for automatic feature learning.
- **Reinforcement learning** -- Train an agent that learns optimal tool
  selection through compression/quality reward signals (DQN, multi-armed
  bandit, contextual bandit approaches).
- **Transformer-based sequence models** -- Model multi-step pipelines as
  sequences, using attention to learn tool chaining strategies.
- **Bayesian optimization** -- Use Gaussian processes or tree-structured Parzen
  estimators for hyperparameter search over compression parameters.
- **Neural compression** -- Explore learned image compression methods as an
  alternative to traditional GIF encoders.
- **Format expansion** -- Extend prediction to modern formats (WebP, AVIF) once
  encoder support is added.
- **Additional tool integrations** -- Pillow, Sharp, libgif are candidates for
  future engine support.

---

## Tool Documentation

- [Gifsicle Manual](https://www.lcdf.org/gifsicle/man.html)
- [ImageMagick GIF Options](https://imagemagick.org/script/formats.php#gif)
- [FFmpeg GIF Filters](https://ffmpeg.org/ffmpeg-filters.html#gif)
- [gifski Documentation](https://gif.ski/)
