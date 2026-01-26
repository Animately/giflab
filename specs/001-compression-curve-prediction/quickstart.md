# Quickstart: Compression Curve Prediction

**Feature**: 001-compression-curve-prediction  
**Date**: 2025-01-26

## Overview

This feature adds the ability to predict GIF compression curves (file size vs lossy level) and color reduction curves (file size vs color count) based on visual features extracted from the GIF.

## Prerequisites

- GifLab installed via Poetry
- At least one compression engine available (gifsicle or animately)
- Python 3.11+

## Quick Commands

### Extract Features from a GIF

```bash
poetry run python -m giflab predict extract-features path/to/image.gif
```

Output: JSON with 20+ visual features.

### Predict Compression Curve

```bash
poetry run python -m giflab predict lossy-curve path/to/image.gif --engine gifsicle
```

Output: Predicted file sizes at lossy levels 0, 20, 40, 60, 80, 100, 120.

### Predict Color Reduction Curve

```bash
poetry run python -m giflab predict color-curve path/to/image.gif --engine gifsicle
```

Output: Predicted file sizes at color counts 256, 128, 64, 32, 16.

### Build Training Dataset

```bash
poetry run python -m giflab predict build-dataset data/raw/ --output data/training/
```

Runs compression experiments and pairs results with extracted features.

### Train Prediction Models

```bash
poetry run python -m giflab predict train --dataset data/training/ --output data/models/
```

Trains gradient boosting models for each engine and curve type.

## Python API

```python
from giflab.prediction import extract_features, predict_lossy_curve, predict_color_curve

# Extract features
features = extract_features("path/to/image.gif")

# Predict curves
lossy_curve = predict_lossy_curve(features, engine="gifsicle")
color_curve = predict_color_curve(features, engine="gifsicle")

# Access predictions
print(f"Size at lossy=40: {lossy_curve.size_at_lossy_40} KB")
print(f"Size at 64 colors: {color_curve.size_at_colors_64} KB")
```

## Directory Structure

```
data/
├── training/
│   ├── features.csv      # Extracted GIF features
│   └── outcomes.csv      # Compression results
└── models/
    ├── gifsicle_lossy_v1.pkl
    ├── gifsicle_color_v1.pkl
    ├── animately_lossy_v1.pkl
    └── animately_color_v1.pkl
```

## Accuracy Targets

| Curve Type | Engine | Target MAPE |
|------------|--------|-------------|
| Lossy | gifsicle | ≤15% |
| Lossy | animately | ≤15% |
| Color | gifsicle | ≤20% |
| Color | animately | ≤20% |

## Next Steps

1. Run `poetry run python -m giflab predict build-dataset` on your GIF collection
2. Train models with `poetry run python -m giflab predict train`
3. Use predictions in your application
