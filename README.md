# GifLab

Training dataset generation for GIF compression curve prediction.

---

## Overview

GifLab processes raw GIFs through 7 compression engines, measures quality with comprehensive GIF-specific metrics, and produces normalized SQLite datasets for training ML models that predict compression curves from visual features alone.

**Data flow**: Raw GIFs → Visual feature extraction (25 features) → Compression across 7 engines → Quality metrics (13 measures) → SQLite → Training data → Curve prediction models

## Quick Start

```bash
# Install dependencies
poetry install

# Run the compression pipeline (generates dataset)
poetry run python -m giflab run --sampling representative

# Train prediction models on collected data
poetry run python -m giflab train

# Export training data for external use
poetry run python -m giflab export --format csv

# View dataset statistics
poetry run python -m giflab stats
```

## The Dataset

GifLab's output is a SQLite database containing everything needed to train compression curve predictors.

### What gets stored

Each GIF produces:

- **Visual features** (25 per GIF) -- spatial, temporal, compressibility, and CLIP content classification scores stored in the `gif_features` table
- **Compression runs** -- file size, quality metrics, and efficiency scores for every engine/parameter combination stored in `compression_runs`
- **Pipelines and presets** -- the exact engine configurations used, tracked in `pipelines` and `param_presets`

### Training records

A training record pairs a GIF's visual features with its compression curves:

| Field | Description |
|-------|-------------|
| `gif_sha` | Content hash for deduplication |
| `features` | 25-dimensional feature vector (GifFeaturesV1) |
| `lossy_curve` | Quality scores at each lossy level per engine |
| `color_curve` | Quality scores at each palette size per engine |
| `engine` | Which of the 7 engines produced this curve |

### Visual features (25)

- **Spatial**: entropy, edge_density, color_complexity, gradient_smoothness, contrast_score, text_density, dct_energy_ratio, color_histogram_entropy, dominant_color_ratio
- **Temporal**: motion_intensity, motion_smoothness, static_region_ratio, temporal_entropy, frame_similarity, inter_frame_mse_mean, inter_frame_mse_std
- **Compressibility**: lossless_compression_ratio, transparency_ratio
- **CLIP content**: screen_capture, vector_art, photography, hand_drawn, 3d_rendered, pixel_art

### Exporting data

```bash
# Export features table to CSV
poetry run python -m giflab export --format csv

# View database statistics
poetry run python -m giflab stats
```

## ML Pipeline

The prediction pipeline has three stages: collect data, train models, predict curves.

### 1. Collect: `giflab run`

Processes GIFs through all engine/parameter combinations, extracts visual features, computes quality metrics, and stores everything in SQLite.

```bash
# Process a directory of GIFs
poetry run python -m giflab run data/raw --workers 8

# Use research presets for targeted experiments
poetry run python -m giflab run --preset frame-focus

# Intelligent sampling across parameter space
poetry run python -m giflab run --sampling representative
```

### 2. Train: `giflab train`

Trains gradient boosting models (CurvePredictionModel) that learn to predict compression curves from visual features.

```bash
# Train on collected data
poetry run python -m giflab train

# Train for a specific engine
poetry run python -m giflab train --engine animately-advanced
```

The trained models predict two curve types:
- **Lossy curves**: quality score at each lossy compression level
- **Color curves**: quality score at each palette size

### 3. Predict: `giflab predict`

Uses trained models to predict compression curves for new GIFs without running actual compression.

```bash
# Predict curves for a new GIF
poetry run python -m giflab predict --input new.gif

# Predict for a specific engine and curve type
poetry run python -m giflab predict --input new.gif --engine gifsicle --curve-type lossy
```

### Why prediction matters

Running a GIF through all 7 engines at every parameter combination is slow. Prediction lets you estimate compression curves from visual features alone, enabling instant engine/parameter selection without trial-and-error compression.

## Dataset Best Practices

When creating or extending a GifLab dataset for ML tasks, follow these rules:

1. **Deterministic extraction** -- lock random seeds; metric functions must be pure
2. **Schema validation** -- export rows that pass `TrainingRecordV1` (pydantic) validation
3. **Version tagging** -- record dataset version, `giflab` semver, and git commit hash
4. **Canonical data-splits** -- maintain GIF-level `train/val/test` JSON split files
5. **Feature scaling** -- persist `scaler.pkl` (z-score or min-max) alongside data
6. **Missing-value handling** -- encode unknown metrics as `np.nan`, not `0.0`
7. **Outlier and drift reports** -- auto-generate an HTML outlier summary and correlation dashboard
8. **Reproducible pipeline** -- provide a `make data` target that builds the dataset end-to-end
9. **Comprehensive logs** -- include parameter checksum and elapsed-time stats in every run

Pull requests touching dataset code must address all items or explain why they do not apply.

## The 7 Engines

| Engine | Tool | Description |
|--------|------|-------------|
| `gifsicle` | gifsicle | Standard GIF optimizer |
| `animately-standard` | animately | Standard lossy (`--lossy`) |
| `animately-advanced` | animately | Advanced lossy (`--advanced-lossy`) |
| `animately-hard` | animately | Hard mode (`--hard --lossy`) |
| `imagemagick` | convert/magick | ImageMagick lossy compression |
| `ffmpeg` | ffmpeg | FFmpeg GIF encoding |
| `gifski` | gifski | High-quality GIF encoder |

Each engine is measured for file size, quality metrics (13 measures), and efficiency score. All results are stored in SQLite.

### Compression capabilities

| Engine | Color | Frame | Lossy |
|--------|-------|-------|-------|
| **gifsicle** | yes | yes | yes |
| **Animately** | yes | yes | yes |
| **ImageMagick** | yes | yes | yes |
| **FFmpeg** | yes | yes | yes |
| **gifski** | no | no | yes |

## Quality Metrics

GifLab uses a 13-metric quality assessment system across multiple dimensions of compression quality.

### Core metrics

| Metric | Type | Purpose |
|--------|------|---------|
| SSIM | Structural similarity | Primary perceptual quality measure |
| MS-SSIM | Multi-scale similarity | Enhanced structural assessment |
| PSNR | Signal quality | Traditional quality measure |
| MSE/RMSE | Pixel error | Direct difference measurement |
| FSIM | Feature similarity | Gradient and phase feature analysis |
| GMSD | Gradient deviation | Gradient-map based assessment |
| CHIST | Color correlation | Histogram-based color fidelity |
| Edge Similarity | Structural | Edge preservation analysis |
| Texture Similarity | Perceptual | Texture pattern correlation |
| Sharpness Similarity | Visual quality | Sharpness preservation |
| Temporal Consistency | Animation | Frame-to-frame stability |

### SSIM calculation modes

| Mode | Frames Sampled | Use Case |
|------|----------------|----------|
| Fast | 3 keyframes | Large datasets (10,000+ GIFs) |
| Optimized | 10-20 frames | Production pipeline |
| Full | All frames | Research analysis |

### Efficiency scoring

Efficiency score (0-1 scale) uses a geometric mean of quality and compression performance:

```
efficiency = (composite_quality^0.5) * (normalized_compression^0.5)
```

Compression ratio is log-normalized and capped at 20x to handle diminishing returns.

## Research Presets

Research presets provide predefined pipeline combinations for common analysis scenarios.

```bash
# List all available presets
poetry run python -m giflab run --list-presets

# Compare frame reduction algorithms
poetry run python -m giflab run --preset frame-focus

# Compare color quantization methods
poetry run python -m giflab run --preset color-optimization

# Quick testing preset
poetry run python -m giflab run --preset quick-test
```

Available presets: `frame-focus`, `color-optimization`, `lossy-quality-sweep`, `tool-comparison-baseline`, `dithering-focus`, `png-optimization`, `quick-test`.

See [Compression Testing Guide](docs/guides/experimental-testing.md) for details.

## Validation and Debugging

GifLab includes automatic validation during compression that detects quality degradation, efficiency problems, frame reduction issues, disposal artifacts, and temporal consistency failures.

```bash
# View detailed failure analysis
poetry run python -m giflab view-failures results/runs/latest/

# Filter by specific error types
poetry run python -m giflab view-failures results/runs/latest/ --error-type gifski

# Get detailed error information
poetry run python -m giflab view-failures results/runs/latest/ --detailed
```

Validation status levels: PASS, WARNING, ERROR, ARTIFACT, UNKNOWN. The system adjusts thresholds based on content type (animation-heavy, smooth gradients, text/graphics, photo-realistic).

See [Compression Testing Guide](docs/guides/experimental-testing.md) for the full validation reference.

## Source Detection

GifLab automatically detects GIF sources based on directory structure:

```
data/raw/
├── tenor/              # Tenor platform GIFs (subdirs by search query)
├── animately/          # Animately platform uploads (flat)
├── tgif_dataset/       # TGIF research dataset (flat)
└── unknown/            # Unclassified GIFs
```

Source platform and metadata are tracked in the CSV output (`source_platform`, `source_metadata` columns).

See [Directory-Based Source Detection Guide](docs/guides/directory-source-detection.md) for details.

## Pareto Frontier Analysis

Pareto analysis identifies mathematically optimal compression pipelines -- trade-offs where you cannot improve quality without increasing file size, or reduce file size without degrading quality.

```bash
# Run experiments with Pareto analysis
poetry run python -m giflab run --sampling representative

# View top performers
poetry run python -m giflab select-pipelines results/runs/latest/enhanced_streaming_results.csv --top 5
```

Dominated pipelines are automatically identified and can be safely eliminated from consideration.

## Pipeline Usage Examples

```bash
# Standard production processing
poetry run python -m giflab run data/raw --workers 8 --resume

# Test all engines with comprehensive sampling
poetry run python -m giflab run --sampling representative

# Quick test for development
poetry run python -m giflab run --sampling quick

# Select top performing pipelines
poetry run python -m giflab select-pipelines results/runs/latest/enhanced_streaming_results.csv --top 3 -o winners.yaml

# Run with selected pipelines
poetry run python -m giflab run data/raw --pipelines winners.yaml
```

## Project Structure

```
giflab/
├── data/                 # Data directories
├── src/giflab/           # Python package
│   ├── cli/              # CLI command definitions
│   ├── prediction/       # ML prediction pipeline
│   │   ├── features.py   # 25 visual feature extractors
│   │   ├── models.py     # CurvePredictionModel (gradient boosting)
│   │   ├── schemas.py    # Pydantic schemas (GifFeaturesV1, etc.)
│   │   └── dataset.py    # Dataset preparation utilities
│   ├── external_engines/ # Compression engine wrappers
│   ├── monitoring/       # Pipeline monitoring
│   ├── caching/          # Result caching
│   ├── sampling/         # GIF sampling strategies
│   ├── storage.py        # SQLite storage layer
│   └── metrics.py        # Quality metric calculations
├── tests/                # Test suite (4-layer)
├── notebooks/            # Analysis notebooks
├── results/              # Experiment results
└── pyproject.toml        # Poetry configuration
```

## Requirements

- Python 3.11+
- Poetry for dependency management
- FFmpeg for video processing

## Cross-Platform Setup

Engine paths are configurable via environment variables or `src/giflab/config.py` under the `EngineConfig` class.

### Environment variables

```bash
export GIFLAB_GIFSICLE_PATH=/usr/local/bin/gifsicle
export GIFLAB_ANIMATELY_PATH=/usr/local/bin/animately
export GIFLAB_IMAGEMAGICK_PATH=/usr/local/bin/magick
export GIFLAB_FFMPEG_PATH=/usr/local/bin/ffmpeg
export GIFLAB_FFPROBE_PATH=/usr/local/bin/ffprobe
export GIFLAB_GIFSKI_PATH=/usr/local/bin/gifski
```

### Tool installation

**macOS**:
```bash
brew install python@3.11 ffmpeg gifsicle imagemagick gifski
# Animately binary included in repository (bin/darwin/arm64/animately)
```

**Linux/Ubuntu**:
```bash
sudo apt install python3.11 ffmpeg gifsicle imagemagick-6.q16
cargo install gifski
# Animately: download from releases, place in bin/linux/x86_64/
```

**Windows**:
```bash
choco install python ffmpeg gifsicle imagemagick
winget install gifski
# Animately: download from releases, place in bin/windows/x86_64/
```

### Verification

```bash
# Test all engines are properly configured
poetry run python -c "from giflab.system_tools import get_available_tools; print(get_available_tools())"

# Run smoke tests
poetry run pytest tests/smoke/ -v
```

## Testing

Tests are organized into four layers. New tests must go in the correct layer.

| Layer | Path | Purpose | Time budget |
|-------|------|---------|-------------|
| smoke | `tests/smoke/` | Imports, types, pure logic | <5s |
| functional | `tests/functional/` | Mocked engines, synthetic GIFs | <2min |
| integration | `tests/integration/` | Real engines, real metrics | <5min |
| nightly | `tests/nightly/` | Memory, perf, stress, golden | No limit |

```bash
make test           # Fast feedback: smoke + functional (<2min)
make test-ci        # CI: + integration (<5min)
make test-nightly   # Everything including perf/memory
make test-file F=tests/functional/test_metrics.py  # Single file
```

See [Testing Best Practices](docs/guides/testing-best-practices.md) for full guidelines.

## Documentation

### Core
- [Project Scope](SCOPE.md) -- Goals, requirements, and architecture overview

### User Guides
- [Beginner's Guide](docs/guides/beginner.md) -- Step-by-step introduction for new users
- [Setup Guide](docs/guides/setup.md) -- Installation and configuration instructions

### Technical Reference
- [Metrics System](docs/technical/metrics-system.md) -- Quality assessment framework
- [EDA Framework](docs/technical/eda-framework.md) -- Data analysis and visualization tools
- [ML Best Practices](docs/technical/ml-best-practices.md) -- Dataset preparation
- [Content Classification](docs/technical/content-classification.md) -- AI-powered content tagging
- [Testing Best Practices](docs/guides/testing-best-practices.md) -- Quality assurance

### Research and Analysis
- [Compression Research](docs/analysis/compression-research.md) -- Engine comparison and optimization
- [Implementation Lessons](docs/analysis/implementation-lessons.md) -- Development insights

## License

MIT License -- see LICENSE file for details.
