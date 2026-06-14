# GifLab

GIF compression engines + trustworthy quality metrics, and the experiment instrument that ranks the best compression pipeline for each kind of GIF.

---

## Overview

GifLab wears two hats:

1. **Measurement library** — stable `compress()` and `measure()` functions over 7 compression engines and a comprehensive GIF-specific quality metric stack (including the calibrated `composite_quality` verdict). This is the public API the sibling [gifprep](https://github.com/animately/gifprep) repo consumes for its *preprocessing* experiments. See [docs/public-api.md](docs/public-api.md).
2. **Compression-pipeline leaderboard instrument** — a matrix-benchmark harness that fans out compression pipelines (frame / colour / lossy, across engines) and ranks the best pipeline **per GIF content-type** on the quality-vs-size trade-off.

It also extracts visual features and stores compression results in normalised SQLite, intended as ML training data.

> **Status.** The leaderboard instrument is being **rebuilt** — the original was removed in commit `648db9a` — and the ML curve-predictor is a **deferred** follow-on (the `train` CLI is currently a stub; see below). This README marks **(shipped)** vs **(planned)** where it matters. For the direction and the shipped/planned split, see [Compression-Pipeline Leaderboard](docs/technical/compression-pipeline-leaderboard.md).

**Data flow (shipped)**: Raw GIFs → Visual feature extraction (25 features) → Compression across 7 engines → Quality metrics → `composite_quality` verdict → SQLite.

## Quick Start

```bash
# Install dependencies
poetry install

# Run the compression + feature-extraction pipeline over a directory of GIFs
poetry run python -m giflab run data/raw                 # single-engine (quick)
poetry run python -m giflab run data/raw --mode full     # all tool combinations

# Export collected data
poetry run python -m giflab export --db data/giflab.db --output out.csv --table runs

# View dataset statistics
poetry run python -m giflab stats --db data/giflab.db
```

> **Heads-up:** `giflab train` exists but is currently a **stub** (it prints row counts, not a trained model). Real curve-prediction training lives under `giflab predict train` (see [ML Pipeline](#ml-pipeline)). Older docs referenced `--sampling`, `--preset`, `select-pipelines`, and `view-failures` — those commands were removed in `648db9a` and are **not** in the current CLI.

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

## ML Pipeline (partial — predictor deferred)

The longer-term vision is a model that predicts compression curves from visual features alone, so engine/parameter selection is instant without trial-and-error compression. **Current state:** data collection and feature extraction are shipped; end-to-end model training is **not finished** (the top-level `giflab train` command is a stub). Treat this section as direction, not a working pipeline.

### 1. Collect: `giflab run` (shipped)

Processes GIFs through engine/parameter combinations, extracts the 25 visual features, computes quality metrics, and stores everything in SQLite.

```bash
poetry run python -m giflab run data/raw                # single-engine per lossy engine
poetry run python -m giflab run data/raw --mode full    # all frame × colour × lossy combinations
```

### 2. Train (planned — `giflab train` is a stub)

`giflab train` currently loads the training split and prints row counts **without fitting a model**. The real (still-maturing) training code is `src/giflab/prediction/models.py`, exposed via the `predict` subgroup:

```bash
poetry run python -m giflab predict train --dataset data/ --engine animately-advanced
```

### 3. Predict: `giflab predict` (partial)

The `predict` subgroup exposes feature extraction and curve prediction:

```bash
poetry run python -m giflab predict extract-features new.gif      # 25-feature vector
poetry run python -m giflab predict lossy-curve new.gif --engine gifsicle
poetry run python -m giflab predict color-curve new.gif --engine gifsicle
```

> The **compression-pipeline leaderboard** (which pipeline actually wins per content-type) is a *separate* effort from this predictor — see [Compression-Pipeline Leaderboard](docs/technical/compression-pipeline-leaderboard.md). The leaderboard is the near-term goal; the predictor is downstream of it.

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

## Compression-Pipeline Leaderboard (direction — in rebuild)

The near-term goal is a **trustworthy leaderboard** that answers: *for this kind of GIF, which compression pipeline gives the smallest file at acceptable quality?* It fans out every pipeline combination (frame / colour / lossy, across engines, including cross-engine chains), ranks them at an **iso-quality** operating point, and reports the best per content-type.

This experiment was built and run in 2025, then **removed in `648db9a`**; it is being **rebuilt fresh** now that the `composite_quality` metric is trustworthy enough to rank on. The building blocks survive — `src/giflab/dynamic_pipeline.py::generate_all_pipelines()` enumerates ~1,200 pipeline structures, and `scripts/audit/` provides the sweep/metric machinery.

> A previous CLI exposed `--preset`, `--sampling`, `select-pipelines`, and `view-failures`. **Those were removed in `648db9a` and are not in the current CLI.** The concepts (sampling, Pareto elimination, presets) return in the rebuild under new names.

See [Compression-Pipeline Leaderboard](docs/technical/compression-pipeline-leaderboard.md) for the full design (deliverable, the quality/size pivot, two-stage screening, validation gate, taxonomy) and the shipped-vs-planned split.

## Validation

GifLab validates compression results in-pipeline (`src/giflab/wrapper_validation/`), detecting quality degradation, efficiency problems, frame-reduction issues, disposal artifacts, and temporal-consistency failures. Validation status levels: PASS, WARNING, ERROR, ARTIFACT, UNKNOWN. Thresholds adapt to content type (animation-heavy, smooth gradients, text/graphics, photo-realistic).

> The standalone `giflab view-failures` CLI viewer was removed in `648db9a`. Validation still runs as part of the pipeline; the failure-inspection CLI is not currently exposed.

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

## Pareto Frontier Analysis (planned — part of the leaderboard)

Pareto analysis identifies optimal compression pipelines — trade-offs where you cannot improve quality without increasing file size, or vice versa — and eliminates dominated pipelines. This is a core mechanism of the [Compression-Pipeline Leaderboard](docs/technical/compression-pipeline-leaderboard.md) (used both as a Pareto-frequency cross-check and to prune the search). The previous implementation (`core/pareto.py`, `select-pipelines`) was removed in `648db9a` and is being rebuilt.

## Pipeline Usage Examples

```bash
# Single-engine pass over a directory (resume-safe by default; skips already-done GIFs)
poetry run python -m giflab run data/raw

# All frame × colour × lossy tool combinations
poetry run python -m giflab run data/raw --mode full

# Force re-processing of everything / re-process only outdated GIFs
poetry run python -m giflab run data/raw --force
poetry run python -m giflab run data/raw --upgrade

# Inspect the collected dataset
poetry run python -m giflab stats --db data/giflab.db
poetry run python -m giflab export --db data/giflab.db --output runs.csv --table runs
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

### Test Fixtures

Test fixture GIFs (`tests/fixtures/*.gif`) are gitignored.  After a fresh
clone or in a new git worktree, regenerate them before running tests:

```bash
make fixtures       # Regenerate all test GIF fixtures (deterministic, no external tools)
```

### Test Layers

Tests are organized into four layers. New tests must go in the correct layer.

| Layer | Path | Purpose | Time budget |
|-------|------|---------|-------------|
| smoke | `tests/smoke/` | Imports, types, pure logic | <5s |
| functional | `tests/functional/` | Mocked engines, synthetic GIFs | <2min |
| integration | `tests/integration/` | Real engines, real metrics | <5min |
| nightly | `tests/nightly/` | Memory, perf, stress, golden | No limit |

```bash
make fixtures       # Regenerate gitignored GIF fixtures (run once after clone / worktree)
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
