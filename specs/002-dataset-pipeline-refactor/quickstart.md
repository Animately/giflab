# Quickstart: Dataset Pipeline Refactor

**Feature**: 002-dataset-pipeline-refactor
**Date**: 2026-02-09
**Purpose**: User guide for the unified GifLab pipeline.

## Prerequisites

```bash
# Install Poetry (if needed)
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Verify setup
poetry run python -c "import giflab; print('GifLab ready!')"

# Verify engine availability
poetry run python -c "from giflab.system_tools import get_available_tools; print(get_available_tools())"
```

### Required Engine Binaries

| Engine | Binary | Install (macOS) |
|--------|--------|----------------|
| gifsicle | `gifsicle` | `brew install gifsicle` |
| animately | `animately` | In-house tool at `/Users/lachlants/bin/` |
| imagemagick | `convert`/`magick` | `brew install imagemagick` |
| ffmpeg | `ffmpeg` | `brew install ffmpeg` |
| gifski | `gifski` | `brew install gifski` |

## Basic Workflow

### 1. Generate Training Dataset

```bash
# Single mode (default): each lossy engine independently
# 11 real lossy tools × 7 levels = 77 runs per GIF
poetry run python -m giflab run data/raw/

# Full mode: all pipeline combinations
# 5 frame × 20 color × 12 lossy = 1,200 combinations per GIF
poetry run python -m giflab run data/raw/ --mode full

# Force reprocess all GIFs
poetry run python -m giflab run data/raw/ --force

# Reprocess only outdated GIFs
poetry run python -m giflab run data/raw/ --upgrade

# Custom database location
poetry run python -m giflab run data/raw/ --db results/experiment.db
```

### 2. Check Database Statistics

```bash
# Show counts per engine, GIF, and pipeline
poetry run python -m giflab stats

# Stats for custom database
poetry run python -m giflab stats --db results/experiment.db
```

### 3. Export Data

```bash
# Export to CSV for analysis
poetry run python -m giflab export --format csv --output results.csv

# Export to JSON
poetry run python -m giflab export --format json --output results.json

# Export from custom database
poetry run python -m giflab export --db results/experiment.db --format csv --output results.csv
```

### 4. Train Prediction Models

```bash
# Train models from database
poetry run python -m giflab train

# Train for a specific engine
poetry run python -m giflab train --engine animately-standard
```

### 5. Predict Compression Curves

```bash
# Predict curves for a single GIF
poetry run python -m giflab predict path/to/file.gif

# Predict for a specific engine
poetry run python -m giflab predict path/to/file.gif --engine gifsicle
```

## CLI Reference

```
giflab
├── run          Generate training dataset from GIFs
│   ├── INPUT_DIR      Directory containing GIF files
│   ├── --db PATH      SQLite database path (default: data/giflab.db)
│   ├── --mode TEXT    "single" (per-engine) or "full" (all combinations)
│   ├── --force        Reprocess all GIFs
│   └── --upgrade      Reprocess outdated GIFs only
│
├── train        Train compression prediction models
│   ├── --db PATH      SQLite database path
│   └── --engine TEXT  Train for specific engine only
│
├── predict      Predict compression curves for a GIF
│   ├── GIF_PATH       Path to GIF file
│   ├── --db PATH      SQLite database path
│   └── --engine TEXT  Predict for specific engine only
│
├── export       Export database to CSV or JSON
│   ├── --db PATH      SQLite database path
│   ├── --format TEXT  "csv" or "json"
│   └── --output PATH  Output file path
│
└── stats        Show database statistics
    └── --db PATH      SQLite database path
```

## The 7 Engines

| Engine | Description | Best For |
|--------|-------------|----------|
| `gifsicle` | Standard GIF optimizer | General-purpose, fast |
| `animately-standard` | Animately's standard lossy | Balanced quality/size |
| `animately-advanced` | PNG-sequence based | High-quality animations |
| `animately-hard` | Aggressive compression | Maximum size reduction |
| `imagemagick` | ImageMagick lossy | Versatile, many options |
| `ffmpeg` | FFmpeg GIF encoding | Video-derived content |
| `gifski` | High-quality encoder | Photographic content |

## Pipeline Modes

### Single Mode (Default)

Runs each lossy engine independently with no frame or color reduction:
- 7 engines × 7 lossy levels = **49 runs per GIF**
- Fast, good for initial dataset building

### Full Mode

Runs all 3-slot pipeline combinations:
- 5 frame tools × 20 color tools × 12 lossy tools = **1,200 combinations per GIF**
- Each at 7 lossy levels = **8,400 parameter variations per GIF**
- Slow but comprehensive

## Database Schema

All results are stored in SQLite at `data/giflab.db` (default):

```
gifs              → GIF files with extracted features
tools             → Registered compression tools
pipelines         → 3-slot pipeline combinations
param_presets     → Parameter combinations (lossy level, colors, frame ratio)
compression_runs  → Results per GIF × pipeline × params
failures          → Failed compression attempts
```

## Troubleshooting

### "No engines found"
Ensure engine binaries are installed and in PATH:
```bash
which gifsicle animately convert ffmpeg gifski
```

### "ModuleNotFoundError"
Always use `poetry run`:
```bash
poetry run python -m giflab run data/raw/
```

### Database locked
Another process may be using the database. Wait and retry, or specify a different `--db` path.

### Animately failures
Ensure you're using flag-based arguments:
```bash
# Correct
animately --input in.gif --output out.gif --lossy 60

# Wrong (will fail silently)
animately in.gif out.gif --lossy 60
```
