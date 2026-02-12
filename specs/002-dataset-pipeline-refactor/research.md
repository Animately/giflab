# Research: Dataset Pipeline Refactor

**Feature**: 002-dataset-pipeline-refactor
**Date**: 2026-02-09
**Purpose**: Document the engine landscape, combination matrix, and current architecture for the refactoring.

## Engine Analysis

### The 7 Lossy Compression Engines

| # | Engine Name | Binary | CLI Flags | Level Range | Notes |
|---|------------|--------|-----------|-------------|-------|
| 1 | `gifsicle` | `gifsicle` | `--lossy=<level>` | 0-200 | Standard GIF optimizer |
| 2 | `animately-standard` | `animately` | `--input <in> --output <out> --lossy <level>` | 0-200 | Animately's standard lossy |
| 3 | `animately-advanced` | `animately` | `--input <in> --output <out> --advanced-lossy <level>` | 0-200 | PNG-sequence based |
| 4 | `animately-hard` | `animately` | `--input <in> --output <out> --hard --lossy <level>` | 0-200 | Hard mode + lossy |
| 5 | `imagemagick` | `convert`/`magick` | quality parameter | 0-100 | ImageMagick lossy |
| 6 | `ffmpeg` | `ffmpeg` | q_scale mapping | 0-100 | FFmpeg GIF encoding |
| 7 | `gifski` | `gifski` | `--quality <level>` | 1-100 | High-quality GIF encoder |

### Animately Flag Details

Animately is a single binary (`animately`) with 3 compression modes:

```bash
# Standard lossy (animately-standard)
animately --input in.gif --output out.gif --lossy 60

# Advanced lossy (animately-advanced)
animately --input in.gif --output out.gif --advanced-lossy 60

# Hard lossy (NEW: animately-hard)
animately --input in.gif --output out.gif --hard --lossy 60
```

**Critical**: Animately requires flag-based arguments (`--input`/`--output`), not positional.

### Current Wrapper State

| NAME | Class | Notes |
|------|-------|-------|
| `animately-standard` | `AnimatelyLossyCompressor` | Standard lossy |
| `animately-advanced` | `AnimatelyAdvancedLossyCompressor` | Advanced lossy |
| `animately-hard` | `AnimatelyHardLossyCompressor` | Hard mode lossy |

## Pipeline Combination Matrix

### Slot 1: Frame Reduction (5 tools)

| # | Tool Name | Class | Notes |
|---|-----------|-------|-------|
| 1 | `gifsicle-frame` | `GifsicleFrameReducer` | |
| 2 | `animately-frame` | `AnimatelyFrameReducer` | |
| 3 | `imagemagick-frame` | `ImageMagickFrameReducer` | |
| 4 | `ffmpeg-frame` | `FFmpegFrameReducer` | |
| 5 | `none-frame` | `NoOpFrameReducer` | No frame reduction |

### Slot 2: Color Reduction (20 tools)

| # | Tool Name | Class | Notes |
|---|-----------|-------|-------|
| 1 | `gifsicle-color` | `GifsicleColorReducer` | |
| 2 | `animately-color` | `AnimatelyColorReducer` | |
| 3 | `imagemagick-color` | `ImageMagickColorReducer` | Default (None) dithering |
| 4 | `imagemagick-color-riemersma` | `ImageMagickColorReducerRiemersma` | |
| 5 | `imagemagick-color-floyd` | `ImageMagickColorReducerFloydSteinberg` | |
| 6 | `ffmpeg-color` | `FFmpegColorReducer` | Default (none) dithering |
| 7 | `ffmpeg-color-sierra2` | `FFmpegColorReducerSierra2` | |
| 8 | `ffmpeg-color-sierra2-4a` | `FFmpegColorReducerSierra2_4a` | FFmpeg default dithering |
| 9 | `ffmpeg-color-sierra3` | `FFmpegColorReducerSierra3` | Full Sierra v3 |
| 10 | `ffmpeg-color-floyd` | `FFmpegColorReducerFloydSteinberg` | |
| 11 | `ffmpeg-color-burkes` | `FFmpegColorReducerBurkes` | Burkes error diffusion |
| 12 | `ffmpeg-color-atkinson` | `FFmpegColorReducerAtkinson` | Classic Apple dithering |
| 13 | `ffmpeg-color-heckbert` | `FFmpegColorReducerHeckbert` | Simple error diffusion |
| 14 | `ffmpeg-color-bayer0` | `FFmpegColorReducerBayerScale0` | |
| 15 | `ffmpeg-color-bayer1` | `FFmpegColorReducerBayerScale1` | |
| 16 | `ffmpeg-color-bayer2` | `FFmpegColorReducerBayerScale2` | |
| 17 | `ffmpeg-color-bayer3` | `FFmpegColorReducerBayerScale3` | |
| 18 | `ffmpeg-color-bayer4` | `FFmpegColorReducerBayerScale4` | |
| 19 | `ffmpeg-color-bayer5` | `FFmpegColorReducerBayerScale5` | |
| 20 | `none-color` | `NoOpColorReducer` | No color reduction |

### Slot 3: Lossy Compression (12 tools)

| # | Tool Name | Class | Notes |
|---|-----------|-------|-------|
| 1 | `gifsicle-lossy` | `GifsicleLossyCompressor` | Default O2 optimization |
| 2 | `gifsicle-lossy-basic` | `GifsicleLossyBasic` | Basic optimization |
| 3 | `gifsicle-lossy-O1` | `GifsicleLossyO1` | Level 1 optimization |
| 4 | `gifsicle-lossy-O2` | `GifsicleLossyO2` | Level 2 optimization |
| 5 | `gifsicle-lossy-O3` | `GifsicleLossyO3` | Level 3 optimization |
| 6 | `animately-standard` | `AnimatelyLossyCompressor` | Standard lossy |
| 7 | `animately-advanced` | `AnimatelyAdvancedLossyCompressor` | Advanced lossy |
| 8 | `animately-hard` | `AnimatelyHardLossyCompressor` | Hard mode lossy |
| 9 | `imagemagick-lossy` | `ImageMagickLossyCompressor` | |
| 10 | `ffmpeg-lossy` | `FFmpegLossyCompressor` | |
| 11 | `gifski-lossy` | `GifskiLossyCompressor` | |
| 12 | `none-lossy` | `NoOpLossyCompressor` | No lossy compression |

### Combination Count

- Frame reduction: 5 tools
- Color reduction: 20 tools
- Lossy compression: 12 tools
- **Total**: 5 × 20 × 12 = **1,200 combinations** per GIF

### COMBINE_GROUP Constraints

Tools use `COMBINE_GROUP` to prevent cross-engine combinations in a single pipeline step. For example, gifsicle frame reduction can combine with any color/lossy tool, but the `COMBINE_GROUP` prevents nonsensical same-slot conflicts.

## Current Architecture Analysis

### Two Parallel Systems

**System 1: Experiment Runner** (`core/runner.py`)
- Purpose: Systematic pipeline testing and analysis
- Output: CSV files (streaming_results.csv, elimination CSVs, pareto CSVs)
- Features: Pipeline elimination, content-type analysis, Pareto frontiers
- Entry: `GifLabRunner.run()`

**System 2: Prediction Runner** (`prediction_runner.py`)
- Purpose: Feature extraction + single/full pipeline execution
- Output: SQLite database
- Features: Feature extraction, pipeline registration, compression runs
- Entry: `run_prediction_pipeline()`

### What the Unified Runner Inherits

From **Prediction Runner**:
- SQLite storage via `GifLabStorage`
- Feature extraction integration
- Pipeline registration from `dynamic_pipeline.py`
- Single/full mode selection
- Force/upgrade flags

From **Experiment Runner**:
- Pareto frontier analysis (optional post-processing)
- Progress tracking and interval saving
- Memory monitoring integration

### Quality Metrics Ecosystem

The metrics system is comprehensive and should not be modified — only integrated:

| Module | Purpose | Integration Point |
|--------|---------|-------------------|
| `metrics.py` | Core orchestrator (SSIM, PSNR, disposal) | Called per compression result |
| `enhanced_metrics.py` | Composite scoring | Post-processing aggregation |
| `optimized_metrics.py` | Vectorized performance | Used for batch processing |
| `parallel_metrics.py` | Multi-core frames | Used for large GIFs |
| `conditional_metrics.py` | Cost-aware selection | Used for time-constrained runs |
| `deep_perceptual_metrics.py` | LPIPS neural quality | Optional GPU-accelerated |
| `temporal_artifacts.py` | Flicker, pumping | GIF-critical detection |
| `gradient_color_artifacts.py` | Banding, posterization | GIF-critical detection |
| `text_ui_validation.py` | Text readability | GIF-critical for UI content |
| `ssimulacra2_metrics.py` | Modern perceptual | High-quality perceptual metric |

### CSV Usage to Remove

| File | CSV Usage | Replacement |
|------|-----------|-------------|
| `core/runner.py` | `streaming_results.csv`, elimination CSVs, pareto CSVs | SQLite via `storage.py` |
| `prediction/dataset.py` | Features CSV export | SQLite + `export` command |
| `prediction/cli.py` | Batch-extract CSV | Merged into main CLI, SQLite |
| `tag_pipeline.py` | Tagging results CSV | SQLite |
| `io.py` | CSV writing utilities | Keep utility, redirect to SQLite |
| `cli/utils.py` | CSV helpers | Keep utility, redirect to export |

## Directories to Remove

| Directory | Files | Reason |
|-----------|-------|--------|
| `benchmarks/` | 5 files | A/B testing suites, never imported |
| `config_profiles/` | 8 files | Deployment configs, never imported |
| `dashboard/` | 0 files | Empty directory |
| `results/` | 158 runs | Old experiment data, not needed for code |
