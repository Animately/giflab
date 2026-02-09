# 🎞️ GifLab

GIF compression and analysis laboratory for systematic performance evaluation.

---

## Overview

GifLab analyzes GIF compression by generating variants with configurable parameters across multiple dimensions:

- **Frame reduction**: Configurable keep ratios from full frames to aggressive reduction
- **Color quantization**: Flexible palette sizes and dithering algorithms  
- **Lossy compression**: Variable quality levels mapped to engine-specific implementations
- **Engines**: **gifsicle**, **Animately**, **ImageMagick**, **FFmpeg**, **gifski**

Each variant is measured for file size, comprehensive quality metrics (11 different measures), render time, and efficiency score.

### Compression Pipeline

GifLab provides a comprehensive compression pipeline with access to all major GIF processing engines:

- **Purpose**: Systematic testing, pipeline analysis, and optimization discovery
- **Usage**: `poetry run python -m giflab run --sampling representative`
- **Focused Presets**: `poetry run python -m giflab run --preset frame-focus`

| Engine | Color | Frame | Lossy |
|--------|-------|-------|--------|
| **gifsicle** | ✅ | ✅ | ✅ |
| **Animately** | ✅ | ✅ | ✅ |
| **ImageMagick** | ✅ | ✅ | ✅ |
| **FFmpeg** | ✅ | ✅ | ✅ |
| **gifski** | ❌ | ❌ | ✅ |

## 🎯 **Research Presets**

Research presets provide predefined pipeline combinations for common analysis scenarios, and you can create custom presets for specific research needs.

### Quick Start with Presets

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

### Available Research Presets

Built-in presets cover common research scenarios:
- **`frame-focus`**: Compare frame reduction algorithms
- **`color-optimization`**: Compare color quantization methods
- **`lossy-quality-sweep`**: Evaluate lossy compression effectiveness
- **`tool-comparison-baseline`**: Cross-engine comparison
- **`dithering-focus`**: Compare dithering algorithms
- **`png-optimization`**: PNG sequence optimization
- **`quick-test`**: Fast development testing

### Custom Preset Creation

You can create custom presets by defining your own pipeline combinations. The preset system supports flexible parameter configurations for targeted research studies.

### Analysis Workflow

GifLab provides a comprehensive analysis workflow for systematic optimization:

1. **Explore** – Generate synthetic GIFs and test pipeline combinations with intelligent sampling or research presets
2. **Validate** – Automatic validation system detects compression issues and data integrity problems
3. **Analyze** – Use built-in analysis tools to identify optimal pipelines for your content type
4. **Scale** – Apply validated pipelines to full datasets

Workflow summary:

| Step | Command | Purpose |
|------|---------|---------|
| 1. Explore | `giflab run --sampling representative` | Test pipeline combinations, eliminate underperformers |
| 2. Validate | Automatic during processing | Quality validation, artifact detection, parameter verification |
| 3. Analyze | `giflab select-pipelines results.csv --top 3` | Identify optimal pipelines using Pareto analysis |
| 4. Scale | `giflab run data/raw/ --pipelines winners.yaml` | Apply validated pipelines to full datasets |

The system includes comprehensive validation to ensure data integrity throughout the compression pipeline.

## 🔍 **Automated Validation & Debugging**

GifLab includes a comprehensive validation system that automatically detects compression issues and provides detailed debugging information to help identify problems quickly.

### Pipeline Validation System

The optimization validation system runs automatically during compression and detects:

- **Quality Degradation**: Monitors composite quality scores against configurable thresholds
- **Efficiency Problems**: Validates efficiency scores and compression ratios
- **Frame Reduction Issues**: Detects incorrect frame counts and FPS inconsistencies  
- **Disposal Artifacts**: Identifies GIF disposal method artifacts and corruption
- **Temporal Consistency**: Validates frame-to-frame stability in animations
- **Multi-metric Combinations**: Detects unusual metric patterns that indicate problems

### Validation Status Levels

| Status | Description | Action |
|--------|-------------|--------|
| **PASS** | All validation checks passed | Continue processing |
| **WARNING** | Minor issues detected | Compression acceptable, review settings |
| **ERROR** | Significant issues found | Compression problematic, investigate |
| **ARTIFACT** | Disposal artifacts detected | Severe corruption, pipeline failure |
| **UNKNOWN** | Validation could not run | System error, check logs |

### Content-Type Aware Validation

The system adjusts validation thresholds based on content type:
- **Animation Heavy**: Different frame reduction tolerances
- **Smooth Gradients**: Adjusted quality thresholds for lossy compression
- **Text/Graphics**: Stricter artifact detection
- **Photo-realistic**: Enhanced temporal consistency checks

### Debugging Failed Pipelines

When validation detects issues, use these commands for debugging:

```bash
# View detailed failure analysis
poetry run python -m giflab view-failures results/runs/latest/

# Filter by specific error types
poetry run python -m giflab view-failures results/runs/latest/ --error-type gifski

# Get detailed error information
poetry run python -m giflab view-failures results/runs/latest/ --detailed
```

The validation system helps catch problems early so you can trace issues back to specific pipeline configurations and fix them quickly.

### The Problem
Different GIF compression tools have varying performance characteristics across content types. Optimal tool selection depends on GIF characteristics such as color complexity, animation patterns, and visual content type.

### The Solution
GifLab's compression pipeline tests multiple algorithmic combinations to:

1. Build datasets of GIF characteristics paired with optimal tool combinations
2. Train ML models to predict optimal compression strategies
3. Create automated tool selection based on content analysis
4. Continuously improve through expanded testing

### Research Approach
- Comprehensive multi-engine testing (gifsicle, Animately, ImageMagick, FFmpeg, gifski)
- Automated content type classification and detection
- Performance prediction modeling for compression/quality trade-offs
- Novel tool combination and parameter optimization

*See [ML Strategy Documentation](docs/technical/ml-strategy.md) for detailed implementation plans.*

### Integration with Production Workflows
After analysis validation, apply findings to your production compression pipeline:
```bash
# 1. Run experiments to identify optimal strategy
poetry run python -m giflab run --sampling representative

# 2. Select top performing pipelines
poetry run python -m giflab select-pipelines results/runs/latest/enhanced_streaming_results.csv --top 3 -o winners.yaml

# 3. Run full pipeline with optimized settings
poetry run python -m giflab run data/raw --pipelines winners.yaml
```

📖 **For detailed documentation, see:** [Compression Testing Guide](docs/guides/experimental-testing.md)

## 🗂️ Directory-Based Source Detection

GifLab automatically detects GIF sources based on directory structure, making it easy to organize and analyze GIFs from different platforms:

### Directory Structure
```
data/raw/
├── tenor/              # GIFs from Tenor platform
│   ├── love/           # "love" search results
│   ├── marketing/      # "marketing" search results
│   └── email_campaign/ # Email campaign GIFs
├── animately/          # GIFs from Animately platform (all user uploads)
│   ├── user_upload_1.gif
│   ├── user_upload_2.gif
│   └── user_upload_3.gif
├── tgif_dataset/       # GIFs from TGIF research dataset
│   ├── research_gif_1.gif
│   ├── research_gif_2.gif
│   └── research_gif_3.gif
└── unknown/            # Ungrouped GIFs
```

### Platform Naming Convention

| Platform | Directory Name | Type | Notes |
|----------|----------------|------|-------|
| Tenor | `tenor/` | Live Platform | Google's GIF search platform |
| Animately | `animately/` | Live Platform | Your compression platform |
| TGIF Dataset | `tgif_dataset/` | Research Dataset | Academic research dataset |
| Unknown | `unknown/` | Fallback | Unclassified or mixed sources |

**Why "tgif_dataset"?** The "_dataset" suffix distinguishes research datasets from live platforms, making the data source clear. Both `tgif/` and `tgif_dataset/` are supported.

**Why flat structure for some platforms?** Animately and TGIF files have uniform characteristics within each platform, so subdirectories don't add meaningful organization. Tenor search queries, however, create content differences worth preserving in the directory structure.

### Quick Start
```bash
# 1. Create directory structure
poetry run python -m giflab organize-directories data/raw/

# 2. Move GIFs to appropriate directories
# (manually or via collection scripts)

# 3. Run analysis with automatic source detection
poetry run python -m giflab run data/raw/

# 4. Optional: Disable source detection
poetry run python -m giflab run data/raw/ --no-detect-source-from-directory
```

### CSV Output
The resulting CSV includes source tracking columns:
- `source_platform`: Platform identifier (tenor, animately, tgif_dataset, unknown)
- `source_metadata`: JSON metadata with query, context, and collection details

📖 **For detailed documentation, see:** [Directory-Based Source Detection Guide](docs/guides/directory-source-detection.md)

## Comprehensive Quality Analysis

GifLab uses an 11-metric quality assessment system that evaluates multiple dimensions of compression quality:

### Core Quality Metrics

| Metric | Type | Purpose |
|--------|------|---------|
| **SSIM** | Structural similarity | Primary perceptual quality measure |
| **MS-SSIM** | Multi-scale similarity | Enhanced structural assessment |
| **PSNR** | Signal quality | Traditional quality measure |
| **MSE/RMSE** | Pixel error | Direct difference measurement |
| **FSIM** | Feature similarity | Gradient and phase feature analysis |
| **GMSD** | Gradient deviation | Gradient-map based assessment |
| **CHIST** | Color correlation | Histogram-based color fidelity |
| **Edge Similarity** | Structural | Edge preservation analysis |
| **Texture Similarity** | Perceptual | Texture pattern correlation |
| **Sharpness Similarity** | Visual quality | Sharpness preservation |
| **Temporal Consistency** | Animation | Frame-to-frame stability |

### SSIM Calculation Modes

| Mode | Frames Sampled | Processing Time | Use Case |
|------|----------------|-----------------|----------|
| **Fast** | 3 keyframes | ~5ms | Large datasets (10,000+ GIFs) |
| **Optimized** | 10-20 frames | ~6-8ms | Production pipeline |
| **Full** | All frames | ~12ms | Research analysis |

### Sampling Strategies

- **Uniform**: Evenly distributed frames across the GIF
- **Keyframe**: Strategic positions (start, 1/4, 1/2, 3/4, end) + uniform fill
- **Adaptive**: Content-aware sampling (future enhancement)

### Configuration Examples

```python
from giflab.config import MetricsConfig

# For large datasets (speed priority)
fast_config = MetricsConfig(
    SSIM_MODE="fast",                    # 3 frames only
    CALCULATE_FRAME_METRICS=False        # Skip detailed metrics
)

# For production pipeline (balanced)
production_config = MetricsConfig(
    SSIM_MODE="optimized",               # Sampled frames
    SSIM_MAX_FRAMES=15,                  # Good balance
    SSIM_SAMPLING_STRATEGY="uniform",    # Representative sampling
    CALCULATE_FRAME_METRICS=True         # Keep detailed metrics
)

# For research (accuracy priority)
research_config = MetricsConfig(
    SSIM_MODE="full",                    # All frames
    CALCULATE_FRAME_METRICS=True         # Full analysis
)
```

### Dataset Processing Estimates

| Dataset Size | Avg Frames | Fast Mode | Optimized Mode | Full Mode |
|-------------|------------|-----------|----------------|-----------|
| 1,000 GIFs  | 50         | **0.03h** | 0.15h         | 0.8h      |
| 5,000 GIFs  | 100        | **0.3h**  | 1.5h          | 8.3h      |
| 10,000 GIFs | 100        | **0.6h**  | 3.0h          | 16.7h     |

*Processing times include 24 variants per GIF (frame ratios × color counts × lossy levels × engines)*

## Expanded Quality Metrics (S10)

In addition to SSIM-based measures, GifLab now computes **eight complementary frame-level metrics** that capture color, texture, edge and gradient fidelity:

| Metric | Abbrev. | Higher-is-better? | Notes |
|--------|---------|-------------------|-------|
| Mean Squared Error | MSE | ❌ (lower) | Pixel-wise error |
| Root Mean Squared Error | RMSE | ❌ (lower) | Square-root of MSE |
| Feature Similarity Index | FSIM | ✅ | Gradient & phase features |
| Gradient Magnitude Similarity Deviation | GMSD | ❌ (lower) | Gradient-map deviation |
| Color-Histogram Correlation | CHIST | ✅ | Channel-wise histogram correlation |
| Edge-Map Jaccard Similarity | EDGE | ✅ | Edge overlap similarity |
| Texture-Histogram Correlation | TEXTURE | ✅ | LBP histogram correlation |
| Sharpness Similarity | SHARP | ✅ | Laplacian variance ratio |

All metrics are exposed via `giflab.metrics.calculate_comprehensive_metrics` and exported to CSV with mean/std/min/max descriptors (plus optional raw values).

## 🎯 Efficiency Scoring System

GifLab calculates an **efficiency score** (0-1 scale) using a geometric mean of quality and compression performance:

```python
efficiency = (composite_quality^0.5) × (normalized_compression^0.5)
```

### Technical Implementation:
- **Geometric mean**: Balanced approach requiring both quality preservation and compression effectiveness
- **Log-normalized compression**: Compression ratio capped at 20x to handle diminishing returns
- **Composite quality input**: Uses the comprehensive 11-metric quality system
- **Range**: 0-1 scale for consistent interpretation


## 🎯 Pareto Frontier Analysis

GifLab uses Pareto frontier analysis to identify mathematically optimal compression pipelines by finding trade-offs where you cannot improve quality without increasing file size, or reduce file size without degrading quality.

### Mathematical Optimality

Pareto analysis eliminates subjective weighting by identifying pipelines that are mathematically optimal for quality-size trade-offs.

### Pipeline Elimination Benefits

1. Eliminates subjective quality-vs-size weighting decisions
2. Provides mathematically rigorous pipeline comparisons
3. Supports content-type specific analysis
4. Works with multiple quality metrics (SSIM, MS-SSIM, composite scores)

### Experimental Commands

```bash
# Run comprehensive experiments with Pareto analysis
poetry run python -m giflab run --sampling representative

# View experiment results and top performers
poetry run python -m giflab select-pipelines results/runs/latest/enhanced_streaming_results.csv --top 5

# Use quick sampling for faster testing during development
poetry run python -m giflab run --sampling quick
```

### Interpretation Guide

Pareto frontier points represent optimal trade-offs:
- Leftmost: Maximum quality achievable
- Rightmost: Maximum compression at acceptable quality  
- Middle: Balanced quality-size compromises

Dominated pipelines are automatically identified and can be safely eliminated from consideration.



## Quick Start

```bash
# Install dependencies (requires Poetry)
poetry install

# Test all compression strategies with intelligent sampling
poetry run python -m giflab run --sampling representative

# Pick top 3 pipelines by SSIM
poetry run python -m giflab select-pipelines results/runs/latest/enhanced_streaming_results.csv --top 3 -o winners.yaml

# Run production compression on full dataset with chosen pipelines
poetry run python -m giflab run data/raw --pipelines winners.yaml

# Add AI-generated tags
poetry run python -m giflab tag results.csv data/raw
```

## Pipeline Usage Examples

### 🏭 Standard Workflow (Reliable, Fast)
```bash
# Standard production processing with gifsicle + Animately
poetry run python -m giflab run data/raw

# Production with custom settings
poetry run python -m giflab run data/raw --workers 8 --resume

# Production with pipeline YAML (advanced)
poetry run python -m giflab run data/raw --pipelines custom_pipelines.yaml
```

### 🧪 Comprehensive Testing (All 5 Engines)
```bash
# Test all 5 engines with comprehensive sampling
poetry run python -m giflab run --sampling representative

# Quick test for development
poetry run python -m giflab run --sampling quick

# Full comprehensive testing (slower but thorough)
poetry run python -m giflab run --sampling full

# Targeted testing with strategic synthetic GIFs
poetry run python -m giflab run --sampling targeted
```


### 📊 Workflow Recommendations

**For Production Work:**
```bash
# Large datasets, proven reliability
poetry run python -m giflab run data/raw --workers 8 --resume
```

**For Engine Comparison:**
```bash  
# Compare all engines to find the best for your content
poetry run python -m giflab run --sampling representative
poetry run python -m giflab select-pipelines results/runs/latest/enhanced_streaming_results.csv --top 3
```

**For Research & Development:**
```bash
# Full comprehensive testing with advanced analysis
poetry run python -m giflab run --sampling full --use-gpu
```

## Project Structure

```
giflab/
├─ data/                 # Data directories
├─ src/giflab/           # Python package
├─ notebooks/            # Analysis notebooks  
├─ tests/                # Test suite
└─ pyproject.toml        # Poetry configuration
```


## Requirements

- Python 3.11+
- Poetry for dependency management
- FFmpeg for video processing

## Cross-Platform Setup

Engine paths are configurable via environment variables or `src/giflab/config.py` under the `EngineConfig` class.

### Environment Variables (Recommended)
Set these environment variables to override default engine paths:
```bash
export GIFLAB_GIFSICLE_PATH=/usr/local/bin/gifsicle
export GIFLAB_ANIMATELY_PATH=/usr/local/bin/animately
export GIFLAB_IMAGEMAGICK_PATH=/usr/local/bin/magick
export GIFLAB_FFMPEG_PATH=/usr/local/bin/ffmpeg
export GIFLAB_FFPROBE_PATH=/usr/local/bin/ffprobe
export GIFLAB_GIFSKI_PATH=/usr/local/bin/gifski
```

### Tool Installation by Platform

#### macOS
```bash
# Install tools via Homebrew
brew install python@3.11 ffmpeg gifsicle imagemagick

# Install gifski for high-quality lossy compression
brew install gifski

# Animately binary included in repository (bin/darwin/arm64/animately)
# Automatically detected - no additional installation required
```

#### Linux/Ubuntu
```bash
# Install via package manager
sudo apt update
sudo apt install python3.11 ffmpeg gifsicle imagemagick-6.q16

# Install gifski via cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install gifski

# Animately: Download from releases and place in bin/linux/x86_64/
# See bin/linux/x86_64/PLACEHOLDER.md for instructions
```

#### Windows
```bash
# Using Chocolatey
choco install python ffmpeg gifsicle imagemagick

# Install gifski via cargo or download binary
winget install gifski

# Animately: Download from releases and place in bin/windows/x86_64/
# See bin/windows/x86_64/PLACEHOLDER.md for instructions
```

#### Repository Binaries
GifLab includes platform-specific binaries for tools not available via package managers:
- **Animately**: Pre-built binaries in `bin/<platform>/<architecture>/`
- **Automatic detection**: Tool discovery checks repository binaries before system PATH
- **Cross-platform support**: macOS ARM64, Linux x86_64, Windows x86_64

### Verification
```bash
# Test all engines are properly configured
poetry run python -c "from giflab.system_tools import get_available_tools; print(get_available_tools())"

# Run smoke tests to verify functionality
poetry run pytest tests/smoke/ -v
```

## 📈 Machine-Learning Dataset Best Practices  
*Why this matters: High-quality, well-documented metrics are the foundation of reliable ML models.*

When you create or extend a GifLab dataset for ML tasks, **follow the checklist below** (detailed rationale in *Section&nbsp;8* of `QUALITY_METRICS_EXPANSION_PLAN.md`). These rules apply to *every* future contribution:

1. **Deterministic extraction** – lock random seeds; metric functions must be pure.
2. **Schema validation** – export rows that pass `MetricRecordV1` (pydantic) validation.
3. **Version tagging** – record dataset version, `giflab` semver, and git commit hash.
4. **Canonical data-splits** – maintain *GIF-level* `train/val/test` JSON split files.
5. **Feature scaling** – persist `scaler.pkl` (z-score or min-max) alongside data.
6. **Missing-value handling** – encode unknown metrics as `np.nan`, not `0.0`.
7. **Outlier & drift reports** – auto-generate an HTML outlier summary and correlation dashboard.
8. **Reproducible pipeline** – provide a `make data` target that builds the dataset + EDA artifacts end-to-end.
9. **Comprehensive logs** – include parameter checksum and elapsed-time stats in every run.

> **🚦 Gate keeper:** Pull requests touching dataset code MUST tick all items or explain why they do not apply.

For implementation details, see `giflab/data_prep.py` (to be added) and the ML checklist in the plan document.

## 🧪 Testing & Development

GifLab maintains strict testing standards to ensure code quality and project cleanliness:

### Testing Guidelines
- **Unit/Integration Tests**: Use `tests/` directory with pytest
- **Manual Testing**: Use `test-workspace/` structure (never root directory!)
- **Debug Sessions**: Create organized sessions in `test-workspace/debug/`
- **Cleanup Protocols**: Regular cleanup prevents project pollution

### Test Markers
GifLab uses pytest markers to organize and selectively run different types of tests:

- **`@pytest.mark.fast`**: Quick tests safe for parallel execution (< 1 second)
- **`@pytest.mark.slow`**: Long-running tests or those requiring external binaries
- **`@pytest.mark.performance`**: Performance benchmark and timing-sensitive tests
- **`@pytest.mark.serial`**: Tests requiring sequential execution (not parallel-safe)
- **`@pytest.mark.integration`**: Integration tests with external tools/dependencies
- **`@pytest.mark.external_tools`**: Tests requiring specific external tools

**Marker-Based Test Execution:**
```bash
# Run only fast tests (parallel execution)
poetry run pytest -m "fast" -n auto

# Run performance tests sequentially
poetry run pytest -m "performance and serial" -n 1

# Run performance tests that can be parallelized
poetry run pytest -m "performance and not serial" -n auto

# Skip slow and external tool tests
poetry run pytest -m "not slow and not external_tools"
```

### Quick Commands
```bash
# Fast feedback: smoke + functional (<2min)
make test

# CI: + integration (<5min)
make test-ci

# Everything including nightly
make test-nightly

# Single file
make test-file F=tests/functional/test_metrics.py

# Create test workspace
make test-workspace

# Emergency cleanup of root directory mess
make clean-testing-mess

# Clean temporary files
make clean-temp
```

**📋 Full Guidelines**: See [Testing Best Practices](docs/guides/testing-best-practices.md) for comprehensive testing standards and protocols.

## 📚 Documentation

### Core
- **[Project Scope](SCOPE.md)** - Goals, requirements, and architecture overview

### User Guides
- **[Beginner's Guide](docs/guides/beginner.md)** - Step-by-step introduction for new users
- **[Setup Guide](docs/guides/setup.md)** - Installation and configuration instructions

### Technical Reference
- **[Metrics System](docs/technical/metrics-system.md)** - Comprehensive quality assessment framework
- **[EDA Framework](docs/technical/eda-framework.md)** - Data analysis and visualization tools
- **[ML Best Practices](docs/technical/ml-best-practices.md)** - Machine learning dataset preparation
- **[Testing Best Practices](docs/guides/testing-best-practices.md)** - Quality assurance and testing approaches
- **[Content Classification](docs/technical/content-classification.md)** - AI-powered content tagging system

### Research & Analysis
- **[Compression Research](docs/analysis/compression-research.md)** - Engine comparison and optimization strategies
- **[Implementation Lessons](docs/analysis/implementation-lessons.md)** - Development insights and best practices

## Common Commands & Troubleshooting

### 🗑️ Clear All Cached Data
If you need to reset all cached pipeline results (SQLite database):
```bash
poetry run python -m giflab run --clear-cache --estimate-time
```
This clears all cached test results from `results/cache/pipeline_results_cache.db`.

### 🔄 Force Fresh Results (Without Clearing Cache)
To run fresh tests while keeping cached data intact:
```bash
poetry run python -m giflab run --use-cache=false --sampling quick
```

### 📊 Check Cache Performance
Cache statistics are automatically shown in pipeline elimination output.

### 🔍 Debug Pipeline Failures
View detailed failure information:
```bash
poetry run python -m giflab view-failures --summary
```

**📚 For more details:** See [Compression Testing Guide](docs/guides/experimental-testing.md)

## License

MIT License – see LICENSE file for details.