# 🎞️ GifLab — Project Scope

---

## 0 Objective

### Core Mission
GifLab is, in priority order:

1. **A measurement library** — stable `compress()` / `measure()` over 7 engines plus a comprehensive quality metric stack (including the calibrated `composite_quality` verdict), consumed by the sibling **gifprep** repo for its preprocessing experiments. See `docs/public-api.md`.
2. **A compression-pipeline leaderboard instrument** — fan out compression pipelines (frame / colour / lossy, across engines) and rank the best pipeline **per GIF content-type** on the quality-vs-size trade-off. *(In rebuild — the original harness was removed in `648db9a`; see `docs/technical/compression-pipeline-leaderboard.md`.)*
3. **A prediction-dataset generator** — analyse GIFs and run compression sweeps at prediction-required granularity, stored in SQLite as ML training data.

**Boundary with gifprep**: gifprep owns *preprocessing* (denoise / cleanup / AI transforms applied *before* compression) and its own Pareto harness; giflab owns the engines, the metrics, and the *compression-pipeline* matrix benchmark. The dependency is one-way (gifprep → giflab).

**Deferred**: the ML curve-predictor (`.pkl` export). The top-level `giflab train` CLI is currently a **stub**; real training lives under `giflab predict train`.

### Compression Parameters
* **Lossy levels** ∈ { 0 · 20 · 40 · 60 · 80 · 100 · 120 } — 7 points for smooth curve prediction
* **Color counts** ∈ { 256 · 128 · 64 · 32 · 16 } — 5 points for color curve prediction
* **Engines** (7 total):
  * `gifsicle` — standard GIF optimizer
  * `animately-standard` — Animately standard lossy (`--lossy`)
  * `animately-advanced` — Animately advanced lossy (`--advanced-lossy`)
  * `animately-hard` — Animately hard mode (`--hard --lossy`)
  * `imagemagick` — ImageMagick lossy compression
  * `ffmpeg` — FFmpeg GIF encoding
  * `gifski` — high-quality GIF encoder

### Visual Features Extracted (25 features)
* **Spatial**: entropy, edge_density, color_complexity, gradient_smoothness, contrast_score, text_density, dct_energy_ratio, color_histogram_entropy, dominant_color_ratio
* **Temporal**: motion_intensity, motion_smoothness, static_region_ratio, temporal_entropy, frame_similarity, inter_frame_mse_mean, inter_frame_mse_std
* **Compressibility**: lossless_compression_ratio, transparency_ratio
* **CLIP Content Classification**: screen_capture, vector_art, photography, hand_drawn, 3d_rendered, pixel_art

### ML-Driven Vision
**Goal**: Train gradient boosting models to predict compression curves (file size at each lossy level) from visual features alone — enabling instant compression estimation without running actual compression.

### Requirements
* Parallel execution, resumable after interruption.
* Corrupt/unreadable GIFs moved to `data/bad_gifs/`.
* Works on macOS and Windows/WSL.
* Keeps each GIF's original file-name **and** a content hash for deduplication.
* **NEW**: Content classification and feature extraction for ML training
* **NEW**: Comprehensive framework for testing additional tools and strategies

---

## 1 Directory Layout

```
giflab/
├─ data/
│   ├─ raw/              ← originals
│   ├─ giflab.db         ← SQLite database (features + compression runs)
│   ├─ models/           ← trained prediction models (.pkl)
│   ├─ bad_gifs/         ← corrupt originals
│   └─ tmp/              ← temp files
├─ logs/                 ← run logs
├─ src/giflab/
│   ├─ storage.py        ← SQLite storage (replaces CSV + cache)
│   ├─ prediction_runner.py ← unified pipeline runner
│   ├─ prediction/
│   │   ├─ features.py   ← visual feature extraction + CLIP
│   │   ├─ models.py     ← gradient boosting model training
│   │   └─ schemas.py    ← Pydantic schemas
│   ├─ config.py
│   ├─ meta.py
│   ├─ lossy.py
│   ├─ metrics.py
│   └─ cli.py            ← `python -m giflab …`
├─ notebooks/
├─ tests/
└─ pyproject.toml
```

---

## 2 CSV Column Schema

| column               | dtype | example                    |
|----------------------|-------|----------------------------|
| gif_sha              | str   | `6c54c899e2b0baf7…`        |
| orig_filename        | str   | `very_weird name (2).gif`  |
| engine               | str   | `gifsicle`                 |
| lossy                | int   | `40`                       |
| frame_keep_ratio     | float | `0.80`                     |
| color_keep_count     | int   | `64`                       |
| kilobytes            | float | `413.72`                   |
| ssim                 | float | `0.936`                    |
| render_ms            | int   | `217`                      |
| orig_kilobytes       | float | `2134.55`                  |
| orig_width           | int   | `480`                      |
| orig_height          | int   | `270`                      |
| orig_frames          | int   | `24`                       |
| orig_fps             | float | `24.0`                     |
| orig_n_colors        | int   | `220`                      |
| entropy (optional)   | float | `4.1`                      |
| source_platform      | str   | `tenor`                    |
| source_metadata      | str   | `{"query":"love","tenor_id":"xyz123"}` |
| tags (optional)      | str   | `vector-art;low-fps`       |
| timestamp            | str   | ISO-8601                   |

*Primary key = (`gif_sha`, `engine`, `lossy`, `frame_keep_ratio`, `color_keep_count`).*

---

## 3 Compression Engine Architecture

**Important**: All compression parameters (lossy, frame_keep, color_keep) must be applied **in a single pass** by the same engine. This is critical for efficiency since GIF recompilaton is inherently lossy.

### Engine Responsibilities

| Engine | Handles |
|--------|---------|
| `gifsicle` | Lossy compression + frame reduction + color reduction |
| `animately-standard` | Standard lossy compression + frame reduction + color reduction |
| `animately-advanced` | Advanced lossy via PNG sequence pipeline |
| `animately-hard` | Hard mode lossy compression (`--hard --lossy`) |
| `imagemagick` | Lossy compression + frame reduction + color reduction (multiple dithering modes) |
| `ffmpeg` | Lossy compression + frame reduction + color reduction (multiple dithering modes) |
| `gifski` | High-quality lossy compression |

### Anti-Pattern (❌ Don't Do This)
```
1. Reduce frames → save temp.gif
2. Apply color reduction → save temp2.gif  
3. Apply lossy compression → save final.gif
```

### Correct Pattern (✅ Do This)
```
1. Single engine call with all parameters:
   - lossy level
   - frame_keep_ratio
   - color_keep_count
   → save final.gif
```

This approach:
- Minimizes quality loss from multiple recompressions
- Reduces I/O overhead
- Allows engines to optimize the full parameter space together

---

## 4 Variant Matrix per GIF

| frame_keep_ratio | color_keep_count | lossy values |
|------------------|------------------|--------------|
| 1.00             | 256              | 0, 40, 120   |
| 1.00             | 64               | 0, 40, 120   |
| 0.80             | 256              | 0, 40, 120   |
| 0.80             | 64               | 0, 40, 120   |

*Total*: 24 renders per GIF.
Add more ratios or palette sizes via `config.py`; the pipeline renders only missing combinations.

---

## 4 CLI

```bash
poetry run python -m giflab run [options]       # compression + feature extraction
poetry run python -m giflab train [options]      # STUB: prints row counts, does not fit a model (use `predict train`)
poetry run python -m giflab export [options]     # export data/models
poetry run python -m giflab stats [options]      # database statistics
poetry run python -m giflab predict [subcommand] # extract-features | train | lossy-curve | color-curve
```

Options:

```
  --workers, -j N   number of processes (default: CPU count)
  --resume          skip existing renders (default: true)
  --fail-dir PATH   folder for bad gifs   (default: data/bad_gifs)
  --dry-run         list work only
```

*SIGINT* finishes active tasks, flushes data, exits.
Re-run with `--resume` continues where it left off, using SHA dedup to skip duplicates.

---

## 5 Implementation Stages (Cursor Tasks)

| Stage   | Deliverable                                                        | Status |
|---------|--------------------------------------------------------------------|--------|
| **S0**  | Repo scaffold, Poetry, black/ruff, pytest.                         | ✅     |
| **S1**  | `meta.py` — extract metadata + SHA + file-name; tests.             | ✅     |
| **S2**  | `lossy.py`.                                                        | ✅     |
| **S3**  | `frame_keep.py`.                                                   | ✅     |
| **S4**  | `color_keep.py`.                                                   | ✅     |
| **S5**  | `metrics.py` — see [Metrics System Documentation](docs/technical/metrics-system.md) for technical details.                                                    | ✅     |
| **S6**  | `pipeline.py` + `io.py` (skip, resume, CSV append, move bad gifs). | ✅     |
| **S7**  | `cli.py` (`run` subcommand).                                       | ✅     |
| **S8**  | `tests/test_resume.py`.                                            | ✅     |
| **S9**  | `tagger.py` + `tag_pipeline.py` + `cli tag`.                       | ✅     |
| **S10** | notebooks 01 & 02.                                                 | ⏳     |

**Status Legend:**
- ⏳ = Not started
- 🔄 = In progress  
- ✅ = Completed
- ❌ = Blocked/Issues

---

## 6 Cross-Platform Setup

| macOS                                      | Windows / WSL                                  |
|--------------------------------------------|------------------------------------------------|
| `brew install python@3.11 ffmpeg gifsicle` | `choco install python ffmpeg gifsicle` or WSL2 |
| Place `animately-cli` binary on PATH       | Same / WSL                                     |

**Engine Paths:**
- Animately engine: `/Users/lachlants/bin/launcher`
- Gifsicle: `gifsicle`

```bash
git clone https://…/giflab.git
cd giflab
poetry install
poetry run python -m giflab run data/raw
```

## 7  Prediction Training Pipeline

### Dataset Generation Framework
- **Purpose**: Generate comprehensive compression datasets across diverse GIF content
- **Scope**: Small-scale validation (~10 GIFs) before large-scale sweeps
- **Engines**: gifsicle, animately (standard/advanced/hard), ImageMagick, FFmpeg, gifski
- **Output**: SQLite databases with compression results and visual features for model training

### Machine Learning Pipeline
1. **Content Classification**: Automatically categorize GIFs (text, photo, animation, graphics)
2. **Feature Extraction**: Extract 25 visual, structural, and semantic features per GIF
3. **Compression Sweeps**: Run parameter sweeps across engines, lossy levels, and color counts
4. **Model Training**: Train gradient boosting models to predict compression curves from features

### Prediction Model Export
GifLab trains and exports prediction models (`.pkl` files) for use by **external tools** (e.g., Animately). The workflow is:

1. **GifLab** runs compression pipelines → collects training data
2. **GifLab** trains gradient boosting models on collected data
3. **GifLab** exports trained models as `.pkl` files
4. **External tools** load models → predict compression curves instantly without running actual compression

> **Important**: GifLab is the data generator, not the consumer of predictions. Predictions are for external consumption.

### Tool Integration Priority
1. **Tier 1 (Immediate)**: ImageMagick, FFmpeg, gifski
2. **Tier 2 (Strategic)**: WebP conversion, AVIF conversion, Pillow
3. **Tier 3 (Research)**: Neural compression, custom hybrid chains

### Prediction Data Requirements
To train accurate compression curve prediction models, the main pipeline must generate data at sufficient granularity:

| Parameter | Current Config | Prediction Requirement | Status |
|-----------|---------------|----------------------|--------|
| Lossy levels | `[0, 40, 120]` | `[0, 20, 40, 60, 80, 100, 120]` | ⚠️ Gap |
| Color counts | `[256, 128, 64, 32, 16, 8]` | `[256, 128, 64, 32, 16]` | ✅ OK |
| Visual features | Basic metadata | 25+ features (see `prediction/schemas.py`) | ⚠️ Gap |

**Recommendation**: Run a dedicated "prediction training" mode with finer lossy granularity, or use `giflab predict build-dataset` which runs its own compression sweeps.

*See [ML Strategy Documentation](docs/technical/ml-strategy.md) for detailed implementation plans.

---

## 8  ML Dataset Quality Requirements

All future stages must comply with the “Machine-Learning Dataset Best Practices” checklist in `README.md` and the detailed guidance in *Section 8* of `QUALITY_METRICS_EXPANSION_PLAN.md`.  In short, any code that produces or mutates metric data **MUST**:

- guarantee deterministic, reproducible extraction;
- validate against `MetricRecordV1` pydantic schema;
- tag outputs with dataset+code versions;
- respect canonical train/val/test GIF splits;
- preserve or update `scaler.pkl` feature-scaling artefacts;
- regenerate outlier and correlation reports when metrics change.

Pull requests that add or modify dataset-related code must include evidence (CI artefacts or notebook screenshots) showing the checklist is satisfied.
