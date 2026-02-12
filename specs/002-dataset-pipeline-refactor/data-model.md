# Data Model: Dataset Pipeline Refactor

**Feature**: 002-dataset-pipeline-refactor
**Date**: 2026-02-09
**Purpose**: Define the unified data schema supporting 7 engines and pipeline combinations.

## Schema Overview

The data model supports the full pipeline from raw GIFs to training dataset:

```
GIF File → GifFeaturesV1 (25+ features)
         → CompressionRun (per pipeline × parameter combo)
             → QualityMetrics (per run)
             → ValidationResult (per run)
         → CompressionCurveV1 (per engine × curve type)
         → TrainingRecordV2 (features + all curves)
```

## Engine Enum (Expanded)

```python
class Engine(str, Enum):
    """The 7 lossy compression engines."""
    GIFSICLE = "gifsicle"
    ANIMATELY_STANDARD = "animately-standard"
    ANIMATELY_ADVANCED = "animately-advanced"
    ANIMATELY_HARD = "animately-hard"
    IMAGEMAGICK = "imagemagick"
    FFMPEG = "ffmpeg"
    GIFSKI = "gifski"
```

### Migration from V1

| V1 Value | V2 Value | Notes |
|----------|----------|-------|
| `gifsicle` | `gifsicle` | Unchanged |
| `animately` | `animately-standard` | Renamed for specificity |
| — | `animately-advanced` | Previously `animately-advanced-lossy` in wrappers |
| — | `animately-hard` | New engine |
| — | `imagemagick` | New in enum (already existed in wrappers) |
| — | `ffmpeg` | New in enum (already existed in wrappers) |
| — | `gifski` | New in enum (already existed in wrappers) |

## SQLite Schema (storage.py)

The existing `storage.py` schema is already normalized. Key tables:

### `tools` table
Stores tool registry (all wrapper classes from `tool_wrappers.py`).

```sql
CREATE TABLE tools (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,     -- e.g., "animately-standard", "gifsicle-lossy"
    category TEXT NOT NULL,         -- "frame_reduction", "color_reduction", "lossy_compression"
    engine TEXT,                    -- e.g., "gifsicle", "animately"
    description TEXT
);
```

### `pipelines` table
Stores 3-slot pipeline combinations.

```sql
CREATE TABLE pipelines (
    id INTEGER PRIMARY KEY,
    identifier TEXT UNIQUE NOT NULL,  -- e.g., "none-frame|none-color|animately-standard"
    frame_tool_id INTEGER REFERENCES tools(id),
    color_tool_id INTEGER REFERENCES tools(id),
    lossy_tool_id INTEGER REFERENCES tools(id)
);
```

### `gifs` table
Stores GIF metadata and extracted features.

```sql
CREATE TABLE gifs (
    id INTEGER PRIMARY KEY,
    sha256 TEXT UNIQUE NOT NULL,
    file_name TEXT NOT NULL,
    file_path TEXT,
    file_size_bytes INTEGER NOT NULL,
    width INTEGER,
    height INTEGER,
    frame_count INTEGER,
    features_json TEXT,              -- JSON blob of GifFeaturesV1
    feature_version TEXT,
    processed_at TIMESTAMP
);
```

### `compression_runs` table
Stores individual compression results.

```sql
CREATE TABLE compression_runs (
    id INTEGER PRIMARY KEY,
    gif_id INTEGER REFERENCES gifs(id),
    pipeline_id INTEGER REFERENCES pipelines(id),
    param_preset_id INTEGER REFERENCES param_presets(id),
    output_size_bytes INTEGER,
    compression_ratio REAL,
    metrics_json TEXT,               -- JSON blob of quality metrics
    validation_status TEXT,          -- "valid", "corrupted", "error"
    duration_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(gif_id, pipeline_id, param_preset_id)
);
```

### `param_presets` table
Stores parameter combinations.

```sql
CREATE TABLE param_presets (
    id INTEGER PRIMARY KEY,
    lossy_level INTEGER,
    color_count INTEGER,
    frame_ratio REAL,
    UNIQUE(lossy_level, color_count, frame_ratio)
);
```

### `failures` table
Stores failed compression attempts for debugging.

```sql
CREATE TABLE failures (
    id INTEGER PRIMARY KEY,
    gif_id INTEGER REFERENCES gifs(id),
    pipeline_id INTEGER REFERENCES pipelines(id),
    param_preset_id INTEGER REFERENCES param_presets(id),
    error_type TEXT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Pydantic Schemas (Updated)

### GifFeaturesV1 — No Changes

The existing `GifFeaturesV1` with 25+ features is retained as-is. All fields remain:

- **Identity**: gif_sha, gif_name, extraction_version, extracted_at
- **Metadata**: width, height, frame_count, duration_ms, file_size_bytes, unique_colors
- **Spatial** (9): entropy, edge_density, color_complexity, gradient_smoothness, contrast_score, text_density, dct_energy_ratio, color_histogram_entropy, dominant_color_ratio
- **Temporal** (6): motion_intensity, motion_smoothness, static_region_ratio, temporal_entropy, frame_similarity, inter_frame_mse_mean, inter_frame_mse_std
- **Compressibility** (1): lossless_compression_ratio
- **Transparency** (1): transparency_ratio

### CompressionCurveV1 — Updated Engine Field

The `engine` field type changes from `Engine` (2 values) to `Engine` (7 values). No structural changes needed — the enum expansion is backward compatible.

### TrainingRecordV2 — Flexible Engine Curves

Replace hardcoded engine-specific fields with a flexible mapping:

```python
class TrainingRecordV2(BaseModel):
    """Paired features and outcomes for model training.

    Schema version: 2.0.0
    """
    schema_version: Literal["2.0.0"] = "2.0.0"

    record_id: str
    gif_sha: str
    dataset_version: str
    split: DatasetSplit
    features: GifFeaturesV1

    # Flexible engine curves (replaces hardcoded gifsicle/animately fields)
    lossy_curves: dict[str, CompressionCurveV1]   # engine name → curve
    color_curves: dict[str, CompressionCurveV1]    # engine name → curve

    created_at: datetime
```

### Migration from TrainingRecordV1

```python
# V1 (hardcoded):
lossy_curve_gifsicle: CompressionCurveV1
lossy_curve_animately: CompressionCurveV1 | None
color_curve_gifsicle: CompressionCurveV1
color_curve_animately: CompressionCurveV1 | None

# V2 (flexible):
lossy_curves: {"gifsicle": ..., "animately-standard": ..., "animately-advanced": ..., ...}
color_curves: {"gifsicle": ..., "animately-standard": ..., ...}
```

## Quality Metrics Schema

Quality metrics are stored as JSON in `compression_runs.metrics_json`. The schema is determined by the metrics ecosystem:

```python
@dataclass
class CompressionMetrics:
    """Quality metrics for a single compression result."""
    # Structural
    ssim: float                    # 0-1, higher is better
    ms_ssim: float | None         # Multi-scale SSIM
    psnr: float                   # dB, higher is better

    # Perceptual
    lpips: float | None           # 0-1, lower is better (neural)
    ssimulacra2: float | None     # Modern perceptual

    # GIF-specific artifacts
    disposal_artifact_score: float | None    # Disposal method issues
    transparency_artifact_score: float | None
    temporal_flicker_score: float | None     # Frame-to-frame flicker
    temporal_pumping_score: float | None     # Size oscillation
    gradient_banding_score: float | None     # Color banding
    posterization_score: float | None        # Color quantization artifacts
    dither_quality_score: float | None       # Dithering effectiveness
    text_readability_score: float | None     # Text/UI clarity

    # Composite
    composite_score: float        # Weighted aggregate
```

## Data Flow

```
1. GIF file discovered in input directory
2. SHA256 computed → check if already in `gifs` table
3. Features extracted → stored in `gifs.features_json`
4. For each pipeline combination:
   a. Check `compression_runs` for existing result → skip if present
   b. Execute pipeline: frame → color → lossy
   c. Validate output (wrapper_validation)
   d. Measure quality metrics
   e. Store in `compression_runs`
   f. On failure, store in `failures`
5. After all pipelines complete:
   a. Build compression curves per engine
   b. Optionally compute Pareto frontiers
```

## Export Formats

### CSV Export (`giflab export --format csv`)

Flat denormalized table with one row per compression run:

| Column | Source |
|--------|--------|
| gif_sha | gifs.sha256 |
| gif_name | gifs.file_name |
| original_size | gifs.file_size_bytes |
| pipeline | pipelines.identifier |
| lossy_level | param_presets.lossy_level |
| color_count | param_presets.color_count |
| output_size | compression_runs.output_size_bytes |
| compression_ratio | compression_runs.compression_ratio |
| ssim | metrics_json.ssim |
| psnr | metrics_json.psnr |
| ... | (all metric fields) |
| feature_* | gifs.features_json.* |

### JSON Export (`giflab export --format json`)

Hierarchical structure grouping curves by GIF and engine.
