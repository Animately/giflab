# Implementation Plan: Dataset Pipeline Refactor

**Branch**: `002-dataset-pipeline-refactor` | **Date**: 2026-02-09 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-dataset-pipeline-refactor/spec.md`

## Summary

Unify GifLab's two parallel systems (experiment runner in `core/runner.py` + dataset builder in `prediction/dataset.py`) into a single pipeline that processes raw GIFs through 7 compression engines, measures quality with comprehensive GIF-specific metrics, and stores everything in normalized SQLite. Add the missing `animately-hard` engine, rename existing animately wrappers for consistency, expand the Engine enum from 2 to 7, and consolidate all output to SQLite with CSV available only as export.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: Click, Pydantic, OpenCV, scikit-image, NumPy, Pillow
**Storage**: SQLite (via `storage.py`)
**Testing**: pytest (4-layer: smoke, functional, integration, nightly)
**Target Platform**: macOS (development), Linux (CI)
**Project Type**: Single project (src/giflab/)
**Performance Goals**: Process 100+ GIFs/hour in full pipeline mode
**Constraints**: All commands via `poetry run`, engine binaries must be installed
**Scale/Scope**: 7 engines × 7 lossy levels × 20 color tools × 5 frame tools = 1,200 combinations per GIF

## Constitution Check

*GATE: The constitution is currently a template. Applying GifLab-specific principles:*

| Principle | Status | Notes |
|-----------|--------|-------|
| Poetry-first | PASS | All commands use `poetry run` |
| Schema versioning | PASS | Pydantic models with `schema_version` field |
| Test-first | PASS | 4-layer test architecture maintained |
| SQLite storage | PASS | All output via `storage.py` |
| Engine wrappers | PASS | All 7 engines follow `ExternalTool` pattern |

## Project Structure

### Documentation (this feature)

```text
specs/002-dataset-pipeline-refactor/
├── spec.md              # User stories, requirements, success criteria
├── plan.md              # This file — technical context, constitution check
├── research.md          # Engine analysis, combination matrix
├── data-model.md        # Unified data schema (7 engines + combinations)
├── quickstart.md        # User guide for unified CLI
├── tasks.md             # Phased task breakdown
├── checklists/
│   └── requirements.md  # Quality checklist
└── contracts/
    └── schemas.py       # Updated Pydantic schemas
```

### Source Code (repository root)

```text
src/giflab/
├── cli/                        # Unified CLI commands
│   ├── __init__.py             # Main CLI group (run, train, predict, export, stats)
│   ├── run_cmd.py              # Unified pipeline runner command
│   ├── export_cmd.py           # NEW: CSV/JSON export command
│   └── utils.py                # CLI utilities
├── core/
│   ├── runner.py               # REFACTOR: Merge with prediction_runner.py
│   └── pareto.py               # Pareto frontier analysis (keep)
├── prediction/
│   ├── schemas.py              # MODIFY: Expand Engine enum to 7
│   ├── dataset.py              # REFACTOR: Merge into unified runner
│   ├── features.py             # Feature extraction (keep)
│   └── cli.py                  # REMOVE: Merged into main CLI
├── tool_wrappers.py            # MODIFY: Add AnimatelyHardLossyCompressor, rename
├── lossy.py                    # MODIFY: Add compress_with_animately_hard()
├── storage.py                  # VERIFY: Schema supports 7 engines
├── dynamic_pipeline.py         # Pipeline combination generation (keep)
├── prediction_runner.py        # REFACTOR: Merge into core/runner.py
├── metrics.py                  # Quality metrics orchestrator (keep)
├── enhanced_metrics.py         # Composite scoring (keep)
├── optimized_metrics.py        # Vectorized metrics (keep)
├── parallel_metrics.py         # Multi-core metrics (keep)
├── conditional_metrics.py      # Smart metric selection (keep)
├── deep_perceptual_metrics.py  # LPIPS neural quality (keep)
├── temporal_artifacts.py       # Flicker, pumping detection (keep)
├── gradient_color_artifacts.py # Banding, posterization (keep)
├── text_ui_validation.py       # Text readability (keep)
├── ssimulacra2_metrics.py      # Modern perceptual metric (keep)
├── wrapper_validation/         # Output corruption detection (keep)
├── optimization_validation/    # Quality floor enforcement (keep)
├── caching/                    # Frame extraction cache (keep)
├── monitoring/                 # Memory safety (keep)
└── synthetic_gifs.py           # Test data generation (keep)

tests/
├── smoke/                      # Imports, types, pure logic (<5s)
├── functional/                 # Mocked engines, synthetic GIFs (<2min)
├── integration/                # Real engines, real metrics (<5min)
└── nightly/                    # Memory, perf, stress (no limit)
```

**Structure Decision**: Single project layout. The refactoring consolidates the dual-pipeline architecture without changing the directory layout. Key modifications are to existing files, not new directories.

## Key Design Decisions

### 1. Engine Naming Convention

Engines:
- `animately-standard` (standard lossy compression)
- `animately-advanced` (advanced lossy compression)
- `animately-hard` (hard mode lossy compression)

All engine names follow the pattern `<tool>-<mode>` where `<tool>` is the binary and `<mode>` describes the algorithm variant.

### 2. Pipeline Combination Model

The existing `dynamic_pipeline.py` already generates 3-slot pipelines (frame → color → lossy). This system is retained and extended to include the new `animately-hard` engine in the lossy slot.

### 3. Unified Runner Architecture

The current two runners:
- `core/runner.py` (`GifLabRunner`) — experiment-oriented, CSV output, elimination analysis
- `prediction_runner.py` (`PredictionRunner`) — prediction-oriented, SQLite output, feature extraction

Merge into a single `PredictionRunner` that:
1. Extracts features (from `prediction/features.py`)
2. Runs pipeline combinations (from `dynamic_pipeline.py`)
3. Measures quality metrics (from `metrics.py` ecosystem)
4. Validates outputs (from `wrapper_validation/`)
5. Stores in SQLite (from `storage.py`)

### 4. Schema Evolution

The `Engine` enum expands from 2 to 7 values. The `TrainingRecordV1` schema currently hardcodes `lossy_curve_gifsicle` and `lossy_curve_animately` fields. The new schema uses a flexible mapping approach: `compression_curves: dict[Engine, CompressionCurveV1]`.

### 5. Storage Consolidation

All CSV writing in `core/runner.py`, `prediction/dataset.py`, `prediction/cli.py`, `tag_pipeline.py`, `io.py`, and `cli/utils.py` is replaced with SQLite writes. A new `export` CLI command reads from SQLite and writes CSV/JSON on demand.

## Complexity Tracking

No constitution violations to track — the refactoring simplifies the existing architecture.
