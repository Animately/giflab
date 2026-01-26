# Implementation Plan: Compression Curve Prediction

**Branch**: `001-compression-curve-prediction` | **Date**: 2025-01-26 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-compression-curve-prediction/spec.md`

## Summary

Build a prediction system that extracts visual features from GIFs and predicts compression curves (file size vs lossy level) and color reduction curves (file size vs color count) without running actual compression. This enables instant feedback for users selecting compression settings on the Animately platform.

**Technical Approach**: Extract 15+ visual features using existing tagger infrastructure, collect training data by pairing features with actual compression outcomes, train gradient boosting models for curve prediction, expose via CLI and Python API.

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: scikit-learn (gradient boosting), numpy, pandas, Pydantic (schemas), existing giflab modules (tagger, metrics, lossy)  
**Storage**: CSV files for training data, pickle for trained models, SQLite cache for feature extraction  
**Testing**: pytest with existing test infrastructure  
**Target Platform**: macOS/Linux CLI, Python library  
**Project Type**: Single project (extends existing giflab)  
**Performance Goals**: Feature extraction <5s per GIF, prediction <100ms per GIF  
**Constraints**: Must integrate with existing pipeline, deterministic feature extraction, schema-validated outputs  
**Scale/Scope**: Initial training on 1,000+ GIFs, prediction for any valid GIF

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Single-Pass Compression | ✅ PASS | Feature extraction does not compress; prediction models trained on single-pass results |
| II. ML-Ready Data | ✅ PASS | All outputs validated against Pydantic schemas, versioned, deterministic |
| III. Poetry-First Execution | ✅ PASS | All commands via `poetry run`, no direct python invocation |
| IV. Test-Driven Quality | ✅ PASS | Tests required for feature extraction, schema validation, prediction accuracy |
| V. Extensible Tool Interfaces | ✅ PASS | Prediction models are engine-specific, new engines can be added |
| VI. LLM-Optimized Codebase | ✅ PASS | Explicit patterns, type hints, docstrings throughout |

**Gate Result**: PASS - No violations, proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/001-compression-curve-prediction/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (Pydantic schemas)
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
src/giflab/
├── prediction/                    # NEW: Prediction module
│   ├── __init__.py
│   ├── features.py               # GifFeatures extraction
│   ├── schemas.py                # Pydantic schemas (GifFeatures, CompressionCurve, TrainingRecord)
│   ├── dataset.py                # Training dataset builder
│   ├── models.py                 # Prediction model training and inference
│   └── cli.py                    # CLI commands for prediction
├── tagger.py                     # EXISTING: Leverage for visual features
├── metrics.py                    # EXISTING: Leverage for quality metrics
└── lossy.py                      # EXISTING: Compression engines

tests/
├── unit/
│   ├── test_prediction_features.py
│   ├── test_prediction_schemas.py
│   ├── test_prediction_dataset.py
│   └── test_prediction_models.py
└── integration/
    └── test_prediction_pipeline.py

data/
├── training/                     # Training datasets
│   ├── features/                 # Extracted features CSV
│   └── outcomes/                 # Compression outcomes CSV
└── models/                       # Trained prediction models
    ├── gifsicle_lossy_v1.pkl
    ├── gifsicle_color_v1.pkl
    ├── animately_lossy_v1.pkl
    └── animately_color_v1.pkl
```

**Structure Decision**: Extends existing giflab structure with new `prediction/` module. Follows existing patterns for CLI integration and test organization.

## Complexity Tracking

No constitution violations - no complexity justification needed.
