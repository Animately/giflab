# Tasks: Compression Curve Prediction

**Input**: Design documents from `/specs/001-compression-curve-prediction/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: Tests are included per Constitution Principle IV (Test-Driven Quality).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/giflab/prediction/`, `tests/unit/` at repository root
- Paths follow existing giflab structure per plan.md

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create prediction module structure and Pydantic schemas

- [ ] T001 Create prediction module directory structure at src/giflab/prediction/
- [ ] T002 Create src/giflab/prediction/__init__.py with public exports
- [ ] T003 [P] Create Pydantic schemas in src/giflab/prediction/schemas.py (GifFeaturesV1, CompressionCurveV1, TrainingRecordV1, PredictionModelMetadataV1)
- [ ] T004 [P] Add prediction module to src/giflab/__init__.py exports
- [ ] T005 [P] Create data directories: data/training/features/, data/training/outcomes/, data/models/

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core feature extraction infrastructure that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T006 Implement base feature extraction utilities in src/giflab/prediction/features.py (frame sampling, dimension handling)
- [ ] T007 [P] Add new compressibility features to tagger: lossless_compression_ratio, inter_frame_mse_mean, inter_frame_mse_std in src/giflab/tagger.py
- [ ] T008 [P] Add color histogram features: color_histogram_entropy, dominant_color_ratio in src/giflab/tagger.py
- [ ] T009 [P] Add DCT energy ratio feature: dct_energy_ratio in src/giflab/tagger.py
- [ ] T010 Create unified feature extraction function extract_gif_features() in src/giflab/prediction/features.py that combines tagger + new features
- [ ] T011 [P] Create tests/unit/test_prediction_schemas.py with schema validation tests
- [ ] T012 [P] Create tests/unit/test_prediction_features.py with feature extraction tests

**Checkpoint**: Feature extraction ready - user story implementation can now begin

---

## Phase 3: User Story 3 - Extract Visual Features for ML Training (Priority: P1) 🎯 MVP

**Goal**: Extract standardized visual features from GIFs for compression prediction

**Independent Test**: Provide a GIF file, verify all 20+ feature fields are populated with valid numeric values

### Tests for User Story 3

- [ ] T013 [P] [US3] Test deterministic extraction (same GIF → same features) in tests/unit/test_prediction_features.py
- [ ] T014 [P] [US3] Test feature value ranges (entropy 0-8, ratios 0-1) in tests/unit/test_prediction_features.py
- [ ] T015 [P] [US3] Test edge cases (1 frame, >500 frames, transparency) in tests/unit/test_prediction_features.py

### Implementation for User Story 3

- [ ] T016 [US3] Implement full GifFeaturesV1 extraction in src/giflab/prediction/features.py
- [ ] T017 [US3] Add frame sampling for large GIFs (>500 frames) in src/giflab/prediction/features.py
- [ ] T018 [US3] Add transparency ratio calculation in src/giflab/prediction/features.py
- [ ] T019 [US3] Add schema validation to extraction output in src/giflab/prediction/features.py
- [ ] T020 [US3] Add CLI command `giflab predict extract-features` in src/giflab/prediction/cli.py
- [ ] T021 [US3] Register CLI command in src/giflab/cli/__init__.py

**Checkpoint**: Feature extraction fully functional and testable independently

---

## Phase 4: User Story 4 - Build Training Dataset (Priority: P2)

**Goal**: Collect compression results paired with visual features for model training

**Independent Test**: Run compression experiments, verify output dataset contains features + outcomes in correct schema

### Tests for User Story 4

- [ ] T022 [P] [US4] Test dataset schema validation in tests/unit/test_prediction_dataset.py
- [ ] T023 [P] [US4] Test train/val/test split logic in tests/unit/test_prediction_dataset.py
- [ ] T024 [P] [US4] Test feature-outcome joining by gif_sha in tests/unit/test_prediction_dataset.py

### Implementation for User Story 4

- [ ] T025 [US4] Create dataset builder class in src/giflab/prediction/dataset.py
- [ ] T026 [US4] Implement compression sweep runner (all lossy levels, all color counts) in src/giflab/prediction/dataset.py
- [ ] T027 [US4] Implement feature-outcome joining by gif_sha in src/giflab/prediction/dataset.py
- [ ] T028 [US4] Implement train/val/test split (80/10/10 by GIF, not row) in src/giflab/prediction/dataset.py
- [ ] T029 [US4] Add dataset versioning and schema validation in src/giflab/prediction/dataset.py
- [ ] T030 [US4] Add CLI command `giflab predict build-dataset` in src/giflab/prediction/cli.py
- [ ] T031 [US4] Add batch processing with progress reporting in src/giflab/prediction/dataset.py

**Checkpoint**: Training dataset generation fully functional

---

## Phase 5: User Story 1 - Predict Compression Curve for New GIF (Priority: P1)

**Goal**: Predict file sizes at each lossy level without running compression

**Independent Test**: Provide a GIF, verify predicted sizes for lossy levels 0-120, compare to actual within 15%

### Tests for User Story 1

- [ ] T032 [P] [US1] Test model loading and inference in tests/unit/test_prediction_models.py
- [ ] T033 [P] [US1] Test prediction output schema validation in tests/unit/test_prediction_models.py
- [ ] T034 [P] [US1] Test confidence score generation in tests/unit/test_prediction_models.py

### Implementation for User Story 1

- [ ] T035 [US1] Create prediction model wrapper class in src/giflab/prediction/models.py
- [ ] T036 [US1] Implement gradient boosting training for lossy curves in src/giflab/prediction/models.py
- [ ] T037 [US1] Implement model serialization (pickle) with metadata in src/giflab/prediction/models.py
- [ ] T038 [US1] Implement predict_lossy_curve() function in src/giflab/prediction/models.py
- [ ] T039 [US1] Add confidence score calculation based on training data coverage in src/giflab/prediction/models.py
- [ ] T040 [US1] Add CLI command `giflab predict lossy-curve` in src/giflab/prediction/cli.py
- [ ] T041 [US1] Add CLI command `giflab predict train` in src/giflab/prediction/cli.py

**Checkpoint**: Lossy curve prediction fully functional

---

## Phase 6: User Story 2 - Predict Color Reduction Curve (Priority: P2)

**Goal**: Predict file sizes at each color count without running compression

**Independent Test**: Provide a GIF, verify predicted sizes for color counts 256-16

### Tests for User Story 2

- [ ] T042 [P] [US2] Test color curve model training in tests/unit/test_prediction_models.py
- [ ] T043 [P] [US2] Test color curve prediction output in tests/unit/test_prediction_models.py

### Implementation for User Story 2

- [ ] T044 [US2] Implement gradient boosting training for color curves in src/giflab/prediction/models.py
- [ ] T045 [US2] Implement predict_color_curve() function in src/giflab/prediction/models.py
- [ ] T046 [US2] Add CLI command `giflab predict color-curve` in src/giflab/prediction/cli.py
- [ ] T047 [US2] Update `giflab predict train` to train all 4 models (2 engines × 2 curve types) in src/giflab/prediction/cli.py

**Checkpoint**: Color curve prediction fully functional

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Integration, documentation, and validation

- [ ] T048 [P] Create integration test for full prediction pipeline in tests/integration/test_prediction_pipeline.py
- [ ] T049 [P] Add prediction module documentation to docs/technical/prediction-system.md
- [ ] T050 Update README.md with prediction feature documentation
- [ ] T051 [P] Add model accuracy validation (MAPE calculation) in src/giflab/prediction/models.py
- [ ] T052 Run quickstart.md validation - verify all CLI commands work
- [ ] T053 Add feature importance export to model metadata in src/giflab/prediction/models.py

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Story 3 (Phase 3)**: Depends on Foundational - Feature extraction (MVP foundation)
- **User Story 4 (Phase 4)**: Depends on US3 - Needs feature extraction to build dataset
- **User Story 1 (Phase 5)**: Depends on US4 - Needs training data to train models
- **User Story 2 (Phase 6)**: Depends on US4 - Needs training data to train models
- **Polish (Phase 7)**: Depends on all user stories

### User Story Dependencies

```
US3 (Feature Extraction) ──┬──> US4 (Build Dataset) ──┬──> US1 (Lossy Prediction)
                           │                          │
                           │                          └──> US2 (Color Prediction)
                           │
                           └──> [Can demo feature extraction independently]
```

### Within Each User Story

- Tests written first (fail before implementation)
- Core logic before CLI integration
- Schema validation throughout

### Parallel Opportunities

**Phase 1 (Setup)**:
- T003, T004, T005 can run in parallel

**Phase 2 (Foundational)**:
- T007, T008, T009 can run in parallel (different feature additions)
- T011, T012 can run in parallel (different test files)

**Phase 3 (US3)**:
- T013, T014, T015 can run in parallel (different test cases)

**Phase 5 & 6 (US1 & US2)**:
- Can run in parallel after US4 completes (different model types)

---

## Implementation Strategy

### MVP First (Feature Extraction Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 3 (Feature Extraction)
4. **STOP and VALIDATE**: Test feature extraction independently
5. Demo: `giflab predict extract-features sample.gif`

### Full Prediction System

1. Complete MVP (Phases 1-3)
2. Add Phase 4: User Story 4 (Dataset Builder)
3. Add Phase 5: User Story 1 (Lossy Prediction)
4. Add Phase 6: User Story 2 (Color Prediction)
5. Complete Phase 7: Polish

---

## Summary

| Metric | Value |
|--------|-------|
| Total Tasks | 53 |
| Phase 1 (Setup) | 5 tasks |
| Phase 2 (Foundational) | 7 tasks |
| Phase 3 (US3 - Features) | 9 tasks |
| Phase 4 (US4 - Dataset) | 10 tasks |
| Phase 5 (US1 - Lossy) | 10 tasks |
| Phase 6 (US2 - Color) | 6 tasks |
| Phase 7 (Polish) | 6 tasks |
| Parallel Opportunities | 18 tasks marked [P] |
| MVP Scope | Phases 1-3 (21 tasks) |
