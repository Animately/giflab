# Tasks: Dataset Pipeline Refactor

**Input**: Design documents from `/specs/002-dataset-pipeline-refactor/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1-US5)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: Spec creation and project cleanup

- [x] T001 [US5] Create spec-kit directory at `specs/002-dataset-pipeline-refactor/`
- [x] T002 [US5] Re-initialize spec-kit for Claude Code (`specify init . --ai claude --here --force`)
- [ ] T003 [P] [US5] Write spec.md, plan.md, research.md, data-model.md, quickstart.md
- [ ] T004 [P] [US5] Write contracts/schemas.py with updated Pydantic models
- [ ] T005 [P] [US5] Write tasks.md (this file) and checklists/requirements.md

---

## Phase 2: Engine Layer (US2 - All 7 Engines)

**Purpose**: Add animately-hard engine wrapper and rename existing animately wrappers

**Checkpoint**: After this phase, all 7 engines should compress a test GIF successfully

### Implementation

- [ ] T006 [US2] Add `AnimatelyHardLossyCompressor` class in `src/giflab/tool_wrappers.py` using `animately --input <in> --output <out> --hard --lossy <level>`
- [ ] T007 [US2] Add `compress_with_animately_hard()` function in `src/giflab/lossy.py`
- [ ] T008 [US2] Update `LossyEngine` enum in `src/giflab/lossy.py` to include `ANIMATELY_HARD`
- [ ] T009 [US2] Update `Engine` enum in `src/giflab/prediction/schemas.py` from 2 to 7 values: `GIFSICLE`, `ANIMATELY_STANDARD`, `ANIMATELY_ADVANCED`, `ANIMATELY_HARD`, `IMAGEMAGICK`, `FFMPEG`, `GIFSKI`
- [ ] T010 [US2] Add `AnimatelyHardLossyCompressor` to tool registry in `src/giflab/tool_wrappers.py` (TOOL_REGISTRY or equivalent)
- [ ] T011 [US2] Verify all 7 engines work: `poetry run pytest tests/smoke/ -x -q`

---

## Phase 3: Schema Evolution (US1 - Training Dataset)

**Purpose**: Update Pydantic schemas and storage for 7-engine support

**Checkpoint**: After this phase, schemas validate data for all 7 engines

### Implementation

- [ ] T015 [P] [US1] Create `TrainingRecordV2` in `src/giflab/prediction/schemas.py` with flexible `lossy_curves: dict[str, CompressionCurveV1]` and `color_curves: dict[str, CompressionCurveV1]`
- [ ] T016 [P] [US1] Add V1→V2 migration utility for `TrainingRecordV1` → `TrainingRecordV2`
- [ ] T017 [US1] Update `GifLabStorage.populate_tools_from_registry()` in `src/giflab/storage.py` to register new `animately-hard` tool and updated names
- [ ] T018 [US1] Verify SQLite schema supports 7 engines: `poetry run pytest tests/functional/ -x -q`

---

## Phase 4: Pipeline Unification (US1 - Training Dataset)

**Purpose**: Merge experiment runner and prediction runner into one unified pipeline

**Checkpoint**: After this phase, `giflab run` produces a complete dataset

### Implementation

- [ ] T019 [US1] Merge `PredictionRunner` (`src/giflab/prediction_runner.py`) functionality into `core/runner.py` or refactor as the single runner
- [ ] T020 [US1] Integrate feature extraction (`prediction/features.py`) into unified runner flow
- [ ] T021 [US1] Integrate quality metrics measurement into unified runner (call `metrics.py` ecosystem per compression result)
- [ ] T022 [US1] Integrate wrapper validation into unified runner (call `wrapper_validation/` per compressed output)
- [ ] T023 [US1] Ensure unified runner uses SQLite only (no CSV writes) via `storage.py`
- [ ] T024 [US1] Support `--mode single` (per-engine) and `--mode full` (all combinations) in unified runner
- [ ] T025 [US1] Support `--force` (reprocess all) and `--upgrade` (reprocess outdated) in unified runner
- [ ] T026 [US1] Verify unified pipeline: `poetry run pytest tests/smoke/ tests/functional/ -x -q`

---

## Phase 5: Storage Consolidation (US3 - SQLite Only)

**Purpose**: Remove CSV as primary output, add CSV export command

**Checkpoint**: After this phase, no pipeline code writes CSV files

### Implementation

- [ ] T027 [US3] Remove CSV writing from `src/giflab/core/runner.py` (streaming_results.csv, elimination CSVs, pareto CSVs)
- [ ] T028 [P] [US3] Remove CSV writing from `src/giflab/prediction/dataset.py`
- [ ] T029 [P] [US3] Remove CSV writing from `src/giflab/prediction/cli.py` (or remove file if merged into main CLI)
- [ ] T030 [US3] Create `src/giflab/cli/export_cmd.py` with `giflab export --format csv|json --output <path>` command
- [ ] T031 [US3] Register `export` command in `src/giflab/cli/__init__.py`
- [ ] T032 [US3] Verify no CSV files are written during pipeline: `poetry run pytest tests/functional/ -x -q`

---

## Phase 6: CLI Simplification (US4 - Simplified CLI)

**Purpose**: Streamline CLI to focused dataset workflow commands

**Checkpoint**: After this phase, `giflab --help` shows only: run, train, predict, export, stats

### Implementation

- [ ] T033 [US4] Update `src/giflab/cli/__init__.py` to register only: `run`, `train`, `predict`, `export`, `stats`
- [ ] T034 [US4] Remove or merge `src/giflab/prediction/cli.py` (batch-extract functionality merged into `run` command)
- [ ] T035 [US4] Update `run` command in `src/giflab/cli/run_cmd.py` to use unified runner
- [ ] T036 [US4] Ensure `train`, `predict` commands reference updated Engine enum (7 engines)
- [ ] T037 [US4] Verify CLI: `poetry run python -m giflab --help`

---

## Phase 7: Cleanup (US5 - Clean Codebase)

**Purpose**: Remove unused directories and files

**Checkpoint**: After this phase, `make test` passes and codebase is clean

### Implementation

- [ ] T038 [P] [US5] Delete `src/giflab/benchmarks/` directory (5 files)
- [ ] T039 [P] [US5] Delete `src/giflab/config_profiles/` directory (8 files)
- [ ] T040 [P] [US5] Delete `src/giflab/dashboard/` directory (empty)
- [ ] T041 [US5] Clean up old results data in `results/` (archive or delete 158 runs)
- [ ] T042 [US5] Remove any orphaned imports referencing deleted modules
- [ ] T043 [US5] Verify cleanup: `make test`

---

## Phase 8: Documentation (Cross-cutting)

**Purpose**: Update documentation to reflect unified pipeline

### Implementation

- [ ] T044 [P] Update `README.md` — reposition from "compression lab" to "training dataset generation for compression curve prediction"
- [ ] T045 [P] Update `pyproject.toml` description
- [ ] T046 [P] Update `SCOPE.md` with all 7 engines and SQLite-only storage
- [ ] T047 Update `CLAUDE.md` with new CLI commands and engine names

---

## Phase 9: Verification

**Purpose**: Full verification that everything works

- [ ] T048 Run `make test` — all smoke + functional tests pass
- [ ] T049 Run `make test-ci` — integration tests pass
- [ ] T050 Run `poetry run python -m giflab run data/raw/ --mode single` — produces SQLite with 7 engines
- [ ] T051 Run `poetry run python -m giflab export --format csv --output test.csv` — valid CSV export
- [ ] T052 Run `poetry run python -m giflab stats` — shows counts for all 7 engines
- [ ] T053 Verify `animately-hard` engine: `poetry run python -c "from giflab.tool_wrappers import AnimatelyHardLossyCompressor; print(AnimatelyHardLossyCompressor.NAME)"`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — complete
- **Phase 2 (Engines)**: No dependencies — can start immediately
- **Phase 3 (Schemas)**: Depends on Phase 2 (Engine enum values)
- **Phase 4 (Pipeline)**: Depends on Phase 2 + Phase 3
- **Phase 5 (Storage)**: Depends on Phase 4 (unified runner must exist)
- **Phase 6 (CLI)**: Depends on Phase 4 + Phase 5
- **Phase 7 (Cleanup)**: Can start after Phase 4 (core pipeline stable)
- **Phase 8 (Docs)**: Can start after Phase 6 (CLI finalized)
- **Phase 9 (Verify)**: Depends on all previous phases

### Critical Path

```
Phase 2 (Engines) → Phase 3 (Schemas) → Phase 4 (Pipeline) → Phase 5 (Storage) → Phase 6 (CLI) → Phase 9 (Verify)
```

### Parallel Opportunities

- T006, T007, T008 can run in parallel (different classes in same file)
- T015, T016 can run in parallel with T017
- T027, T028, T029 can run in parallel (different files)
- T038, T039, T040 can run in parallel (independent directories)
- T044, T045, T046 can run in parallel (independent files)
- Phase 7 (Cleanup) can run in parallel with Phase 6 (CLI)
