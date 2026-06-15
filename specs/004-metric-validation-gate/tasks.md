# Tasks: Metric Validation Gate

**Feature**: `004-metric-validation-gate` | **Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)
**Input**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

## Format: `[ID] [P?] [Story] Description`

- **[P]** = parallelizable (different file, no incomplete dependency)
- **[US#]** = the user story a task serves (user-story phases only)
- All commands run via `poetry run`; the gate is **read-only over `src/giflab/`** (validates the metric, never edits it).

## Path Conventions

- New study package: `scripts/audit/validation_gate/`
- Reused (do not modify): `scripts/audit/_common.py`, `scripts/audit/build_sample.py`, `scripts/audit/report.py`, `giflab.metrics.calculate_comprehensive_metrics`, shipped `compress()`
- Tests: `tests/smoke/`, `tests/functional/`
- Run outputs: `docs/metrics-audit/<date>/validation-gate/`

---

## Phase 1: Setup (Shared Infrastructure)

- [ ] T001 Create the `scripts/audit/validation_gate/` package (`__init__.py`) and a run-folder path helper in `scripts/audit/validation_gate/paths.py`
- [ ] T002 [P] Confirm `scipy.stats` is importable in the Poetry env for rank-correlation; if absent, add it to `pyproject.toml` (it is otherwise a transitive dep of the metrics stack)
- [ ] T003 [P] Add `.gitignore` entries so raw study outputs stay untracked: `docs/metrics-audit/*/validation-gate/study_outputs/`

## Phase 2: Foundational (Blocking Prerequisites)

**Blocks all user stories — must complete first.**

- [ ] T004 Define pydantic schemas for all entities (StudyGif, StudyOutput, HumanJudgement, AgreementResult, QualityFloor, DisagreementRecord, GateVerdict) per data-model.md in `scripts/audit/validation_gate/schemas.py` (NaN-honest fields; `ContentBucket` + `OperationAxis` enums)
- [ ] T005 [P] Schema validation smoke test (round-trip, NaN handling, enum constraints) in `tests/smoke/test_validation_gate_schemas.py`
- [ ] T006 CSV/manifest I/O helpers (read/write manifest + judgements, resume-safe, dropping `_common.py`-style header collisions) in `scripts/audit/validation_gate/io.py`
- [ ] T007 [P] Verdict helper: compute `composite_quality` + the 11 weighted sub-metrics for an (orig, comp) pair via `calculate_comprehensive_metrics(force_all_metrics=True)` full-stack config (R5), recording `metric_config_id` + `giflab_commit`, in `scripts/audit/validation_gate/verdict.py`

---

## Phase 3: User Story 1 - Go/no-go on metric trustworthiness (Priority: P1) 🎯 MVP

**Goal**: produce a single GO/NO-GO/LOW-CONFIDENCE verdict + per-content-type rank-agreement numbers.
**Independent test**: run the gate end-to-end on the study set; `report.md` contains the verdict + a per-bucket agreement table (SC-001).

### Tests for User Story 1

- [ ] T008 [P] [US1] Smoke test the rank-agreement maths — per-GIF Spearman ρ with tie-average ranks, NaN-verdict exclusion, and the GO/NO-GO/LOW-CONFIDENCE decision rule — in `tests/smoke/test_validation_gate_maths.py`
- [ ] T009 [P] [US1] Functional test `build_study generate` with a **mocked** `compress()` + `ingest_rankings` validation in `tests/functional/test_validation_gate_pipeline.py`

### Implementation for User Story 1

- [ ] T010 [US1] `build_study select`: stratified study-GIF selection (reuse `build_sample.py`) → `study_manifest.csv` with a **blank `content_bucket`** column (FR-008), in `scripts/audit/validation_gate/build_study.py`
- [ ] T011 [US1] `build_study generate`: one-axis-at-a-time quality ladders via `compress()` (R4), compute verdict + sub-metrics (T007), append rows, resume-safe, **fail loud on any unlabelled GIF**, in `scripts/audit/validation_gate/build_study.py` (depends on T006, T007, T010)
- [ ] T012 [P] [US1] `ranking_sheet`: render the offline, self-contained HTML contact sheet — per GIF, outputs inline in **randomised** order (seed recorded), never showing the verdict — reusing `report.py` thumbnails, in `scripts/audit/validation_gate/ranking_sheet.py`
- [ ] T013 [P] [US1] `ingest_rankings`: parse the sheet export / hand-edited CSV → validated `HumanJudgement` records (ranking covers exactly the GIF's outputs; boundary id is one of them; ties allowed), in `scripts/audit/validation_gate/ingest_rankings.py`
- [ ] T014 [US1] `analyse` (agreement): per-GIF + per-bucket Spearman ρ (+ Kendall τ) with tie-average ranks (reuse `_common.tie_average_unit_ranks`), NaN-excluded and counted → `agreement.csv`, in `scripts/audit/validation_gate/analyse.py` (depends on T004, T006)
- [ ] T015 [US1] `render_report`: apply the `GateVerdict` decision rule (GO iff mean ρ ≥ 0.85 and worst bucket > 0.5; else NO-GO; LOW-CONFIDENCE near-threshold) + agreement tables + figures → `report.md`, reusing `report.py` helpers, in `scripts/audit/validation_gate/render_report.py` (depends on T014)

**Checkpoint**: US1 alone is a shippable MVP — it answers the gating question (go/no-go) and is the only milestone that must pass before any M2 work.

---

## Phase 4: User Story 2 - Calibrate the per-content-type quality floor (Priority: P2)

**Goal**: a per-bucket quality floor (median + spread) usable as the leaderboard operating point.
**Independent test**: `floors.csv` has a floor + spread for every represented bucket (SC-003).

### Tests for User Story 2

- [ ] T016 [P] [US2] Smoke test floor derivation — median + IQR of per-GIF boundary `composite_quality`, null-boundary ("no crossing") handling, and the low-confidence flag for thin buckets — in `tests/smoke/test_validation_gate_floors.py`

### Implementation for User Story 2

- [ ] T017 [US2] Extend `analyse` with per-bucket floor calibration (R2): floor = median, spread = IQR of boundary `composite_quality`; flag low-confidence buckets → `floors.csv`, in `scripts/audit/validation_gate/analyse.py` (depends on T014)
- [ ] T018 [US2] Extend `render_report` with the per-bucket floors table + section, in `scripts/audit/validation_gate/render_report.py` (depends on T015, T017)

---

## Phase 5: User Story 3 - Surface systematic verdict biases (Priority: P3)

**Goal**: a disagreement report attributing each systematic divergence to a sub-metric, grouped by bucket × operation type.
**Independent test**: `disagreements.csv` lists each beyond-margin divergence with bucket, operation axis, and dominant sub-metric (SC-004).

### Tests for User Story 3

- [ ] T019 [P] [US3] Smoke test disagreement attribution — beyond-margin gating, dominant **weighted** sub-metric selection (R6), and bucket × operation-axis grouping — in `tests/smoke/test_validation_gate_disagreements.py`

### Implementation for User Story 3

- [ ] T020 [US3] Extend `analyse` with disagreement attribution: beyond-margin human↔verdict divergences → dominant weighted sub-metric (R6), grouped by `content_bucket` × `operation_axis` → `disagreements.csv`, in `scripts/audit/validation_gate/analyse.py` (depends on T014)
- [ ] T021 [US3] Extend `render_report` with the disagreement-report section, in `scripts/audit/validation_gate/render_report.py` (depends on T015, T020)

---

## Phase 6: Polish & Cross-Cutting Concerns

- [ ] T022 [P] Add a `make` target / poetry entry points for the gate steps and cross-link them from [quickstart.md](./quickstart.md)
- [ ] T023 [P] Add a short note to `docs/metrics-audit/` (or the dated run folder README) describing the validation-gate run convention
- [ ] T024 Run `make test` (smoke + functional), `poetry run ruff check`, and `poetry run mypy` over `scripts/audit/validation_gate/`; fix issues (no `src/giflab/` changes)
- [ ] T025 Update the Obsidian task `[[giflab-metric-validation-gate]]` + `[[THREAD-leaderboard]]` to link spec 004 and record the build is scoped

---

## Dependencies & Execution Order

- **Setup (Phase 1)** → **Foundational (Phase 2)** → **User Stories (Phases 3–5)** → **Polish (Phase 6)**.
- **Phase 2 blocks everything** (schemas, I/O, verdict helper are shared).
- **US1 (P1)** is the MVP and must complete (and pass) before US2/US3 add value — and before any M2 leaderboard work begins.
- **US2 and US3 are independent of each other**: both only extend `analyse` + `render_report` using US1's `study_manifest.csv` + `human_judgements.csv`. They can be built in either order once US1's T014/T015 exist. (They touch the same two files, so coordinate edits — not `[P]` against each other.)

## Parallel Opportunities

- Phase 1: T002, T003 in parallel.
- Phase 2: T005, T007 in parallel (T004 → T005; T006 independent).
- US1: T008, T009 (tests) in parallel; T012, T013 (`ranking_sheet`, `ingest_rankings`) in parallel — different files, both independent of T011's generation once schemas exist.
- Tests T008, T016, T019 are each `[P]` (separate test files).

## Implementation Strategy

**MVP = Phase 1 + Phase 2 + Phase 3 (US1).** That alone produces the go/no-go verdict — the gate's whole reason to exist. Stop and run it; the verdict decides whether M2 proceeds or metric-fixing takes priority. Add US2 (floors) next (the leaderboard needs them) and US3 (bias report) last (most valuable on a NO-GO). The human rating step (quickstart §4) sits between US1's `ranking_sheet`/`ingest` — budget one focused session (SC-005).

## Task Summary

- **Total**: 25 tasks (T001–T025)
- **Per story**: Setup 3 · Foundational 4 · US1 8 · US2 3 · US3 3 · Polish 4
- **Tests**: 5 (T005, T008, T009, T016, T019)
- **Suggested MVP**: T001–T015 (Setup + Foundational + US1) → first go/no-go verdict
