# Implementation Plan: Metric Validation Gate

**Branch**: `004-metric-validation-gate` | **Date**: 2026-06-14 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/004-metric-validation-gate/spec.md`

## Summary

Establish whether `composite_quality` is trustworthy enough to rank compression outputs, and calibrate the per-content-type quality floor the leaderboard ranks at — a go/no-go before any leaderboard machinery is built. Technical approach: a **thin, reproducible, single-run study script set** layered onto the existing audit harness (`scripts/audit/`). It (1) generates a small study set of compressed outputs spanning a quality range across hand-labelled content buckets, (2) computes `composite_quality` per output, (3) renders an offline ranking sheet a single rater fills in (per-GIF quality order + one acceptability boundary), (4) ingests those judgements and computes per-GIF rank agreement, per-bucket quality floors, and an operation-type disagreement report, and (5) emits a dated verdict + report under `docs/metrics-audit/<date>/validation-gate/`. No leaderboard runner, cross-engine chaining, auto-labeller, or ML.

## Technical Context

**Language/Version**: Python 3.11 (Poetry)
**Primary Dependencies**: existing giflab internals only — `scripts/audit/_common.py` (compression + CSV-append + resume helpers), `scripts/audit/build_sample.py` (stratified sampler over `~/Documents/GIFs`), `scripts/audit/report.py` (markdown + matplotlib figure rendering); `giflab.metrics.calculate_comprehensive_metrics(force_all_metrics=True)` for the `composite_quality` verdict; the shipped `compress()` API / existing tool wrappers for study-output generation. New maths: `scipy.stats` (already a transitive dep via the metrics stack) for rank-correlation, or a small in-repo implementation. `pydantic` (already used in `prediction/schemas.py`) for the manifest/result schemas.
**Storage**: filesystem only — versioned artifacts under `docs/metrics-audit/<date>/validation-gate/` (CSV + markdown + PNG), consistent with prior audit runs. No SQLite (see Constitution Check / Complexity Tracking).
**Testing**: pytest, 4-layer — pure agreement/floor maths in `tests/smoke/`; study-set generation + ranking-sheet ingestion with mocked compression in `tests/functional/`.
**Target Platform**: macOS/Linux dev machine (single-run research study, run under `gentle`).
**Project Type**: single project (script set under `scripts/audit/`, outputs under `docs/`).
**Performance Goals**: not latency-bound; study scale ≈ 15–20 GIFs × 4–6 compressions (≈ 60–120 outputs), generatable in one `gentle` session. Metric computation reuses the existing (LPIPS/SSIMULACRA2-capable) stack.
**Constraints**: deterministic + reproducible (fixed GIF set, fixed compression params, recorded metric config + giflab commit — FR-010); fully offline (no web service for ranking — an HTML/CSV sheet the rater fills in); must NOT modify any metric implementation (this gate only *measures* `composite_quality`).
**Scale/Scope**: one content bucket = the 5 compression-behaviour buckets from `docs/technical/compression-pipeline-leaderboard.md`, hand-assigned; one rater; one dated study run that can be re-run.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **I. Dataset-First** — PASS (indirect-but-essential). The gate produces no training data itself, but it validates the `composite_quality` verdict that *all* downstream dataset/selection quality depends on. It is the explicit guard for Principle VII.
- **II. ML-Ready Data** — PASS. Study manifest + result records validate against versioned `pydantic` schemas; the study is deterministic and reproducible (FR-010).
- **III. Poetry-First** — PASS. All commands run via `poetry run`.
- **IV. SQLite-Only Storage** — **DEVIATION (justified)**. Outputs are CSV/markdown/PNG under `docs/metrics-audit/<date>/`, not SQLite. See Complexity Tracking.
- **V. Engine Wrapper Pattern** — PASS. Study outputs are produced through the existing wrappers / `compress()`; no new engine code.
- **VI. Test-First (4-Layer)** — PASS. Tests added to the correct layers; `make test` must pass.
- **VII. Quality Metrics Preservation** — PASS (directly serves it). The gate *validates* the metric and MUST NOT modify metric code. If the gate returns NO-GO, metric fixes are a separate, follow-up effort under the usual metric-accuracy discipline.

**Result**: PASS with one justified deviation (Principle IV). No blocking violations.

## Project Structure

### Documentation (this feature)

```text
specs/004-metric-validation-gate/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output (how to run the gate end-to-end)
├── contracts/           # Phase 1 — script/CLI contracts (no REST API)
├── checklists/
│   └── requirements.md  # from /speckit.specify
├── spec.md
└── tasks.md             # /speckit.tasks output (not created here)
```

### Source Code (repository root)

```text
scripts/audit/validation_gate/        # NEW — the thin study script set
├── __init__.py
├── build_study.py        # select study GIFs (reuse build_sample.py) + hand-label manifest
│                         #   → generate quality-spread compressions (reuse _common.py / compress())
│                         #   → compute composite_quality per output (calculate_comprehensive_metrics)
├── ranking_sheet.py      # render offline HTML/CSV contact sheet (per-GIF, frames + outputs)
├── ingest_rankings.py    # parse the filled sheet → HumanJudgement records
├── analyse.py            # rank agreement (per-GIF + per-bucket), per-bucket floors, disagreement-by-operation
└── render_report.py      # verdict + tables + figures (reuse report.py helpers)

scripts/audit/_common.py                # REUSED (compression, CSV append, resume)
scripts/audit/build_sample.py           # REUSED (stratified sampler)
scripts/audit/report.py                 # REUSED (markdown + figure helpers)

src/giflab/                             # UNCHANGED — read-only consumer of metrics + compress()

tests/
├── smoke/test_validation_gate_maths.py        # pure: rank agreement, floor derivation, verdict rule
└── functional/test_validation_gate_pipeline.py # mocked compress(): study-set gen + sheet ingestion

docs/metrics-audit/<date>/validation-gate/      # OUTPUTS (gitignored raw data + committed report.md)
├── study_manifest.csv     # GIF → bucket, output → params + composite_quality (reproducibility)
├── ranking_sheet.html     # the artifact the rater fills in
├── human_judgements.csv   # ingested rankings + acceptability boundaries
├── agreement.csv          # per-GIF + per-bucket rank agreement
├── floors.csv             # per-bucket quality floor + spread
└── report.md              # the verdict + disagreement report (committed)
```

**Structure Decision**: Single project. The gate is a new `scripts/audit/validation_gate/` package that *reuses* the established audit harness, keeping it thin and consistent with prior metric-audit runs. `src/giflab/` is a read-only dependency — the gate imports `compress`/`calculate_comprehensive_metrics` but changes no production code. Outputs follow the existing `docs/metrics-audit/<date>/` convention.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Principle IV (SQLite-only storage) — outputs are CSV/markdown/PNG under `docs/metrics-audit/<date>/` | This is a one-off **research artifact** (a dated audit verdict + human-readable report), not training-data pipeline output. It follows the precedent already set by `scripts/audit/` (every prior metrics-audit lives as CSV + `report.md` under `docs/metrics-audit/`). | Forcing the verdict/floors into SQLite would diverge from every prior audit run, add a schema for throwaway study data, and make the human-readable report a second-class artifact — more complexity for a single-run study, no queryability benefit. |
