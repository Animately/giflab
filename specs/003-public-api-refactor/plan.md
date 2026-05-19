# Implementation Plan: Public API for External Consumers

**Branch**: `003-public-api-refactor` | **Date**: 2026-05-19 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/003-public-api-refactor/spec.md`

## Summary

Expose two new top-level functions on the `giflab` package — `compress(input, output, engine, params) → CompressResult` and `measure(reference, candidate, metrics) → MeasureResult` — as a stable public API that the sibling `gifprep` repository (and future external consumers) can depend on. Implementation is a thin dispatch layer in a new module `src/giflab/public_api.py` that delegates to existing internals (`tool_wrappers.py` lossy compressors for `compress`; `metrics.calculate_comprehensive_metrics` for `measure`). No engine, metric, CLI, dataset, or schema code changes. Ship behind a tagged release (`v0.3.0`) so consumers can pin.

## Technical Context

**Language/Version**: Python 3.11 (per `pyproject.toml`)
**Primary Dependencies**: No new dependencies. Reuses existing internals: `tool_wrappers.AnimatelyLossyCompressor` + 4 sibling compressors, `metrics.calculate_comprehensive_metrics`, `external_engines.common.run_command`, `error_handling.GifLabError` hierarchy.
**Storage**: N/A (filesystem read/write only; no DB)
**Testing**: pytest via Poetry; new tests land in `tests/functional/` (mocked-engine dispatch tests) and `tests/integration/` (real-engine + real-metric end-to-end tests). Per Constitution Principle VI.
**Target Platform**: Python library on macOS dev / Linux CI. Consumed by other Python projects.
**Project Type**: Single-project library refactor.
**Performance Goals**: Zero overhead beyond a function call + dataclass construction (<1 ms wrapper overhead). Engine and metric performance unchanged.
**Constraints**: Must not modify CLI, dataset pipeline, matrix benchmark, SQLite schema, feature extraction, tool interfaces, parameter grids, or metrics implementations (per FR-018, FR-019). Must not introduce preprocessing concepts (per FR-019). API must be importable without triggering heavy module loads (LPIPS model, torch) — lazy import preserved.
**Scale/Scope**: ~150 LOC new module, ~6-line addition to `__init__.py`, ~3 new test files, 1 new doc (`docs/public-api.md`), 1 changelog entry, 1 release tag.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Note |
|---|---|---|
| I. Dataset-First | ✅ PASS | Refactor unlocks `gifprep` (the AI Preprocessing initiative) to consume giflab's compression + metric capabilities as a library. Indirectly serves the dataset mission by enabling preprocessing experiments that feed back into training data quality. |
| II. ML-Ready Data | ✅ PASS | No data pipeline outputs added or modified. Existing schemas untouched. |
| III. Poetry-First | ✅ PASS | All new commands documented as `poetry run ...`. Tests run under Poetry. |
| IV. SQLite-Only Storage | ✅ PASS | No new storage. `compress` writes a GIF file (caller-specified path); `measure` returns a value object. Neither touches the SQLite store. |
| V. Engine Wrapper Pattern | ✅ PASS | `compress` dispatches to existing `*LossyCompressor` classes; does not introduce a new engine, alter the wrapper protocol, or bypass `apply()`. The 5 public engines map 1:1 to existing wrapper classes (the 7 internal engines remain, two — `gifsicle_dithered`, `ffmpeg_dithered` variants — are simply not surfaced publicly in v1). |
| VI. Test-First (4-Layer) | ✅ PASS | New tests placed in `tests/functional/test_public_api.py` (mocked dispatch) and `tests/integration/test_public_api_e2e.py` (real engines + real metrics). No tests in `tests/` root. |
| VII. Quality Metrics Preservation | ✅ PASS | `measure` calls `calculate_comprehensive_metrics` unmodified and selects keys from its returned dict. The 13-metric ecosystem is untouched; we expose 7 of those keys as the public surface. |

**Animately CLI flag-based**: Preserved — the public `compress` function dispatches through `AnimatelyLossyCompressor`, which already uses flag-based invocation.
**External binaries graceful degradation**: When a requested engine binary is missing, the wrapper raises; the public `compress` re-raises as `EngineUnavailableError` (a subclass of the existing `EngineError`) so consumers get a clear, typed signal.

**Result: GATE PASSED. No violations. Complexity Tracking section omitted.**

## Project Structure

### Documentation (this feature)

```text
specs/003-public-api-refactor/
├── plan.md              # This file (/speckit.plan command output)
├── spec.md              # Already committed (23692d1)
├── checklists/
│   └── requirements.md  # Already committed
├── research.md          # Phase 0 output (this command)
├── data-model.md        # Phase 1 output (this command)
├── quickstart.md        # Phase 1 output (this command)
├── contracts/
│   └── public_api.md    # Phase 1 output (this command) — function signatures + invariants
└── tasks.md             # Phase 2 output (/speckit.tasks command — NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/giflab/
├── __init__.py                  # MODIFIED: add 6-line export block for public API
├── public_api.py                # NEW: compress(), measure(), CompressResult, MeasureResult, typed exceptions
├── tool_wrappers.py             # UNCHANGED: dispatch target for compress()
├── metrics.py                   # UNCHANGED: dispatch target for measure()
├── external_engines/
│   └── common.py                # UNCHANGED
└── error_handling.py            # UNCHANGED: GifLabError hierarchy extended in public_api.py only

tests/
├── functional/
│   └── test_public_api.py       # NEW: mocked-engine + mocked-metrics dispatch tests, error paths
└── integration/
    └── test_public_api_e2e.py   # NEW: real engines (animately + gifsicle minimum), real metrics

docs/
├── gifprep-integration.md       # EXISTING (9830d70): marked "implemented — see docs/public-api.md" at end of refactor
└── public-api.md                # NEW: the live contract — signatures, invariants, supported engines/metrics, version pinning guidance

CHANGELOG.md                     # NEW or UPDATED: pin the v0.3.0 public surface
pyproject.toml                   # MODIFIED: version bump 0.2.0 → 0.3.0
```

**Structure Decision**: Single-project library refactor. All work concentrates in one new module (`src/giflab/public_api.py`) + minimal `__init__.py` edits + tests in the appropriate 4-layer test directories. Mirrors the layout used for `001-compression-curve-prediction` and `002-dataset-pipeline-refactor`.

## Post-Design Constitution Re-Check

*Run after Phase 1 artifacts (research.md, data-model.md, contracts/, quickstart.md) are complete.*

All seven principles re-checked against the concrete design:

| Principle | Phase 1 evidence | Status |
|---|---|---|
| I. Dataset-First | Refactor unlocks gifprep (AI Preprocessing initiative) without altering giflab's dataset pipeline. | ✅ PASS |
| II. ML-Ready Data | data-model.md adds value objects that cross the public boundary only. No persisted entities. SQLite untouched. | ✅ PASS |
| III. Poetry-First | quickstart.md uses `poetry install`; all test commands in plan use `poetry run`. | ✅ PASS |
| IV. SQLite-Only Storage | Neither `compress` nor `measure` writes to SQLite. `compress` writes a single GIF file at the caller's path. | ✅ PASS |
| V. Engine Wrapper Pattern | research.md R-1 explicitly dispatches via the 5 existing `*LossyCompressor` classes through their `apply()` method. No new wrapper, no protocol change. | ✅ PASS |
| VI. Test-First (4-Layer) | research.md R-7 places tests in `tests/functional/` and `tests/integration/` per the layer rules. | ✅ PASS |
| VII. Quality Metrics Preservation | research.md R-2 reuses `calculate_comprehensive_metrics` unmodified and projects its result dict. R-2 caveat (FR-009 gating) may add a single `MetricsConfig` switch but does not modify any metric implementation. | ✅ PASS |

**No new violations introduced in Phase 1. Complexity Tracking remains empty. Plan is ready for `/speckit.tasks`.**

## Generated Artifacts

| Artifact | Path | Status |
|---|---|---|
| Plan | `specs/003-public-api-refactor/plan.md` | this file |
| Research | `specs/003-public-api-refactor/research.md` | created |
| Data model | `specs/003-public-api-refactor/data-model.md` | created |
| Contract | `specs/003-public-api-refactor/contracts/public_api.md` | created |
| Quickstart | `specs/003-public-api-refactor/quickstart.md` | created |
| Agent context | `CLAUDE.md` | updated (active-feature block appended) |
