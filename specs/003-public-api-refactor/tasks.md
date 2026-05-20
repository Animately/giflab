---
description: "Task list for 003-public-api-refactor implementation"
---

# Tasks: Public API for External Consumers

**Input**: Design documents from `/specs/003-public-api-refactor/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/public_api.md, quickstart.md (all present)

**Tests**: Tests are REQUIRED per Constitution Principle VI (Test-First, 4-Layer Architecture). New tests land in `tests/smoke/`, `tests/functional/`, and `tests/integration/` according to the layer rules in CLAUDE.md.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing. US1 (compress) is the MVP. US2 (measure) and US3 (versioned release) layer on top.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- File paths are absolute or repo-root-relative

## Path Conventions

Single-project library refactor. All source lives under `src/giflab/`, all tests under `tests/<layer>/`. Docs under `docs/`. See plan.md § Project Structure.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project version bump in preparation for the new public surface. No new dependencies — public API reuses existing internals.

- [X] T001 Bump giflab version `0.2.0` → `0.3.0` in `pyproject.toml`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Create the empty module that both US1 and US2 will fill. Without this, neither story can begin implementation.

**⚠️ CRITICAL**: No user story implementation tasks can begin until T002 is complete.

- [X] T002 Create `src/giflab/public_api.py` with module docstring ("Public API surface for external giflab consumers — see docs/public-api.md") and lightweight top-of-file imports only: `from __future__ import annotations`, `from dataclasses import dataclass`, `from pathlib import Path`, `from typing import Any, Literal`, `from giflab.error_handling import GifLabError`. **Do not** import `metrics` or `tool_wrappers` at module top — those imports go inside function bodies (research.md R-5 lazy-import strategy).

**Checkpoint**: Foundation ready — US1 and US2 implementation can now begin.

---

## Phase 3: User Story 1 - Compress (Priority: P1) 🎯 MVP

**Goal**: External consumer can call `giflab.compress(input, output, engine, params)` and receive a structured `CompressResult` from any of the 5 supported engines (animately, gifsicle, gifski, imagemagick, ffmpeg).

**Independent Test**: From a separate Python process, `from giflab import compress` then call it with a sample GIF + gifsicle + `{"colors": 64}` and verify the output file exists at the requested path and `result.output_bytes > 0`.

### Tests for User Story 1

> Write tests FIRST; they should FAIL until T007–T011 land.

- [X] T003 [P] [US1] Create `tests/functional/test_public_api_compress.py` with mocked-wrapper dispatch tests covering: (a) happy path for each of the 5 engines (parametrized), (b) `UnknownEngineError` raised before any I/O when engine string is invalid, (c) `EngineUnavailableError` raised when wrapper's `apply()` reports missing binary, (d) `CompressResult` field shape matches data-model.md (output_path echoed, output_bytes from stat, render_ms from wrapper, engine echoed, engine_version from `WrapperClass.version()`, params is an immutable copy), (e) `CompressResult` is frozen (mutation raises `FrozenInstanceError`), (f) caller's `params` dict can be mutated after the call without affecting the result.
- [X] T004 [P] [US1] Create `tests/integration/test_public_api_compress_e2e.py` with real-engine end-to-end tests: compress a synthetic test GIF using `animately` and `gifsicle` at non-trivial params (`lossy_level=40` for animately; `colors=64` for gifsicle). Skip each engine gracefully if its binary is missing via `pytest.importorskip`-style check. Verify output file exists, `output_bytes > 0`, `render_ms > 0`, `engine_version` is non-empty.
- [X] T005 [P] [US1] Create `tests/smoke/test_public_api_imports.py` with a single test asserting that `from giflab import compress, CompressResult, SUPPORTED_ENGINES, UnknownEngineError, EngineUnavailableError` succeeds. Use `sys.modules` snapshotting to assert that after the import, `lpips` and `torch` are NOT in `sys.modules` (enforces the lazy-import strategy from research.md R-5).

### Implementation for User Story 1

- [X] T006 [US1] In `src/giflab/public_api.py`, add the `CompressResult` frozen dataclass with fields `output_path: Path`, `output_bytes: int`, `render_ms: int`, `engine: str`, `engine_version: str`, `params: dict[str, Any]`. Per data-model.md, store `params` as a shallow copy (`dict(params)`) in `__post_init__` to honour the immutability invariant.
- [X] T007 [US1] In `src/giflab/public_api.py`, add `SUPPORTED_ENGINES: tuple[str, ...] = ("animately", "gifsicle", "gifski", "imagemagick", "ffmpeg")` and `EngineIdentifier = Literal["animately", "gifsicle", "gifski", "imagemagick", "ffmpeg"]`. Add a module-level `_ENGINE_DISPATCH` dict mapping each string to the corresponding wrapper class imported lazily from `giflab.tool_wrappers`: `AnimatelyLossyCompressor`, `GifsicleLossyCompressor`, `GifskiLossyCompressor`, `ImageMagickLossyCompressor`, `FFmpegLossyCompressor`. Wrap the import + dict construction in a `_get_engine_dispatch()` helper that builds the dict on first call and caches it, so module import remains lightweight.
- [X] T008 [US1] In `src/giflab/public_api.py`, add exception classes `UnknownEngineError(GifLabError)` and `EngineUnavailableError(GifLabError)`. Constructors should accept the offending engine string and (for UnknownEngineError) the supported list, producing messages like `"Unknown engine 'foo'. Supported: animately, gifsicle, gifski, imagemagick, ffmpeg"` and `"Engine 'gifski' binary not found on PATH"`.
- [X] T009 [US1] In `src/giflab/public_api.py`, implement the `compress(input_path, output_path, engine, params=None) -> CompressResult` function. Order of operations: (1) validate engine is in `SUPPORTED_ENGINES`, raise `UnknownEngineError` if not; (2) lazy-look up the wrapper class via `_get_engine_dispatch()`; (3) check `WrapperClass.available()`, raise `EngineUnavailableError` if not; (4) check input_path exists, raise `FileNotFoundError`; (5) instantiate the wrapper, call `wrapper.apply(input_path, output_path, params=params or {})`; (6) build and return `CompressResult` using `output_path.stat().st_size` for `output_bytes`, `apply()` return dict for `render_ms`, and `WrapperClass.version()` for `engine_version`. Per contracts/public_api.md, the function must not catch generic exceptions — let wrapper errors propagate.
- [X] T010 [US1] In `src/giflab/__init__.py`, append a new export block (after the existing `tool_wrappers` block, before the file ends): `from .public_api import compress, CompressResult, SUPPORTED_ENGINES, EngineIdentifier, UnknownEngineError, EngineUnavailableError`. Keep the import lazy-safe — `public_api.py` itself does not eagerly import heavy modules, so this addition does not regress the existing lightweight-import contract.
- [X] T011 [US1] Run `poetry run pytest tests/smoke/test_public_api_imports.py tests/functional/test_public_api_compress.py tests/integration/test_public_api_compress_e2e.py -v` and confirm all tests pass (integration tests for unavailable engines should report as skipped, not failed).

**Checkpoint**: US1 complete. An external consumer can `from giflab import compress` and run a single compression engine end-to-end. MVP is shippable here even if US2 is not yet done — gifprep can begin wiring the compression half of its harness against a `v0.3.0-rc1` pre-release.

---

## Phase 4: User Story 2 - Measure (Priority: P2)

**Goal**: External consumer can call `giflab.measure(reference, candidate, metrics)` and receive a `MeasureResult` populating exactly the requested metrics from the 7 supported (ssim, ms_ssim, psnr, lpips, gmsd, fsim, chist), without paying the cost of unrequested metrics.

**Independent Test**: From a separate Python process, call `measure(ref, cand, metrics=["ssim"])` and verify only `result.ssim` is non-None, all other fields are None. Then call `measure(ref, cand, metrics=["ssim", "psnr"])` and verify both are populated, others remain None. Process must not have loaded the LPIPS model in either call (assert via `sys.modules`).

### Tests for User Story 2

> Write tests FIRST; they should FAIL until T015–T020 land.

- [X] T012 [P] [US2] Create `tests/functional/test_public_api_measure.py` with mocked-`calculate_comprehensive_metrics` tests covering: (a) happy path for each of the 7 metrics requested individually (parametrized), (b) multi-metric requests (`["ssim", "psnr"]`, `["ssim", "ms_ssim", "lpips"]`, all 7), (c) `MeasureResult` field shape matches data-model.md — requested metrics populated with floats, non-requested fields are None, (d) `UnknownMetricError` raised before any computation when metric string is invalid, (e) `ValueError` raised when `metrics=[]`, (f) duplicate metric names in the list are tolerated, (g) all-or-nothing semantics: if the mock raises for one metric, no partial result is returned, the raised exception's `context["metric"]` names the failing metric, (h) **FR-009 cost-avoidance**: when `metrics=["ssim"]`, the mock's `calculate_comprehensive_metrics` is called with a `MetricsConfig` that has LPIPS disabled (verify via call-args inspection on the mock).
- [X] T013 [P] [US2] Create `tests/integration/test_public_api_measure_e2e.py` with real-metric tests: use a known reference GIF and a slightly-compressed candidate (build the candidate inline via `compress(...)` from US1). Test (a) `metrics=["ssim"]` returns a float in `[0.0, 1.0]`, (b) `metrics=["ssim", "psnr"]` returns both, (c) `metrics=["ms_ssim"]` returns a float, (d) `metrics=["chist"]` returns a float. Skip LPIPS in integration if model download is unavailable in CI.

### Implementation for User Story 2

- [X] T014 [US2] In `src/giflab/public_api.py`, add `SUPPORTED_METRICS: tuple[str, ...] = ("ssim", "ms_ssim", "psnr", "lpips", "gmsd", "fsim", "chist")` and `MetricIdentifier = Literal["ssim", "ms_ssim", "psnr", "lpips", "gmsd", "fsim", "chist"]`.
- [X] T015 [US2] In `src/giflab/public_api.py`, add the `MeasureResult` frozen dataclass with 7 optional float fields (one per metric in SUPPORTED_METRICS), all defaulting to `None`. Per data-model.md, fields are `ssim`, `ms_ssim`, `psnr`, `lpips`, `gmsd`, `fsim`, `chist`, each typed `float | None`.
- [X] T016 [US2] In `src/giflab/public_api.py`, add `UnknownMetricError(GifLabError)` with constructor accepting the offending metric string and supported list, producing a message like `"Unknown metric 'foo'. Supported: ssim, ms_ssim, psnr, lpips, gmsd, fsim, chist"`.
- [X] T017 [US2] Read `src/giflab/config.py` and `src/giflab/metrics.py::MetricsConfig` to identify which existing config switches gate each of the 7 metrics. For any metric in SUPPORTED_METRICS that **cannot** be individually disabled via an existing switch, add a new boolean switch to `MetricsConfig` (e.g., `ENABLE_LPIPS: bool = True`) and add a corresponding guard around the metric's computation in `metrics.py`. **Scope guard**: this is the only modification permitted to `metrics.py` in this refactor — add a single `if not config.ENABLE_<METRIC>: skip` line per missing switch, nothing else. Document any new switches added in research.md R-2 as a follow-up note.
- [X] T018 [US2] In `src/giflab/public_api.py`, implement the `measure(reference_path, candidate_path, metrics) -> MeasureResult` function. Order of operations: (1) validate `metrics` is non-empty, raise `ValueError` if empty; (2) validate every entry is in `SUPPORTED_METRICS`, raise `UnknownMetricError` on first unknown; (3) validate both paths exist, raise `FileNotFoundError` otherwise; (4) lazy-import `calculate_comprehensive_metrics` and `MetricsConfig` from `giflab.metrics`; (5) build a `MetricsConfig` instance with switches turned OFF for metrics NOT in the requested set (using the switch names confirmed/added in T017); (6) call `calculate_comprehensive_metrics(reference_path, candidate_path, config=config)`; (7) on any metric computation failure, wrap as a `GifLabError` with `context={"metric": <name>}` and raise (all-or-nothing per spec.md Edge Cases); (8) construct and return `MeasureResult` projecting the dict's `"ssim"`, `"ms_ssim"`, etc. keys into the requested fields, leaving non-requested fields as `None`.
- [X] T019 [US2] In `src/giflab/__init__.py`, extend the public-API export block from T010 to add: `measure, MeasureResult, SUPPORTED_METRICS, MetricIdentifier, UnknownMetricError`. Final exported set: 11 names from `public_api`.
- [X] T020 [US2] Run `poetry run pytest tests/functional/test_public_api_measure.py tests/integration/test_public_api_measure_e2e.py -v` and confirm all tests pass. Also re-run T011's test set to confirm US2 changes did not regress US1.

**Checkpoint**: US2 complete. An external consumer can now call both `compress` and `measure`. The full Pareto-tradeoff harness shape from quickstart.md § 4 works end-to-end.

---

## Phase 5: User Story 3 - Versioned Release (Priority: P3)

**Goal**: A tagged release of giflab pins the public API surface so external consumers (gifprep first) can depend on a specific version.

**Independent Test**: After merge, verify `git tag` shows `v0.3.0`, `CHANGELOG.md` documents the v0.3.0 public surface, `docs/public-api.md` exists as the canonical contract, and `docs/gifprep-integration.md` header marks the refactor as implemented.

### Implementation for User Story 3

- [X] T021 [US3] Create `docs/public-api.md` from `specs/003-public-api-refactor/contracts/public_api.md`. Polish for external readers: drop the "Feature" / "Module" header metadata, drop internal cross-references to research.md, keep the function signatures, behaviour, parameters, raises, determinism, thread-safety, versioning, and "What this contract does not promise" sections. Add a short intro paragraph explaining giflab's role and the one-way library dependency from gifprep.
- [X] T022 [US3] Update `docs/gifprep-integration.md` — replace the `**Status**:` line with `**Status**: Implemented as of v0.3.0 — see [public-api.md](./public-api.md) for the live contract. This document is preserved as the historical proposal.`. Leave the rest of the file unchanged.
- [X] T023 [US3] Create or update `CHANGELOG.md` in the repo root with a `## v0.3.0 — 2026-MM-DD` entry pinning the public API surface: list each exported name (`compress`, `measure`, `CompressResult`, `MeasureResult`, `SUPPORTED_ENGINES`, `SUPPORTED_METRICS`, `EngineIdentifier`, `MetricIdentifier`, `UnknownEngineError`, `UnknownMetricError`, `EngineUnavailableError`), list the 5 supported engines and 7 supported metrics, note the all-or-nothing measurement semantics, and reference `docs/public-api.md` as the source of truth. Substitute the actual merge date for `MM-DD`.
- [ ] T024 [US3] (Post-merge, manual) After the PR merges to main, tag the release: `git tag -a v0.3.0 -m "Public API for external consumers"` then `git push origin v0.3.0`. Document this step in the PR description; do not execute it during implementation.

**Checkpoint**: All user stories complete. gifprep can now update its `pyproject.toml` to pin `giflab = "^0.3.0"` and proceed with `specs/002-benchmark-harness/spec.md`.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Verify the full test suite and quickstart end-to-end before the PR opens.

- [X] T025 [P] Run `poetry run black src/giflab/public_api.py src/giflab/__init__.py tests/smoke/test_public_api_imports.py tests/functional/test_public_api_compress.py tests/functional/test_public_api_measure.py tests/integration/test_public_api_compress_e2e.py tests/integration/test_public_api_measure_e2e.py` to format.
- [X] T026 [P] Run `poetry run ruff check src/giflab/public_api.py tests/` and fix any issues.
- [X] T027 [P] Run `poetry run mypy src/giflab/public_api.py` and resolve type errors.
- [X] T028 Run the full test suite: `make test` (smoke + functional, must pass) then `make test-ci` (adds integration, must pass).
- [X] T029 Execute `specs/003-public-api-refactor/quickstart.md` § 4 end-to-end against a real sample GIF — copy the code block into a scratch script, run with `poetry run python`, confirm the Pareto-shape output matches expectations. This validates SC-001 (consumer can integrate in fewer than 10 lines without internal imports).

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: T001 has no dependencies. Can run anytime before tag.
- **Foundational (Phase 2)**: T002 has no dependencies. **BLOCKS all US1/US2 implementation tasks.**
- **US1 (Phase 3)**: Depends on T002.
- **US2 (Phase 4)**: Depends on T002. May begin in parallel with US1 once T002 lands, but shares `src/giflab/public_api.py` and `src/giflab/__init__.py` with US1 — sequential execution is simpler.
- **US3 (Phase 5)**: Documentation-only; depends on US1 and US2 being feature-complete so the CHANGELOG can list the actual exported surface accurately.
- **Polish (Phase 6)**: Depends on US1, US2, US3 all complete.

### User Story Dependencies

- **US1 (P1, Compress)**: Independent of US2 and US3 in terms of the contract surface — but shares files with US2 (see note above).
- **US2 (P2, Measure)**: Independent test surface, but task T020 also re-runs US1's test set as a regression check.
- **US3 (P3, Versioned release)**: Depends on US1 and US2 being shipped to populate the CHANGELOG correctly.

### Within Each User Story

- Tests (T003–T005, T012–T013) MUST be written first and confirmed to FAIL before implementation tasks for that story begin (Constitution Principle VI — Test-First).
- Within implementation: dataclasses (T006, T015) and identifier constants (T007, T014) before exception types (T008, T016) before functions (T009, T018) before `__init__.py` export (T010, T019) before test-run verification (T011, T020).

### Parallel Opportunities

- **Within US1**: T003, T004, T005 can be written in parallel (different test files, no production code yet).
- **Within US2**: T012 and T013 can be written in parallel (different test files).
- **Cross-story file conflict**: T010 and T019 both edit `src/giflab/__init__.py`. T009 and T018 both edit `src/giflab/public_api.py`. **These are not [P] across stories.** Run US1 implementation fully, then US2 implementation, to avoid merge conflicts.
- **Polish**: T025, T026, T027 can run in parallel (different commands, no shared state).

---

## Parallel Example: User Story 1 Tests

```bash
# Write all US1 test files in parallel (no shared file, no production code dependency yet):
Task: "Functional dispatch tests for compress() in tests/functional/test_public_api_compress.py"
Task: "Integration end-to-end tests for compress() in tests/integration/test_public_api_compress_e2e.py"
Task: "Import-only smoke test in tests/smoke/test_public_api_imports.py"

# Then run them and confirm they fail with the expected ImportError:
poetry run pytest tests/smoke/test_public_api_imports.py tests/functional/test_public_api_compress.py tests/integration/test_public_api_compress_e2e.py -v
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001 — version bump).
2. Complete Phase 2: Foundational (T002 — empty module).
3. Complete Phase 3: User Story 1 (T003–T011).
4. **STOP and VALIDATE**: External consumer can compress GIFs through the public API. This alone unblocks the compression half of gifprep's harness.
5. Optionally cut a pre-release tag (`v0.3.0-rc1`) so gifprep can prototype against the compress surface while measure is in flight.

### Incremental Delivery

1. Setup + Foundational → infrastructure ready (1 file, 1 line bump).
2. US1 → compress works → gifprep's harness compresses → demo Pareto-shape compression-only sweep (no quality scores yet).
3. US2 → measure works → gifprep gets quality scores → full Pareto report.
4. US3 → tagged release → gifprep pins by version → contract is stable.
5. Each story is shippable independently as a pre-release; only US3's tag is the final cut.

### Parallel Team Strategy

With multiple developers, this refactor is too small to benefit much from parallelism — single-developer sequential execution will finish faster than coordinating around shared files. Recommended: one developer end-to-end.

---

## Notes

- [P] tasks = different files, no dependencies.
- [Story] label maps task to specific user story for traceability.
- All Python commands MUST use `poetry run` per Constitution Principle III.
- Tests verified to FAIL before implementation (Constitution Principle VI).
- Commit after each completed task or logical group (e.g., commit after T002, commit after T011 with all of US1, commit after T020 with all of US2, commit after T023 with all of US3 documentation).
- Stop at any checkpoint to validate independently. The MVP checkpoint after T011 is the natural place to cut a pre-release if you want gifprep to start integrating early.
- Avoid: cross-story file conflicts (US1 and US2 share `public_api.py` and `__init__.py` — sequential is simpler), vague tasks, parameter validation duplication between public API and engine wrappers (let wrappers own their param schemas per research.md R-3 and spec.md Assumptions).
