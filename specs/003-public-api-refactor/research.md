# Phase 0: Research & Decisions

**Feature**: 003-public-api-refactor
**Date**: 2026-05-19
**Status**: Complete

The spec carries no `[NEEDS CLARIFICATION]` markers — assumptions were resolved during `/speckit.specify` and documented in `spec.md` § Assumptions. This document records the **dispatch decisions** that translate those assumptions into concrete internals to wrap.

---

## R-1 — Engine dispatch table

**Decision**: `compress(engine, ...)` looks up the engine string in a frozen dict of `{engine_str → wrapper_class}` and instantiates the wrapper, then calls `apply(input_path, output_path, params=params)`.

**Rationale**: All 5 public engines already have a primary lossy-compression wrapper in `src/giflab/tool_wrappers.py` following the `ExternalTool` protocol (Constitution Principle V). Dispatch by string keeps the public API a thin shell — no parameter parsing duplication, no engine-specific branching in `public_api.py`.

| Public `engine` string | Wrapper class | Source file |
|---|---|---|
| `"animately"` | `AnimatelyLossyCompressor` | `src/giflab/tool_wrappers.py` |
| `"gifsicle"` | `GifsicleLossyCompressor` | `src/giflab/tool_wrappers.py` |
| `"gifski"` | `GifskiLossyCompressor` | `src/giflab/tool_wrappers.py` |
| `"imagemagick"` | `ImageMagickLossyCompressor` | `src/giflab/tool_wrappers.py` |
| `"ffmpeg"` | `FFmpegLossyCompressor` | `src/giflab/tool_wrappers.py` |

**Alternatives considered**:
- *Class-by-class branching* — rejected: more code, harder to extend.
- *Dynamic introspection via `capability_registry`* — rejected for v1: the registry covers all three pipeline slots (frame, colour, lossy), and the public `compress` is intentionally lossy-only. A registry lookup would broaden the API contract beyond the spec.
- *Expose all 7 internal engines (incl. dithered variants)* — rejected per `/speckit.specify` clarification: keep v1 surface small (5 engines), promote later.

---

## R-2 — Metric subset selection

**Decision**: `measure(metrics=[...])` validates the requested metric names against an allowlist of 7, calls `calculate_comprehensive_metrics(reference, candidate)` to get the full result dict, then projects exactly the requested keys into `MeasureResult`. Non-requested fields are `None`.

**Rationale**: `calculate_comprehensive_metrics` already computes conditionally based on internal `MetricsConfig` flags — but those flags govern advanced metrics, not the basic 7. Projecting the result is the simplest correct mapping and avoids touching `metrics.py`.

**Caveat — FR-009 (don't compute unrequested metrics)**: A naive projection still triggers full metric computation. To honour FR-009, we need to gate expensive metrics (LPIPS specifically — loads a torch model, dominates cost) **before** calling `calculate_comprehensive_metrics`.

**Implementation approach**: Build a `MetricsConfig` override that disables metric families the consumer didn't request. `MetricsConfig` is the existing toggle layer; we set its switches based on the requested-metric set before passing it through. The exact switch names will be confirmed during `/speckit.tasks` by reading `metrics.MetricsConfig`; if a clean switch doesn't exist for one of the 7, we add it (counts as a small extension, not a metrics-code modification, since we're only adding a guard around an existing computation).

**Alternatives considered**:
- *Always compute all 7, project the requested subset* — rejected: violates FR-009 (consumers requesting only SSIM should not pay LPIPS cost).
- *Compute each metric independently in `public_api.py`* — rejected: would duplicate logic already in `metrics.py` and violate Constitution Principle VII (Quality Metrics Preservation).

| Public `metric` string | Source key in `calculate_comprehensive_metrics` result dict |
|---|---|
| `"ssim"` | `"ssim"` |
| `"ms_ssim"` | `"ms_ssim"` |
| `"psnr"` | `"psnr"` |
| `"lpips"` | `"lpips"` |
| `"gmsd"` | `"gmsd"` |
| `"fsim"` | `"fsim"` |
| `"chist"` | `"chist"` |

---

## R-3 — `CompressResult` field sourcing

**Decision**: Build the result by combining the wrapper's `apply()` return dict (which provides `render_ms`, `kilobytes`, `command`, `engine`) with caller-known fields (`output_path`, `params`) and one lookup (`engine_version` via the wrapper class's `version()` classmethod).

**Rationale**: `apply()` already returns the metadata the public surface needs (validated via `validate_wrapper_apply_result()`). Bytes are derived from `output_path.stat().st_size` for precision (the existing `kilobytes` field is rounded). Version comes from the wrapper's existing `version()` classmethod — no subprocess call in the hot path is needed if `version()` is cached at module load.

**Field mapping**:

| `CompressResult` field | Source |
|---|---|
| `output_path` | Caller-passed (echo) |
| `output_bytes` | `output_path.stat().st_size` |
| `render_ms` | `wrapper.apply()["render_ms"]` |
| `engine` | Caller-passed (echo) |
| `engine_version` | `WrapperClass.version()` |
| `params` | Caller-passed (echo, immutable copy) |

---

## R-4 — Exception hierarchy

**Decision**: Add three exception types to `public_api.py`, all subclassing the existing `GifLabError` from `src/giflab/error_handling.py`:

- `UnknownEngineError(GifLabError)` — raised before any file I/O when `engine` is not in the dispatch table.
- `UnknownMetricError(GifLabError)` — raised before any computation when a metric is not in the allowlist.
- `EngineUnavailableError(GifLabError)` — raised when the engine binary is missing on `PATH`; wraps the underlying wrapper failure for a typed signal.

**Rationale**: Reuse the existing hierarchy (Constitution doesn't mandate a new one, and `GifLabError` already supports `cause` + `context` for traceability). New typed errors are the minimum the spec requires (FR-014, FR-015) without restructuring `error_handling.py`. Other failures (missing input file, engine subprocess non-zero exit) bubble up via the existing wrapper exceptions — consumers already need to handle those.

**Alternatives considered**:
- *One `PublicAPIError` umbrella* — rejected: spec asks for clear, typed errors that distinguish unknown engine vs unknown metric.
- *Move exceptions to `error_handling.py`* — rejected: keeps public-API exception names co-located with the public functions that raise them. Easier for consumers to import (`from giflab import UnknownEngineError`).

---

## R-5 — Lazy import strategy

**Decision**: `public_api.py` does **not** import heavy modules at module load. `compress` and `measure` import their internals inside the function body (or via a module-level lazy lookup).

**Rationale**: The existing `__init__.py` (lines 122–183) deliberately keeps top-level imports lightweight to avoid triggering torch/LPIPS model loads on `import giflab`. The new public API must preserve this — a consumer who only wants `compress` should not pay LPIPS load cost.

**Implementation**: Lazy import inside function bodies for `metrics` (which transitively loads LPIPS), eager import for `tool_wrappers` (already imported by `__init__.py`).

---

## R-6 — Version pinning & release

**Decision**: Ship the refactor as **`v0.3.0`** (minor bump from current `0.2.0` in `pyproject.toml`). Add a `CHANGELOG.md` entry pinning the public surface. Tag the release after merge.

**Rationale**: Semver minor for a new public API surface. The 0.x prefix signals continued pre-1.0 evolution — appropriate while gifprep is still iterating. Promotion to 1.0 happens when the contract has survived a full gifprep integration cycle without breakage.

**Alternatives considered**:
- *Patch bump (0.2.1)* — rejected: new public surface is not a patch.
- *1.0.0* — rejected: premature stability commitment before gifprep has consumed the contract end-to-end.

---

## R-7 — Test coverage placement

**Decision**: Two test files in the existing 4-layer architecture:

- **`tests/functional/test_public_api.py`** — uses mocked wrappers and mocked `calculate_comprehensive_metrics`. Covers: happy path for each engine string, happy path for each metric subset (1 metric, 3 metrics, all 7), unknown-engine error, unknown-metric error, missing-binary error, `CompressResult`/`MeasureResult` field shape, immutability of result dataclasses, lazy-import preservation (importing `giflab.compress` should not import `lpips`).
- **`tests/integration/test_public_api_e2e.py`** — real engines, real metrics. Covers: `animately` and `gifsicle` end-to-end (the two engines most likely to be available in CI), `measure(metrics=["ssim"])` end-to-end, `measure(metrics=["ssim", "psnr"])` end-to-end. Skips gracefully if a binary is missing.

**Rationale**: Aligns with Constitution Principle VI and the project's own guidance (`tests/smoke/` is too thin for dispatch logic; `tests/nightly/` is overkill for a stable, small surface). Mirrors the `001-compression-curve-prediction` test layout.

---

## R-8 — Documentation home

**Decision**: After this refactor ships, `docs/public-api.md` becomes the **live contract**. `docs/gifprep-integration.md` gets a one-line header update — "Implemented; see `docs/public-api.md`" — and is otherwise preserved as historical context (the proposal that prompted the refactor).

**Rationale**: Spec FR-017 + the integration doc's own "How to proceed" step 7. Avoids two competing sources of truth.

---

## Summary

All decisions are documented; no open questions remain. Phase 1 produces:

1. `data-model.md` — `CompressResult`, `MeasureResult`, `EngineIdentifier`, `MetricIdentifier`
2. `contracts/public_api.md` — function signatures, invariants, error contracts
3. `quickstart.md` — end-to-end script showing both functions from an external consumer's perspective
