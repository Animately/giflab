# Feature Specification: Public API for External Consumers

**Feature Branch**: `003-public-api-refactor`
**Created**: 2026-05-19
**Status**: Draft
**Input**: User description: contents of `docs/gifprep-integration.md` — proposal to expose `giflab.compress` and `giflab.measure` as a stable, documented public API so sibling repositories (initially [gifprep](https://github.com/Animately/gifprep)) can depend on giflab as a library.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - External consumer runs a single compression engine (Priority: P1)

A downstream Python project (e.g., gifprep's benchmark harness) needs to compress a GIF with a specific engine and parameters, without reading giflab's internal modules or learning how the engine wrappers, parameter grids, or dataset pipeline are structured. The consumer calls one importable function, passes the input path, output path, engine name, and a parameter dictionary, and receives back a structured result describing the produced file.

**Why this priority**: This is the foundational capability. The entire two-repo split hinges on gifprep being able to invoke any of giflab's compression engines from outside the repository. Without `compress`, no Pareto-tradeoff harness in gifprep can run. Every later scenario depends on it.

**Independent Test**: Install giflab from its tagged release into a clean virtual environment, write a 5-line script that imports `compress`, points at a sample GIF, runs one engine at a non-trivial setting, and reads back the result fields. No giflab source is touched; only the public API is used.

**Acceptance Scenarios**:

1. **Given** giflab is installed as a dependency in a separate project, **When** the consumer calls the public compression function with a valid input file, a writable output path, a known engine name, and engine-appropriate parameters, **Then** a compressed GIF is written to the output path and a structured result is returned containing the output path, output byte size, render duration, engine name, engine version, and the parameters used.
2. **Given** the consumer requests a known engine with an empty/default parameter set, **When** the function runs, **Then** the engine executes with its documented defaults and the result reflects those defaults in its `params` field.
3. **Given** the consumer requests an unknown engine name, **When** the function is called, **Then** a clear, typed error is raised before any file work begins, naming the unknown engine and listing the supported engines.

---

### User Story 2 - External consumer measures quality between two GIFs (Priority: P2)

The consumer has produced a candidate GIF (typically via `compress` from story 1, or via their own preprocessing + compression chain) and needs to score its quality against a reference GIF using one or more of giflab's quality metrics. The consumer specifies which metrics they want; unrequested metrics are not computed (to avoid paying their cost — e.g., LPIPS model loading).

**Why this priority**: Necessary for the Pareto tradeoff that gifprep's benchmark harness produces (size vs. quality at matched levels), but only meaningful once compressed candidates exist. Strictly downstream of story 1.

**Independent Test**: With giflab installed as a dependency, call the public measurement function on a known reference + candidate pair, requesting first `[ssim]` alone, then `[ssim, ms_ssim]`, and verify the returned result populates exactly the requested fields and leaves the rest unpopulated.

**Acceptance Scenarios**:

1. **Given** a reference GIF and a candidate GIF, **When** the consumer calls the public measurement function with a list of metric names that are all supported, **Then** a structured result is returned populating each requested metric with a numeric score and leaving non-requested metrics unpopulated.
2. **Given** the consumer requests an unknown metric name, **When** the function is called, **Then** a clear, typed error is raised before computation begins, naming the unknown metric and listing the supported metrics.
3. **Given** the consumer requests only `ssim`, **When** the function runs, **Then** the more expensive metrics (notably LPIPS, which requires model loading) are not invoked.

---

### User Story 3 - Consumer pins to a versioned API contract (Priority: P3)

The consumer needs assurance that the public surface they bind against will not change unannounced. They pin their dependency on giflab to a specific released version. Future giflab work on internal modules (engines, metrics, dataset pipeline) does not break the consumer's pinned version. Future intentional changes to the public surface ship as new versions, documented in a changelog.

**Why this priority**: Stability is a contract-level concern, not a runtime feature. It enables the two-repo split to survive over time but is not required for the first end-to-end demo.

**Independent Test**: Verify a tagged giflab release exists on the repository, a changelog entry documents the public API surface for that tag, and gifprep's `specs/002-benchmark-harness/spec.md` is updated to reference that exact tag.

**Acceptance Scenarios**:

1. **Given** the refactor has shipped, **When** the consumer reads the repository's release page, **Then** they find a version tag with a changelog entry that pins the exact public API surface (function names, parameters, result shapes, supported engines, supported metrics).
2. **Given** a future internal refactor of an engine wrapper or metric implementation, **When** the consumer continues to use the pinned version, **Then** the consumer's code continues to function unchanged.

---

### Edge Cases

- The input file does not exist or is unreadable → typed error before engine invocation, naming the missing path.
- The output path is in a directory that does not exist or is unwritable → typed error before engine invocation.
- The output path exists already → the function overwrites it without prompting (this is the documented invariant; consumers manage their own collision policy).
- The engine produces a file but exits non-zero (engine-internal failure) → the underlying engine error surfaces to the consumer; no swallowed exceptions.
- A requested metric is supported in principle but its computation fails on the supplied frames (e.g., dimension mismatch, model load failure for LPIPS) → the failure surfaces to the consumer with the metric name attached. Partial results from other requested metrics are not returned (an "all-or-nothing" measurement call simplifies the consumer's error handling).
- The two functions are called concurrently from multiple threads/processes against different files → safe, since each call operates on independent paths and shares no mutable state.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST expose a single top-level function named `compress` that runs one compression engine on an input GIF and writes the result to a caller-specified output path.
- **FR-002**: The `compress` function MUST accept an input path, an output path, an engine identifier, and an engine-specific parameter mapping.
- **FR-003**: The `compress` function MUST return a structured result containing the output path, the output file size in bytes, the render duration in milliseconds, the engine identifier, the engine version string, and the parameter mapping that was used.
- **FR-004**: System MUST support these engine identifiers through the public `compress` function: `animately`, `gifsicle`, `gifski`, `imagemagick`, `ffmpeg`.
- **FR-005**: System MUST expose a single top-level function named `measure` that computes quality metrics between a reference GIF and a candidate GIF.
- **FR-006**: The `measure` function MUST accept a reference path, a candidate path, and a list of requested metric identifiers.
- **FR-007**: The `measure` function MUST return a structured result with a field for each supported metric; fields corresponding to requested metrics MUST be populated with a numeric score, fields for non-requested metrics MUST be left unpopulated.
- **FR-008**: System MUST support these metric identifiers through the public `measure` function: `ssim`, `ms_ssim`, `psnr`, `lpips`, `gmsd`, `fsim`, `chist`.
- **FR-009**: The `measure` function MUST NOT invoke the computation for any metric that was not requested (to avoid the cost of expensive optional metrics such as LPIPS model loading).
- **FR-010**: Both functions and both result types MUST be importable from the top-level `giflab` package without requiring the consumer to know any internal module path.
- **FR-011**: Neither function MUST mutate its input file in any way.
- **FR-012**: The `compress` function MUST be deterministic: identical input + engine + parameters MUST produce identical output bytes, to the extent the underlying engine itself is deterministic.
- **FR-013**: The `compress` function MUST overwrite the output path if a file already exists there.
- **FR-014**: System MUST raise a clear, typed error before any file I/O when the consumer passes an unknown engine identifier; the error MUST name the unknown identifier and list the supported identifiers.
- **FR-015**: System MUST raise a clear, typed error before any computation when the consumer passes an unknown metric identifier; the error MUST name the unknown identifier and list the supported identifiers.
- **FR-016**: System MUST be released with an explicit version tag after this refactor lands, so external consumers can pin against a specific public API surface.
- **FR-017**: The public API surface (function names, parameters, result shapes, supported engines, supported metrics, invariants) MUST be documented in a single canonical location within the repository, distinct from the in-flight tactical refactor document.
- **FR-018**: The existing CLI, dataset generation pipeline, matrix benchmark, SQLite schema, feature extraction code, internal tool interfaces, and engine parameter grids MUST NOT change as part of this refactor.
- **FR-019**: The existing three-slot tool interface (`frame_reduction`, `color_reduction`, `lossy_compression`) and dynamic pipeline structure MUST NOT be restructured; preprocessing concepts MUST NOT be introduced into giflab.

### Key Entities *(include if feature involves data)*

- **CompressResult**: The structured return value of the `compress` function. Carries the produced output path, the produced output file size in bytes, the render duration in milliseconds, the engine identifier used, the engine version string, and the parameter mapping that was used. Immutable.
- **MeasureResult**: The structured return value of the `measure` function. Carries one optional numeric field per supported metric (`ssim`, `ms_ssim`, `psnr`, `lpips`, `gmsd`, `fsim`, `chist`). Each field is populated iff that metric was requested in the call. Immutable.
- **EngineIdentifier**: A bounded set of engine name strings recognised by `compress`. Exactly the five listed in FR-004 for this release.
- **MetricIdentifier**: A bounded set of metric name strings recognised by `measure`. Exactly the seven listed in FR-008 for this release.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A new external project can install giflab from a tagged release into a clean virtual environment and successfully call both `compress` and `measure` in a script of fewer than 10 lines without importing any internal giflab module.
- **SC-002**: gifprep's benchmark harness, currently blocked on this work, successfully pins to a published giflab version tag and exercises both public functions across all 5 supported engines and at least 2 of the 7 supported metrics, producing a Pareto report end-to-end.
- **SC-003**: 100% of the 5 documented engines and 100% of the 7 documented metrics are reachable through the public surface; no consumer needs to drop down to internal modules to reach any of them.
- **SC-004**: 0 existing functional tests in the giflab test suite regress as a result of this refactor (CLI, dataset generation, matrix benchmark, metrics, engines all continue to behave identically).
- **SC-005**: A developer unfamiliar with giflab internals can read the public-API documentation and integrate both functions into a new project in under 30 minutes.
- **SC-006**: After release, future internal-only refactors (engine wrapper internals, metric implementation details, dataset pipeline) ship without breaking the pinned consumer at the released tag.

## Assumptions

- The five engines listed in FR-004 (`animately`, `gifsicle`, `gifski`, `imagemagick`, `ffmpeg`) cover the current scope. The integration document mentions these explicitly. The codebase additionally exposes dithered variants of `gifsicle` and `ffmpeg` internally; these are intentionally excluded from the first public release surface to keep the API stable and small. They can be added in a later minor version if a consumer needs them.
- The seven metrics listed in FR-008 align with the consumer's stated needs in the integration document. The full `calculate_comprehensive_metrics` function produces 13 metrics internally; the additional six remain available through internal APIs but are not promised by the public surface in this release.
- "Engine version string" (FR-003) is whatever the underlying engine reports (e.g., the output of `--version` or equivalent), captured at call time.
- Released with a version tag (FR-016) means a git tag on the giflab repository; the form (e.g., `v0.2.0`) follows the project's existing versioning convention.
- Engine-specific parameter validation (the contents of the `params` mapping for each engine) is delegated to the existing engine wrappers; the public `compress` function does not re-validate them. This avoids duplicating the parameter schemas that already live in internal wrappers.
- The all-or-nothing semantics for partial metric failure (Edge Cases) is a deliberate choice: simpler consumer error handling, and a failing metric typically indicates a real problem the consumer should know about rather than silently absorb.

## Out of Scope

- Adding new compression engines.
- Adding new quality metrics.
- Any preprocessing capability (denoising, cleanup, generative reimagining) — preprocessing is gifprep's concern, always.
- Pipeline chaining (compress→compress, preprocess→compress as a single call) — consumers compose these themselves.
- Async/streaming variants of either function.
- Multi-file batch APIs — consumers loop over files themselves.
- Changes to the existing CLI surface (`python -m giflab run`, etc.).
- Changes to the dataset-generation pipeline, matrix benchmark, SQLite schema, or feature extraction.
- Restructuring of internal tool interfaces (`tool_interfaces.py`, `dynamic_pipeline.py`).
