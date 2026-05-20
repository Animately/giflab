# Contract: `giflab` Public API

**Feature**: 003-public-api-refactor
**Module**: `giflab.public_api` (re-exported from `giflab`)
**Stability**: Stable from `v0.3.0` onward. Breaking changes ship as a minor version bump until 1.0, then as a major bump.

This document is the **contract**. It binds the giflab maintainers (what we promise) and the consumer (what they can rely on). After the refactor ships, this document moves to `docs/public-api.md` and becomes the long-lived public-facing reference.

---

## Module imports

All public symbols are importable from the top-level package. Consumers MUST NOT import from `giflab.public_api` directly — that module path may change without notice; the top-level names will not.

```python
from giflab import (
    compress,
    measure,
    CompressResult,
    MeasureResult,
    SUPPORTED_ENGINES,
    SUPPORTED_METRICS,
    UnknownEngineError,
    UnknownMetricError,
    EngineUnavailableError,
)
```

---

## `compress`

```python
def compress(
    input_path: Path,
    output_path: Path,
    engine: EngineIdentifier,
    params: dict[str, Any] | None = None,
) -> CompressResult: ...
```

**Behaviour**:

- Runs the named compression engine on `input_path`, writing the result to `output_path`.
- Returns a `CompressResult` describing the produced file.
- Overwrites `output_path` if a file already exists there.
- Does not mutate `input_path`.

**Parameters**:

| Name | Type | Required | Notes |
|---|---|---|---|
| `input_path` | `pathlib.Path` | yes | Must exist and be readable. |
| `output_path` | `pathlib.Path` | yes | Parent directory must exist and be writable. File at this path will be overwritten. |
| `engine` | `str` (one of `SUPPORTED_ENGINES`) | yes | See R-1 in research.md for the mapping to internal wrappers. |
| `params` | `dict[str, Any] \| None` | no (default `None`) | Engine-specific parameter mapping. `None` is treated as empty dict (engine defaults). Schema is engine-specific — see the engine wrapper docstrings. |

**Returns**: `CompressResult` (see data-model.md).

**Raises**:

| Exception | Condition |
|---|---|
| `UnknownEngineError` | `engine` not in `SUPPORTED_ENGINES`. Raised **before** any file I/O. |
| `EngineUnavailableError` | Engine binary not found on `PATH`. |
| `FileNotFoundError` | `input_path` does not exist. |
| `ValueError` | `params` contains keys the engine wrapper rejects. (Engine-specific.) |
| Subclass of `GifLabError` | Any other engine-side failure (subprocess non-zero exit, etc.). |

**Determinism**: For a given (input file bytes, engine, params) tuple, the output bytes are deterministic to the extent the underlying engine is deterministic. The public wrapper introduces no nondeterminism of its own.

**Thread safety**: Safe to call concurrently from multiple threads or processes, provided the `input_path` and `output_path` arguments do not collide across calls. The function holds no shared mutable state.

---

## `measure`

```python
def measure(
    reference_path: Path,
    candidate_path: Path,
    metrics: list[MetricIdentifier],
) -> MeasureResult: ...
```

**Behaviour**:

- Computes the requested quality metrics between `reference_path` and `candidate_path`.
- Returns a `MeasureResult` with the requested metric fields populated; non-requested fields are `None`.
- Does not invoke the computation for any metric not in `metrics` (FR-009).
- Does not mutate either input file.

**Parameters**:

| Name | Type | Required | Notes |
|---|---|---|---|
| `reference_path` | `pathlib.Path` | yes | Must exist and be readable. |
| `candidate_path` | `pathlib.Path` | yes | Must exist and be readable. |
| `metrics` | `list[str]` (each one of `SUPPORTED_METRICS`) | yes | Must be non-empty. Order is not significant. Duplicates are tolerated and have no effect. |

**Returns**: `MeasureResult` (see data-model.md).

**Raises**:

| Exception | Condition |
|---|---|
| `UnknownMetricError` | Any element of `metrics` not in `SUPPORTED_METRICS`. Raised **before** any computation. |
| `ValueError` | `metrics` is empty. |
| `FileNotFoundError` | Either path does not exist. |
| Subclass of `GifLabError` | Computation failure for any requested metric (per the all-or-nothing semantics from spec.md Edge Cases). The requested metric set is on `context["metrics"]`; when a single metric can be attributed as the failure, `context["metric"]` is also set. |

**Cost**: Requesting only cheap metrics (`ssim`, `psnr`, `chist`) is fast. Requesting `lpips` triggers a PyTorch model load on first call within a process and dominates cost. Subsequent calls within the same process reuse the cached model.

**Thread safety**: Safe to call concurrently. The LPIPS model cache is thread-safe.

---

## Result types

See [data-model.md](../data-model.md) for full field listings of `CompressResult` and `MeasureResult`.

Both are `@dataclass(frozen=True)`. Hash-equal results compare equal. Mutating a field raises `dataclasses.FrozenInstanceError`.

---

## Exception types

All public exceptions inherit from `giflab.error_handling.GifLabError` and carry the existing `cause` and `context` attributes for traceability.

| Exception | Module | Inherits |
|---|---|---|
| `UnknownEngineError` | `giflab.public_api` | `GifLabError` |
| `UnknownMetricError` | `giflab.public_api` | `GifLabError` |
| `EngineUnavailableError` | `giflab.public_api` | `GifLabError` |

Consumers MAY catch any of these as `GifLabError` if they want uniform handling.

---

## Versioning

This contract is bound to giflab `v0.3.0` and later (within the 0.x major). Breaking changes — removing an engine from `SUPPORTED_ENGINES`, removing a metric from `SUPPORTED_METRICS`, changing a return field, renaming a public symbol — ship as a minor bump (e.g., `v0.4.0`) with a CHANGELOG entry. Additive changes — adding a new supported engine or metric, adding an optional parameter with a default — ship as a patch bump.

Consumers SHOULD pin to a version string in their dependency manifest (e.g., `giflab = "^0.3.0"` in their `pyproject.toml`).

---

## What this contract does **not** promise

- Internal module paths (`giflab.public_api`, `giflab.tool_wrappers`, `giflab.metrics`, etc.) may move or be renamed at any time. Consumers must use top-level imports.
- Internal exception types other than the three listed above are not part of the contract.
- The CLI surface (`python -m giflab run ...`) is not part of the public API — it is a separate, separately-versioned surface for direct use.
- Engine-specific `params` dict schemas are documented in the engine wrappers' docstrings, but those schemas are not promised to be stable across giflab versions — consumers writing engine-specific param dicts accept that responsibility.
