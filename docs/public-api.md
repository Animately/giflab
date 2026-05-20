# giflab Public API

**Stability**: Stable from `v0.3.0` onward.
**Module**: All public symbols are importable from the top-level `giflab` package.

giflab is a Python library for GIF compression, quality metrics, and ML training-dataset generation. This document is the **contract** for external consumers — projects that depend on giflab as a library rather than using its CLI directly. The primary current consumer is [gifprep](https://github.com/Animately/gifprep), a sibling repository that adds preprocessing strategies on top of giflab's compression and measurement primitives. The dependency is one-way: gifprep imports giflab; giflab does not depend on gifprep.

---

## Imports

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

Consumers **MUST NOT** import from `giflab.public_api` directly — that module path may change without notice; the top-level names will not.

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
| `engine` | `str` (one of `SUPPORTED_ENGINES`) | yes | Engine identifier. |
| `params` | `dict[str, Any] \| None` | no (default `None`) | Engine-specific parameter mapping. `None` is treated as empty dict (engine defaults). Schema is engine-specific — engines that require `lossy_level` will raise `ValueError` if it is missing. |

**Returns**: `CompressResult` (see [Result types](#result-types) below).

**Raises**:

| Exception | Condition |
|---|---|
| `UnknownEngineError` | `engine` not in `SUPPORTED_ENGINES`. Raised **before** any file I/O. |
| `EngineUnavailableError` | Engine binary not found on `PATH`. |
| `FileNotFoundError` | `input_path` does not exist. |
| `ValueError` | `params` is missing keys the engine wrapper requires. |
| Subclass of `GifLabError` | Any other engine-side failure (subprocess non-zero exit, etc.). |

**Determinism**: For a given (input file bytes, engine, params) tuple, the output bytes are deterministic to the extent the underlying engine is deterministic. The public wrapper introduces no nondeterminism of its own.

**Thread safety**: Safe to call concurrently from multiple threads or processes, provided the `input_path` and `output_path` arguments do not collide across calls.

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
- The expensive LPIPS metric (loads a PyTorch model) is only computed when requested.
- Does not mutate either input file.

**Parameters**:

| Name | Type | Required | Notes |
|---|---|---|---|
| `reference_path` | `pathlib.Path` | yes | Must exist and be readable. |
| `candidate_path` | `pathlib.Path` | yes | Must exist and be readable. |
| `metrics` | `list[str]` (each one of `SUPPORTED_METRICS`) | yes | Must be non-empty. Order is not significant. Duplicates are tolerated and have no effect. |

**Returns**: `MeasureResult`.

**Raises**:

| Exception | Condition |
|---|---|
| `UnknownMetricError` | Any element of `metrics` not in `SUPPORTED_METRICS`. Raised **before** any computation. |
| `ValueError` | `metrics` is empty. |
| `FileNotFoundError` | Either path does not exist. |
| Subclass of `GifLabError` | Computation failure for any requested metric (all-or-nothing semantics). The exception's `context["metrics"]` lists the metrics that were requested. |

**Cost model**:

- Requesting only the six "cheap" metrics (`ssim`, `ms_ssim`, `psnr`, `gmsd`, `fsim`, `chist`) is fast — they share intermediate computation and run as a single pass over the frames.
- Requesting `lpips` triggers a PyTorch model load on first call within a process and dominates cost. Subsequent calls within the same process reuse the cached model.

**Thread safety**: Safe to call concurrently.

---

## Result types

Both result types are `@dataclass(frozen=True)`. Hash-equal results compare equal. Mutating a field raises `dataclasses.FrozenInstanceError`.

### `CompressResult`

| Field | Type | Notes |
|---|---|---|
| `output_path` | `pathlib.Path` | Echoed from the call. |
| `output_bytes` | `int` | Exact byte count of the produced file. |
| `render_ms` | `int` | Engine subprocess duration in milliseconds. |
| `engine` | `str` | Echoed from the call. |
| `engine_version` | `str` | Version string the engine binary reports, or `"unknown"`. |
| `params` | `dict[str, Any]` | Shallow copy of the params passed in. Mutation of the caller's dict after the call does not affect this result. |

### `MeasureResult`

Seven optional float fields, one per supported metric: `ssim`, `ms_ssim`, `psnr`, `lpips`, `gmsd`, `fsim`, `chist`. Each field is populated iff that metric was requested in the call; otherwise `None`.

---

## Supported sets

```python
SUPPORTED_ENGINES: tuple[str, ...] = (
    "animately", "gifsicle", "gifski", "imagemagick", "ffmpeg",
)

SUPPORTED_METRICS: tuple[str, ...] = (
    "ssim", "ms_ssim", "psnr", "lpips", "gmsd", "fsim", "chist",
)
```

These tuples are authoritative for what the public API accepts in this release. To check support at runtime:

```python
from giflab import SUPPORTED_ENGINES, SUPPORTED_METRICS
assert "gifsicle" in SUPPORTED_ENGINES
```

---

## Exception types

All public exceptions inherit from `giflab.error_handling.GifLabError` and carry `cause` and `context` attributes for traceability.

| Exception | Module | Inherits |
|---|---|---|
| `UnknownEngineError` | `giflab.public_api` | `GifLabError` |
| `UnknownMetricError` | `giflab.public_api` | `GifLabError` |
| `EngineUnavailableError` | `giflab.public_api` | `GifLabError` |

Catch them uniformly with `except GifLabError:` if you want.

---

## Versioning

This contract is bound to giflab `v0.3.0` and later (within the 0.x major). Breaking changes — removing an engine from `SUPPORTED_ENGINES`, removing a metric from `SUPPORTED_METRICS`, changing a return field, renaming a public symbol — ship as a minor bump (e.g., `v0.4.0`) with a `CHANGELOG.md` entry. Additive changes — adding a new supported engine or metric, adding an optional parameter with a default — ship as a patch bump.

Consumers should pin to a version string in their dependency manifest:

```toml
[tool.poetry.dependencies]
giflab = "^0.3.0"
```

---

## What this contract does **not** promise

- Internal module paths (`giflab.public_api`, `giflab.tool_wrappers`, `giflab.metrics`, etc.) may move or be renamed at any time. Use top-level imports.
- Internal exception types other than the three listed above are not part of the contract.
- The CLI surface (`python -m giflab run ...`) is not part of the public API — it is a separate, separately-versioned surface for direct use.
- Engine-specific `params` dict schemas are documented in the engine wrappers' docstrings but are not promised to be stable across giflab versions — consumers writing engine-specific param dicts accept that responsibility.
