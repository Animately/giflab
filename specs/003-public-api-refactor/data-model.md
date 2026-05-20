# Phase 1: Data Model

**Feature**: 003-public-api-refactor
**Date**: 2026-05-19

The public API surface is a function-call library, not a stateful system — the "data model" is the set of value objects exchanged across the public boundary plus the bounded identifier sets the functions accept.

All entities are **immutable** (`@dataclass(frozen=True)`) and reside in `src/giflab/public_api.py`.

---

## Entity 1: `CompressResult`

Return value of `compress()`. Describes the file that was produced and the engine work that produced it.

| Field | Type | Source | Notes |
|---|---|---|---|
| `output_path` | `pathlib.Path` | Caller-passed | Echoed for symmetry with calls that capture only the result. |
| `output_bytes` | `int` | `output_path.stat().st_size` | Exact byte count, not rounded kilobytes. |
| `render_ms` | `int` | Wrapper `apply()` return | Wall-clock duration of the engine subprocess only — excludes Python-side dispatch overhead. |
| `engine` | `str` | Caller-passed | Echoed to make the result self-describing. |
| `engine_version` | `str` | `WrapperClass.version()` | The version string the underlying binary reports. Falls back to `"unknown"` if the binary does not implement a version flag. |
| `params` | `dict[str, Any]` | Caller-passed | Stored as an immutable copy. The same dict passed in produces the result — useful for benchmark harnesses that record exactly which params produced which output file. |

**Invariants**:

- All fields are non-null on return.
- `output_path` references a file that exists on disk when the result is returned (the file was just written).
- `output_bytes >= 0`. `output_bytes == 0` is unusual but legal (e.g., a degenerate input).
- `render_ms >= 0`.
- `params` is shallowly copied at construction time so mutation of the caller's dict after the call does not affect the result.

**Validation rules** (enforced at `compress()` boundary, before result construction):

- `engine` is one of the 5 supported identifiers.
- `input_path` exists and is readable.
- `output_path` parent directory exists and is writable.

---

## Entity 2: `MeasureResult`

Return value of `measure()`. Carries one optional numeric field per supported metric. Each field is `None` unless the caller requested that metric.

| Field | Type | Populated when |
|---|---|---|
| `ssim` | `float \| None` | `"ssim"` in `metrics` argument |
| `ms_ssim` | `float \| None` | `"ms_ssim"` in `metrics` argument |
| `psnr` | `float \| None` | `"psnr"` in `metrics` argument |
| `lpips` | `float \| None` | `"lpips"` in `metrics` argument |
| `gmsd` | `float \| None` | `"gmsd"` in `metrics` argument |
| `fsim` | `float \| None` | `"fsim"` in `metrics` argument |
| `chist` | `float \| None` | `"chist"` in `metrics` argument |

**Invariants**:

- For every metric `m` in the caller's `metrics` argument, `getattr(result, m) is not None`.
- For every metric `m` *not* in the caller's `metrics` argument, `getattr(result, m) is None`.
- All numeric values are finite (not `NaN`, not ±inf). A metric that would return `NaN` raises instead (see Edge Cases in spec.md).

**Validation rules** (enforced at `measure()` boundary, before result construction):

- Every string in `metrics` is one of the 7 supported identifiers.
- `metrics` is non-empty (calling `measure` with no requested metrics is a programming error — raises `ValueError`).
- `reference_path` and `candidate_path` both exist and are readable.

---

## Entity 3: `EngineIdentifier`

A bounded set of strings recognised by `compress()`. Defined as a `Literal` type for static analysis and a runtime tuple for the dispatch table key set.

```python
EngineIdentifier = Literal["animately", "gifsicle", "gifski", "imagemagick", "ffmpeg"]
SUPPORTED_ENGINES: tuple[str, ...] = ("animately", "gifsicle", "gifski", "imagemagick", "ffmpeg")
```

**Invariant**: `SUPPORTED_ENGINES` is exactly the set of keys in the engine dispatch table (research.md R-1). Adding to one without the other is a build error.

---

## Entity 4: `MetricIdentifier`

A bounded set of strings recognised by `measure()`. Same shape as `EngineIdentifier`.

```python
MetricIdentifier = Literal["ssim", "ms_ssim", "psnr", "lpips", "gmsd", "fsim", "chist"]
SUPPORTED_METRICS: tuple[str, ...] = ("ssim", "ms_ssim", "psnr", "lpips", "gmsd", "fsim", "chist")
```

**Invariant**: Every identifier in `SUPPORTED_METRICS` corresponds to a populated field on `MeasureResult`.

---

## Entity Relationships

```text
compress(input_path, output_path, engine: EngineIdentifier, params) → CompressResult
                                          │
                                          └──── dispatched through engine table to
                                                tool_wrappers.<EngineLossyCompressor>.apply()

measure(reference_path, candidate_path, metrics: list[MetricIdentifier]) → MeasureResult
                                                  │
                                                  └──── projected from
                                                        metrics.calculate_comprehensive_metrics()
```

The two functions share no state. A `CompressResult` is not consumed by `measure` directly — the consumer uses `result.output_path` to call `measure(reference_path, result.output_path, ...)`.

---

## Out-of-scope entities

- No persisted entities. Nothing is written to SQLite by either function.
- No request/response wrappers beyond the two result dataclasses.
- No retry, batch, async, or stream variants in v1.
