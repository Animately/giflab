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
    *,
    apply_content_ceiling: bool = True,
) -> CompressResult: ...
```

**Behaviour**:

- Runs the named compression engine on `input_path`, writing the result to `output_path`.
- Returns a `CompressResult` describing the produced file.
- Overwrites `output_path` if a file already exists there.
- Does not mutate `input_path`.
- May clamp the requested lossy level down for certain content (see [Content-aware lossy ceiling](#content-aware-lossy-ceiling) below) and may surface non-fatal warnings on `CompressResult.warnings`.

**Parameters**:

| Name | Type | Required | Notes |
|---|---|---|---|
| `input_path` | `pathlib.Path` | yes | Must exist and be readable. |
| `output_path` | `pathlib.Path` | yes | Parent directory must exist and be writable. File at this path will be overwritten. |
| `engine` | `str` (one of `SUPPORTED_ENGINES`) | yes | Engine identifier. |
| `params` | `dict[str, Any] \| None` | no (default `None`) | Engine-specific parameter mapping. `None` is treated as empty dict (engine defaults). Schema is engine-specific — engines that require `lossy_level` will raise `ValueError` if it is missing. |
| `apply_content_ceiling` | `bool` (keyword-only) | no (default `True`) | When `True`, apply the [content-aware lossy ceiling](#content-aware-lossy-ceiling). Pass `False` to bypass it (audit sweeps that need their lossy grid run verbatim). |

#### Content-aware lossy ceiling

When the engine is `animately` and `params` carries a positive `lossy_level`, `compress()` classifies the input's **original** frames (data-viz / photographic / film-grain) and clamps the requested `lossy_level` **down** to a per-class maximum — it never raises the level. This protects content the 2026-05-26 outlier deep-dive identified as fragile: flat-colour animations band under heavy lossy, and near-256-colour photographic / film-grain content posterises above modest levels.

| Content class | Ceiling | Config field (`giflab.config.ClassifierConfig`) |
|---|---|---|
| data-viz / flat-colour animation | `40` (conservative; non-lossless) | `MAX_LOSSY_DATA_VIZ` |
| photographic gradient/image | `20` | `MAX_LOSSY_PHOTOGRAPHIC` |
| film grain / sensor noise | `30` | `MAX_LOSSY_FILM_GRAIN` |
| other / unclassified | none | — |

- When a clamp happens, a human-readable warning string is appended to `CompressResult.warnings`.
- If the requested level is already at or below the ceiling, nothing is clamped and no warning is emitted.
- **Data-viz scope & the conservative ceiling.** Pre-compression *single-frame* primitives cannot reliably isolate a categorical chart from a flat logo / cartoon / UI / line-art frame — they are numerically identical (a 4-colour synthetic chart equals a 4-colour flat cartoon). The classifier therefore treats `data-viz` as a broad "flat-colour animation" class. Forcing that whole population to lossless (the original `0` ceiling) silently defeated lossy compression for the single most common GIF category, so the data-viz ceiling is a **conservative `40`**: ordinary flat content at typical lossy levels (≤40) passes through untouched, and only genuinely extreme requests are clamped where flat-colour banding becomes severe. The detection score is a strict conjunction (geometric mean of flat-structure × limited-palette × low-grain × animation-length), so photographic and film-grain content can never trip it. Set `MAX_LOSSY_DATA_VIZ` lower only with a discriminating signal (e.g. a palette-histogram or pair-metric) that actually isolates categorical charts.
- The ceilings are **animately-calibrated only**. Other lossy engines (`gifsicle`, etc.) skip classification entirely — `gifsicle`'s native lossy scale is a 3× multiple of the public scale and has not been calibrated. (TODO: calibrate per-engine ceilings.)
- Classification is **fail-soft**: a corrupt or unreadable input is classified `other`, applies no ceiling, and never blocks the compress.
- `compress()` also emits a frame-drop warning on `CompressResult.warnings` when the engine output has fewer frames than the input (a cheap input-vs-output frame-count comparison; it does not run the metrics pipeline).
- Pass `apply_content_ceiling=False` to bypass the ceiling entirely. Audit sweeps (`scripts/audit/`) set this so their monotonicity / corpus lossy grids run at the exact requested levels.

**Returns**: `CompressResult` (see [Result types](#result-types) below).

**Raises**:

| Exception | Condition |
|---|---|
| `UnknownEngineError` | `engine` not in `SUPPORTED_ENGINES`. Raised **before** any file I/O. |
| `EngineUnavailableError` | Engine binary not found on `PATH`. |
| `FileNotFoundError` | `input_path` does not exist. |
| `ValueError` | `params` is missing keys the engine wrapper requires. |
| Subclass of `GifLabError` | Any other engine-side failure (subprocess non-zero exit, etc.). |

**Determinism**: For a given (input file bytes, engine, params, `apply_content_ceiling`) tuple, the output bytes are deterministic to the extent the underlying engine is deterministic. The content-aware lossy ceiling is itself deterministic — the classifier samples frames at fixed `np.linspace` indices (never random) and the clamp is a pure function of the classification and the requested level. The public wrapper introduces no nondeterminism of its own.

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

### Transparency handling — dual (white + black) composite

GIF/PNG inputs that carry transparency are scored against **two** background
composites and merged pessimistically. The motivation is a measurement bias:
compositing transparent pixels onto a single fixed white background swamps
differences in **dark content** (near-black detail on a transparent field is
high-contrast against white, so a perturbation that would be obvious against a
dark background is nearly invisible). White-only scoring therefore
systematically over-rates dark-content GIFs.

Behaviour:

- **Opaque inputs** (no transparency) are scored with a single white-composite
  pass exactly as before — **zero added cost** (the second pass is never run).
- **Transparent inputs** are composited onto white **and** black; each pass
  runs the full metrics pipeline, then the results are merged **per metric** by
  taking the value that is **worse for quality** (direction derived live from
  the composite formula). The merged `composite_quality` **and** `efficiency`
  are then recomputed from the worst-of values. Because the worst metric can
  come from a different background per metric, the merged composite can fall
  below *either* single-background composite — a compressor cannot game the
  score by being friendly to one background. `compression_ratio` is identical
  across passes (it is file-size-derived), so `efficiency` stays consistent and
  reflects the lower composite.
- Internal `render_ms` for the metrics dict is re-timed at the file level, so
  it roughly **doubles** on transparent inputs (it covers both extractions and
  both metric passes). This is the internal metrics-dict `render_ms`, **not**
  `CompressResult.render_ms` (which is engine subprocess time, see below) — and
  it is **not** projected onto the public `MeasureResult` (which carries no
  `render_ms`).
- The **public `measure()` surface is unaffected**: it projects only the seven
  metric fields; the worst-of merge happens beneath that projection, so a caller
  requesting e.g. `ssim` simply receives the (worst-of, transparency-aware)
  value with the same field shape.

> Note (out of scope, follow-up): under the experimental Phase 6 optimized
> metrics path (`GIFLAB_ENABLE_PHASE6_OPTIMIZATION=true`), a temporal-metric
> *failure* currently emits a `0.0` best-case sentinel rather than `NaN`. Both
> dual-composite passes go through that same path, so the worst-of merge
> faithfully propagates that pre-existing fabrication **without amplifying it**
> (picking `0.0` from both passes yields `0.0`, identical to single-pass Phase
> 6). Fixing the sentinel to emit `NaN` per the metrics-accuracy policy is a
> separate follow-up, not part of this change.

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
| `params` | `dict[str, Any]` | Shallow copy of the **effective** params used. Mutation of the caller's dict after the call does not affect this result. If the [content-aware lossy ceiling](#content-aware-lossy-ceiling) clamped `lossy_level`, this dict reflects the clamped value, not the requested one. |
| `warnings` | `tuple[str, ...]` | Non-fatal advisories about the compression. Empty when nothing noteworthy happened. Currently carries the lossy-ceiling clamp message and the frame-drop message (see [`compress`](#compress)). Consumers should treat the contents as human-readable strings, not a stable machine schema. |

### `MeasureResult`

Seven optional float fields, one per supported metric: `ssim`, `ms_ssim`, `psnr`, `lpips`, `gmsd`, `fsim`, `chist`. Each field is populated iff that metric was requested in the call; otherwise `None`.

**Units and ranges**:

| Field | Range | Notes |
|---|---|---|
| `ssim`, `ms_ssim`, `fsim`, `chist` | `[0.0, 1.0]` | Higher is better. |
| `gmsd` | `[0.0, 1.0]` | Lower is better. |
| `psnr` | `[0.0, 50.0]` dB | Higher is better. Values are reported in decibels; `50.0` is the cap (the internal `PSNR_MAX_DB` setting) and represents "effectively identical" candidates. |
| `lpips` | `[0.0, 1.0]` | Lower is better. Perceptual distance per the LPIPS model; this is the mean across frames. |

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
