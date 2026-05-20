# Changelog

All notable changes to giflab are documented here. Versioning follows semver within the 0.x major; see [docs/public-api.md](./docs/public-api.md) for the versioning policy that applies to the public API surface.

## v0.3.0 — 2026-05-19

### Added: Public API for external consumers

Two new top-level functions and their result types, designed to let external Python projects (initially [gifprep](https://github.com/Animately/gifprep)) depend on giflab as a library.

**Public symbols** (importable from `giflab`):

- `compress(input_path, output_path, engine, params) -> CompressResult`
- `measure(reference_path, candidate_path, metrics) -> MeasureResult`
- `CompressResult` — frozen dataclass with `output_path`, `output_bytes`, `render_ms`, `engine`, `engine_version`, `params`
- `MeasureResult` — frozen dataclass with one optional float field per supported metric
- `SUPPORTED_ENGINES: tuple[str, ...]` — the 5 engines the public API recognises
- `SUPPORTED_METRICS: tuple[str, ...]` — the 7 metrics the public API recognises
- `EngineIdentifier`, `MetricIdentifier` — `Literal` types for static analysis
- `UnknownEngineError`, `UnknownMetricError`, `EngineUnavailableError` — typed exceptions, all inherit from `GifLabError`

**Supported engines** (`SUPPORTED_ENGINES`):

`animately`, `gifsicle`, `gifski`, `imagemagick`, `ffmpeg`.

**Supported metrics** (`SUPPORTED_METRICS`):

`ssim`, `ms_ssim`, `psnr`, `lpips`, `gmsd`, `fsim`, `chist`.

**Semantics**:

- `compress()` is a thin dispatch wrapper over the existing engine wrappers in `src/giflab/tool_wrappers.py`. It does not introduce new engines or alter parameter schemas.
- `measure()` is a thin projection wrapper over `calculate_comprehensive_metrics`. Requesting metrics returns a `MeasureResult` populating exactly those fields; non-requested fields are `None`.
- Metric computation is all-or-nothing: a failure in any requested metric raises `GifLabError` with `context["metrics"]` listing the requested set. Partial results are not returned.
- LPIPS computation (the only individually expensive metric) is gated — requesting only cheap metrics does not load the PyTorch model.
- Both functions are deterministic to the extent the underlying engines / metrics are.

**Source of truth**: [docs/public-api.md](./docs/public-api.md). The integration proposal that prompted this work is preserved at [docs/gifprep-integration.md](./docs/gifprep-integration.md).

### Changed

- `calculate_comprehensive_metrics` and `calculate_comprehensive_metrics_from_frames` accept a new `force_all_metrics: bool = False` kwarg. Callers that pass `True` skip the conditional metrics optimization and receive the full result dict. The public `measure()` function uses this to honour its contract. Existing callers that omit the kwarg continue to use the conditional optimization (or the legacy `GIFLAB_FORCE_ALL_METRICS` env var) unchanged.

### Not changed

- CLI (`python -m giflab run ...`).
- Dataset generation pipeline.
- Matrix benchmark.
- SQLite schema.
- Feature extraction.
- Engine parameter grids.
- Internal tool interfaces (`tool_interfaces.py`, `dynamic_pipeline.py`).
- Quality metrics implementations.
