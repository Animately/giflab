# Changelog

All notable changes to giflab are documented here. Versioning follows semver within the 0.x major; see [docs/public-api.md](./docs/public-api.md) for the versioning policy that applies to the public API surface.

## v0.4.0 — 2026-06-09

### Changed (BREAKING): single-stream metric legacy bare-key aliases removed

The legacy bare single-stream metric keys that *read* like original-vs-compressed pair signals but only ever measured the **compressed** stream have been removed. The honest `*_compressed` keys are now the canonical names. This is a breaking change to the metric **key schema** (hence the minor bump), affecting any consumer that read these keys off the result dict or from the SQLite store.

**Removed bare keys** (use the `*_compressed` equivalent instead): `temporal_consistency`, `disposal_artifacts`, `flicker_excess`, `flicker_frame_ratio`, `flat_flicker_ratio`, `flat_region_count`, `temporal_pumping_score`, `quality_oscillation_frequency`, `lpips_t_mean` / `lpips_t_p95` / `lpips_t_max`.

- Statistical siblings (`_std` / `_min` / `_max`) re-rooted onto `*_compressed`; `_pre` / `_post` / `_delta` / `_original` provenance kept.
- Phase 6 optimized path emits `temporal_consistency_compressed` only (honest compressed-stream value).
- `storage.py`: `QUALITY_METRIC_COLUMNS` + table schema renamed to `*_compressed` (idempotent `ALTER` auto-adds the new columns; stale bare columns in old DBs are ignored).
- The public `measure()` surface was **not** affected by this change — it projects only the bare cheap-metric keys (`ssim`, `ms_ssim`, `psnr`, `gmsd`, `fsim`, `chist`), none of which were single-stream aliases.

### Added: `composite_quality` on the public `measure()` surface

`composite_quality` — giflab's calibrated weighted-aggregate **verdict number** — is now a recognised metric on the public API. The intended downstream consumer is gifprep, which adopts it as its single deterministic quality gate.

- New entry in `SUPPORTED_METRICS`: `"composite_quality"`.
- New `MeasureResult.composite_quality: float | None = None` field (backward-compatible — existing `MeasureResult` construction and callers that never request the metric are unaffected; the field defaults to `None`).
- New `MetricIdentifier` `Literal` member and public→internal key mapping (the bare `composite_quality` key, projected through unchanged — no `_mean` sibling, no denormalisation).
- **Determinism fix.** `composite_quality` redistributes a NaN contributor's weight across the measured contributors, so its value is *contributor-set-dependent*. The only request-gated contributor is `lpips` (`ENHANCED_LPIPS_WEIGHT` 0.04). Requesting `composite_quality` now **forces the LPIPS computation on** regardless of whether `"lpips"` is also requested, so `measure(["composite_quality"])`, `measure(["composite_quality", "lpips"])`, and `measure(["composite_quality", "ssim", "psnr"])` all return the **same** value for a given file pair. Cost consequence: `composite_quality` is **not** a cheap metric — it pays the LPIPS model load (see the [Cost model](./docs/public-api.md#cost-model)).
- **NaN / environmental contract.** `composite_quality` may be `NaN` when the majority of present contributor weight is unmeasurable (per `COMPOSITE_NAN_THRESHOLD`) — surfaced as a `NaN` float, never a fabricated sentinel. The `ssimulacra2` contributor (3% weight) is binary-gated, not request-gated, so `composite_quality` is deterministic **per environment** but can differ across machines that do/don't have the `ssimulacra2` binary on `PATH`. The per-dimension weights are now a documented public contract — see [`composite_quality` — weights are a public contract](./docs/public-api.md#composite_quality--weights-are-a-public-contract).

### Not changed

- CLI, dataset generation pipeline, matrix benchmark, feature extraction, engine parameter grids, internal tool interfaces, and the underlying metric *implementations* (`composite_quality` is exposed, not reimplemented).

## v0.3.2 — 2026-05-22

### Fixed: Public API FR-009 perceptual-cost contract

`measure(metrics=["ssim"])` (or any subset of the six "cheap" metrics) no longer triggers a PyTorch / LPIPS model load via the temporal-artifacts pipeline. This restores the FR-009 promise in [docs/public-api.md](./docs/public-api.md) that requesting only cheap metrics is fast.

**Headline impact**: ~3.3× per-cell speedup measured against gifprep's benchmark harness (67.1s for 9 cells vs the previous ~24.7s/cell baseline), restoring it to gifprep's SC-001 <10 min budget on the full corpus.

**Changes**:

- New `MetricsConfig.ENABLE_TEMPORAL_ARTIFACTS: bool = True` flag (default preserves dataset-pipeline behaviour). `measure()` derives the flag from a new `_TEMPORAL_NEEDING_METRICS` constant (empty in this release) so callers requesting only metrics that don't need temporal computation opt out automatically.
- `calculate_enhanced_temporal_metrics` now reuses the existing `get_temporal_detector` singleton (mirrors `deep_perceptual_metrics._global_validator`). Eliminates the misleading per-call `LPIPS model initialized successfully` log even when `LPIPSModelCache` was already caching the model.
- `deep_perceptual_metrics` is now gated at the call site so the `No LPIPS scores obtained` WARN no longer fires spuriously when `ENABLE_DEEP_PERCEPTUAL=False`.
- `temporal_artifacts.zero_temporal_metrics(frame_count)` centralises the zero-valued temporal metrics dict that previously had two different shapes across call sites.
- Conditional-path parity: `ssimulacra2` is now gated by `ENABLE_SSIMULACRA2` in the conditional path too, matching the lpips + temporal gates.

### Not changed

- CLI, dataset generation pipeline, matrix benchmark, SQLite schema, feature extraction, engine parameter grids, internal tool interfaces, quality metrics implementations.

## v0.3.1 — 2026-05-21

### Fixed: deep_perceptual_metrics.py was silently gitignored

`src/giflab/deep_perceptual_metrics.py` existed on disk but was excluded from the v0.3.0 release because the unanchored gitignore rule `deep_*` at `.gitignore:179` matched it. Every external consumer pinned at `@v0.3.0` received hardcoded `lpips_quality_mean = 0.5` as a fallback for any content, regardless of actual perceptual difference. Cross-corpus LPIPS variance is restored — values now range 0.0026..0.0664 on the 5-GIF local corpus (vs flat 0.5 in v0.3.0).

**Changes**:

- Anchored the unanchored gitignore rules (`deep_*`, `debug_*`, `final_*`, `verification_*`, `comprehensive_*`, `gpu_*`, `clean_test_*`, `debug_*.png`, `step*.gif`, `pipeline_*.gif`) to repository root with a leading `/` so source files cannot be accidentally caught.
- Added `src/giflab/deep_perceptual_metrics.py` (704 lines) to git tracking.
- Added `scripts/debug_pipeline.py` (329 lines) to git tracking.
- Added `tests/smoke/test_package_completeness.py` as a recurrence guard.
- Synced stale `__version__` in `src/giflab/__init__.py` (was lagging at 0.1.0).

### Fixed: pre-existing CI test failures

Five unrelated test failures that had been present on `main` for an extended period:

- `detect_flicker_excess` — incorrect expected value.
- LPIPS patch targets — module-path drift.
- `test_with_custom_config` — config schema sync.
- `test_extract_features_cli` — CLI invocation pattern.

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
