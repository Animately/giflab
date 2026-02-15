# Internal Modules Reference

Documentation for GifLab's internal infrastructure modules that support the compression pipeline.

---

## `monitoring/` -- Performance Monitoring Infrastructure

**Location**: `src/giflab/monitoring/`

**Purpose**: Provides metrics collection, alerting, memory pressure management, and cache effectiveness analysis for the compression pipeline. This is a standalone infrastructure module -- not directly imported by production pipeline code, but used in tests and available for instrumentation.

### Key Components

| File | Class/Function | Description |
|------|---------------|-------------|
| `metrics_collector.py` | `MetricsCollector` | Central metrics collection with ring buffers and aggregation. Records counters, gauges, histograms, and timers. |
| `metrics_collector.py` | `MetricsAggregator` | Aggregates raw metric points into summaries with percentiles. |
| `backends.py` | `InMemoryBackend`, `SQLiteBackend`, `StatsDBackend` | Pluggable storage backends for metrics data. Created via `create_backend()`. |
| `alerting.py` | `AlertManager` | Evaluates alert rules against collected metrics and tracks active/historical alerts. |
| `alerting.py` | `AlertNotifier` | Dispatches alert notifications to registered handlers (log, console). |
| `memory_monitor.py` | `SystemMemoryMonitor` | Monitors system memory usage with configurable polling. Uses psutil when available. |
| `memory_monitor.py` | `CacheMemoryTracker` | Tracks per-cache memory usage and cache operation effectiveness. |
| `memory_monitor.py` | `MemoryPressureManager` | Detects memory pressure levels and triggers eviction callbacks. |
| `decorators.py` | `@track_timing`, `@track_counter`, `@track_gauge`, `@track_histogram` | Decorator-based metric instrumentation for functions. |
| `decorators.py` | `MetricTracker` | Context-manager-based metric tracking. |
| `integration.py` | `instrument_all_systems()`, `remove_instrumentation()` | Instruments frame cache, validation cache, resize cache, sampling, and metrics calculation. |
| `memory_integration.py` | `MemoryPressureIntegration` | Connects memory monitoring to cache eviction. |
| `cache_effectiveness.py` | `CacheEffectivenessMonitor` | Tracks cache hit rates, baseline comparisons, and windowed statistics. |
| `effectiveness_analysis.py` | `CacheEffectivenessAnalyzer` | Analyzes cache effectiveness data and generates optimization recommendations. |
| `baseline_framework.py` | `PerformanceBaselineFramework` | A/B testing framework with statistical significance testing and confidence intervals. |

### Singletons

Access via module-level functions:
- `get_metrics_collector()` / `reset_metrics_collector()`
- `get_alert_manager()` / `get_alert_notifier()`
- `get_system_memory_monitor()` / `get_cache_memory_tracker()` / `get_memory_pressure_manager()`
- `get_cache_effectiveness_monitor()`
- `get_effectiveness_analyzer()`
- `get_baseline_framework()`

### Integration Points

- Not imported by main production code
- Tested in `tests/functional/test_monitoring.py`, `tests/functional/test_cache_effectiveness.py`, `tests/nightly/test_memory_monitoring.py`

---

## `wrapper_validation/` -- Compression Output Validation

**Location**: `src/giflab/wrapper_validation/`

**Purpose**: Validates that compression wrapper outputs meet correctness requirements. This module is **actively integrated** into the production pipeline via `tool_wrappers.py` and `metrics.py`.

### Key Components

| File | Class/Function | Description |
|------|---------------|-------------|
| `core.py` | `WrapperOutputValidator` | Validates individual wrapper outputs: frame count after reduction, color count after reduction, timing preservation, and file integrity. |
| `types.py` | `ValidationResult` | Dataclass holding validation status, warnings, and errors. |
| `types.py` | `TimingValidationError`, `TimingImportError`, `TimingCalculationError`, `TimingFileError` | Typed exceptions for validation failures. |
| `quality_validation.py` | `QualityThresholdValidator` | Validates quality degradation stays within acceptable bounds. Checks metric outliers and quality variance. |
| `timing_validation.py` | `TimingGridValidator` | Validates frame timing integrity: duration extraction, grid alignment, timing drift, and alignment accuracy. |
| `timing_validation.py` | `validate_frame_timing_for_operation()` | Convenience function for per-operation timing validation. |
| `pipeline_validation.py` | `PipelineStageValidator` | Validates multi-stage pipeline executions: per-stage validation, inter-stage consistency, PNG sequence integrity. |
| `integration.py` | `validate_wrapper_apply_result()` | Called from `tool_wrappers.py` after each wrapper `apply()` call. |
| `integration.py` | `validate_pipeline_execution_result()` | Validates complete pipeline execution results. |
| `integration.py` | `create_validation_report()` | Generates human-readable validation reports. |

### Integration Points

- **`src/giflab/tool_wrappers.py`**: Imports `validate_wrapper_apply_result` to validate each wrapper's output after `apply()`.
- **`src/giflab/metrics.py`**: Imports timing validation for frame timing analysis.
- Tested in `tests/functional/test_wrapper_validation.py`, `tests/functional/test_timing_validation.py`, `tests/integration/test_pipeline_stage_validation.py`, `tests/integration/test_engine_equivalence.py`

---

## `optimization_validation/` -- Compression Result Validation

**Location**: `src/giflab/optimization_validation/`

**Purpose**: Validates compression results against configurable quality and efficiency thresholds. Checks for artifacts, temporal consistency, and perceptual quality. Used in integration and nightly tests but not directly imported by production pipeline code.

### Key Components

| File | Class/Function | Description |
|------|---------------|-------------|
| `validation_checker.py` | `ValidationChecker` | Comprehensive compression result validator. Checks frame reduction, FPS consistency, quality thresholds, efficiency thresholds, disposal artifacts, temporal consistency, multi-metric combinations, temporal artifacts, deep perceptual metrics, and SSIMulacra2. |
| `config.py` | `get_default_validation_config()` | Returns default threshold configuration. |
| `config.py` | `load_validation_config()`, `save_validation_config()` | Load/save validation configs from files. |
| `data_structures.py` | `ValidationResult` | Holds validation status, issues, warnings, and metrics. Provides `has_errors()`, `has_warnings()`, `is_acceptable()`, `get_summary()`, `get_detailed_report()`. |
| `data_structures.py` | `ValidationConfig` | Configurable thresholds for all validation checks. Can load from file. |
| `data_structures.py` | `ValidationStatus` | Enum: pass/warn/fail status. |
| `data_structures.py` | `ValidationMetrics` | Collected metrics from a validation run. |
| `data_structures.py` | `ValidationIssue`, `ValidationWarning` | Structured issue/warning types. |

### Integration Points

- Not imported by main production code
- Used in test suites: `tests/integration/test_phase3_integration.py`, `tests/integration/test_temporal_artifact_detection_e2e.py`, `tests/integration/test_temporal_validation_integration.py`, `tests/functional/test_temporal_artifacts_robustness.py`, `tests/nightly/test_pipeline_integration.py`
