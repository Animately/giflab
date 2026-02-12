# Feature Specification: Dataset Pipeline Refactor

**Feature Branch**: `002-dataset-pipeline-refactor`
**Created**: 2026-02-09
**Status**: Draft
**Input**: Unify the legacy experiment/optimization pipeline and the newer prediction module into a single pipeline that produces high-quality training datasets for compression curve prediction.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Generate Training Dataset from Raw GIFs (Priority: P1)

A developer has a directory of raw GIFs and wants to generate a complete training dataset. They run `giflab run data/raw/` which extracts visual features from each GIF, runs configurable compression algorithm combinations across 7 engines, measures quality metrics and file sizes, validates outputs, and stores everything in a normalized SQLite database. The unified pipeline replaces the previous two-system architecture (experiment runner + dataset builder).

**Why this priority**: This is the core value proposition — a single command that produces the complete training dataset. Without this, there is no data for model training.

**Independent Test**: Can be tested by providing a directory of GIFs and verifying the SQLite database contains features, compression outcomes for all configured engine combinations, and quality metrics.

**Acceptance Scenarios**:

1. **Given** a directory of valid GIFs and the default configuration, **When** `giflab run data/raw/` executes, **Then** the SQLite database contains one features record per GIF and compression results for each configured pipeline combination
2. **Given** a GIF processed with 7 lossy engines at levels [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], **When** results are stored, **Then** each result includes file size, quality metrics (SSIM, PSNR, etc.), and validation status
3. **Given** a previously processed GIF, **When** the pipeline runs again without `--force`, **Then** existing results are skipped (idempotent)
4. **Given** the `--mode full` flag, **When** the pipeline runs, **Then** all 1,200 possible pipeline combinations (5 frame × 20 color × 12 lossy) are tested

---

### User Story 2 - Compress with All 7 Engines (Priority: P1)

The system supports 7 distinct lossy compression engines, each producing different quality/size tradeoffs:

| Engine | Tool | Flag |
|--------|------|------|
| `gifsicle` | gifsicle | `--lossy <level>` |
| `animately-standard` | animately | `--lossy <level>` |
| `animately-advanced` | animately | `--advanced-lossy <level>` |
| `animately-hard` | animately | `--hard --lossy <level>` |
| `imagemagick` | convert/magick | quality parameter |
| `ffmpeg` | ffmpeg | q_scale mapping |
| `gifski` | gifski | `--quality <level>` |

**Normalization Contract**: `lossy_level` is a 0-100 integer representing compression intensity (0 = best quality, 100 = maximum compression). Each engine maps this to its native parameter space:

| Engine | Normalized 0→100 | Native param | Formula |
|--------|---|---|---|
| Gifsicle | 0→100 | `--lossy 0`→`--lossy 300` | `native = normalized × 3` |
| Animately std | 0→100 | `--lossy 0`→`--lossy 100` | `native = normalized` |
| Animately adv | 0→100 | config `0`→`100` | `native = normalized` |
| Animately hard | 0→100 | `--lossy 0`→`--lossy 100` | `native = normalized` |
| ImageMagick | 0→100 | `-quality 100`→`-quality 0` | `native = 100 - normalized` |
| FFmpeg | 0→100 | `-q:v 1`→`-q:v 31` | `native = 1 + round(normalized × 30 / 100)` |
| Gifski | 0→100 | `--quality 100`→`--quality 1` | `native = max(1, 100 - normalized)` |

Sample points: `[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]` (11 points, 10-unit intervals)

**Why this priority**: Equal to P1 because the dataset's value comes from comparing compression across all available engines. The `animately-hard` engine and consistent naming are foundational to the 7-engine model.

**Independent Test**: Can be tested by compressing a single GIF with each engine and verifying valid compressed output.

**Acceptance Scenarios**:

1. **Given** a valid GIF and the `animately-hard` engine, **When** compression runs at lossy level 60, **Then** the output is a valid GIF smaller than the input, produced via `animately --input in.gif --output out.gif --hard --lossy 60`
2. **Given** the Engine enum, **When** all 7 engines are listed, **Then** they are: `gifsicle`, `animately-standard`, `animately-advanced`, `animately-hard`, `imagemagick`, `ffmpeg`, `gifski`

---

### User Story 3 - SQLite-Only Storage with CSV Export (Priority: P2)

All pipeline output goes to a normalized SQLite database via `storage.py`. CSV is available only as an optional export for external analysis tools (`giflab export --format csv`). This replaces the current mixed CSV/SQLite output where CSV is the primary format in several modules.

**Why this priority**: Storage consolidation depends on the pipeline being unified (P1) but is needed before the dataset can be reliably consumed by training tools.

**Independent Test**: Can be tested by running the pipeline and verifying all results are in SQLite, then running `giflab export` and verifying valid CSV output.

**Acceptance Scenarios**:

1. **Given** the unified pipeline runs, **When** results are stored, **Then** no CSV files are written by the pipeline itself
2. **Given** a populated SQLite database, **When** `giflab export --format csv --output results.csv` runs, **Then** a valid CSV with all features and compression outcomes is written
3. **Given** a populated SQLite database, **When** `giflab stats` runs, **Then** it shows counts per engine, per GIF, and per pipeline combination

---

### User Story 4 - Simplified CLI for Dataset Workflow (Priority: P2)

The CLI provides a focused set of commands for the dataset workflow: `run` (generate data), `train` (train models), `predict` (predict curves), `export` (export data), `stats` (database statistics). Legacy commands that don't serve the dataset mission are removed.

**Why this priority**: CLI simplification follows from the pipeline unification and helps users understand the intended workflow.

**Independent Test**: Can be tested by running `giflab --help` and verifying only the dataset-focused commands appear.

**Acceptance Scenarios**:

1. **Given** the CLI, **When** `giflab --help` runs, **Then** it shows: `run`, `train`, `predict`, `export`, `stats`
2. **Given** the `run` command, **When** called with `--mode single`, **Then** it runs each lossy engine independently (7 engines × 11 levels = 77 runs per GIF)
3. **Given** the `run` command, **When** called with `--mode full`, **Then** it runs all pipeline combinations (frame × color × lossy)

---

### User Story 5 - Clean Codebase (Priority: P3)

Remove genuinely unused code: `benchmarks/` (5 files of A/B testing suites), `config_profiles/` (8 deployment-specific config files never imported), `dashboard/` (empty directory), old results data (158 runs in `results/`), and legacy CLI commands. This reduces cognitive overhead without removing any functional code.

**Why this priority**: Cleanup is cosmetic and should only happen after the pipeline is unified and working.

**Independent Test**: Can be tested by verifying removed directories don't exist and `make test` still passes.

**Acceptance Scenarios**:

1. **Given** the cleanup is complete, **When** `make test` runs, **Then** all tests pass
2. **Given** the cleanup is complete, **When** the codebase is inspected, **Then** `benchmarks/`, `config_profiles/`, and `dashboard/` directories no longer exist
3. **Given** the cleanup is complete, **When** no imports reference removed modules, **Then** there are zero broken import paths

---

### Edge Cases

- What happens when an engine binary is missing? (Graceful skip with warning, engine results omitted from dataset)
- What happens when `animately --hard` produces a larger file than the input? (Record the result — the ML model needs to learn this)
- What happens when a GIF has only 1 frame? (Process normally, temporal features default to 0, frame reduction skipped)
- What happens when SQLite database is locked? (Retry with backoff, fail after 30s with clear error)
- What happens when a pipeline combination produces a corrupted GIF? (Validation catches it, mark as failed, continue)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support 7 distinct lossy compression engines: gifsicle, animately-standard, animately-advanced, animately-hard, imagemagick, ffmpeg, gifski
- **FR-002**: System MUST generate pipeline combinations from 3 slots: frame reduction (5 tools), color reduction (20 tools), lossy compression (12 tools)
- **FR-003**: System MUST extract 18 visual features per GIF using the existing `GifFeaturesV1` schema
- **FR-004**: System MUST measure comprehensive GIF-specific quality metrics per compression result (SSIM, PSNR, disposal artifacts, transparency, temporal artifacts, gradient/color artifacts, text/UI validation)
- **FR-005**: System MUST store all pipeline output in SQLite via `storage.py`
- **FR-006**: System MUST NOT write CSV files as primary output (CSV only via explicit export command)
- **FR-007**: System MUST validate compressed GIF outputs for corruption before recording results
- **FR-008**: System MUST skip previously processed GIF/pipeline combinations unless `--force` is specified
- **FR-009**: System MUST support `--upgrade` flag to reprocess only GIFs processed with an older version
- **FR-010**: System MUST provide CLI commands: `run`, `train`, `predict`, `export`, `stats`
- **FR-011**: System MUST add `AnimatelyHardLossyCompressor` using `animately --input <in> --output <out> --hard --lossy <level>`

### Key Entities

- **Pipeline**: A 3-slot combination of (frame_reduction, color_reduction, lossy_compression) tools. Identified by a stable string identifier (e.g., `gifsicle-frame|animately-color|animately-standard`).

- **CompressionRun**: A single execution of a Pipeline on a GIF at a specific parameter set. Records input file size, output file size, quality metrics, validation status, and timing.

- **Engine**: One of 7 named lossy compression algorithms. Each maps to a specific external tool binary and command-line flags.

- **GifFeatures**: 18 visual characteristics extracted from a GIF (spatial, temporal, compressibility, transparency). Schema-versioned via `GifFeaturesV1`.

- **QualityMetrics**: Comprehensive GIF-specific quality measurements including structural similarity, perceptual quality, artifact detection (disposal, transparency, temporal, gradient/color, text/UI).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All 7 engines produce valid compressed output for a test GIF (verified by wrapper validation)
- **SC-002**: `giflab run` in single mode produces a SQLite database with features + compression results for all 7 engines
- **SC-003**: `giflab run --mode full` generates results for all 1,200 pipeline combinations (or correctly skips combinations where tools are unavailable)
- **SC-004**: No CSV files are written by the pipeline (only by explicit `giflab export`)
- **SC-005**: `make test` passes with all existing tests (no regressions)
- **SC-006**: Quality metrics ecosystem (11+ metrics) produces comprehensive scores for all compression results
- **SC-007**: Pipeline is idempotent — re-running without `--force` produces no duplicate records

## Assumptions

- All 7 engine binaries (gifsicle, animately, convert/magick, ffmpeg, gifski) are available on the development machine
- The existing quality metrics ecosystem is comprehensive and doesn't need modification — only integration into the unified pipeline
- The existing `GifFeaturesV1` schema with 18 features is sufficient for ML training
- SQLite can handle datasets up to 100K compression runs without performance issues
- The `--hard` flag in animately combines with `--lossy` (confirmed by user)
- Existing test infrastructure (4-layer: smoke, functional, integration, nightly) remains unchanged
