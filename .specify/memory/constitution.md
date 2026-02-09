<!--
SYNC IMPACT REPORT
==================
Version change: 1.1.0 → 1.2.0
Modified principles: IV. Test-First Development — added 4-layer test architecture and test discipline rules
Added sections: Test Architecture (4-Layer), Test Discipline
Removed sections: None
Templates requiring updates: ✅ None
Follow-up TODOs: None
-->

# GifLab Constitution

## Core Principles

### I. Single-Pass Compression (NON-NEGOTIABLE)

All compression parameters (lossy level, frame reduction, color reduction) MUST be applied in a single engine call. Chained re-compressions are forbidden.

**Rationale**: Each GIF decode/encode cycle introduces quality loss. Single-pass compression minimizes artifacts and allows engines to optimize the full parameter space together.

**Enforcement**: Tool wrappers MUST accept all parameters in one `apply()` call. Pipeline execution MUST NOT save intermediate GIF files between compression steps.

### II. ML-Ready Data

All pipeline outputs MUST be deterministic, reproducible, and validated against Pydantic schemas. Dataset versioning is mandatory.

**Requirements**:
- Validate against `MetricRecordV1` schema before CSV/DB write
- Tag outputs with `giflab_version` and `code_commit`
- Respect canonical train/val/test GIF splits
- Preserve `scaler.pkl` feature-scaling artifacts when modifying metrics

**Rationale**: GifLab's purpose is generating training data for ML models. Non-reproducible or schema-violating data corrupts downstream models.

### III. Poetry-First Execution

All Python commands MUST use `poetry run`. The `animately` CLI MUST use flag-based arguments (`--input`, `--output`). Positional arguments will fail silently.

**Enforcement**: Never invoke `python` directly. Always `poetry run python -m giflab ...` or `poetry run pytest`.

**Rationale**: Ensures consistent dependency resolution and prevents environment contamination.

### IV. Test-First Development (NON-NEGOTIABLE)

All features MUST have tests written before or alongside implementation. Test coverage is mandatory, not optional.

**Coverage Requirements**:
- **Core logic**: ≥90% line coverage for models, features, schemas, dataset modules
- **Integration**: End-to-end tests using synthetic test fixtures
- **CLI commands**: At least one happy-path test per command
- **New features**: Tests MUST be included in the same PR as the feature

**Test Approach**:
- Use synthetic GIFs (solid, gradient, noise, animation patterns) for deterministic testing
- Compare predicted outcomes against actual compression results where applicable
- Mock external dependencies (GPU, network) but test real compression engines when available

**Enforcement**:
- Tests MUST exist for new compression engines, metrics, or pipeline logic
- Tests MUST NOT be deleted or weakened without explicit justification
- PRs without adequate test coverage MUST NOT be merged
- `poetry run pytest --cov` MUST be run before any merge

**Test Architecture (4-Layer)**:
Tests are organized into four layers with strict placement rules:

| Layer | Path | Purpose | Time budget | When to run |
|-------|------|---------|-------------|-------------|
| smoke | `tests/smoke/` | Imports, types, pure logic | <5s total | Every save |
| functional | `tests/functional/` | Mocked engines, synthetic GIFs | <2min total | Every commit |
| integration | `tests/integration/` | Real engines, real metrics | <5min total | CI / pre-merge |
| nightly | `tests/nightly/` | Memory, perf, stress, golden | No limit | Nightly schedule |

**Test Discipline**:
- New test files MUST be placed in the correct layer. No test files in `tests/` root.
- Each test MUST cover a **distinct failure mode**. Permutation tests, parameter exhaustion, and internal-function tests are prohibited.
- Prefer adding assertions to existing tests over creating new test functions.
- Test count growth MUST be justified: if a file exceeds 30 tests, review for consolidation.
- `make test` (smoke + functional) MUST complete in <2 minutes. PRs that break this budget MUST NOT be merged.
- Layer-specific conftest files handle isolation. The root conftest provides only shared fixtures.

**Quality Metrics**: The 11-metric quality system (SSIM, MS-SSIM, PSNR, FSIM, GMSD, CHIST, Edge Similarity, Texture Similarity, Sharpness Similarity, Temporal Consistency, Composite Quality) is the source of truth for compression quality. Calculations MUST match documented formulas.

### V. Extensible Tool Interfaces

New compression engines MUST implement the `ExternalTool` abstract base class. Tools declare capabilities via the capability registry.

**Interface Contract**:
- Implement `available()`, `version()`, `apply()` methods
- Declare `COMBINE_GROUP` for pipeline optimization
- Register in `capability_registry.py`

**Rationale**: Enables adding new engines (gifski, WebP, AVIF) without modifying core pipeline logic.

### VI. LLM-Optimized Codebase

This codebase is maintained exclusively by LLM agents. Human readability is secondary to machine parseability.

**Implications**:
- Prefer explicit over implicit patterns
- Use consistent naming conventions machine agents can pattern-match
- Docstrings and type hints are mandatory for LLM context
- Comments explain "why" not "what" (LLMs can read the code)

## ML Dataset Requirements

All code producing or mutating metric data MUST:

1. Guarantee deterministic, reproducible extraction
2. Validate against `MetricRecordV1` Pydantic schema
3. Tag outputs with dataset + code versions
4. Respect canonical train/val/test GIF splits
5. Preserve or update `scaler.pkl` feature-scaling artifacts
6. Regenerate outlier and correlation reports when metrics change

PRs modifying dataset-related code MUST include evidence (CI artifacts or test output) showing compliance.

## Development Workflow

1. **Feature Branches**: All work on `feature/<name>` or numbered branches (e.g., `001-feature-name`)
2. **Test-First**: Write or update tests before implementing features
3. **Spec-First**: New features SHOULD use `/speckit.specify` before implementation
4. **Coverage Check**: Run `poetry run pytest --cov` and verify coverage targets before merge
5. **CI Required**: All tests must pass before merge
6. **Resume Support**: Pipelines MUST support `--resume` for interrupted runs

## Governance

This constitution supersedes all other development practices. Amendments require:

1. Documentation of the change rationale
2. Version increment (MAJOR for principle changes, MINOR for additions, PATCH for clarifications)
3. Update to dependent templates if principles affect spec/plan/task generation

Runtime development guidance is in `CLAUDE.md`. Architecture details are in `SCOPE.md`.

**Version**: 1.2.0 | **Ratified**: 2025-06-26 | **Last Amended**: 2026-02-09
