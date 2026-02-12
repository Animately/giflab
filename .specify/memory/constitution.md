# GifLab Constitution

## Core Principles

### I. Dataset-First
Every feature starts with the question: "Does this help produce better training data?" GifLab's primary mission is generating high-quality training datasets for compression curve prediction. Code that doesn't serve this mission should be questioned.

### II. ML-Ready Data
All pipeline outputs MUST validate against versioned Pydantic schemas (`prediction/schemas.py`). Schema versioning enables backward-compatible evolution. Training data must be deterministic and reproducible.

### III. Poetry-First
ALL Python commands MUST use `poetry run`. No bare `python`, `pip`, or `pytest` commands. This is non-negotiable for dependency consistency.

### IV. SQLite-Only Storage
All pipeline output goes to SQLite via `storage.py`. CSV is available only as an optional export command, never as primary output. This ensures data integrity and queryability.

### V. Engine Wrapper Pattern
All compression engines follow the `ExternalTool` wrapper pattern in `tool_wrappers.py`. New engines MUST implement `available()`, `version()`, `apply()`, and `combines_with()`. The 7 engines are: gifsicle, animately-standard, animately-advanced, animately-hard, imagemagick, ffmpeg, gifski.

### VI. Test-First (4-Layer Architecture)
Tests are organized into four layers (smoke, functional, integration, nightly). New tests MUST go in the correct layer. `make test` must always pass before merging.

### VII. Quality Metrics Preservation
The comprehensive GIF-specific metrics ecosystem (11+ metrics including temporal, gradient, text/UI artifacts) is essential infrastructure. Modifications to metrics code require careful validation to avoid breaking ML training data consistency.

## Constraints

- **Animately CLI**: Always use flag-based arguments (`--input`, `--output`), never positional
- **External binaries**: Graceful degradation when engine binaries are missing
- **Schema evolution**: New schema versions must include migration utilities from previous versions
- **Combination constraints**: `COMBINE_GROUP` prevents nonsensical cross-engine combinations within a single pipeline slot

## Governance

Constitution principles supersede all other practices. Amendments require documentation and test verification.

**Version**: 1.0.0 | **Ratified**: 2026-02-09 | **Last Amended**: 2026-02-09
