# Requirements Checklist: Dataset Pipeline Refactor

**Purpose**: Quality validation checklist for the pipeline refactoring
**Created**: 2026-02-09
**Feature**: [spec.md](../spec.md)

## Engine Layer

- [ ] CHK001 `AnimatelyHardLossyCompressor` class exists in `tool_wrappers.py`
- [ ] CHK002 `AnimatelyHardLossyCompressor.NAME` is `"animately-hard"`
- [ ] CHK003 Hard compressor uses `--hard --lossy <level>` flags (not positional args)
- [ ] CHK004 `AnimatelyLossyCompressor.NAME` renamed to `"animately-standard"`
- [ ] CHK005 `AnimatelyAdvancedLossyCompressor.NAME` renamed to `"animately-advanced"`
- [ ] CHK006 All 7 engine wrappers registered in tool registry
- [ ] CHK007 Each engine produces valid compressed GIF output for a test input

## Schema Layer

- [ ] CHK008 `Engine` enum has exactly 7 values: gifsicle, animately-standard, animately-advanced, animately-hard, imagemagick, ffmpeg, gifski
- [ ] CHK009 `TrainingRecordV2` uses `dict[str, CompressionCurveV1]` for curves (not hardcoded fields)
- [ ] CHK010 V1→V2 migration maps `animately` to `animately-standard`
- [ ] CHK011 All Pydantic models pass validation with sample data
- [ ] CHK012 `LossyEngine` enum in `lossy.py` includes `ANIMATELY_HARD`

## Pipeline Layer

- [ ] CHK013 Single unified runner replaces dual system (no `prediction_runner.py` and `core/runner.py` doing same work)
- [ ] CHK014 Feature extraction integrated into unified runner
- [ ] CHK015 Quality metrics measured for each compression result
- [ ] CHK016 Wrapper validation applied to each compressed output
- [ ] CHK017 `--mode single` runs 7 engines × 7 levels = 49 per GIF
- [ ] CHK018 `--mode full` runs all pipeline combinations
- [ ] CHK019 `--force` reprocesses all GIFs
- [ ] CHK020 `--upgrade` reprocesses only outdated GIFs
- [ ] CHK021 Pipeline is idempotent (re-running without --force produces no duplicates)

## Storage Layer

- [ ] CHK022 All pipeline output goes to SQLite via `storage.py`
- [ ] CHK023 No CSV files written during pipeline execution
- [ ] CHK024 `giflab export --format csv` produces valid CSV
- [ ] CHK025 `giflab export --format json` produces valid JSON
- [ ] CHK026 SQLite schema supports all 7 engines
- [ ] CHK027 `compression_runs` table has UNIQUE constraint on (gif_id, pipeline_id, param_preset_id)

## CLI Layer

- [ ] CHK028 `giflab --help` shows exactly: run, train, predict, export, stats
- [ ] CHK029 No legacy commands appear in CLI help
- [ ] CHK030 `giflab run` defaults to `--mode single`
- [ ] CHK031 `giflab stats` shows per-engine and per-pipeline counts

## Cleanup Layer

- [ ] CHK032 `benchmarks/` directory removed
- [ ] CHK033 `config_profiles/` directory removed
- [ ] CHK034 `dashboard/` directory removed
- [ ] CHK035 No broken imports after cleanup
- [ ] CHK036 `make test` passes after all changes

## Documentation Layer

- [ ] CHK037 `README.md` reflects "training dataset generation" mission
- [ ] CHK038 `pyproject.toml` description updated
- [ ] CHK039 `SCOPE.md` lists all 7 engines
- [ ] CHK040 `CLAUDE.md` references updated engine names and CLI commands

## Regression

- [ ] CHK041 All smoke tests pass: `poetry run pytest tests/smoke/ -x -q`
- [ ] CHK042 All functional tests pass: `poetry run pytest tests/functional/ -x -q`
- [ ] CHK043 All integration tests pass: `poetry run pytest tests/integration/ -x -q`
- [ ] CHK044 Quality metrics ecosystem unchanged and functional
- [ ] CHK045 Caching system unchanged and functional
- [ ] CHK046 Monitoring system unchanged and functional

## Notes

- Check items off as completed: `[x]`
- CHK007 depends on all engine binaries being installed on the machine
- CHK043 may be slow (~5min) — run after all other checks pass
- Items are numbered sequentially for easy reference in task tracking
