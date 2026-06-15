# Claude Code Configuration for GifLab

This file provides project-specific guidance for AI assistants working with the GifLab codebase.

## 🚨 **CRITICAL: Always Use Poetry**

**This project uses Poetry for ALL Python command execution.** Never run Python commands directly.

### ❌ **WRONG** - These will fail with ModuleNotFoundError:
```bash
python -m giflab run data/raw
python -m pytest tests/
python -c "from giflab.metrics import calculate_metrics"
PYTHONPATH=src python -m giflab run
```

### ✅ **CORRECT** - Always use Poetry:
```bash
poetry run python -m giflab run data/raw
poetry run pytest tests/
poetry run python -c "from giflab.metrics import calculate_metrics"
```

## Why Poetry is Required

1. **Dependencies**: `click`, `pytest`, `numpy`, etc. exist only in Poetry's virtual environment
2. **Path Resolution**: Poetry ensures proper module discovery and PYTHONPATH setup
3. **Version Consistency**: Poetry locks dependency versions for reproducible builds
4. **Project Structure**: `src/giflab/` package structure requires Poetry's path handling

## 🎯 **Metric accuracy is load-bearing — read this before touching any metric**

The metrics in `src/giflab/metrics.py`, `src/giflab/enhanced_metrics.py`, `src/giflab/temporal_artifacts.py`, and `src/giflab/gradient_color_artifacts.py` are the foundation everything else builds on — `gifprep`, the prediction-features pipeline (`src/giflab/prediction/`), the audit harness (`scripts/audit/`), `composite_quality`, and any downstream selection or ranking. **A small error in a metric becomes a systematic error in everything downstream.** Lossy compression scores, audit verdicts, ML training data, gifprep's quality decisions — all inherit metric mistakes.

When implementing or modifying a metric, **prefer the more accurate approach over the pragmatic shortcut, even if it means more code**. The 2026-05-22 metrics audit surfaced multiple bugs of the same shape: a fix correctly addressed the named failure mode but introduced a new failure mode adjacent to it (cliff-edge thresholds, sentinel-instead-of-NaN, single-stream-mislabeled-as-pair). The pattern to avoid:

- **Continuous over discrete.** Avoid hard thresholds that produce cliff-edges (`if x > T: worst else: best`). Any threshold has a discontinuity that downstream consumers cannot smooth over. Use smooth degradation instead — e.g. `score = max(0.0, 1.0 - distance / scale)`. If you find yourself picking a threshold value, ask whether the same code could be a smooth function with no threshold.
- **NaN over fabricated values.** When a metric genuinely can't be computed (mismatched frame counts, missing binary, subprocess crashes, undefined-on-input), return `float("nan")` and aggregate with `np.nanmean` / `np.nanpercentile` / `np.nanmin`. Never substitute sentinels like `0.5`, `1.0`, `-1.0`, or `50.0` — they silently corrupt every aggregate and propagate as fake data through `composite_quality`, validators, and CSV exports. Downstream NaN-handling guards must use NaN-aware comparisons (`any([nan, nan])` is `True` in Python — that catches most people out).
- **Pair-wise over single-stream — and labelled honestly.** If a key is named like a pair-comparison (`temporal_consistency`, `texture_similarity`, `disposal_artifacts`), it must measure original-vs-compressed. If it actually only measures the compressed stream, name it `_compressed` and document the limitation. Never let a single-stream value be weighted into `composite_quality` as if it were a pair signal — a perfectly-static-black compressed output will silently win.
- **Honest error paths end-to-end.** When a metric fails, propagate the failure all the way through. Validator guards that read possibly-NaN values must use NaN-aware comparisons. CSV serialisation must round-trip NaN. Composite formulas must redistribute weight when an input is missing, not silently default it to zero or one.
- **Same key shape across paths.** `calculate_comprehensive_metrics`, `_from_frames`, the Phase 6 optimized path, and any future fast path must emit the same key schema for the same conceptual content. If the optimized path can't compute `_pre`/`_post` separately, document that prominently rather than silently aliasing — silent equivalence lies to the consumer about what was actually measured.

**"Within reason"** means: don't burn weeks on academic perfection for a metric used in one diagnostic-only place. But for anything that feeds `composite_quality`, the audit pipeline, ML feature extraction, or selection/ranking — the accuracy choice always wins. Extra 30 lines now saves "why is our composite_quality lower than it should be" investigations later, and worse: it saves us from publishing benchmarks built on quietly wrong numbers.

If a PR has a "pragmatic" version and a "more accurate" version, document both in the PR body and pick the more accurate one. Only fall back to pragmatic if the accuracy choice has a real cost (e.g. >2× runtime regression). Performance and accuracy usually don't trade off — the cleaner implementation is often faster too.

## Common Command Patterns

### Testing
```bash
# Fast feedback (default: smoke + functional, <2min)
make test

# CI (+ integration, <5min)
make test-ci

# Everything including nightly
make test-nightly

# Single file
make test-file F=tests/functional/test_metrics.py

# Direct pytest (layer-specific)
poetry run pytest tests/smoke/ tests/functional/ -x -q
poetry run pytest tests/integration/ -n auto -q
```

### GifLab Operations
```bash
# Compression + feature extraction over a directory of GIFs
poetry run python -m giflab run data/raw                 # single-engine (quick)
poetry run python -m giflab run data/raw --mode full     # all frame × colour × lossy combos

# Inspect the collected dataset
poetry run python -m giflab stats --db data/giflab.db
poetry run python -m giflab export --db data/giflab.db --output runs.csv --table runs
```
> Note: `--preset`, `--sampling`, and `select-pipelines` were removed in `648db9a`; the leaderboard harness that used them is being rebuilt (see `docs/technical/compression-pipeline-leaderboard.md`).

### Development Tools
```bash
# Code quality
poetry run black src/ tests/
poetry run ruff check src/ tests/
poetry run mypy src/

# Interactive Python
poetry run python
poetry run jupyter notebook
```

## Project Structure Notes

- **Source code**: `src/giflab/` (package structure)
- **Tests**: `tests/` (4-layer: smoke, functional, integration, nightly)
- **Configuration**: `pyproject.toml` (Poetry + tool config)
- **Dependencies**: Managed entirely through Poetry
- **Scripts**: Defined in `[tool.poetry.scripts]` section

## Environment Setup

The project is already properly configured. Just ensure Poetry is installed:

```bash
# Install Poetry (if needed)
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Verify setup
poetry run python -c "import giflab; print('✅ GifLab ready!')"
```

## Makefile Integration

The project includes a Makefile that properly uses Poetry. You can also use:

```bash
make test           # Fast feedback: smoke + functional (<2min)
make test-ci        # CI: + integration (<5min)
make test-nightly   # Everything including perf/memory
make test-file F=tests/functional/test_metrics.py  # Single file
make data           # Run compression pipeline
```

All Makefile targets internally use `poetry run` commands.

## Animately CLI Tool Usage

### 🎯 **Critical: Animately Requires Flag-Based Arguments**

**Animately is an in-house CLI tool** with multiple versions in `/Users/lachlants/bin/`. 

### ❌ **WRONG** - Positional arguments will FAIL silently:
```bash
animately input.gif output.gif --lossy 60
```

### ✅ **CORRECT** - Always use flag-based syntax:
```bash
animately --input input.gif --output output.gif --lossy 60
```

### Version Management
- **Current recommended version**: `1.1.20.0` (Sep 4, 2025) - enhanced logging
- **Location**: `/Users/lachlants/bin/animately` → symlinked to best version
- **Available versions**: 
  - `animately_1.1.6.0_experimental` (oldest, basic)
  - `animately_1.1.18.0` (stable)
  - `animately_1.1.20.0` (July 21) - basic logging
  - `animately_1.1.20.0` (Sep 4) - **RECOMMENDED** - enhanced logging with timestamps

### Common Usage Patterns
```bash
# Basic compression
animately --input source.gif --output compressed.gif

# Lossy compression  
animately --input source.gif --output lossy.gif --lossy 60

# Advanced lossy with color reduction
animately --input source.gif --output advanced.gif --advanced-lossy 40 --colors 128

# With scaling and cropping
animately --input source.gif --output processed.gif --scale 0.5 --crop "100,100,200,200"
```

### Troubleshooting Animately
- **Exit code 1/2 without error**: Check if using positional instead of flag arguments
- **"No input file path provided"**: Missing `--input` flag
- **Silent failure**: Verify flag syntax: `--input file.gif --output out.gif`

## Python/Poetry Troubleshooting

### "ModuleNotFoundError: No module named 'click'"
- **Cause**: Using `python -m` instead of `poetry run python -m`
- **Fix**: Always prefix with `poetry run`

### "ModuleNotFoundError: No module named 'giflab'"
- **Cause**: Wrong working directory or missing Poetry
- **Fix**: Ensure you're in project root with `pyproject.toml`, then use `poetry run`

### "Command not found: giflab"
- **Cause**: Trying to use `giflab` directly instead of module syntax
- **Fix**: Use `poetry run python -m giflab` or `poetry run giflab`

## Test Architecture (4-Layer)

Tests are organized into four layers. New tests MUST go in the correct layer — never in `tests/` root.

| Layer | Path | Purpose | Time budget |
|-------|------|---------|-------------|
| smoke | `tests/smoke/` | Imports, types, pure logic | <5s |
| functional | `tests/functional/` | Mocked engines, synthetic GIFs | <2min |
| integration | `tests/integration/` | Real engines, real metrics | <5min |
| nightly | `tests/nightly/` | Memory, perf, stress, golden | No limit |

### Where to Put New Tests
- **Pure logic, schemas, imports** → `tests/smoke/`
- **Needs mocks or synthetic GIFs** → `tests/functional/`
- **Needs real compression engines** → `tests/integration/`
- **Performance, memory, stress** → `tests/nightly/`

### Intentionally Skipped Tests (Normal Behavior)
These tests are **correctly skipped** during normal test runs:

1. **Golden Results Test** - `tests/nightly/test_gradient_color_regression.py`
   - **Skip reason**: "Use --update-golden to save new golden results"
   - **How to run**: `poetry run pytest --update-golden tests/nightly/test_gradient_color_regression.py::*golden*`
   - **Purpose**: Generates reference data for regression testing

2. **Stress Tests** - `tests/nightly/test_gradient_color_performance.py`
   - **Skip reason**: "Stress tests require GIFLAB_STRESS_TESTS=1"
   - **How to run**: `GIFLAB_STRESS_TESTS=1 poetry run pytest tests/nightly/test_gradient_color_performance.py::TestStressTesting`
   - **Purpose**: Performance testing with large images and many frames

### Test Troubleshooting
- **Skipped animately tests**: Check that animately uses `--input`/`--output` flags (not positional args)
- **Missing `compress` method**: ExternalTool classes use `apply()` method, not `compress()`
- **Test timeouts**: Use `-n auto` for parallel execution: `poetry run pytest -n auto`

## For AI Assistants: Key Reminders

1. **ALWAYS** use `poetry run` for Python execution
2. **NEVER** use bare `python`, `pip`, or `pytest` commands
3. **ANIMATELY**: Always use `--input file.gif --output out.gif` (never positional args)
4. **CHECK** that you're prefixing commands with `poetry run`
5. **TEST** commands with `poetry run` if unsure
6. **REMEMBER** this is a Poetry project - dependencies require the virtual environment
7. **METRICS**: Read the "Metric accuracy is load-bearing" section above before touching `metrics.py`, `enhanced_metrics.py`, `temporal_artifacts.py`, or `gradient_color_artifacts.py`. Continuous over discrete, NaN over sentinels, pair-wise honestly labelled. The accuracy choice wins unless it costs >2× runtime.

## ⚠️ Known gotchas

**Windows is not a CI target — there is intentionally no Windows job.** GifLab runs on macOS/Linux only; the `Windows External Tools Tests` job was removed from `ci.yml` (PR #51, 2026-06-09) because it was a perpetually-red, non-required check that added noise without a signal we act on. CI runs only `Fast Tests` ×2 (the required gate), `Core Tests` ×2, `External Tools (Ubuntu)`, `macOS Compatibility`, and `Lint and Type Check`. **A missing Windows check is expected — do not "restore" it or treat its absence as a regression.** Don't add Windows-specific CI, test skips, or platform shims unless someone explicitly makes Windows a target.

**`gh pr edit --body ...` can silently exit 1.** GitHub's "Projects (classic) deprecated" GraphQL error causes `gh pr edit` to exit non-zero even when the body would otherwise update fine. Agents that don't check the exit code think the edit succeeded — but the body never lands. Observed in the [[giflab-rollout-2026-05-26]] rollout during PR #22's round-2 follow-up.

**Workaround**: use the REST API directly:

```
gh api -X PATCH repos/Animately/giflab/pulls/<N> -f body="<body content>"
```

The REST endpoint doesn't touch projects-classic and exits 0 on success. Verify the change landed via `gh pr view <N> --json body`.

**Squash-merging a stale branch silently reverts newer same-file work.** A PR branched from an older `main` carries a full snapshot of every file it touches. When squash-merged, GitHub replaces those files with the branch's snapshot — so any change another PR landed in the *same file* after this branch was cut is silently overwritten, with no merge conflict. This bit us when PR #31 (branched before #30) squash-merged and wiped out #30's `edge_similarity` median-aggregation code in `metrics.py`; #30's test survived but its production code was gone, and `composite_quality` quietly reverted to a non-monotonic signal (restored via PR #32).

**Rule**: always `git fetch origin && git rebase origin/main` an audit-fix branch immediately before squash-merge, then re-run `make test` on the rebased branch. `main` now enforces "require branches up to date before merging" (branch protection, gated on the green `Fast Tests` checks), which makes GitHub demand the rebase — but do it locally first so you catch any same-file conflict and verify tests before pushing.

## Agent Teams

Agent teams are enabled for this project. When using them, follow these guidelines.

### When to Use Agent Teams
- **Research and review**: multiple teammates investigate different aspects simultaneously
- **New modules or features**: teammates each own a separate piece without conflicts
- **Debugging with competing hypotheses**: teammates test different theories in parallel
- **Cross-layer coordination**: changes spanning frontend, backend, and tests

### When NOT to Use Agent Teams
- Sequential tasks or same-file edits — use a single session or subagents instead
- Simple tasks where coordination overhead exceeds the benefit
- Work with many dependencies between steps

### Best Practices
- **Give teammates enough context**: they load CLAUDE.md but don't inherit the lead's conversation history — include task-specific details in the spawn prompt
- **Size tasks appropriately**: self-contained units that produce a clear deliverable (a function, a test file, a review)
- **Avoid file conflicts**: break work so each teammate owns a different set of files
- **Wait for teammates**: don't start implementing tasks yourself — let teammates complete their work
- **Start with research/review**: for unfamiliar tasks, use teams for investigation before implementation
- **Use delegate mode**: press `Shift+Tab` to restrict the lead to coordination-only (no direct code changes)
- **Require plan approval for risky work**: have teammates plan before implementing, then approve or reject

### Task Management
- Aim for 5-6 tasks per teammate to keep everyone productive
- Tasks have three states: pending, in progress, completed
- Tasks can have dependencies — blocked tasks auto-unblock when dependencies complete
- Teammates self-claim the next unassigned, unblocked task after finishing one

### Display Modes
- **In-process** (default): `Shift+Up/Down` to navigate teammates, `Enter` to view, `Escape` to interrupt, `Ctrl+T` for task list
- **Split panes**: requires tmux or iTerm2 — set `"teammateMode": "tmux"` in settings

---

*This configuration ensures reliable, reproducible development workflows for both humans and AI assistants.*

## Active Technologies
- Python 3.11 (per `pyproject.toml`) + No new dependencies. Reuses existing internals: `tool_wrappers.AnimatelyLossyCompressor` + 4 sibling compressors, `metrics.calculate_comprehensive_metrics`, `external_engines.common.run_command`, `error_handling.GifLabError` hierarchy. (003-public-api-refactor)
- N/A (filesystem read/write only; no DB) (003-public-api-refactor)

## Recent Changes
- 003-public-api-refactor: Added Python 3.11 (per `pyproject.toml`) + No new dependencies. Reuses existing internals: `tool_wrappers.AnimatelyLossyCompressor` + 4 sibling compressors, `metrics.calculate_comprehensive_metrics`, `external_engines.common.run_command`, `error_handling.GifLabError` hierarchy.
