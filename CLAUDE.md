# Claude Code Configuration for GifLab

This file provides project-specific guidance for AI assistants working with the GifLab codebase.

## 🚨 **CRITICAL: Always Use Poetry**

**This project uses Poetry for ALL Python command execution.** Never run Python commands directly.

### ❌ **WRONG** - These will fail with ModuleNotFoundError:
```bash
python -m giflab run --preset quick-test
python -m pytest tests/
python -c "from giflab.metrics import calculate_metrics"
PYTHONPATH=src python -m giflab run
```

### ✅ **CORRECT** - Always use Poetry:
```bash
poetry run python -m giflab run --preset quick-test
poetry run pytest tests/
poetry run python -c "from giflab.metrics import calculate_metrics"
```

## Why Poetry is Required

1. **Dependencies**: `click`, `pytest`, `numpy`, etc. exist only in Poetry's virtual environment
2. **Path Resolution**: Poetry ensures proper module discovery and PYTHONPATH setup
3. **Version Consistency**: Poetry locks dependency versions for reproducible builds
4. **Project Structure**: `src/giflab/` package structure requires Poetry's path handling

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
# Pipeline analysis and optimization
poetry run python -m giflab run --preset frame-focus
poetry run python -m giflab run --sampling representative

# Large-scale processing
poetry run python -m giflab run data/raw --workers 8

# Analysis tools
poetry run python -m giflab select-pipelines results.csv --top 3
```

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