# Test Performance Optimization Guide

**Purpose**: Document GifLab's test infrastructure and the 4-layer architecture that replaced the original env-var-based system.

## Overview

GifLab's test suite evolved from a flat directory with environment-variable gates to a **4-layer directory architecture** that enforces speed budgets structurally. Tests are placed in layers based on their isolation level and execution cost, with each layer having its own conftest for isolation.

## Historical Context

### The Problem (Pre-Optimization)
- **Individual tests**: Up to 82+ seconds each due to broken mock patterns
- **Full test suite**: 30+ minutes, making development iteration painful
- **No structural separation**: All tests in a flat `tests/` directory
- **Env-var gates**: `GIFLAB_ULTRA_FAST`, `GIFLAB_MOCK_ALL_ENGINES`, `GIFLAB_MAX_PIPES` controlled behavior — fragile and easy to misconfigure

### The Solution
Replace environment-variable gates with a **directory-based architecture** where test placement determines execution context. Layer-specific conftest files handle isolation automatically.

## 4-Layer Architecture

| Layer | Path | Tests | Purpose | Time budget | When to run |
|-------|------|-------|---------|-------------|-------------|
| smoke | `tests/smoke/` | ~88 | Imports, types, pure logic | <5s | Every save |
| functional | `tests/functional/` | ~939 | Mocked engines, synthetic GIFs | <2min | Every commit |
| integration | `tests/integration/` | ~353 | Real engines, real metrics | <5min | CI / pre-merge |
| nightly | `tests/nightly/` | ~140 | Memory, perf, stress, golden | No limit | Nightly schedule |

**Total**: ~1520 tests. **Fast feedback** (smoke+functional): ~1027 tests in <2 minutes.

### Makefile Targets

```bash
# Fast feedback (default: smoke + functional)
make test

# CI (+ integration)
make test-ci

# Everything including nightly
make test-nightly

# Single file
make test-file F=tests/functional/test_metrics.py
```

### Layer Placement Rules

- **smoke**: Pure Python logic, no I/O, no mocks, no external tools. Tests here validate imports, type schemas, and deterministic computations.
- **functional**: Uses mocks and synthetic GIFs. Tests business logic through mocked engine interfaces. No real compression tools.
- **integration**: Requires real compression engines (gifsicle, animately, etc.). Tests actual tool behavior and cross-engine interactions.
- **nightly**: Performance benchmarks, memory leak detection, stress tests, golden regression data. No time budget — runs on schedule.

## Key Optimization: Mock Pattern Fix

The single most impactful optimization was fixing broken mock patterns. This lesson remains relevant.

### The 2061x Speedup

```python
# BROKEN PATTERN (82.47s execution)
@patch('giflab.core.GifLabRunner')
def test_integration(self, mock_class, tmp_path):
    mock_instance = MagicMock()
    mock_class.return_value = mock_instance

    # BUG: Creates real object instead of using mock
    eliminator = GifLabRunner(tmp_path)
    result = eliminator.run_experimental_analysis()

# FIXED PATTERN (0.04s execution)
@patch('giflab.core.GifLabRunner')
def test_integration(self, mock_class, tmp_path):
    mock_instance = MagicMock()
    mock_class.return_value = mock_instance

    # FIX: Use the mock class
    eliminator = mock_class(tmp_path)
    result = eliminator.run_experimental_analysis()
```

**Impact**: 82.47s → 0.04s per test. This single fix transformed the entire suite.

## Mock Pattern Reference

### Pattern A: Class-Level Mocking (Recommended)
```python
@patch('giflab.core.GifLabRunner')
def test_integration(self, mock_class, tmp_path):
    mock_instance = MagicMock()
    mock_class.return_value = mock_instance

    eliminator = mock_class(tmp_path)
    result = eliminator.run_experimental_analysis()

    assert result is not None
    mock_instance.run_experimental_analysis.assert_called_once()
```

### Pattern B: Method-Level Mocking
```python
def test_integration(self, tmp_path):
    eliminator = GifLabRunner(tmp_path)
    with patch.object(eliminator, 'run_experimental_analysis') as mock_method:
        mock_method.return_value = mock_result
        result = eliminator.run_experimental_analysis()
```

### Pattern C: Fixture-Based (Cleanest)
```python
@pytest.fixture
def fast_experimental_runner(tmp_path, monkeypatch):
    runner = GifLabRunner(tmp_path)
    monkeypatch.setattr(runner, '_run_comprehensive_testing', lambda: mock_result)
    monkeypatch.setattr(runner, '_execute_pipeline_matrix', lambda: [])
    return runner

def test_integration(fast_experimental_runner):
    result = fast_experimental_runner.run_experimental_analysis()
```

## CI/CD Integration

### Workflow Strategy
```yaml
# .github/workflows/tests.yml
jobs:
  test:
    strategy:
      matrix:
        layer: [fast, ci, nightly]

    steps:
      - name: Fast Tests (smoke + functional, <2min)
        if: matrix.layer == 'fast'
        run: make test

      - name: CI Tests (+ integration, <5min)
        if: matrix.layer == 'ci'
        run: make test-ci

      - name: Nightly Tests (everything)
        if: matrix.layer == 'nightly'
        run: make test-nightly
```

### Trigger Strategy
- **Fast tests**: Every pull request
- **CI tests**: Main branch merges
- **Nightly tests**: Scheduled nightly runs

## Performance Thresholds

| Layer | Time budget | Enforcement |
|-------|-------------|-------------|
| smoke | <5s | conftest isolation |
| functional | <2min | conftest isolation |
| integration | <5min | CI timeout |
| nightly | No limit | Scheduled only |

If `make test` exceeds 2 minutes, the PR MUST NOT be merged (per constitution v1.2.0).

## Smart Pipeline Sampling

### Equivalence Class Strategy
Group similar pipelines and test representatives:
```python
LOSSY_EQUIVALENCE_CLASSES = {
    "low_lossy": ["lossy_10", "lossy_15", "lossy_20"],      # Test lossy_15
    "mid_lossy": ["lossy_30", "lossy_40", "lossy_50"],      # Test lossy_40
    "high_lossy": ["lossy_60", "lossy_70", "lossy_80"],     # Test lossy_70
}
```

### Risk-Based Prioritization
- **High-risk**: Always test (new engines, edge cases, cross-engine interactions)
- **Medium-risk**: Sample 50% (established engines with minor variations)
- **Low-risk**: Sample 10% (redundant or legacy configurations)

## Lessons Learned

1. **Mock patterns matter**: A single broken pattern caused 2061x performance degradation
2. **Structure beats configuration**: Directory-based layers are more reliable than env-var gates
3. **Layered approach works**: Different test layers serve different needs effectively
4. **Conftest isolation scales**: Layer-specific conftest files automatically enforce boundaries

---

**Last Updated**: February 2026
**Architecture**: 4-layer (smoke / functional / integration / nightly)
**Status**: Active — maintained via constitution v1.2.0
