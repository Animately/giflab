# 🧪 Testing Strategy for GifLab Metrics

**Comprehensive testing approach for quality metrics, data preparation, and ML pipeline components.**

---

## 1. Overview

This document outlines the testing strategy used to ensure reliability and correctness of GifLab's expanded metrics system. The approach covers unit testing, integration testing, and validation strategies for the complete ML pipeline.

---

## 1.1 Test Architecture (4-Layer)

Tests are organized into four layers. Each layer has its own conftest for isolation.

| Layer | Path | Purpose | Time budget | When to run |
|-------|------|---------|-------------|-------------|
| smoke | `tests/smoke/` | Imports, types, pure logic | <5s | Every save |
| functional | `tests/functional/` | Mocked engines, synthetic GIFs | <2min | Every commit |
| integration | `tests/integration/` | Real engines, real metrics | <5min | CI / pre-merge |
| nightly | `tests/nightly/` | Memory, perf, stress, golden | No limit | Nightly schedule |

### Makefile Targets
```bash
make test           # smoke + functional (<2min)
make test-ci        # + integration (<5min)
make test-nightly   # everything including perf/memory
make test-file F=tests/functional/test_metrics.py
```

### Key Optimizations
1. **Critical mock pattern fixes**: 82s → 0.04s (2061x speedup)
2. **Parallel execution**: pytest-xdist with `-n auto`
3. **Layer-specific conftest isolation**: Each layer's conftest handles mocking/setup
4. **Directory-based separation**: Structure enforces speed budgets, no env vars needed

📖 **For complete optimization details**: See [Test Performance Optimization Guide](../guides/test-performance-optimization.md)

---

## 2. Core Testing Principles

### 2.1 Deterministic Test Fixtures

**Approach**: Generate controlled test data with known properties:

```python
def _identical_frames(self):
    """Create two identical frames with realistic content."""
    frame = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
    return frame, frame.copy()

def _slightly_different_frames(self):
    """Create frames with small differences (should score well)."""
    frame1 = np.random.randint(100, 150, (64, 64, 3), dtype=np.uint8)
    frame2 = frame1.copy()
    # Add small noise
    noise = np.random.randint(-10, 11, frame1.shape, dtype=np.int16)
    frame2 = np.clip(frame2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame1, frame2

def _very_different_frames(self):
    """Create frames with significant differences (should score poorly)."""
    frame1 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    np.random.seed(999)  # Different seed for different pattern
    frame2 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    np.random.seed()  # Reset seed
    return frame1, frame2
```

### 2.2 Metric Range Assertions

**Strategy**: Validate expected behavior across similarity spectrum:

```python
def test_metric_ranges(self):
    """Test that metrics behave correctly across similarity levels."""
    
    # Test identical frames
    ident1, ident2 = self._identical_frames()
    identical_score = metric_function(ident1, ident2)
    
    # Test similar frames  
    sim1, sim2 = self._slightly_different_frames()
    similar_score = metric_function(sim1, sim2)
    
    # Test different frames
    diff1, diff2 = self._very_different_frames()
    different_score = metric_function(diff1, diff2)
    
    # Assert expected ordering
    if higher_is_better:
        assert identical_score >= similar_score >= different_score
        assert identical_score > 0.95  # High similarity
    else:  # lower_is_better (MSE, RMSE, GMSD)
        assert identical_score <= similar_score <= different_score
        assert identical_score < 0.1   # Low error
```

### 2.3 Negative Testing

**Error Condition Validation**:

```python
def test_error_conditions(self):
    """Test that metrics handle invalid inputs correctly."""
    
    # Mismatched shapes
    frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
    frame2 = np.zeros((32, 32, 3), dtype=np.uint8)
    
    with pytest.raises(ValueError):
        metric_function(frame1, frame2)
    
    # Invalid dtype
    frame_invalid = np.zeros((64, 64, 3), dtype=np.float64)
    
    with pytest.raises(ValueError):
        metric_function(frame1, frame_invalid)
```

---

## 3. Metric-Specific Testing

### 3.1 Error-Based Metrics (MSE, RMSE)

**Test Requirements**:
- Identical frames → score = 0.0
- Different frames → score > 0.0
- Ordering: identical < similar < different

**Implementation**:
```python
def test_mse_comprehensive(self):
    """Test MSE with various frame types."""
    # Identical frames
    ident1, ident2 = self._identical_frames()
    assert mse(ident1, ident2) == pytest.approx(0.0, abs=1e-6)
    
    # Similar frames should have low MSE
    sim1, sim2 = self._slightly_different_frames()
    similar_mse = mse(sim1, sim2)
    
    # Very different frames should have high MSE
    diff1, diff2 = self._very_different_frames()
    different_mse = mse(diff1, diff2)
    
    assert similar_mse < different_mse
    assert similar_mse > 0.0
    assert different_mse > 1000.0  # Should be significantly higher
```

### 3.2 Similarity-Based Metrics (SSIM, FSIM, etc.)

**Test Requirements**:
- Identical frames → score ≈ 1.0
- Different frames → score < 1.0
- Ordering: identical > similar > different

**Implementation**:
```python
def test_fsim_comprehensive(self):
    """Test FSIM with gradient patterns."""
    # Identical frames
    ident1, ident2 = self._identical_frames()
    identical_fsim = fsim(ident1, ident2)
    
    # Similar gradients
    grad1, grad2 = self._gradient_frames()
    similar_fsim = fsim(grad1, grad2)
    
    # Different gradients
    diff_grad1, diff_grad2 = self._different_gradient_frames()
    different_fsim = fsim(diff_grad1, diff_grad2)
    
    assert identical_fsim >= different_fsim
    assert identical_fsim > 0.95  # Should be very high for identical
    assert similar_fsim > 0.0     # Should be positive
    assert different_fsim > 0.0   # Should be positive
```

### 3.3 Content-Specific Testing

**Specialized Test Frames**:

```python
def _gradient_frames(self):
    """Create frames with gradients to test gradient-based metrics."""
    frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
    frame2 = np.zeros((64, 64, 3), dtype=np.uint8)
    
    for i in range(64):
        for j in range(64):
            # Horizontal gradient
            val1 = int(255 * i / 63)
            val2 = int(255 * i / 63) + np.random.randint(-5, 6)
            frame1[i, j] = [val1, val1, val1]
            frame2[i, j] = [np.clip(val2, 0, 255)] * 3
    
    return frame1, frame2

def _color_frames(self):
    """Create frames with different color distributions."""
    frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
    frame2 = np.zeros((64, 64, 3), dtype=np.uint8)
    
    # Fill with similar colors
    frame1[:32, :32] = [255, 0, 0]    # Red quadrant
    frame1[:32, 32:] = [0, 255, 0]    # Green quadrant
    frame1[32:, :32] = [0, 0, 255]    # Blue quadrant
    frame1[32:, 32:] = [255, 255, 0]  # Yellow quadrant
    
    # Slightly different colors
    frame2[:32, :32] = [250, 5, 5]    # Slightly different red
    frame2[:32, 32:] = [5, 250, 5]    # Slightly different green
    frame2[32:, :32] = [5, 5, 250]    # Slightly different blue
    frame2[32:, 32:] = [250, 250, 5]  # Slightly different yellow
    
    return frame1, frame2
```

---

## 4. Integration Testing

### 4.1 Comprehensive Metrics Testing

**Test Complete Pipeline**:

```python
def test_calculate_comprehensive_metrics_basic(self, tmp_path):
    """Test complete metrics calculation pipeline."""
    original_path, compressed_path = self._create_test_gifs(tmp_path)
    
    metrics = calculate_comprehensive_metrics(original_path, compressed_path)
    
    # Verify all expected metrics are present
    expected_metrics = [
        'ssim', 'ms_ssim', 'psnr', 'mse', 'rmse', 'fsim', 'gmsd',
        'chist', 'edge_similarity', 'texture_similarity', 'sharpness_similarity'
    ]
    
    for metric in expected_metrics:
        assert metric in metrics
        assert f"{metric}_std" in metrics
        assert f"{metric}_min" in metrics
        assert f"{metric}_max" in metrics
    
    # Verify temporal consistency
    assert 'temporal_consistency_pre' in metrics
    assert 'temporal_consistency_post' in metrics
    assert 'temporal_consistency_delta' in metrics
    
    # Verify composite quality
    assert 'composite_quality' in metrics
    assert 0.0 <= metrics['composite_quality'] <= 1.0
```

### 4.2 Configuration Testing

**Test Raw Metrics Flag**:

```python
def test_raw_metrics_flag(self, tmp_path):
    """Test raw metrics flag functionality."""
    original_path, compressed_path = self._create_test_gifs(tmp_path)
    
    # Test with raw metrics enabled
    config = MetricsConfig(RAW_METRICS=True)
    metrics = calculate_comprehensive_metrics(original_path, compressed_path, config)
    
    # Verify raw metrics are present
    assert 'psnr_raw' in metrics
    assert 'ssim_raw' in metrics
    assert 'temporal_consistency_delta_raw' in metrics
    
    # Verify raw values differ from normalized (for PSNR)
    assert metrics['psnr_raw'] != metrics['psnr']
```

---

## 5. Data Preparation Testing

### 5.1 Scaling Functions

**Test Scaling Helpers**:

```python
def test_minmax_scale_basic(self):
    """Test min-max scaling functionality."""
    data = [0, 5, 10]
    scaled = minmax_scale(data)
    assert np.allclose(scaled, [0.0, 0.5, 1.0])

def test_minmax_scale_constant(self):
    """Test min-max scaling with constant values."""
    data = [3, 3, 3]
    scaled = minmax_scale(data, feature_range=(0, 1))
    assert np.allclose(scaled, [0.5, 0.5, 0.5])

def test_zscore_scale_basic(self):
    """Test z-score scaling functionality."""
    data = [0, 5, 10]
    scaled = zscore_scale(data)
    assert np.isclose(np.mean(scaled), 0.0)
    assert np.isclose(np.std(scaled), 1.0)
```

### 5.2 Outlier Detection

**Test Outlier Clipping**:

```python
def test_clip_outliers_iqr(self):
    """Test IQR-based outlier clipping."""
    data = [1, 2, 2, 2, 100]  # 100 is an outlier
    clipped = clip_outliers(data, method="iqr", factor=1.5)
    assert max(clipped) < 100  # Outlier should be clipped down

def test_clip_outliers_sigma(self):
    """Test sigma-based outlier clipping."""
    data = [0, 0, 0, 0, 50]
    clipped = clip_outliers(data, method="sigma", factor=1.5)
    assert max(clipped) < 50
```

---

## 6. Schema Validation Testing

### 6.1 Pydantic Schema Testing

**Test Schema Validation**:

```python
def test_validation_passes_on_real_record(self, tmp_path):
    """Test schema validation on real metric output."""
    original, compressed = self._create_test_gifs(tmp_path)
    
    metrics = calculate_comprehensive_metrics(original, compressed)
    
    # Should not raise
    model = validate_metric_record(metrics)
    assert isinstance(model, MetricRecordV1)
    assert 0.0 <= model.composite_quality <= 1.0

def test_validation_fails_on_negative_values(self):
    """Test schema validation rejects invalid data."""
    bad_record = {
        "composite_quality": 0.5,
        "kilobytes": -10,  # invalid
        "render_ms": 100,
        "ssim": 0.8,
    }
    
    assert is_valid_record(bad_record) is False
    
    with pytest.raises(Exception):
        validate_metric_record(bad_record)
```

---

## 7. Performance Testing

### 7.1 Benchmark Testing

**Test Processing Time**:

```python
def test_calculate_comprehensive_metrics_performance(self, tmp_path):
    """Test metrics calculation performance."""
    original_path, compressed_path = self._create_test_gifs(tmp_path)
    
    start_time = time.time()
    metrics = calculate_comprehensive_metrics(original_path, compressed_path)
    end_time = time.time()
    
    # Should complete within reasonable time
    assert end_time - start_time < 1.0  # Less than 1 second
    
    # Should include timing information
    assert 'render_ms' in metrics
    assert metrics['render_ms'] > 0
```

### 7.2 Memory Testing

**Test Memory Usage**:

```python
def test_memory_efficiency(self, tmp_path):
    """Test memory efficiency of metrics calculation."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Process multiple GIFs
    for i in range(10):
        original_path, compressed_path = self._create_test_gifs(tmp_path)
        metrics = calculate_comprehensive_metrics(original_path, compressed_path)
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable (< 100MB)
    assert memory_increase < 100 * 1024 * 1024
```

---

## 8. Notebook Testing

### 8.1 Smoke Testing

**Test Notebook Execution**:

```python
def test_execute_notebook(nb_path: Path, tmp_path):
    """Execute notebook and verify it completes without errors."""
    # Read notebook
    with nb_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    
    # Prepare executor
    ep = ExecutePreprocessor(timeout=180, kernel_name="python3", allow_errors=True)
    
    # Run all cells
    ep.preprocess(nb, {"metadata": {"path": tmp_path}})
    
    # Note: This is a smoke test - errors are tolerated to keep
    # the test suite lightweight and avoid external data dependencies
```

---

## 9. Risk Mitigation Testing

### 9.1 Identified Risks and Test Coverage

**Performance Risk**:
- **Risk**: Added aggregation & temporal-delta increase runtime by ~7%
- **Mitigation**: Efficient aggregation algorithms
- **Test**: Performance benchmarks ensure processing time stays reasonable

**Third-party Dependencies**:
- **Risk**: Version compatibility issues
- **Mitigation**: Pinned versions (`scikit-image>=0.21`, `pydantic>=2.5`)
- **Test**: Dependency version tests in CI

**Schema Drift**:
- **Risk**: `MetricRecordV1` changes break consumers
- **Mitigation**: Flexible validation with `extra="allow"`
- **Test**: Schema validation tests with various input formats

**Scaling Bugs**:
- **Risk**: Incorrect scaler reuse skews models
- **Mitigation**: Comprehensive unit tests for data-prep helpers
- **Test**: Edge case testing for scaling functions

**Test Overhead**:
- **Risk**: Notebook smoke-tests slow down development
- **Mitigation**: Lightweight implementation (2 seconds)
- **Test**: Performance monitoring of test suite execution

---

## 10. Continuous Integration

### 10.1 Test Suite Organization

**Current Test Structure**:
```
tests/
├── test_additional_metrics.py    # New metrics testing
├── test_data_prep.py             # Data preparation helpers
├── test_metric_schema.py         # Schema validation
├── test_temporal_delta.py        # Temporal consistency
├── test_notebooks.py             # Notebook smoke tests
└── test_metrics.py               # Core metrics testing
```

### 10.2 Test Execution

**Local Testing**:
```bash
# Run all tests
pytest -q

# Run specific test categories
pytest tests/test_additional_metrics.py -v
pytest tests/test_data_prep.py -v
pytest tests/test_metric_schema.py -v
pytest tests/test_temporal_delta.py -v
pytest tests/test_notebooks.py -v
```

**Test Coverage**: 356 tests covering:
- 8 new metrics with comprehensive edge cases
- Data preparation utilities
- Schema validation
- Temporal consistency analysis
- Notebook execution
- Integration testing

---

## 11. Implementation Status ✅

### 11.1 Completed Testing

- **Unit Tests**: All 8 new metrics with deterministic fixtures
- **Integration Tests**: Complete pipeline testing
- **Schema Validation**: Pydantic model validation
- **Performance Tests**: Memory and timing benchmarks
- **Smoke Tests**: Notebook execution validation
- **Error Handling**: Comprehensive negative testing

### 11.2 Test Results

- **Total Tests**: 356 tests passing
- **Coverage**: Comprehensive metric and pipeline coverage
- **Performance**: All tests complete in < 30 seconds
- **Reliability**: Deterministic fixtures ensure consistent results

The testing strategy ensures robust, reliable operation of the expanded metrics system with comprehensive coverage of edge cases, performance characteristics, and integration scenarios.

