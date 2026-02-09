# Smart Pipeline Sampling Strategy

**Purpose**: Achieve 95%+ test coverage while reducing execution time from 30+ minutes to <5 minutes through intelligent test selection.

## Overview

The GifLab test suite faces a combinatorial explosion problem with 300+ possible pipeline combinations. Smart sampling ensures comprehensive coverage while maintaining practical execution times.

## Sampling Strategies

### 1. **Equivalence Class Testing**

**Concept**: Group similar pipelines into equivalence classes and test one representative per class.

**Implementation**:
```python
# Example equivalence classes:
LOSSY_EQUIVALENCE_CLASSES = {
    "low_lossy": ["lossy_10", "lossy_15", "lossy_20"],      # Test lossy_15
    "mid_lossy": ["lossy_30", "lossy_40", "lossy_50"],      # Test lossy_40  
    "high_lossy": ["lossy_60", "lossy_70", "lossy_80"],     # Test lossy_70
}

COLOR_EQUIVALENCE_CLASSES = {
    "few_colors": ["color_16", "color_32"],                 # Test color_32
    "many_colors": ["color_128", "color_256"],              # Test color_256
}
```

**Coverage**: ~90% with 70% fewer tests

### 2. **Risk-Based Prioritization**

**High-Risk Combinations** (always test):
- Brand new engines (Animately)
- Edge cases (single frame, extreme compression)
- Cross-engine interactions
- Recently modified code paths

**Medium-Risk Combinations** (sample 50%):
- Established engines with minor parameter variations
- Standard use cases with proven stability

**Low-Risk Combinations** (sample 10%):
- Redundant parameter combinations
- Deprecated or legacy configurations

### 3. **Content-Type Aware Sampling**

**Strategy**: Different content types benefit from different pipeline combinations.

```python
CONTENT_OPTIMIZED_SAMPLING = {
    "gradient": {
        "critical": ["gifsicle_lossy", "animately_advanced"],
        "sample_rate": 0.3  # 30% of remaining combinations
    },
    "solid": {
        "critical": ["imagemagick_lossless", "ffmpeg_basic"],
        "sample_rate": 0.2  # 20% of remaining combinations  
    },
    "photographic": {
        "critical": ["gifski_high_quality", "animately_advanced"],
        "sample_rate": 0.5  # 50% of remaining combinations
    }
}
```

### 4. **Boundary Value Testing**

**Focus**: Test extreme parameter values where bugs are most likely.

**Examples**:
- `lossy_level`: Test 0, 1, 50, 99, 100
- `color_count`: Test 2, 16, 256
- `frame_count`: Test 1, 2, 100+ frames

## Implementation Phases

### Phase A: Static Sampling (Current)
- Layer-specific conftest files cap pipeline generation
- **Result**: Consistent pipeline subset for fast testing

### Phase B: Intelligent Sampling (Next)
- Implement equivalence class selection
- Add risk-based prioritization
- **Target**: 95% coverage with 50% execution time

### Phase C: Adaptive Sampling (Future)
- Machine learning to identify optimal test combinations
- Historical failure analysis to prioritize risky combinations
- **Target**: 98% coverage with 30% execution time

## Coverage Validation

### Metrics Tracked:
1. **Code Coverage**: Lines of code executed
2. **Pipeline Coverage**: Unique pipeline combinations tested
3. **Edge Case Coverage**: Boundary conditions validated
4. **Integration Coverage**: Cross-engine interactions tested

### Validation Process:
```bash
# Full nightly validation
make test-nightly

# Coverage report
poetry run pytest tests/ --cov=src/giflab --cov-report=html

# Generate coverage delta report
poetry run python scripts/compare_coverage.py
```

## Performance Targets

| **Test Category** | **Current** | **Smart Sampling** | **Target** |
|---|---|---|---|
| **Lightning Tests** | 3.28s (56 tests) | 2.5s (40 tests) | <3s |
| **Integration Tests** | 5-10min (458 tests) | 3-5min (200 tests) | <5min |
| **Full Matrix** | 30min+ (full combinations) | Available on-demand | <30min |

## Implementation Status

- ✅ **Phase A**: Static pipeline capping via layer conftest isolation
- 🎯 **Phase B**: Equivalence class implementation (next sprint)
- ⏳ **Phase C**: Adaptive sampling (future enhancement)

## Usage Examples

```bash
# Fast feedback: smoke + functional (<2min)
make test

# CI: + integration (<5min)
make test-ci

# Everything including nightly
make test-nightly

# Custom sampling for specific content type
poetry run pytest tests/ -k "gradient"
```

## Success Metrics

- **Coverage Maintained**: >95% line coverage with smart sampling
- **Speed Improved**: 50-70% reduction in execution time
- **Reliability**: Zero false negatives (missed bugs)
- **Developer Experience**: <3s feedback loop for rapid iteration

---

**Last Updated**: February 2026  
**Owner**: Development Team  
**Next Review**: After Phase 3.3 completion