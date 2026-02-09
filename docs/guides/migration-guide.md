# GifLab Phase 3 Optimizations Migration Guide

## Overview

This guide provides step-by-step instructions for migrating existing GifLab deployments to use the new Phase 3 performance optimizations. The optimizations reduce processing overhead from 4.73x to <1.5x while maintaining metric accuracy within ±0.1%.

## Migration Impact Summary

### What's Changed
- **Performance:** 2-2.5x faster processing for most GIFs
- **Memory:** Better managed but requires ~500MB for model caching
- **API:** Minimal changes, mostly backward compatible
- **Behavior:** Conditional metric skipping for high-quality GIFs (configurable)

### What's Not Changed
- Metric calculation algorithms (accuracy maintained)
- Input/output formats
- Command-line interface
- Default behavior (optimizations can be disabled)

## Pre-Migration Checklist

### System Requirements
- [ ] Python 3.8+ (unchanged)
- [ ] Available memory: 1GB minimum (previously 512MB)
- [ ] CPU cores: 4+ recommended for parallel processing
- [ ] Disk space: 100MB for cached models

### Dependency Updates
```bash
# Update dependencies
poetry update

# Verify installation
poetry run python -c "import giflab; print(giflab.__version__)"
```

### Backup Current Configuration
```bash
# Save current environment
env | grep GIFLAB > giflab_env_backup.txt

# Backup any custom configurations
cp -r /path/to/config /path/to/config.backup
```

## Migration Steps

### Step 1: Test in Staging Environment

#### 1.1 Deploy to Staging
```bash
# Deploy new version to staging
git checkout main
git pull origin main
poetry install
```

#### 1.2 Run Validation Tests
```bash
# Run comprehensive test suite
poetry run pytest tests/ -v

# Run performance benchmarks
poetry run python tests/scripts/benchmarks/benchmark_comprehensive.py

# Verify memory stability
poetry run pytest tests/nightly/test_memory_stability.py
```

#### 1.3 Compare Results
```bash
# Process test GIFs with old version
GIFLAB_DISABLE_ALL_OPTIMIZATIONS=true poetry run python -m giflab run test_data/

# Process with new optimizations
poetry run python -m giflab run test_data/

# Compare outputs (should be within ±0.1%)
```

### Step 2: Gradual Production Rollout

#### 2.1 Phase A: Canary Deployment (10% Traffic)
```bash
# Enable optimizations for canary
export GIFLAB_USE_MODEL_CACHE=true
export GIFLAB_ENABLE_PARALLEL_METRICS=true
export GIFLAB_ENABLE_CONDITIONAL_METRICS=true

# Conservative settings
export GIFLAB_QUALITY_HIGH_THRESHOLD=0.95  # Very conservative
export GIFLAB_MAX_PARALLEL_WORKERS=2       # Limited parallelism
```

**Monitor for 24-48 hours:**
- Error rates
- Processing times
- Memory usage
- Result accuracy

#### 2.2 Phase B: Expanded Rollout (50% Traffic)
```bash
# Adjust based on canary results
export GIFLAB_QUALITY_HIGH_THRESHOLD=0.9   # Standard threshold
export GIFLAB_MAX_PARALLEL_WORKERS=4       # Increased parallelism
```

**Monitor for 24-48 hours:**
- Performance improvements
- Resource utilization
- User feedback

#### 2.3 Phase C: Full Deployment (100% Traffic)
```bash
# Full optimization settings
export GIFLAB_USE_MODEL_CACHE=true
export GIFLAB_ENABLE_PARALLEL_METRICS=true
export GIFLAB_ENABLE_CONDITIONAL_METRICS=true
export GIFLAB_MAX_PARALLEL_WORKERS=$(nproc)  # Use all cores
```

### Step 3: Configuration Migration

#### For Docker Deployments
```dockerfile
# Dockerfile updates
ENV GIFLAB_USE_MODEL_CACHE=true
ENV GIFLAB_ENABLE_PARALLEL_METRICS=true
ENV GIFLAB_ENABLE_CONDITIONAL_METRICS=true
ENV GIFLAB_MAX_PARALLEL_WORKERS=4
```

#### For Kubernetes Deployments
```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: giflab-config
data:
  GIFLAB_USE_MODEL_CACHE: "true"
  GIFLAB_ENABLE_PARALLEL_METRICS: "true"
  GIFLAB_ENABLE_CONDITIONAL_METRICS: "true"
  GIFLAB_MAX_PARALLEL_WORKERS: "4"
```

#### For SystemD Services
```ini
# /etc/systemd/system/giflab.service
[Service]
Environment="GIFLAB_USE_MODEL_CACHE=true"
Environment="GIFLAB_ENABLE_PARALLEL_METRICS=true"
Environment="GIFLAB_ENABLE_CONDITIONAL_METRICS=true"
Environment="GIFLAB_MAX_PARALLEL_WORKERS=4"
```

## API Changes and Compatibility

### New Functions
```python
# New cleanup function for proper resource management
from giflab.metrics import cleanup_all_validators

# Call at application shutdown
cleanup_all_validators()
```

### Backward Compatibility
All existing code continues to work without modifications. Optimizations are applied automatically when enabled via environment variables.

```python
# Existing code - no changes needed
from giflab.metrics import calculate_comprehensive_metrics

results = calculate_comprehensive_metrics(
    original_frames, 
    compressed_frames,
    include_advanced=True
)
```

### Optional: Explicit Optimization Control
```python
# New: Programmatic control over optimizations
import os

# Disable conditional metrics for specific operations
os.environ['GIFLAB_ENABLE_CONDITIONAL_METRICS'] = 'false'
critical_results = calculate_comprehensive_metrics(...)

# Re-enable for normal operations
os.environ['GIFLAB_ENABLE_CONDITIONAL_METRICS'] = 'true'
normal_results = calculate_comprehensive_metrics(...)
```

## Rollback Procedures

### Immediate Rollback
If critical issues occur, disable optimizations immediately:

```bash
# Disable all optimizations
export GIFLAB_DISABLE_ALL_OPTIMIZATIONS=true

# Or selectively disable
export GIFLAB_USE_MODEL_CACHE=false
export GIFLAB_ENABLE_PARALLEL_METRICS=false
export GIFLAB_ENABLE_CONDITIONAL_METRICS=false
```

### Full Version Rollback
```bash
# Revert to previous version
git checkout v1.2.3  # Previous stable version
poetry install

# Restore environment
source giflab_env_backup.txt

# Restart services
systemctl restart giflab
```

## Monitoring and Validation

### Key Metrics to Monitor

#### Performance Metrics
```bash
# Monitor processing times
grep "Metrics calculation time" /var/log/giflab.log | \
  awk '{print $NF}' | \
  datamash mean 1 perc:50 1 perc:95 1 perc:99 1
```

#### Memory Usage
```bash
# Track memory patterns
ps aux | grep giflab | awk '{print $6/1024 " MB"}'

# Or with systemd
systemctl status giflab | grep Memory
```

#### Cache Effectiveness
```python
# Check cache hit rates
from giflab.model_cache import LPIPSModelCache

cache = LPIPSModelCache()
info = cache.get_model_cache_info()
print(f"Cache hits: {info['cache_hits']}")
print(f"Cache misses: {info['cache_misses']}")
print(f"Hit rate: {info['hit_rate']:.2%}")
```

### Validation Scripts

#### Accuracy Validation
```python
#!/usr/bin/env python
"""validate_accuracy.py - Ensure optimization accuracy"""

import os
from giflab.metrics import calculate_comprehensive_metrics
import numpy as np

def validate_accuracy(test_gif_path, tolerance=0.001):
    # Load test data
    original, compressed = load_test_gif(test_gif_path)
    
    # Calculate with optimizations disabled
    os.environ['GIFLAB_DISABLE_ALL_OPTIMIZATIONS'] = 'true'
    baseline = calculate_comprehensive_metrics(original, compressed)
    
    # Calculate with optimizations enabled
    os.environ.pop('GIFLAB_DISABLE_ALL_OPTIMIZATIONS')
    optimized = calculate_comprehensive_metrics(original, compressed)
    
    # Compare results
    for metric, baseline_value in baseline.items():
        optimized_value = optimized[metric]
        if isinstance(baseline_value, (int, float)):
            diff = abs(baseline_value - optimized_value)
            if diff > tolerance:
                print(f"WARNING: {metric} differs by {diff}")
                return False
    
    print("✓ All metrics within tolerance")
    return True
```

#### Performance Validation
```bash
#!/bin/bash
# validate_performance.sh - Ensure performance improvements

# Baseline test
export GIFLAB_DISABLE_ALL_OPTIMIZATIONS=true
BASELINE_TIME=$(time -p poetry run python -m giflab run test.gif 2>&1 | grep real | awk '{print $2}')

# Optimized test  
unset GIFLAB_DISABLE_ALL_OPTIMIZATIONS
OPTIMIZED_TIME=$(time -p poetry run python -m giflab run test.gif 2>&1 | grep real | awk '{print $2}')

# Calculate speedup
SPEEDUP=$(echo "scale=2; $BASELINE_TIME / $OPTIMIZED_TIME" | bc)
echo "Speedup: ${SPEEDUP}x"

if (( $(echo "$SPEEDUP < 1.5" | bc -l) )); then
    echo "WARNING: Speedup less than expected"
    exit 1
fi
```

## Common Migration Issues

### Issue 1: Increased Memory Usage
**Symptom:** Memory usage higher than before

**Solution:**
```bash
# Reduce memory footprint
export GIFLAB_MAX_PARALLEL_WORKERS=2  # Fewer workers
export GIFLAB_CACHE_SIZE_MB=256       # Smaller cache
```

### Issue 2: Different Results
**Symptom:** Metrics differ from previous version

**Solution:**
```bash
# Ensure full metric calculation
export GIFLAB_ENABLE_CONDITIONAL_METRICS=false
export GIFLAB_FORCE_ALL_METRICS=true
```

### Issue 3: Import Errors
**Symptom:** ModuleNotFoundError after update

**Solution:**
```bash
# Clean install
poetry cache clear . --all
poetry install --no-cache
```

### Issue 4: Slower Performance
**Symptom:** No performance improvement or degradation

**Diagnosis:**
```bash
# Check if optimizations are active
poetry run python -c "
from giflab.metrics import get_optimization_status
status = get_optimization_status()
for key, value in status.items():
    print(f'{key}: {value}')
"
```

## Post-Migration Tasks

### 1. Update Documentation
- [ ] Update internal wikis
- [ ] Update API documentation
- [ ] Update runbooks

### 2. Update Monitoring
- [ ] Add new metrics dashboards
- [ ] Update alert thresholds
- [ ] Configure performance tracking

### 3. Team Training
- [ ] Conduct optimization overview session
- [ ] Review new configuration options
- [ ] Share best practices

### 4. Performance Baseline
- [ ] Establish new performance baselines
- [ ] Document expected processing times
- [ ] Set SLA targets

## Support and Troubleshooting

### Getting Help
- GitHub Issues: [Project Repository]/issues
- Documentation: See [Performance Tuning Guide](./performance-tuning-guide.md)
- Logs: Check `/var/log/giflab/` for detailed diagnostics

### Debug Mode
Enable detailed logging for troubleshooting:
```bash
export GIFLAB_DEBUG=true
export GIFLAB_LOG_LEVEL=DEBUG
export GIFLAB_ENABLE_PROFILING=true
```

### Health Check Endpoint
```python
# health_check.py
from giflab.metrics import get_optimization_status
from giflab.model_cache import LPIPSModelCache

def health_check():
    status = {
        "optimizations": get_optimization_status(),
        "cache": LPIPSModelCache().get_model_cache_info(),
        "version": giflab.__version__
    }
    return status
```

## Migration Timeline Example

| Phase | Duration | Traffic | Configuration | Success Criteria |
|-------|----------|---------|---------------|------------------|
| Testing | 2 days | 0% | Staging only | All tests pass |
| Canary | 2 days | 10% | Conservative | Error rate <0.1% |
| Partial | 3 days | 50% | Standard | Speedup >1.5x |
| Full | Ongoing | 100% | Optimized | SLA met |

## Appendix: Configuration Reference

See [Configuration Reference](../reference/configuration-reference.md) for complete environment variable documentation.

---

*Last updated: 2025-09-09*
*Version: 1.0.0*
*For GifLab v2.0.0+ with Phase 3 Optimizations*