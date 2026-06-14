> **⚠️ ARCHIVED / OBSOLETE.** This guide documents the experimental pipeline-elimination harness (presets / `--sampling` / `select-pipelines` / `eliminate-pipelines`) **removed in commit `648db9a`**. Those commands no longer exist. The concepts return in the rebuilt leaderboard under new names — see [`docs/technical/compression-pipeline-leaderboard.md`](../../technical/compression-pipeline-leaderboard.md). Preserved for historical reference only.

---

# Elimination Pipeline Sampling Strategies

The elimination pipeline supports 6 different **sampling strategies** that control how comprehensively pipeline combinations are tested. This guide explains each strategy and when to use them.

## Strategy Overview

| Strategy | Coverage | Speed | Best For |
|----------|----------|-------|----------|
| `full` | 100% | Slowest | Final validation, research |
| `representative` | ~15% | Balanced | **Default** - good coverage/speed balance |
| `factorial` | ~8% | Fast | Statistical design of experiments |
| `progressive` | ~25% | Medium | Multi-stage refinement |
| `targeted` | ~12% | Fast | High-value expansion testing |
| `quick` | ~5% | Fastest | Development, debugging |

## Detailed Strategy Descriptions

### 1. `full` - Full Brute Force
**Coverage:** 100% of pipeline combinations  
**Speed:** Slowest (can take hours/days)  
**Min samples per tool:** All available

```bash
giflab eliminate-pipelines --sampling-strategy full
```

**When to use:**
- Final validation before production deployment
- Research studies requiring complete coverage
- When you have unlimited time and computational resources
- Validating that shorter sampling strategies didn't miss important patterns

**⚠️ Warning:** This can test thousands of pipeline combinations and take a very long time.

### 2. `representative` - Representative Sampling ⭐ **DEFAULT**
**Coverage:** ~15% of pipeline combinations  
**Speed:** Balanced (moderate duration)  
**Min samples per tool:** 5+ samples

```bash
giflab eliminate-pipelines --sampling-strategy representative
# or simply:
giflab eliminate-pipelines
```

**When to use:**
- **Most common choice** - provides good coverage without excessive runtime
- Regular pipeline optimization and maintenance
- When you want thorough testing but have time constraints
- General-purpose elimination runs

**How it works:** Intelligently selects representative samples from each tool category to ensure comprehensive coverage while maintaining reasonable execution time.

### 3. `factorial` - Statistical Design of Experiments
**Coverage:** ~8% of pipeline combinations  
**Speed:** Fast  
**Min samples per tool:** 3+ samples

```bash
giflab eliminate-pipelines --sampling-strategy factorial
```

**When to use:**
- When you want statistically rigorous results with minimal testing
- Research applications where statistical validity is important
- Time-constrained environments where you need reliable insights quickly
- Initial exploration of a new tool set

**How it works:** Uses statistical design of experiments principles to select combinations that maximize information gain while minimizing test count.

### 4. `progressive` - Multi-stage Elimination
**Coverage:** ~25% of pipeline combinations (varies across stages)  
**Speed:** Medium  
**Min samples per tool:** 4+ samples

```bash
giflab eliminate-pipelines --sampling-strategy progressive
```

**When to use:**
- When you want to progressively refine results through multiple stages
- Complex optimization scenarios where you want to eliminate obvious poor performers early
- When you have medium time availability and want thorough results
- Research scenarios requiring staged analysis

**How it works:** Runs multiple elimination stages, using results from earlier stages to inform later testing decisions.

### 5. `targeted` - Strategic Expansion
**Coverage:** ~12% of pipeline combinations  
**Speed:** Fast  
**Min samples per tool:** 4+ samples

```bash
giflab eliminate-pipelines --sampling-strategy targeted
```

**When to use:**
- Testing focused on high-value size and temporal variations
- When you want to emphasize specific aspects (file sizes, frame counts)
- Specialized testing scenarios with known optimization priorities
- Follow-up testing after initial broad analysis

**How it works:** 
- Uses a reduced synthetic GIF set (17 vs 25 GIFs)
- Focuses on key size variations and frame count extremes
- Strategic selection emphasizing practical optimization scenarios

### 6. `quick` - Fast Development Testing
**Coverage:** ~5% of pipeline combinations  
**Speed:** Fastest (minutes)  
**Min samples per tool:** 2+ samples

```bash
giflab eliminate-pipelines --sampling-strategy quick
```

**When to use:**
- **Development and debugging** - rapid iteration during code changes
- Smoke testing to ensure basic functionality works
- Initial exploration of new tools or configurations
- CI/CD pipelines where you need fast feedback
- Quick validation before running longer tests

**⚠️ Limitation:** May miss edge cases and subtle performance differences due to minimal coverage.

## Choosing the Right Strategy

### Decision Tree

```
Are you developing/debugging code?
├─ YES → Use `quick`
└─ NO
   │
   Do you have unlimited time and need complete coverage?
   ├─ YES → Use `full`
   └─ NO
      │
      Do you need statistically rigorous results quickly?
      ├─ YES → Use `factorial`
      └─ NO
         │
         Do you want multi-stage refinement?
         ├─ YES → Use `progressive`
         └─ NO
            │
            Do you want to focus on size/temporal variations?
            ├─ YES → Use `targeted`
            └─ NO → Use `representative` (default)
```

### By Time Available

- **< 30 minutes:** `quick`
- **30 minutes - 2 hours:** `factorial` or `targeted`
- **2-6 hours:** `representative` (recommended)
- **6-12 hours:** `progressive`
- **12+ hours:** `full`

### By Use Case

- **Production optimization:** `representative` or `progressive`
- **Research/academia:** `factorial` or `full`
- **Development/debugging:** `quick`
- **Specialized testing:** `targeted`
- **Complete validation:** `full`

## Advanced Usage

### Combining with Other Options

```bash
# Quick test with limited pipelines
giflab eliminate-pipelines --sampling-strategy quick --max-pipelines 50

# Representative test focused on dithering only
giflab eliminate-pipelines --sampling-strategy representative --test-dithering-only

# Factorial design with GPU acceleration
giflab eliminate-pipelines --sampling-strategy factorial --use-gpu

# Progressive elimination with fresh cache
giflab eliminate-pipelines --sampling-strategy progressive --clear-cache
```

### Time Estimation

Before running any strategy, you can estimate execution time:

```bash
giflab eliminate-pipelines --sampling-strategy representative --estimate-time
```

This shows:
- Number of pipeline combinations to test
- Estimated execution time
- Synthetic GIFs that will be used
- No actual testing is performed

### Monitoring Progress

All strategies support progress monitoring and resumption:

```bash
# Resume interrupted run
giflab eliminate-pipelines --sampling-strategy representative --resume

# Monitor with monitoring script (see scripts/experimental/)
python scripts/experimental/simple_monitor.py &
giflab eliminate-pipelines --sampling-strategy representative
```

## Results and Caching

All sampling strategies benefit from the smart caching system:

- **Cache hits:** Previously tested pipeline combinations are retrieved instantly
- **Cache misses:** New combinations are tested and cached for future runs
- **Cache invalidation:** Automatic when code changes (git commit hash)

See [Elimination Results Tracking](elimination-results-tracking.md) for details on working with results from different sampling strategies.

## Performance Comparison

Based on typical usage with ~25 synthetic GIFs and ~1000+ available pipeline combinations:

| Strategy | Pipelines Tested | Typical Duration | Quality of Results |
|----------|------------------|------------------|-------------------|
| `full` | 1000+ | 6-24 hours | Excellent (complete) |
| `representative` | ~150 | 1-4 hours | Very Good (balanced) |
| `progressive` | ~250 | 2-6 hours | Very Good (staged) |
| `targeted` | ~120 | 45min-2hours | Good (focused) |
| `factorial` | ~80 | 30min-90min | Good (statistical) |
| `quick` | ~50 | 10-30 minutes | Fair (basic coverage) |

*Duration varies significantly based on:*
- Number of available pipeline combinations
- System performance and GPU availability
- Cache hit ratio from previous runs
- Complexity of synthetic GIF test set

## FAQ

**Q: Which strategy should I use for my first elimination run?**  
A: Start with `representative` - it provides the best balance of thoroughness and speed for most use cases.

**Q: Can I switch strategies between runs?**  
A: Yes! The caching system works across all strategies, so you can run `quick` first, then `representative` to expand coverage incrementally.

**Q: Will different strategies give different elimination results?**  
A: Generally similar pipelines will be eliminated, but more thorough strategies may catch edge cases that quicker strategies miss.

**Q: How do I know if a quick strategy missed something important?**  
A: Compare results between strategies, or run `representative` after `quick` to see if additional pipelines get eliminated.

**Q: Can I create custom sampling strategies?**  
A: Currently no, but you can use `--max-pipelines` with any strategy to create custom coverage levels. 