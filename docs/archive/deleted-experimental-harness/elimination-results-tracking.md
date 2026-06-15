> **⚠️ ARCHIVED / OBSOLETE.** This guide documents the experimental pipeline-elimination harness (presets / `--sampling` / `select-pipelines` / `eliminate-pipelines`) **removed in commit `648db9a`**. Those commands no longer exist. The concepts return in the rebuilt leaderboard under new names — see [`docs/technical/compression-pipeline-leaderboard.md`](../../technical/compression-pipeline-leaderboard.md). Preserved for historical reference only.

---

# Elimination Results Tracking

The pipeline elimination system now saves timestamped results to preserve the history of all your elimination runs, making it easy to track changes and compare different testing scenarios.

## Directory Structure

When you run `giflab eliminate-pipelines`, results are saved in a timestamped structure:

```
results/runs/latest/
├── run_20241204_143052/          # Timestamped run directory
│   ├── elimination_test_results.csv              # Standard CSV name
│   ├── elimination_test_results_20241204_143052.csv  # Timestamped CSV
│   ├── elimination_summary.json                  # Run summary
│   ├── run_metadata.json                        # Run metadata
│   ├── failed_pipelines.json                    # Failed pipeline details
│   └── failure_analysis_report.txt              # Failure analysis
├── run_20241204_151230/          # Another run directory
│   └── ... (same structure)
├── latest -> run_20241204_151230  # Symlink to latest run
├── pipeline_results_cache.db                    # NEW: Smart cache database
└── elimination_history_master.csv               # All runs combined
```

## Key Features

### 1. Timestamped Runs
- Each run gets its own directory: `run_YYYYMMDD_HHMMSS`
- No more overwriting previous results
- Easy to compare different elimination runs

### 2. Latest Results Symlink
- `latest` symlink always points to the most recent run
- Scripts can reference `results/runs/latest/latest/` for current results
- Works on Linux/macOS (Windows may not support symlinks)

### 3. Master History File
- `elimination_history_master.csv` contains ALL runs in one file
- Each row includes `run_timestamp` and `run_id` columns
- Perfect for analyzing trends across multiple runs

### 4. Run Metadata
- `run_metadata.json` tracks:
  - Start time
  - GPU usage
  - Python version
  - Git commit (if available)
  - Directory paths

### 5. Smart Caching (NEW)
- `pipeline_results_cache.db` stores previously tested pipeline results
- SQLite database for fast lookups
- Automatic cache invalidation when code changes (git commit)
- Massive speed improvements for repeated testing
- Cache statistics and time savings reported

## Smart Caching System

### How It Works
The caching system stores results using a composite key:
- Pipeline ID + GIF name + test parameters + git commit hash
- SQLite database provides millisecond lookups
- Automatic invalidation prevents stale results

### Cache Management

#### 🗑️ Clear All Cache Data
To completely reset all cached pipeline results:
```bash
poetry run python -m giflab eliminate-pipelines --clear-cache --estimate-time
```
**What this clears:**
- All successful pipeline test results
- All failed pipeline test results  
- Stored in: `results/runs/latest/pipeline_results_cache.db`

#### Other Cache Commands
```bash
# Normal run with cache (default)
poetry run python -m giflab eliminate-pipelines --sampling-strategy representative

# Force fresh results (ignore cache, but keep cached data)
poetry run python -m giflab eliminate-pipelines --no-cache

# Check cache statistics in output logs (automatically shown)
```

### Cache Benefits
- **Speed**: 2-5x faster for subsequent runs
- **Consistency**: Same test conditions yield identical results  
- **Safety**: Automatic invalidation on code changes
- **Transparency**: Cache hit/miss statistics reported

### When Cache Helps Most
- **Iterative testing**: Testing different sampling strategies
- **Parameter tuning**: Adjusting thresholds with same pipelines
- **Debugging**: Re-running failed test scenarios
- **Research**: Comparing different elimination approaches

## Working with Historical Data

### Access Latest Results
```bash
# Always use the latest run
cd results/runs/latest/latest
cat elimination_summary.json
```

### Compare Multiple Runs
```python
import pandas as pd

# Load master history (all runs)
df = pd.read_csv('results/runs/latest/elimination_history_master.csv')

# Group by run to compare
runs = df.groupby('run_id')
print(f"Total runs: {len(runs)}")

# Compare quality metrics across runs
quality_comparison = df.groupby('run_id')['composite_quality'].mean()
print(quality_comparison)

# Find which pipelines are consistently eliminated
eliminated_pipelines = df.groupby('pipeline_id')['success'].count()
print("Most frequently tested pipelines:")
print(eliminated_pipelines.sort_values(ascending=False).head(10))
```

### Analyze Trends
```python
# Track elimination rates over time
elimination_stats = df.groupby('run_id').agg({
    'success': ['count', 'sum'],
    'composite_quality': 'mean',
    'run_timestamp': 'first'
}).reset_index()

elimination_stats.columns = ['run_id', 'total_tests', 'successful_tests', 'avg_quality', 'timestamp']
elimination_stats['failure_rate'] = (elimination_stats['total_tests'] - elimination_stats['successful_tests']) / elimination_stats['total_tests']

print("Elimination run trends:")
print(elimination_stats[['run_id', 'timestamp', 'failure_rate', 'avg_quality']])
```

### Browse Specific Runs
```bash
# List all runs
ls results/runs/latest/run_*

# Compare two specific runs
diff results/runs/latest/run_20241204_143052/elimination_summary.json \
     results/runs/latest/run_20241204_151230/elimination_summary.json

# Find runs with specific characteristics
grep -l "many_colors" results/runs/latest/run_*/elimination_summary.json
```

## Cache Management Examples

### Analyze Cache Performance
```python
import sqlite3
import pandas as pd

# Connect to cache database
conn = sqlite3.connect('results/runs/latest/pipeline_results_cache.db')

# Check cache statistics
cache_stats = pd.read_sql_query("""
    SELECT git_commit, COUNT(*) as entries, 
           MIN(created_at) as first_cached,
           MAX(created_at) as last_cached
    FROM pipeline_results 
    GROUP BY git_commit
    ORDER BY last_cached DESC
""", conn)

print("Cache by git commit:")
print(cache_stats)

# Find most frequently tested pipelines
frequent_pipelines = pd.read_sql_query("""
    SELECT pipeline_id, COUNT(*) as test_count
    FROM pipeline_results 
    GROUP BY pipeline_id
    ORDER BY test_count DESC
    LIMIT 10
""", conn)

print("\nMost frequently cached pipelines:")
print(frequent_pipelines)
```

### Cache Cleanup
```bash
# Manual cache cleanup (if needed)
# Note: Cache auto-invalidates on code changes, but you can manually clean old entries

# Check cache size
ls -lh results/runs/latest/pipeline_results_cache.db

# Clear cache completely
giflab eliminate-pipelines --clear-cache --estimate-time

# Run without cache for debugging
giflab eliminate-pipelines --no-cache --sampling-strategy quick
```

## Migration from Old Format

If you have existing `elimination_results.csv` files from before this update:

1. The old files are not automatically migrated
2. New runs will create the timestamped structure
3. Old files can be manually imported into the new system if needed

## Tips for Long-term Use

### Regular Cleanup
- Old runs accumulate over time
- Consider archiving runs older than 30 days:
```bash
find elimination_results -name "run_*" -type d -mtime +30 -exec rm -rf {} \;
```

### Backup Important Runs
- Tag important runs by renaming directories:
```bash
mv results/runs/latest/run_20241204_143052 results/runs/latest/run_20241204_143052_baseline
```

### Cache Optimization
- Cache database grows over time but provides significant speed benefits
- Cache auto-invalidates on code changes (git commit)
- Consider periodic cache clearing for major version updates:
```bash
giflab eliminate-pipelines --clear-cache
```

### CI/CD Integration
- Use the master history file for automated quality tracking
- Monitor failure rates across runs
- Alert on significant performance degradations
- Cache makes CI runs much faster for repeated testing

## Troubleshooting

### Symlink Issues (Windows)
If the `latest` symlink doesn't work:
- Use the most recent `run_*` directory directly
- Or use PowerShell as Administrator to enable symlink support

### Cache Issues
If cache seems corrupted or giving unexpected results:
- Use `--clear-cache` to reset and start fresh
- Check git commit status if cache seems stale
- Use `--no-cache` for debugging to bypass cache entirely

### Disk Space
Large elimination runs can consume significant disk space:
- Monitor `results/runs/latest/` directory size
- Consider compressing old runs: `tar -czf run_archive.tar.gz run_20241204_*`
- Cache database is typically small (< 50MB) but monitor growth

### Master History File Corruption
If the master CSV gets corrupted:
- Delete `elimination_history_master.csv`
- Re-run elimination to start fresh
- Or manually reconstruct from individual run CSVs

## Debugging Pipeline Failures

### New: `debug-failures` Command

The system now includes a powerful debugging command for analyzing pipeline failures:

```bash
# Show summary of all failures
giflab debug-failures --summary

# Show detailed failures
giflab debug-failures

# Filter by error type
giflab debug-failures --error-type ffmpeg

# Show only recent failures
giflab debug-failures --recent-hours 24

# Debug specific pipeline
giflab debug-failures --pipeline "imagemagick_floyd_16colors"

# Combine filters
giflab debug-failures --error-type timeout --recent-hours 12
```

### Failure Analysis Examples

```bash
# Quick overview of what's failing
giflab debug-failures --summary

# Focus on FFmpeg issues
giflab debug-failures --error-type ffmpeg --recent-hours 48

# Find the most problematic pipelines
giflab debug-failures | grep "Most problematic"
```

### Structured Failure Storage

Failures are now stored in the SQLite database with:
- **Error categorization**: ffmpeg, imagemagick, gifski, timeout, etc.
- **Pipeline context**: Steps, tools used, test parameters
- **Timing information**: When failures occurred
- **Full tracebacks**: For detailed debugging

### Manual SQL Queries

For advanced analysis, you can query the database directly:

```python
import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('results/runs/latest/pipeline_results_cache.db')

# Find most common failure types
failures_by_type = pd.read_sql_query("""
    SELECT error_type, COUNT(*) as count, 
           COUNT(*) * 100.0 / (SELECT COUNT(*) FROM pipeline_failures) as percentage
    FROM pipeline_failures 
    GROUP BY error_type 
    ORDER BY count DESC
""", conn)
print(failures_by_type)

# Find failing pipelines by GIF type
failing_gifs = pd.read_sql_query("""
    SELECT gif_name, error_type, COUNT(*) as failures
    FROM pipeline_failures 
    GROUP BY gif_name, error_type 
    ORDER BY failures DESC
    LIMIT 10
""", conn)
print(failing_gifs)

# Track failure trends over time
failure_trends = pd.read_sql_query("""
    SELECT DATE(created_at) as date, error_type, COUNT(*) as failures
    FROM pipeline_failures 
    WHERE created_at >= date('now', '-7 days')
    GROUP BY date, error_type 
    ORDER BY date DESC, failures DESC
""", conn)
print(failure_trends)
```