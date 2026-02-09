# GifLab Scripts

This directory contains utility scripts for GifLab development and monitoring.

## Performance Monitoring

### `monitor_test_performance.py`

Comprehensive test performance monitor with regression detection and alerting.

#### Usage

```bash
# Monitor fast tests with default settings
python scripts/monitor_test_performance.py fast

# Monitor integration tests with custom config
python scripts/monitor_test_performance.py integration --config scripts/test-performance-config.json

# Monitor full test suite
python scripts/monitor_test_performance.py full
```

#### Features

- **Performance Timing**: Accurate test execution timing
- **Threshold Validation**: Configurable performance thresholds
- **History Tracking**: JSON-based performance history storage
- **Trend Analysis**: Detect gradual performance degradation
- **Alert Integration**: Slack webhook support for immediate notifications
- **Regression Reports**: Detailed analysis and recommendations

#### Configuration

Create a custom configuration file (see `test-performance-config.json` for example):

```json
{
  "thresholds": {
    "fast": 10,
    "integration": 300,
    "full": 1800
  },
  "alert_on_regression": true,
  "save_history": true,
  "slack_webhook_url": "https://hooks.slack.com/...",
  "regression_tolerance": 1.5
}
```

#### Performance Thresholds

| Test Tier | Default Threshold | Purpose |
|-----------|------------------|---------|
| **fast** | 10s | Development iteration |
| **integration** | 5min (300s) | Pre-commit validation |
| **full** | 30min (1800s) | Release validation |

#### CI/CD Integration

The script is automatically integrated into GitHub Actions via `.github/workflows/test-performance-monitoring.yml`:

- Runs on every PR and push to main
- Daily scheduled performance checks
- Automatic artifact collection and reporting
- Build failure on significant regressions

#### Performance History

Performance data is stored in `test-performance-history.json`:

```json
[
  {
    "timestamp": 1704067200,
    "test_tier": "fast",
    "duration": 6.5,
    "success": true,
    "threshold": 10,
    "threshold_met": true
  }
]
```

#### Troubleshooting Performance Issues

If performance alerts trigger:

1. **Check Recent History**:
   ```bash
   cat test-performance-history.json | jq '.[-5:]'
   ```

2. **Profile Slow Tests**:
   ```bash
   poetry run pytest tests/smoke/ tests/functional/ --durations=0 | head -20
   ```

3. **Check Mock Patterns**:
   - Review `tests/conftest.py` for proper mock application
   - Ensure external engine mocking is working correctly

## Analysis Scripts

### `analysis/`

Contains comprehensive analysis scripts for processing experimental results:

- **Enhanced Metrics Analysis**: Complete analysis suite for 11-metric enhanced quality system
- **Frame Reduction Studies**: Deep dive analysis of frame reduction algorithm behavior  
- **Efficiency Insights**: Focused analysis of efficiency scoring and performance
- **Dataset Breakdowns**: Comprehensive dataset analysis with enhanced metrics

See [analysis/README.md](analysis/README.md) for detailed usage instructions.

## Experimental Scripts

### `experimental/`

Contains experimental monitoring and development tools:

- `simple_monitor.py`: Basic pipeline elimination progress monitoring
- `monitor_config.py`: Configuration management for monitoring tools

See individual script documentation for detailed usage instructions.

## Development Guidelines

### Adding New Scripts

1. **Follow naming convention**: Use underscores, descriptive names
2. **Include docstrings**: Comprehensive module and function documentation
3. **Add CLI interface**: Use `argparse` for command-line interaction
4. **Error handling**: Graceful error handling and user feedback
5. **Update this README**: Document new scripts and their purpose

### Performance Considerations

- Keep scripts lightweight and fast-loading
- Minimize external dependencies
- Use appropriate logging levels
- Consider parallel execution where beneficial

### Testing Scripts

Scripts should include basic self-tests:

```python
if __name__ == "__main__":
    # Include basic functionality tests
    pass
```

---

**Last Updated**: January 2025
**Maintainer**: GifLab Development Team