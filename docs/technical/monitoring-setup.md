# GifLab Monitoring and Operations Setup Guide

## Overview

This guide provides comprehensive instructions for setting up monitoring, alerting, and operational procedures for GifLab with Phase 3 performance optimizations. Proper monitoring is essential for maintaining performance gains and identifying issues early.

## Key Performance Indicators (KPIs)

### Primary Metrics

#### 1. Processing Time
- **Metric:** Time to process single GIF (seconds)
- **Target:** <2x baseline for optimized paths
- **Alert Threshold:** >5x baseline
- **Collection Method:** Application logs, timing instrumentation

#### 2. Memory Usage
- **Metric:** Peak memory consumption (MB)
- **Target:** <750MB for standard operations
- **Alert Threshold:** >1GB sustained
- **Collection Method:** Process monitoring, resource metrics

#### 3. Cache Hit Rate
- **Metric:** Model cache hit percentage
- **Target:** >80% for batch processing
- **Alert Threshold:** <50%
- **Collection Method:** Cache statistics logging

#### 4. Optimization Effectiveness
- **Metric:** Percentage of GIFs using optimized path
- **Target:** >60% for typical workloads
- **Alert Threshold:** <30%
- **Collection Method:** Conditional metrics logging

### Secondary Metrics

- **Parallel Processing Efficiency:** CPU utilization across workers
- **Queue Depth:** Pending GIFs in processing queue
- **Error Rate:** Failed processing attempts
- **Metric Accuracy:** Deviation from baseline calculations

## Monitoring Implementation

### 1. Application Instrumentation

#### Basic Metrics Collection
```python
# metrics_collector.py
import time
import psutil
import json
from datetime import datetime
from giflab.metrics import calculate_comprehensive_metrics
from giflab.model_cache import LPIPSModelCache

class MetricsCollector:
    def __init__(self, output_file="/var/log/giflab/metrics.json"):
        self.output_file = output_file
        self.metrics = []
    
    def collect_processing_metrics(self, gif_path, original_frames, compressed_frames):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Process with metrics collection
        results = calculate_comprehensive_metrics(
            original_frames, 
            compressed_frames,
            include_advanced=True
        )
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Collect cache statistics
        cache = LPIPSModelCache()
        cache_info = cache.get_model_cache_info()
        
        metric_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "gif_path": gif_path,
            "processing_time": end_time - start_time,
            "memory_used": end_memory - start_memory,
            "peak_memory": end_memory,
            "cache_hits": cache_info.get("cache_hits", 0),
            "cache_misses": cache_info.get("cache_misses", 0),
            "cache_hit_rate": cache_info.get("hit_rate", 0),
            "metrics_calculated": len(results),
            "optimization_used": self._check_optimization_used(results)
        }
        
        self.metrics.append(metric_data)
        self._write_metrics()
        
        return results
    
    def _check_optimization_used(self, results):
        # Check if conditional processing was used
        skipped_metrics = [
            'lpips_score', 'ssimulacra2_score', 
            'deep_features_similarity', 'perceptual_hash_similarity'
        ]
        skipped_count = sum(1 for m in skipped_metrics if m not in results)
        return skipped_count > 0
    
    def _write_metrics(self):
        with open(self.output_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
```

#### Advanced Instrumentation with OpenTelemetry
```python
# telemetry.py
from opentelemetry import trace, metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider

# Setup tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Setup metrics
metric_reader = PrometheusMetricReader()
metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))
meter = metrics.get_meter(__name__)

# Define metrics
processing_time_histogram = meter.create_histogram(
    name="giflab_processing_time",
    description="GIF processing time in seconds",
    unit="s"
)

memory_usage_gauge = meter.create_gauge(
    name="giflab_memory_usage",
    description="Memory usage in MB",
    unit="MB"
)

cache_hit_counter = meter.create_counter(
    name="giflab_cache_hits",
    description="Number of cache hits"
)

def instrument_processing(func):
    def wrapper(*args, **kwargs):
        with tracer.start_as_current_span("process_gif") as span:
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Record metrics
            processing_time_histogram.record(duration)
            memory_usage_gauge.set(end_memory)
            
            # Add span attributes
            span.set_attribute("processing.duration", duration)
            span.set_attribute("memory.used", end_memory - start_memory)
            
            return result
    return wrapper
```

### 2. System Monitoring Setup

#### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'giflab'
    static_configs:
      - targets: ['localhost:9090']
    metric_relabels:
      - source_labels: [__name__]
        regex: 'giflab_.*'
        action: keep

  - job_name: 'node_exporter'
    static_configs:
      - targets: ['localhost:9100']
```

#### Grafana Dashboard Configuration
```json
{
  "dashboard": {
    "title": "GifLab Performance Monitoring",
    "panels": [
      {
        "title": "Processing Time (P50, P95, P99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.5, giflab_processing_time_bucket)",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, giflab_processing_time_bucket)",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, giflab_processing_time_bucket)",
            "legendFormat": "P99"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "targets": [
          {
            "expr": "giflab_memory_usage",
            "legendFormat": "Current Memory"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "rate(giflab_cache_hits[5m]) / (rate(giflab_cache_hits[5m]) + rate(giflab_cache_misses[5m]))",
            "legendFormat": "Hit Rate %"
          }
        ]
      },
      {
        "title": "Optimization Usage",
        "targets": [
          {
            "expr": "rate(giflab_conditional_skips[5m]) / rate(giflab_total_processed[5m])",
            "legendFormat": "Optimization %"
          }
        ]
      }
    ]
  }
}
```

### 3. Log Aggregation

#### Structured Logging Configuration
```python
# logging_config.py
import logging
import json
from pythonjsonlogger import jsonlogger

def setup_logging():
    # Create logger
    logger = logging.getLogger('giflab')
    logger.setLevel(logging.INFO)
    
    # Create JSON formatter
    formatter = jsonlogger.JsonFormatter(
        fmt='%(timestamp)s %(level)s %(name)s %(message)s',
        rename_fields={'timestamp': '@timestamp'}
    )
    
    # File handler for JSON logs
    file_handler = logging.FileHandler('/var/log/giflab/app.json')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler for debugging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(console_handler)
    
    return logger

# Usage
logger = setup_logging()

def log_processing_metrics(gif_path, duration, memory_used, cache_hit_rate):
    logger.info("Processing completed", extra={
        "event": "processing_complete",
        "gif_path": gif_path,
        "duration_seconds": duration,
        "memory_mb": memory_used,
        "cache_hit_rate": cache_hit_rate,
        "optimization_enabled": os.environ.get('GIFLAB_ENABLE_CONDITIONAL_METRICS', 'true')
    })
```

#### Filebeat Configuration
```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/giflab/*.json
  json.keys_under_root: true
  json.add_error_key: true
  fields:
    service: giflab
    environment: production

output.elasticsearch:
  hosts: ["localhost:9200"]
  index: "giflab-%{+yyyy.MM.dd}"

processors:
  - add_host_metadata:
      when.not.contains:
        tags: forwarded
  - add_docker_metadata: ~
```

## Alerting Configuration

### Alert Rules

#### Prometheus Alert Rules
```yaml
# alerts.yml
groups:
  - name: giflab_performance
    interval: 30s
    rules:
      - alert: HighProcessingTime
        expr: histogram_quantile(0.95, giflab_processing_time_bucket) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High GIF processing time"
          description: "P95 processing time is {{ $value }}s (threshold: 10s)"
      
      - alert: HighMemoryUsage
        expr: giflab_memory_usage > 1000
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}MB (threshold: 1000MB)"
      
      - alert: LowCacheHitRate
        expr: |
          rate(giflab_cache_hits[5m]) / 
          (rate(giflab_cache_hits[5m]) + rate(giflab_cache_misses[5m])) < 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low cache hit rate"
          description: "Cache hit rate is {{ $value }}% (threshold: 50%)"
      
      - alert: OptimizationNotWorking
        expr: |
          rate(giflab_conditional_skips[5m]) / 
          rate(giflab_total_processed[5m]) < 0.3
        for: 15m
        labels:
          severity: info
        annotations:
          summary: "Low optimization usage"
          description: "Only {{ $value }}% of GIFs using optimized path"
```

#### PagerDuty Integration
```python
# pagerduty_integration.py
import requests
import json

class PagerDutyAlert:
    def __init__(self, integration_key):
        self.integration_key = integration_key
        self.url = "https://events.pagerduty.com/v2/enqueue"
    
    def send_alert(self, severity, summary, details):
        payload = {
            "routing_key": self.integration_key,
            "event_action": "trigger",
            "payload": {
                "summary": summary,
                "severity": severity,
                "source": "giflab-monitoring",
                "custom_details": details
            }
        }
        
        response = requests.post(self.url, json=payload)
        return response.status_code == 202
    
    def check_and_alert(self, metrics):
        # Check processing time
        if metrics['p95_processing_time'] > 10:
            self.send_alert(
                "warning",
                "High GIF processing time",
                {
                    "p95_time": metrics['p95_processing_time'],
                    "threshold": 10
                }
            )
        
        # Check memory usage
        if metrics['peak_memory'] > 1000:
            self.send_alert(
                "critical",
                "High memory usage",
                {
                    "peak_memory_mb": metrics['peak_memory'],
                    "threshold_mb": 1000
                }
            )
```

## Operational Procedures

### Daily Operations Checklist

```markdown
## Daily Operations Checklist

### Morning (9 AM)
- [ ] Check overnight alert summary
- [ ] Review performance dashboard for anomalies
- [ ] Verify cache hit rates > 80%
- [ ] Check error logs for patterns
- [ ] Review queue depth and processing backlogs

### Midday (12 PM)
- [ ] Monitor peak load performance
- [ ] Check memory usage trends
- [ ] Verify optimization effectiveness
- [ ] Review any manual interventions needed

### Evening (5 PM)
- [ ] Review daily metrics summary
- [ ] Check for memory leaks (growing trend)
- [ ] Verify cleanup processes ran
- [ ] Document any incidents or anomalies
```

### Incident Response Runbooks

#### Runbook: High Memory Usage
```markdown
# Runbook: High Memory Usage

## Symptoms
- Memory usage > 1GB sustained
- Potential OOM errors
- Slow processing

## Diagnosis
1. Check current memory usage:
   ```bash
   ps aux | grep giflab | awk '{sum+=$6} END {print sum/1024 " MB"}'
   ```

2. Check for memory leaks:
   ```bash
   grep "memory_mb" /var/log/giflab/app.json | tail -100 | \
     jq '.memory_mb' | awk '{sum+=$1; count++} END {print "Avg: " sum/count}'
   ```

3. Identify large GIFs being processed:
   ```bash
   grep "processing_complete" /var/log/giflab/app.json | \
     jq 'select(.memory_mb > 500) | .gif_path'
   ```

## Resolution
1. **Immediate:** Reduce parallel workers
   ```bash
   export GIFLAB_MAX_PARALLEL_WORKERS=2
   systemctl restart giflab
   ```

2. **Short-term:** Force cleanup
   ```bash
   export GIFLAB_FORCE_MODEL_CLEANUP=true
   export GIFLAB_CLEANUP_INTERVAL_SECONDS=60
   ```

3. **Long-term:** Analyze workload patterns and adjust configuration

## Escalation
- If memory > 1.5GB: Page on-call engineer
- If OOM errors occur: Immediate escalation to team lead
```

#### Runbook: Poor Performance
```markdown
# Runbook: Poor Performance

## Symptoms
- P95 processing time > 10 seconds
- Queue depth increasing
- User complaints about slow processing

## Diagnosis
1. Check optimization status:
   ```bash
   curl http://localhost:8080/health | jq '.optimizations'
   ```

2. Verify cache hit rate:
   ```bash
   grep "cache_hit_rate" /var/log/giflab/app.json | tail -10 | \
     jq '.cache_hit_rate' | awk '{sum+=$1} END {print sum/NR}'
   ```

3. Check if conditional processing is working:
   ```bash
   grep "optimization_enabled" /var/log/giflab/app.json | tail -100 | \
     jq '.optimization_enabled' | sort | uniq -c
   ```

## Resolution
1. **Verify optimizations enabled:**
   ```bash
   export GIFLAB_ENABLE_CONDITIONAL_METRICS=true
   export GIFLAB_ENABLE_PARALLEL_METRICS=true
   export GIFLAB_USE_MODEL_CACHE=true
   ```

2. **Increase parallel workers:**
   ```bash
   export GIFLAB_MAX_PARALLEL_WORKERS=8
   ```

3. **Lower quality thresholds for more aggressive optimization:**
   ```bash
   export GIFLAB_QUALITY_HIGH_THRESHOLD=0.85
   ```

## Escalation
- If no improvement after 30 minutes: Page on-call
- If degradation continues: Consider rollback
```

### Health Check Endpoints

```python
# health_check.py
from flask import Flask, jsonify
import psutil
from giflab.metrics import get_optimization_status
from giflab.model_cache import LPIPSModelCache

app = Flask(__name__)

@app.route('/health')
def health():
    """Basic health check"""
    return jsonify({"status": "healthy"}), 200

@app.route('/health/detailed')
def health_detailed():
    """Detailed health check with metrics"""
    process = psutil.Process()
    cache = LPIPSModelCache()
    
    health_data = {
        "status": "healthy",
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
        "optimizations": get_optimization_status(),
        "cache": cache.get_model_cache_info(),
        "config": {
            "parallel_enabled": os.environ.get('GIFLAB_ENABLE_PARALLEL_METRICS', 'true'),
            "conditional_enabled": os.environ.get('GIFLAB_ENABLE_CONDITIONAL_METRICS', 'true'),
            "cache_enabled": os.environ.get('GIFLAB_USE_MODEL_CACHE', 'true'),
        }
    }
    
    # Determine health status
    if health_data["memory_mb"] > 1000:
        health_data["status"] = "degraded"
        health_data["issues"] = ["high_memory"]
    
    if health_data["cache"]["hit_rate"] < 0.5:
        health_data["status"] = "degraded"
        health_data["issues"] = health_data.get("issues", []) + ["low_cache_hits"]
    
    status_code = 200 if health_data["status"] == "healthy" else 503
    return jsonify(health_data), status_code

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    # This would be handled by prometheus_client
    from prometheus_client import generate_latest
    return generate_latest()
```

## Performance Testing Integration

### Continuous Performance Testing
```yaml
# .gitlab-ci.yml or .github/workflows/performance.yml
performance_test:
  stage: test
  script:
    - poetry install
    - export GIFLAB_ENABLE_PROFILING=true
    
    # Run performance benchmarks
    - poetry run python tests/performance/benchmark_comprehensive.py > perf_results.json
    
    # Check against thresholds
    - |
      python -c "
      import json
      with open('perf_results.json') as f:
          results = json.load(f)
      if results['average_speedup'] < 1.5:
          print('Performance regression detected!')
          exit(1)
      "
    
    # Store results
    - cp perf_results.json $CI_PROJECT_DIR/
  artifacts:
    paths:
      - perf_results.json
    reports:
      performance: perf_results.json
```

### Load Testing
```python
# load_test.py
import concurrent.futures
import time
import random
from giflab.metrics import calculate_comprehensive_metrics

def process_gif(gif_id):
    """Simulate GIF processing"""
    start = time.time()
    
    # Load test GIF
    frames = load_test_gif(f"test_data/gif_{gif_id}.gif")
    
    # Process
    results = calculate_comprehensive_metrics(
        frames['original'],
        frames['compressed']
    )
    
    duration = time.time() - start
    return {
        "gif_id": gif_id,
        "duration": duration,
        "success": True
    }

def run_load_test(concurrent_requests=10, total_requests=100):
    """Run load test with specified concurrency"""
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = []
        for i in range(total_requests):
            future = executor.submit(process_gif, i % 10)  # Cycle through 10 test GIFs
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    # Analyze results
    durations = [r['duration'] for r in results]
    print(f"Average duration: {sum(durations)/len(durations):.2f}s")
    print(f"P95 duration: {sorted(durations)[int(len(durations)*0.95)]:.2f}s")
    print(f"P99 duration: {sorted(durations)[int(len(durations)*0.99)]:.2f}s")
    
    return results

if __name__ == "__main__":
    # Test with increasing load
    for concurrency in [1, 5, 10, 20]:
        print(f"\nTesting with {concurrency} concurrent requests:")
        run_load_test(concurrency, 50)
```

## Maintenance Procedures

### Weekly Maintenance
```bash
#!/bin/bash
# weekly_maintenance.sh

echo "Starting weekly maintenance..."

# 1. Clean up old logs
find /var/log/giflab -name "*.json" -mtime +7 -delete

# 2. Vacuum cache
python -c "
from giflab.model_cache import LPIPSModelCache
cache = LPIPSModelCache()
cache.cleanup(force=True)
print('Cache cleaned')
"

# 3. Check for updates
poetry show --outdated

echo "Weekly maintenance completed"
```

### Monthly Review
```markdown
## Monthly Performance Review Template

### Date: ___________

### Performance Metrics Summary
- Average P50 processing time: _____
- Average P95 processing time: _____
- Average memory usage: _____
- Cache hit rate: _____
- Optimization usage rate: _____

### Incidents
- Total incidents: _____
- Critical incidents: _____
- Average resolution time: _____

### Improvements Implemented
- [ ] Configuration changes: _____
- [ ] Code optimizations: _____
- [ ] Infrastructure updates: _____

### Recommendations for Next Month
1. _____
2. _____
3. _____

### Action Items
- [ ] _____
- [ ] _____
- [ ] _____
```

## Appendix: Monitoring Tools Comparison

| Tool | Purpose | Pros | Cons | Recommended For |
|------|---------|------|------|-----------------|
| Prometheus + Grafana | Metrics & Visualization | Open source, powerful queries | Complex setup | Production |
| ELK Stack | Log Aggregation | Comprehensive, scalable | Resource intensive | Large deployments |
| Datadog | Full observability | Easy setup, great UI | Expensive | Enterprise |
| New Relic | APM | Deep insights, auto-instrumentation | Costly, vendor lock-in | SaaS products |
| Custom Scripts | Basic monitoring | Simple, customizable | Limited features | Development/Small scale |

---

*Last updated: 2025-09-09*
*Version: 1.0.0*
*For GifLab v2.0.0+ with Phase 3 Optimizations*