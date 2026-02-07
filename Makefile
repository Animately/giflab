# GifLab project Makefile

# -----------------------------------------------------------------------------
# VARIABLES (override on command-line, e.g. `make data RAW_DIR=/path/to/raw` )
# -----------------------------------------------------------------------------

RAW_DIR ?= data/raw
CSV_DIR ?= data/csv
EDA_DIR ?= data/eda
DATE    := $(shell date +%Y%m%d_%H%M%S)
CSV_PATH := $(CSV_DIR)/results_$(DATE).csv

# -----------------------------------------------------------------------------
# TARGETS
# -----------------------------------------------------------------------------

.PHONY: data help clean-temp clean-testing-mess test-workspace test test-ci test-nightly test-file benchmark-baseline benchmark-compare benchmark-ci performance-status performance-baseline performance-monitor performance-ci

data: ## Run compression pipeline on RAW_DIR and generate EDA artefacts
	@echo "🎞️  Running GifLab compression pipeline (raw=$(RAW_DIR))…"
	@echo "🔍 Validating RAW_DIR..."
	@poetry run python -c "from giflab.input_validation import validate_raw_dir; validate_raw_dir('$(RAW_DIR)')"
	@mkdir -p $(CSV_DIR) $(EDA_DIR)
	poetry run giflab run $(RAW_DIR) --csv $(CSV_PATH) --renders-dir data/renders --workers 0 --resume
	@echo "📊 Results CSV: $(CSV_PATH)"
	@echo "📈 Generating EDA artefacts…"
	@poetry run python -c "from giflab.eda import generate_eda; from pathlib import Path; print('   • Generating EDA into: $(EDA_DIR)'); artefacts = generate_eda(Path('$(CSV_PATH)'), Path('$(EDA_DIR)')); print(f'   • Generated {len(artefacts)} artefacts'); [print(f'     - {name}: {path}') for name, path in artefacts.items()]"
	@echo "✅ Dataset extraction + EDA complete."

test-workspace: ## Create proper test workspace structure
	@echo "🏗️  Creating test workspace structure..."
	@mkdir -p test-workspace/{manual,debug,temp,samples}
	@echo "✅ Test workspace created. See docs/guides/testing-best-practices.md for usage."

clean-temp: ## Clean temporary test files older than 7 days
	@echo "🧹 Cleaning temporary test files..."
	@find test-workspace/temp -type f -mtime +7 -delete 2>/dev/null || true
	@echo "✅ Temporary files cleaned."

clean-testing-mess: ## Emergency cleanup of testing files in root directory
	@echo "🚨 Cleaning up testing mess in root directory..."
	@echo "⚠️  This will delete all test/debug directories and files in root!"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	@echo "🗑️  Removing test directories..."
	@rm -rf debug_* test_* final_* verification_* comprehensive_* deep_* gpu_* clean_test_*
	@echo "🗑️  Removing test files..."
	@rm -f *_test.gif *test*.gif debug_*.png step*.gif pipeline_*.gif
	@echo "🗑️  Removing PNG export directories..."
	@rm -rf *png_export* *png_frames* *png_sequence* *png_from_* *png_fix*
	@echo "✅ Root directory cleaned! Use 'make test-workspace' to create proper structure."

test: ## Fast feedback: smoke + functional (<30s)
	poetry run pytest tests/smoke/ tests/functional/ -x -q

test-ci: ## CI: + integration with process isolation (<5min)
	poetry run pytest tests/smoke/ tests/functional/ tests/integration/ \
		-n auto --dist loadfile -q

test-nightly: ## Nightly: everything including perf/memory
	poetry run pytest tests/ -n auto --dist loadfile

test-file: ## Run a single test file: make test-file F=tests/functional/test_metrics.py
	poetry run pytest $(F) -v

benchmark-baseline: ## Run Phase 4.3 performance baseline measurements
	@echo "📊 Running Phase 4.3 performance baseline measurements..."
	@echo "🔧 Configuration: Caching disabled (baseline mode)"
	poetry run python src/giflab/benchmarks/phase_4_3_benchmarking.py --iterations 3
	@echo "✅ Baseline performance measurements complete!"

benchmark-compare: ## Compare cached vs non-cached performance (requires both baselines)
	@echo "📈 Running Phase 4.3 performance comparison analysis..."
	poetry run python src/giflab/benchmarks/performance_comparison.py --save-report benchmark_results/latest_comparison.json
	@echo "✅ Performance comparison analysis complete!"

benchmark-ci: ## Run performance regression detection for CI
	@echo "🚨 Running performance regression detection for CI..."
	@echo "📊 Quick baseline measurement (current configuration)..."
	@poetry run python src/giflab/benchmarks/phase_4_3_benchmarking.py --iterations 2 --scenario small_gif_basic
	@echo "✅ Performance CI check complete! (Validates current performance baseline)"
	@echo "💡 Use 'make benchmark-compare' to analyze cached vs non-cached performance"

# -----------------------------------------------------------------------------
# Phase 7: Performance Monitoring & Regression Detection
# -----------------------------------------------------------------------------

performance-status: ## Show performance monitoring status and recent alerts
	@echo "📊 Performance Monitoring Status..."
	poetry run python -m giflab performance status --verbose

performance-baseline: ## Create or update performance baselines
	@echo "📈 Creating Performance Baselines..."
	poetry run python -m giflab performance baseline create --iterations 3

performance-monitor: ## Start continuous performance monitoring (requires baselines)
	@echo "🔍 Starting Performance Monitoring..."
	@echo "⚠️  This requires GIFLAB_ENABLE_PERFORMANCE_MONITORING=true"
	poetry run python -m giflab performance monitor start

performance-ci: ## Run performance regression check for CI/CD pipelines
	@echo "🚨 Running Performance CI Check..."
	poetry run python -m giflab performance ci gate --threshold 0.15
	@echo "✅ Performance regression check complete!"

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+: .*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ": |## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$3}' 