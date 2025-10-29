#!/usr/bin/env bash
# Run performance tests and generate reports for CI/CD
#
# Usage:
#   ./scripts/performance/run_performance_tests.sh [--baseline] [--compare]
#
# Options:
#   --baseline    Save results as new baseline
#   --compare     Compare against baseline and fail on regression
#   --report      Generate HTML report
#
# Environment variables:
#   BENCHMARK_BASELINE_DIR: Directory to store/load baseline results (default: .benchmarks/baseline)
#   BENCHMARK_REGRESSION_THRESHOLD: % threshold for regression detection (default: 10)

set -euo pipefail

# Configuration
BASELINE_DIR="${BENCHMARK_BASELINE_DIR:-.benchmarks/baseline}"
CURRENT_DIR=".benchmarks/current"
REPORT_DIR=".benchmarks/reports"
REGRESSION_THRESHOLD="${BENCHMARK_REGRESSION_THRESHOLD:-10}"

MODE="run"
GENERATE_REPORT=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline)
            MODE="baseline"
            shift
            ;;
        --compare)
            MODE="compare"
            shift
            ;;
        --report)
            GENERATE_REPORT=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--baseline] [--compare] [--report]"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$BASELINE_DIR" "$CURRENT_DIR" "$REPORT_DIR"

echo "🚀 Running performance tests..."
echo "   Mode: $MODE"
echo "   Baseline dir: $BASELINE_DIR"
echo "   Current dir: $CURRENT_DIR"
echo ""

# Run performance tests with benchmark output
# IMPORTANT: Disable xdist (parallel execution) for benchmarks
export PYTEST_ADDOPTS=""

if [ "$MODE" = "baseline" ]; then
    echo "📊 Running tests and saving as baseline..."
    uv run pytest tests/performance/ \
        -p no:xdist \
        -o addopts="" \
        --benchmark-only \
        --benchmark-autosave \
        --benchmark-save=baseline \
        --benchmark-save-data \
        --benchmark-storage="file://$BASELINE_DIR" \
        -v

    echo "✅ Baseline saved to $BASELINE_DIR"

elif [ "$MODE" = "compare" ]; then
    echo "📊 Running tests and comparing against baseline..."

    # Check if baseline exists
    if [ ! -d "$BASELINE_DIR" ] || [ -z "$(ls -A $BASELINE_DIR 2>/dev/null)" ]; then
        echo "❌ No baseline found in $BASELINE_DIR"
        echo "   Run with --baseline first to create a baseline"
        exit 1
    fi

    # Run tests and compare
    uv run pytest tests/performance/ \
        -p no:xdist \
        -o addopts="" \
        --benchmark-only \
        --benchmark-autosave \
        --benchmark-compare=baseline \
        --benchmark-compare-fail="min:${REGRESSION_THRESHOLD}%" \
        --benchmark-storage="file://$CURRENT_DIR" \
        -v

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "⚠️  PERFORMANCE REGRESSION DETECTED!"
        echo "   One or more benchmarks are ${REGRESSION_THRESHOLD}% slower than baseline"
        exit 1
    else
        echo "✅ No significant performance regression detected"
    fi

else
    # Just run tests without comparison
    echo "📊 Running performance tests..."
    uv run pytest tests/performance/ \
        -p no:xdist \
        -o addopts="" \
        --benchmark-only \
        --benchmark-autosave \
        --benchmark-storage="file://$CURRENT_DIR" \
        -v
fi

# Generate HTML report if requested
if [ "$GENERATE_REPORT" = true ]; then
    echo ""
    echo "📝 Generating HTML report..."

    uv run pytest-benchmark compare \
        --storage="file://$CURRENT_DIR" \
        --csv="$REPORT_DIR/benchmark_results.csv" \
        || true

    echo "✅ Report saved to $REPORT_DIR"
fi

echo ""
echo "✨ Performance test complete!"
