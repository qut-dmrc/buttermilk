# Performance Testing Scripts

Quick reference for performance testing scripts.

## Scripts

### `run_performance_tests.sh`

Bash script for running performance tests with baseline management.

```bash
# Create baseline
./scripts/performance/run_performance_tests.sh --baseline

# Compare against baseline (fails on regression)
./scripts/performance/run_performance_tests.sh --compare

# Generate HTML report
./scripts/performance/run_performance_tests.sh --compare --report
```

**Environment Variables:**

- `BENCHMARK_BASELINE_DIR`: Baseline storage directory (default: `.benchmarks/baseline`)
- `BENCHMARK_REGRESSION_THRESHOLD`: Regression % threshold (default: `10`)

### `monitor_performance.py`

Python script for advanced performance monitoring and CI/CD integration.

```bash
# Save baseline
python scripts/performance/monitor_performance.py --baseline

# Compare with custom threshold
python scripts/performance/monitor_performance.py --compare --threshold 5

# Just run tests
python scripts/performance/monitor_performance.py
```

**Arguments:**

- `--baseline`: Save results as new baseline
- `--compare`: Compare against baseline and fail on regression
- `--threshold N`: Set regression threshold to N% (default: 10)

**GitHub Integration:**

- Set `GITHUB_TOKEN` to enable PR comments
- Set `GITHUB_REPOSITORY` and `GITHUB_PR_NUMBER` for PR integration

## Quick Commands

```bash
# Development workflow
./scripts/performance/run_performance_tests.sh --baseline  # Before changes
# ... make your changes ...
./scripts/performance/run_performance_tests.sh --compare   # After changes

# CI/CD workflow (automated in GitHub Actions)
python scripts/performance/monitor_performance.py --compare --threshold 10

# View results
cat .benchmarks/current/*.json | jq '.benchmarks[] | {name: .name, mean: .stats.mean}'
```

## See Also

- [Performance Testing Guide](../../docs/PERFORMANCE_TESTING.md) - Complete documentation
- [Issue #284](https://github.com/qut-dmrc/buttermilk/issues/284) - Startup optimization tracking
