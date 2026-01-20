# Performance Testing Guide

This guide explains how to run, monitor, and maintain performance tests for Buttermilk startup optimization (Issue #284).

## Overview

We use **pytest-benchmark** for systematic performance testing with these goals:

- **ChromaDB lazy initialization**: < 1s (down from 23s)
- **MCP server startup**: < 30s (hard requirement)
- **Background warmup**: completes after configured delay

## Quick Start

### Run Performance Tests

```bash
# Run all performance tests
uv run pytest tests/performance/ -v

# Run specific test
uv run pytest tests/performance/test_startup_performance.py::test_chromadb_lazy_init_benchmark -v

# Run only benchmark tests (skip manual timing tests)
uv run pytest tests/performance/ --benchmark-only -v

# Skip performance tests in regular test runs
uv run pytest tests/ -m "not performance"
```

### Benchmark Modes

#### 1. Simple Run (No Comparison)

```bash
uv run pytest tests/performance/ --benchmark-only -v
```

This runs benchmarks and shows results, but doesn't compare against baseline.

#### 2. Save as Baseline

```bash
# Using shell script
./scripts/performance/run_performance_tests.sh --baseline

# Using Python script
python scripts/performance/monitor_performance.py --baseline
```

This runs tests and saves results as the new baseline for future comparisons.

#### 3. Compare Against Baseline

```bash
# Using shell script
./scripts/performance/run_performance_tests.sh --compare

# Using Python script
python scripts/performance/monitor_performance.py --compare --threshold 10
```

This runs tests, compares against baseline, and **fails if regressions > 10%** are detected.

## Performance Tests

### Test Structure

```
tests/performance/
├── __init__.py
├── conftest.py                          # Benchmark configuration
└── test_startup_performance.py          # Main performance tests
```

### Available Tests

| Test                                        | Purpose                            | Target                |
| ------------------------------------------- | ---------------------------------- | --------------------- |
| `test_chromadb_lazy_init_benchmark`         | Benchmark model creation time      | < 1s                  |
| `test_chromadb_first_access_benchmark`      | Benchmark first collection access  | Varies                |
| `test_chromadb_lazy_init_timing`            | Manual timing verification         | < 1s                  |
| `test_background_warmup_timing`             | Verify warmup delay and completion | Completes after delay |
| `test_full_init_async_startup_performance`  | Full initialization via fixture    | < 30s                 |
| `test_minimal_init_async_performance`       | Direct init_async measurement      | < 30s                 |
| `test_chromadb_collection_access_benchmark` | Cached collection access speed     | < 1ms                 |

## Understanding Results

### pytest-benchmark Output

```
------------------------------------ benchmark: chromadb-init ------------------------------------
Name (time in ms)                                    Min      Max     Mean  StdDev  Median   IQR
----------------------------------------------------------------------------------------------------
test_chromadb_lazy_init_benchmark                 0.8234  1.2456  0.9123  0.0567  0.9034  0.0234
----------------------------------------------------------------------------------------------------
```

**Key metrics:**

- **Mean**: Average time across all runs (primary metric)
- **StdDev**: Standard deviation (lower is more consistent)
- **Min/Max**: Range of execution times
- **IQR**: Interquartile range (consistency metric)

### Comparison Output

```
--------------------- benchmark 'chromadb-init': 2 tests ---------------------
Name (time in ms)                       Min     Max    Mean  StdDev  Median
-------------------------------------------------------------------------------
test_chromadb_lazy_init (NOW)        0.8234  1.2456  0.9123  0.0567  0.9034
test_chromadb_lazy_init (baseline)  20.1234 23.4567 21.8901  0.8234 21.7645

Ratio: 0.04x (24.0x faster) ✅
```

## CI/CD Integration

### GitHub Actions Workflow

The `.github/workflows/performance-tests.yml` workflow:

1. **Runs on**: PRs and pushes to main/dev
1. **Downloads baseline**: From cache (if available)
1. **Runs tests**: Compares against baseline
1. **Uploads results**: As artifacts
1. **Comments on PR**: If regressions detected
1. **Fails build**: If regression > 10%

### Baseline Management

Baselines are stored in GitHub Actions cache:

```bash
# On main/dev push: Save new baseline
performance-baseline-main-abc123

# On PR: Compare against main baseline
performance-baseline-main
```

## Local Development Workflow

### 1. Before Making Performance Changes

```bash
# Create baseline before your changes
git checkout main
./scripts/performance/run_performance_tests.sh --baseline

# Switch to your branch
git checkout my-feature-branch
```

### 2. After Making Changes

```bash
# Compare against baseline
./scripts/performance/run_performance_tests.sh --compare

# If regressions are acceptable, update baseline
./scripts/performance/run_performance_tests.sh --baseline
```

### 3. Iterative Development

```bash
# Run tests continuously during development
uv run pytest tests/performance/ --benchmark-only -v

# Quick feedback loop
uv run pytest tests/performance/test_startup_performance.py::test_chromadb_lazy_init_benchmark -v
```

## Advanced Usage

### Custom Benchmark Configuration

Edit `tests/performance/conftest.py`:

```python
@pytest.fixture(scope="function")
def benchmark(benchmark):
    benchmark.pedantic(
        iterations=20,  # More iterations
        rounds=10,  # More rounds
        warmup_rounds=2,  # More warmup
    )
    return benchmark
```

### Profiling Individual Tests

```python
import cProfile
import pstats


def test_with_profiling():
    profiler = cProfile.Profile()
    profiler.enable()

    # Your code here
    embeddings = ChromaDBEmbeddings(...)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(20)  # Top 20 functions
```

### Memory Profiling

```bash
# Install memory_profiler
uv add --dev memory-profiler

# Add @profile decorator to functions
# Run with:
python -m memory_profiler tests/performance/test_startup_performance.py
```

## Regression Thresholds

Default regression threshold: **10%**

This can be adjusted:

```bash
# More strict (5% threshold)
python scripts/performance/monitor_performance.py --compare --threshold 5

# More lenient (20% threshold)
./scripts/performance/run_performance_tests.sh --compare --threshold 20

# Set via environment variable
export BENCHMARK_REGRESSION_THRESHOLD=15
./scripts/performance/run_performance_tests.sh --compare
```

## Troubleshooting

### "No baseline found"

```bash
# Solution: Create a baseline first
./scripts/performance/run_performance_tests.sh --baseline
```

### Tests are too slow/variable

1. **Reduce iterations**: Edit `conftest.py` to use fewer iterations
1. **Disable warmup**: Use `enable_background_warmup=False` in tests
1. **Run on consistent hardware**: CI environments can be variable

### Benchmark reports missing

```bash
# Generate HTML report
./scripts/performance/run_performance_tests.sh --report

# Check .benchmarks/reports/ directory
ls -la .benchmarks/reports/
```

## Best Practices

### 1. **Mark Performance Tests**

```python
@pytest.mark.performance
@pytest.mark.slow
def test_something(): ...
```

This allows selective execution:

```bash
# Run only performance tests
pytest -m performance

# Skip performance tests
pytest -m "not performance"
```

### 2. **Disable Warmup for Testing**

```python
embeddings = ChromaDBEmbeddings(
    enable_background_warmup=False,  # For consistent test timing
)
```

### 3. **Use Temporary Directories**

```python
def test_something(tmp_path):
    embeddings = ChromaDBEmbeddings(
        persist_directory=str(tmp_path / "test_db"),
    )
```

### 4. **Assert on Requirements**

```python
def test_startup_performance():
    assert elapsed < 30.0, f"Startup {elapsed}s exceeds 30s requirement"
```

## Performance Monitoring Dashboard

Results can be visualized using:

1. **GitHub Actions artifacts**: Download `.benchmarks/` results
1. **CSV export**: `pytest-benchmark compare --csv=results.csv`
1. **Custom visualization**: Parse JSON results from `.benchmarks/current/`

## Related Documentation

- [Issue #284: Startup Optimization](https://github.com/qut-dmrc/buttermilk/issues/284)
- [pytest-benchmark docs](https://pytest-benchmark.readthedocs.io/)
- [Testing Philosophy](TESTING_PHILOSOPHY.md)
