"""Configuration for performance tests."""


def pytest_configure(config):
    """Register custom markers for performance tests."""
    config.addinivalue_line(
        "markers",
        "performance: marks tests as performance tests (deselect with '-m \"not performance\"')",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow running tests",
    )

    # Disable xdist for performance tests (benchmarks can't run in parallel)
    if config.getoption("markexpr") == "performance" or "performance" in str(
        config.invocation_params.dir
    ):
        config.option.numprocesses = None
        config.option.dist = "no"


def pytest_benchmark_update_machine_info(config, machine_info):
    """Add custom machine info to benchmark results."""
    # Add any custom machine/environment info
    machine_info["test_suite"] = "buttermilk-startup-optimization"


# Configure pytest-benchmark defaults via pytest ini options
def pytest_benchmark_scale_unit(config, unit, benchmarks, best, worst, sort):  # noqa: PLR0913
    """Custom scale unit for benchmark results."""
    return "ms", 1000, "{value:.2f}"  # Display in milliseconds with 2 decimal places
