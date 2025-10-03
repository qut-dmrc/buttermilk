#!/usr/bin/env python3
"""Simple test runner to validate bootstrap sequence architecture tests.

This script runs a subset of our bootstrap tests to ensure they work correctly
and validate the architecture fix. It can be run standalone to verify the
test implementation.
"""

import sys

import pytest

if __name__ == "__main__":
    # Run specific bootstrap tests
    test_args = [
        "tests/unit/test_bootstrap_sequence_architecture.py::TestExecutionContextCreation::test_execution_context_singleton_behavior",
        "tests/unit/test_bootstrap_sequence_architecture.py::TestInfrastructureSharing::test_execution_context_has_own_infrastructure",
        "tests/unit/test_cli_bootstrap_order.py::TestCLIBootstrapOrder::test_cli_main_bootstrap_order",
        "tests/unit/test_bootstrap_logging_consistency.py::TestBootstrapLoggingConsistency::test_execution_context_logging_initialization",
        "-v"
    ]
    
    # Run the tests
    exit_code = pytest.main(test_args)
    sys.exit(exit_code)
