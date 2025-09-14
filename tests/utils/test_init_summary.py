"""Test summary and documentation for initialization utility tests.

This file documents the comprehensive test coverage for the new initialization
functions in buttermilk.utils.nb and buttermilk.utils.cli.

Test Coverage Summary:
===================

1. **Backwards Compatibility** (test_nb_init.py):
   - Old nb_init() interface continues to work
   - Name extraction from overrides ("name=value" format)
   - Name extraction from "bm.session_info.name=value" format
   - Default name fallback behavior

2. **New Simple Interfaces** (test_nb_init.py, test_cli_init.py):
   - nb.init() simplifies notebook initialization
   - init() simplifies CLI script initialization
   - cli.bootstrap_session_with_config() returns both BM and config objects
   - Proper parameter passthrough

3. **ConfigurationBootstrapper Integration** (test_bootstrap_integration.py):
   - All functions use ConfigurationBootstrapper internally
   - Proper bootstrap sequence: full_context -> session_context
   - Infrastructure sharing between bootstrap phases
   - Error propagation from bootstrap failures

4. **Return Values** (all test files):
   - nb_init() returns backwards-compatible object with .bm and .config
   - nb.init() returns BM instance directly
   - init() returns BM instance directly
   - cli.bootstrap_session_with_config() returns tuple of (BM, config)

5. **Error Handling** (all test files):
   - Proper exception propagation
   - Error logging with descriptive messages
   - No silent failures

6. **Path Handling** (test_nb_init.py, test_cli_init.py):
   - Notebook: defaults to ../../conf from nb.py location
   - CLI: defaults to ./conf from current working directory
   - Custom paths work correctly
   - Absolute path conversion

7. **Context-Specific Behavior**:
   - Notebook functions add "run=notebook" override
   - CLI functions add "run=cli" override
   - Script name auto-detection for CLI functions
   - Fallback naming strategies

8. **Integration Testing** (test_nb_init.py, test_cli_init.py):
   - End-to-end functionality with real (minimal) configuration
   - Temporary test configurations
   - Actual BM instance creation and validation

Test Files:
===========

- `test_nb_init.py`: Tests for notebook initialization functions
- `test_cli_init.py`: Tests for CLI initialization functions
- `test_bootstrap_integration.py`: Tests for ConfigurationBootstrapper integration
- `test_init_summary.py`: This documentation file

Running Tests:
=============

# Run all initialization tests
uv run pytest tests/utils/test_*init*.py -v

# Run specific test categories
uv run pytest tests/utils/test_nb_init.py -v
uv run pytest tests/utils/test_cli_init.py -v
uv run pytest tests/utils/test_bootstrap_integration.py -v

# Run integration tests only
uv run pytest tests/utils/ -k "integration" -v

Key Design Validations:
======================

1. **Architecture Consistency**: All functions use the same ConfigurationBootstrapper
   internally, ensuring consistent behavior across initialization methods.

2. **Backwards Compatibility**: Existing nb_init() usage continues to work without
   changes, protecting existing notebooks and scripts.

3. **Simple Interfaces**: New init() functions provide clean, one-line initialization
   for the most common use cases while supporting advanced configuration.

4. **Error Transparency**: All functions properly propagate configuration and
   bootstrap errors rather than hiding them, supporting the project's fail-fast
   philosophy.

5. **Context Awareness**: Functions automatically configure themselves for their
   intended context (notebook vs CLI) with appropriate defaults and overrides.
"""

# This file serves as documentation and can include basic smoke tests
# if needed, but the main purpose is to document the comprehensive
# test coverage across the initialization utility functions.

import pytest


def test_all_init_test_files_exist():
    """Smoke test to verify all initialization test files exist."""
    from pathlib import Path

    test_dir = Path(__file__).parent

    # Verify all expected test files exist
    expected_files = [
        "test_nb_init.py",
        "test_cli_init.py",
        "test_bootstrap_integration.py",
        "test_init_summary.py",  # This file
    ]

    for file_name in expected_files:
        test_file = test_dir / file_name
        assert test_file.exists(), f"Expected test file {file_name} not found"
        assert test_file.stat().st_size > 0, f"Test file {file_name} is empty"


def test_init_modules_importable():
    """Smoke test to verify the initialization modules can be imported."""
    # Test that the modules we're testing can be imported
    try:
        from buttermilk import BM, init, nb_init
        from buttermilk._core.config_bootstrap import ConfigurationBootstrapper
    except ImportError as e:
        pytest.fail(f"Failed to import required modules: {e}")
