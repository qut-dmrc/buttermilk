"""Tests for main.py example script that demonstrate project validation logic.

These tests verify that the main.py example script scenarios work correctly
and demonstrate the project validation behavior in action using a real
ExecutionContext.
"""

import pytest

from buttermilk import init
from buttermilk._core import execution_context
from buttermilk._core.dmrc import get_bm, set_bm
from buttermilk._core.log import reset_logging_configuration


@pytest.fixture(autouse=True)
def reset_singletons():
    """Fixture to reset the Buttermilk singleton, ExecutionContext, and logging."""
    # Reset logging
    reset_logging_configuration()

    # Reset BM singleton
    try:
        set_bm(None)
    except RuntimeError:
        pass

    # Reset ExecutionContext singleton
    execution_context._global_execution_context = None
    execution_context._execution_context_initialized = False

    yield

    # Teardown: Reset again after test
    reset_logging_configuration()
    try:
        set_bm(None)
    except RuntimeError:
        pass
    execution_context._global_execution_context = None
    execution_context._execution_context_initialized = False


class TestMainScriptRealExecution:
    """Test scenarios from the main.py example script using a real ExecutionContext."""

    def test_session_initialization_sequence(self):
        """Test the complete sequence of session initialization and project handling."""
        # 1. First session fails without a project because the testing.yaml doesn't specify one.
        # We override the config to remove the default project name.
        with pytest.raises(RuntimeError, match="project parameter is required"):
            init(job="first_analysis", overrides=["bm.session_info.name=null"])

        # 2. First session with an explicit project succeeds
        bm1 = init(job="first_analysis", project="project_alpha")
        assert bm1.session_info.project_name == "project_alpha"
        assert bm1.session_info.job == "first_analysis"
        assert get_bm().session_info.project_name == "project_alpha"

        # 3. Second session inherits the project from the execution context
        bm2 = init(job="second_analysis")
        assert bm2.session_info.project_name == "project_alpha"
        assert bm2.session_info.job == "second_analysis"

        # 4. Third session with a different project is not allowed in the same execution context
        with pytest.raises(RuntimeError, match="Project name mismatch"):
            init(job="third_analysis", project="project_beta")
