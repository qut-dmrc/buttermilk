"""Tests for main.py example script that demonstrate project validation logic.

These tests verify that the main.py example script scenarios work correctly
and demonstrate the project validation behavior in action using a real
ExecutionContext.
"""

import pytest

from buttermilk import init
from buttermilk._core.singleton import get_instance, reset_instance


@pytest.fixture(autouse=True)
def reset_buttermilk_singleton():
    """Fixture to reset the Buttermilk singleton before and after each test."""
    reset_instance()
    yield
    reset_instance()


class TestMainScriptRealExecution:
    """Test scenarios from the main.py example script using a real ExecutionContext."""

    def test_session_initialization_sequence(self):
        """Test the complete sequence of session initialization and project handling."""
        # 1. First session fails without a project
        with pytest.raises(RuntimeError, match="project parameter is required"):
            init(job="first_analysis")

        # 2. First session with an explicit project succeeds
        bm1 = init(job="first_analysis", project="project_alpha")
        assert bm1.session_info.project_name == "project_alpha"
        assert bm1.session_info.job == "first_analysis"
        assert get_instance().session_info.project_name == "project_alpha"

        # 3. Second session inherits the project from the execution context
        bm2 = init(job="second_analysis")
        assert bm2.session_info.project_name == "project_alpha"
        assert bm2.session_info.job == "second_analysis"

        # 4. Third session with a different project switches the context
        bm3 = init(job="third_analysis", project="project_beta")
        assert bm3.session_info.project_name == "project_beta"
        assert bm3.session_info.job == "third_analysis"
        assert get_instance().session_info.project_name == "project_beta"

        # 5. Fourth session inherits the new project
        bm4 = init(job="fourth_analysis")
        assert bm4.session_info.project_name == "project_beta"
        assert bm4.session_info.job == "fourth_analysis"

    def test_project_mismatch_error(self):
        """Test that changing a project without resetting the context raises an error."""
        # Initialize with a project
        init(job="first_analysis", project="project_alpha")

        # Try to initialize with a different project in the same context
        with pytest.raises(RuntimeError, match="Project name mismatch"):
            init(job="conflicting_analysis", project="project_beta")