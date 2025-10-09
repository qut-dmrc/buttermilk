"""Test the BM singleton pattern."""
import pytest

from buttermilk import (
    logger,  # noqa
    init_async,
)
from buttermilk._core.bm_init import BM
from buttermilk._core.dmrc import get_bm

from projects.buttermilk.tests.integration.test_tracing import EXPECTED_PROJECT_NAME


def test_conf(real_bm):
    """Test that the test configuration is loaded correctly."""
    # Test the actual nested configuration structure
    assert real_bm is not None, "BM instance should not be None"
    assert real_bm.session_info.job == "testing", "BM instance job should be 'testing'"
    assert real_bm.session_info.project_name == "buttermilk"


def test_session_scoped_instances(real_bm):
    """Test that creating new session-scoped instances works correctly."""
    # Initial values
    assert real_bm.session_info.job == "testing", "Initial job should be 'testing'"

    # Creating new session-scoped instances should work (new architecture)
    new_session = BM(session_info={"project_name": "test-project", "job": "new_task"})

    # Verify new session has different ID but works correctly
    assert new_session.session_info.job == "new_task", "New session should have new job"
    assert new_session.session_info.project_name == "test-project", "New session should have new name"
    assert new_session.session_info.session_id != real_bm.session_info.session_id, "Different sessions have different IDs"

    # Original singleton should be unchanged
    assert real_bm.session_info.job == "testing", "Original singleton should be unchanged"

@pytest.fixture
def second_module_access():
    """Function simulating another module accessing BM."""
    return get_bm()

def test_singleton_between_modules(real_bm, second_module_access):
    """Test that BM stays a singleton when accessed from different module functions."""
    # First initialize BM
    bm1 = real_bm
    assert real_bm.session_info.job == "testing"

    # Now import a module that will access BM (this simulates another module using BM)
    # We'll use a function for simplicity
    bm2 = second_module_access()

    # Both should be the same instance
    assert bm1 is bm2, "BM should be the same instance across different module functions"

    # Properties should be the same (using session_info)
    assert bm2.session_info.project_name == "buttermilk", "Property 'name' should be maintained across modules"
    assert bm2.session_info.job == "testing", "Property 'job' should be maintained across modules"
    assert bm2.session_info.session_id == bm1.session_info.session_id, "Property 'session_id' should be maintained across modules"


@pytest.mark.asyncio
async def test_multiple_sessions_same_project(real_bm):
    """Test that multiple sessions in same project share project name but have unique run IDs.

    All sessions should:
    - Use same project_name in ExecutionContext
    - Have unique session IDs with formatted job names
    - Write to different save_dirs with job-timestamp format
    """

    # Create first session
    bm1 = real_bm
    assert bm1.session_info.job == "testing"

    # Create second session
    job2 = "analysis_v2"
    bm2 = await init_async(
        job=job2,
    )

    # Both should use same project
    assert bm1.session_info.project_name == EXPECTED_PROJECT_NAME
    assert bm2.session_info.project_name == EXPECTED_PROJECT_NAME

    # But have different session IDs
    assert bm1.session_info.session_id != bm2.session_info.session_id

    # And different save_dirs with formatted job names
    save_dir1 = bm1.session_info.save_dir
    save_dir2 = bm2.session_info.save_dir

    assert save_dir1 != save_dir2
    assert job2 in save_dir1
    assert job2 in save_dir2

    # Verify they don't use session- prefix
    assert "session-" not in save_dir1
    assert "session-" not in save_dir2
