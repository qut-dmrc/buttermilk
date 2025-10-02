"""Tests for async initialization pathway.

This module tests the new async-first initialization pathway for Buttermilk.
"""

import pytest

from buttermilk import init_async
from buttermilk._core.bm_init import create_session_bm_async


@pytest.mark.asyncio
async def test_create_session_bm_async():
    """Test async BM session creation."""
    # Create a BM instance asynchronously
    bm = await create_session_bm_async(
        name="test_project",
        job="test_job",
        platform="test"
    )

    assert bm is not None
    assert bm.session_info.project_name == "test_project"
    assert bm.session_info.job == "test_job"
    assert bm.session_info.platform == "test"
    assert bm.session_info.session_id is not None


@pytest.mark.asyncio
async def test_bm_async_init_creates_save_dir():
    """Test that async init properly sets up save directory."""
    bm = await create_session_bm_async(
        name="test_project",
        job="test_job"
    )

    # Verify save_dir was set during async init
    assert bm.session_info.save_dir is not None
    assert "test_project" in bm.session_info.save_dir
    assert "test_job" in bm.session_info.save_dir


@pytest.mark.asyncio
async def test_bm_ensure_initialized():
    """Test BM ensure_initialized() waits for async init."""
    bm = await create_session_bm_async(
        name="test_project",
        job="test_job"
    )

    # Should complete immediately since we already awaited creation
    await bm.ensure_initialized()

    # Verify initialization is complete
    assert bm._initialization_complete.is_set()
    assert bm._initialization_error is None


def test_sync_wrappers_still_work():
    """Test that sync wrappers remain functional for backward compatibility."""
    from buttermilk._core.bm_init import create_session_bm

    # Sync wrapper should still work
    bm = create_session_bm(
        name="test_project",
        job="test_job",
        platform="test"
    )

    assert bm is not None
    assert bm.session_info.project_name == "test_project"
    assert bm.session_info.job == "test_job"
