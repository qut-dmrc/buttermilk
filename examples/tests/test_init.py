"""Initializing the buttermilk package."""

import pytest

from buttermilk import init, init_async

# Expected values from testing.yaml config
EXPECTED_PROJECT_NAME = "buttermilk_testing"


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


def test_sync():
    """Test sync init() wrapper.

    This will use the default config.yaml in the package."""
    bm = init(job="test_sync", project_name=EXPECTED_PROJECT_NAME)
    assert bm is not None
    assert bm.session_info.project_name == EXPECTED_PROJECT_NAME
    assert bm.session_info.job == "test_sync"


@pytest.mark.anyio
async def test_async():
    """Test init_async(), preferred and faster lazy init pathway."""
    bm = await init_async(job="test_async", project_name=EXPECTED_PROJECT_NAME)
    logger = bm.logger
    logger.debug("logging seems to work", structured_logging=True)
    assert bm.session_info.project_name == EXPECTED_PROJECT_NAME
    assert bm.session_info.job == "test_async"


@pytest.mark.anyio
async def test_config():
    """Test async init_async() - the primary initialization path."""
    bm = await init_async(config_dir="../config", config_name="pipeline")
    assert bm.session_info.project_name == "pipeline_example"

    # Test that log file path includes project name.
    # Expected format: /tmp/buttermilk_exec-{timestamp}-{uuid}.jsonl
    # Todo: add a property to bm that has the log file path and test it.

    # Test that bm.save_dir uses proper session-based format.
    # Expected format: {base}/{project_name}/{job}/session-{timestamp}-{uuid}/
    assert "pipeline_example" in bm.session_info.save_dir, f"save_dir should contain project 'pipeline_example', got: {bm.session_info.save_dir}"

    assert bm.cfg.pipeline.source.vector_store == bm.cfg.pipeline.processors[2]
