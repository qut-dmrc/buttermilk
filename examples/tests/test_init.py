"""Initializing the buttermilk package.

These tests demonstrate buttermilk initialization using the testing.yaml config.
They use a module-scoped fixture to share the BM instance across tests.
"""

import pytest


@pytest.fixture(scope="module")
def bm_instance():
    """Module-scoped BM instance using testing.yaml config.

    Handles potential logging configuration conflicts if other tests ran before this.
    """
    from buttermilk import init

    try:
        return init(config_name="testing")
    except RuntimeError as e:
        if "Console logging has already been configured" in str(e):
            # Logging module was left configured (possibly by other tests) but Context wasn't initialized.
            # Reset logging state and try again.
            from buttermilk._core.log import reset_logging_configuration

            reset_logging_configuration()
            return init(config_name="testing")
        raise e


def test_sync(bm_instance):
    """Test that BM instance is properly initialized.

    Uses the testing.yaml config which sets project_name to 'buttermilk'.
    """
    assert bm_instance is not None
    assert bm_instance.session_info.project_name == "buttermilk"
    assert bm_instance.session_info.job == "testing"


@pytest.mark.anyio
async def test_async(bm_instance):
    """Test async access to BM instance."""
    logger = bm_instance.logger
    logger.debug("logging seems to work", structured_logging=True)
    assert bm_instance.session_info.project_name == "buttermilk"


def test_config(bm_instance):
    """Test that BM has proper configuration loaded."""
    # Verify the session has proper paths set up
    assert bm_instance.session_info.save_dir is not None
    assert "buttermilk" in bm_instance.session_info.save_dir, (
        f"save_dir should contain project 'buttermilk', got: {bm_instance.session_info.save_dir}"
    )
