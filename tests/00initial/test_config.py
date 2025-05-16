from unittest.mock import AsyncMock, patch

import pytest
from cloudpathlib import AnyPath
from omegaconf import DictConfig

from buttermilk import buttermilk as bm
from buttermilk._core.log import logger  # noqa


def test_has_test_info(bm: BM):
    assert bm.run_info.name == "buttermilk"
    assert bm.run_info.job == "testing"
    assert bm.save_dir is not None
    assert bm.save_dir != ""


def test_config_llms(bm: BM):
    models = bm.llms
    assert models


def test_save_dir(bm):
    assert "runs/buttermilk/testing/" in bm.save_dir
    assert AnyPath(bm.save_dir)


def test_singleton(bm):
    obj1 = bm
    obj2 = bm

    assert id(obj1) == id(obj2), "variables contain different instances."


def test_singleton_from_fixture(bm):
    obj2 = bm

    assert id(bm) == id(obj2), "variables contain different instances."


def test_time_to_instantiate():
    import time

    start = time.time()
    obj = bm
    end = time.time()
    time_taken = end - start
    print(f"Time taken: {time_taken:.2f} seconds")
    assert time_taken < 1, "Took too long to instantiate BM"


# Use relative import for the module under test

@pytest.mark.anyio
async def test_get_ip_updates_ip(bm):
    """Test that get_ip fetches and updates the _ip attribute."""
    mock_ip_address = "192.168.1.100"

    # Patch the external get_ip utility function
    with patch("buttermilk.utils.get_ip", new_callable=AsyncMock) as mock_get_ip:
        mock_get_ip.return_value = mock_ip_address

        # Create a minimal configuration required for BM initialization
        # Avoid including configs that trigger complex setups (like logging, clouds)
        minimal_cfg = DictConfig({"name": "test_app", "job": "test_job"})

        # Ensure _ip is initially None
        assert bm._ip is None

        # Call the async method
        await bm.get_ip()

        # Assert that the _ip attribute was updated
        assert bm._ip == mock_ip_address

        # Assert that the utility function was called exactly once
        mock_get_ip.assert_called_once()


@pytest.mark.anyio
async def test_get_ip_caches_ip(m):
    """Test that get_ip caches the result and doesn't refetch."""
    mock_ip_address = "10.0.0.50"

    # Patch the external get_ip utility function
    with patch("buttermilk.utils.get_ip", new_callable=AsyncMock) as mock_get_ip:
        mock_get_ip.return_value = mock_ip_address

        # Create a minimal configuration required for BM initialization
        minimal_cfg = DictConfig({"name": "test_app_cache", "job": "test_job_cache"})

        # Call the async method the first time
        await bm.get_ip()

        # Call the async method a second time
        await bm.get_ip()

        # Assert that the utility function was still called only once
        # This verifies the caching logic (if not self._ip:)
        mock_get_ip.assert_called_once()

        # Assert that the _ip attribute is set correctly after calls
        assert bm._ip == mock_ip_address
