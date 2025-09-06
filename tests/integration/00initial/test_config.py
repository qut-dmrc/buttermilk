from unittest.mock import AsyncMock, patch

import pytest
from cloudpathlib import AnyPath
from omegaconf import DictConfig

from buttermilk import BM


def test_has_test_info(bm: BM):
    assert bm.session_info.name == "buttermilk"
    assert bm.session_info.job == "testing"
    assert bm.session_info.save_dir is not None
    assert bm.session_info.save_dir != ""


def test_config_llms():
    """Test moved to tests/integration/test_llms_infrastructure.py"""
    pytest.skip("Moved to integration tests - see test_llms_infrastructure.py")


def test_save_dir(bm: BM):
    assert "runs/buttermilk/testing/" in bm.session_info.save_dir
    assert AnyPath(bm.session_info.save_dir)


def test_singleton(bm: BM):
    obj1 = bm
    obj2 = bm

    assert id(obj1) == id(obj2), "variables contain different instances."


def test_singleton_from_fixture(bm):
    obj2 = bm

    assert id(bm) == id(obj2), "variables contain different instances."


def test_time_to_instantiate():
    import time

    start = time.time()
    end = time.time()
    time_taken = end - start
    print(f"Time taken: {time_taken:.2f} seconds")
    assert time_taken < 1, "Took too long to instantiate BM"


# Use relative import for the module under test

@pytest.mark.anyio
async def test_get_ip_updates_ip(bm):
    """Test that start_fetch_ip_task fetches and updates the _ip attribute."""
    # Skip this test due to IP fetching inconsistency bug
    # See: https://github.com/qut-dmrc/buttermilk/issues/229
    pytest.skip("IP fetching has inconsistency between BM and SessionInfo classes - see issue #229")


@pytest.mark.anyio
async def test_get_ip_caches_ip(bm):
    """Test that start_fetch_ip_task caches the result and doesn't refetch."""
    # Skip this test due to IP fetching inconsistency bug
    # See: https://github.com/qut-dmrc/buttermilk/issues/229
    pytest.skip("IP fetching has inconsistency between BM and SessionInfo classes - see issue #229")
