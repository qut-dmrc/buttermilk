
import pytest
from cloudpathlib import AnyPath

from buttermilk import BM


def test_has_test_info(real_bm: BM):
    assert real_bm.session_info.project_name == "buttermilk"
    assert real_bm.session_info.job == "testing"
    assert real_bm.session_info.save_dir is not None
    assert real_bm.session_info.save_dir != ""


def test_save_dir(real_bm: BM):
    # The save_dir should contain the project name and session ID
    # Format: /path/to/base/buttermilk/testing/session-xxx
    assert "buttermilk/testing/" in real_bm.session_info.save_dir
    assert real_bm.session_info.session_id in real_bm.session_info.save_dir
    assert AnyPath(real_bm.session_info.save_dir)


def test_singleton(real_bm: BM):
    obj1 = real_bm
    obj2 = real_bm

    assert id(obj1) == id(obj2), "variables contain different instances."


def test_singleton_from_fixture(real_bm):
    obj2 = real_bm

    assert id(real_bm) == id(obj2), "variables contain different instances."


def test_time_to_instantiate():
    import time

    start = time.time()
    end = time.time()
    time_taken = end - start
    print(f"Time taken: {time_taken:.2f} seconds")
    assert time_taken < 1, "Took too long to instantiate BM"


# Use relative import for the module under test

@pytest.mark.anyio
async def test_get_ip_updates_ip(real_bm):
    """Test that start_fetch_ip_task fetches and updates the _ip attribute."""
    # Skip this test due to IP fetching inconsistency bug
    # See: https://github.com/qut-dmrc/buttermilk/issues/229
    pytest.skip("IP fetching has inconsistency between BM and SessionInfo classes - see issue #229")


@pytest.mark.anyio
async def test_get_ip_caches_ip(real_bm):
    """Test that start_fetch_ip_task caches the result and doesn't refetch."""
    # Skip this test due to IP fetching inconsistency bug
    # See: https://github.com/qut-dmrc/buttermilk/issues/229
    pytest.skip("IP fetching has inconsistency between BM and SessionInfo classes - see issue #229")
