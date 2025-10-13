"""Test the BM singleton pattern."""
import pytest

from buttermilk import (
    init_async,
    logger,  # noqa
)
from buttermilk._core.bm_init import BM
from buttermilk._core.dmrc import get_bm


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
    assert bm1.session_info.project_name == bm2.session_info.project_name

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


@pytest.mark.asyncio
async def test_init_async_without_config_dir_uses_default():
    """Test that init_async() without config_dir uses the default resolution."""
    # Act: Initialize without config_dir
    bm = await init_async(job="test_default_config", project_name="buttermilk")

    # Assert: Should successfully initialize with default config
    assert bm is not None
    assert bm.session_info.job == "test_default_config"
    assert bm.session_info.project_name == "buttermilk"


@pytest.mark.asyncio
async def test_init_async_with_relative_config_dir(tmp_path, monkeypatch):
    """Test that init_async(config_dir='conf') resolves against CWD."""
    # Arrange: Create custom config in temp directory
    project_dir = tmp_path / "myproject"
    project_dir.mkdir()
    conf_dir = project_dir / "conf"
    conf_dir.mkdir()

    # Copy minimal config to test location
    import shutil
    from pathlib import Path

    src_conf = Path(__file__).parent.parent.parent.parent / "buttermilk" / "conf"
    shutil.copytree(src_conf, conf_dir, dirs_exist_ok=True)

    # Change to project directory
    monkeypatch.chdir(project_dir)

    # Act: Initialize with relative path
    bm = await init_async(job="test_relative", project_name="buttermilk", config_dir="conf")

    # Assert: Should use the config from CWD/conf
    assert bm is not None
    assert bm.session_info.job == "test_relative"


@pytest.mark.asyncio
async def test_init_async_with_absolute_config_dir(tmp_path):
    """Test that init_async(config_dir='/abs/path') uses absolute path."""
    # Arrange: Create custom config at absolute path
    abs_conf_dir = tmp_path / "absolute_config" / "conf"
    abs_conf_dir.mkdir(parents=True)

    # Copy minimal config to test location
    import shutil
    from pathlib import Path

    src_conf = Path(__file__).parent.parent.parent.parent / "buttermilk" / "conf"
    shutil.copytree(src_conf, abs_conf_dir, dirs_exist_ok=True)

    # Act: Initialize with absolute path
    bm = await init_async(job="test_absolute", project_name="buttermilk", config_dir=str(abs_conf_dir))

    # Assert: Should use the specified absolute config path
    assert bm is not None
    assert bm.session_info.job == "test_absolute"


@pytest.mark.asyncio
async def test_multiple_sessions_different_config_dirs(tmp_path, monkeypatch):
    """Test that different sessions can use different config directories."""
    # Arrange: Create two config directories
    import shutil
    from pathlib import Path

    src_conf = Path(__file__).parent.parent.parent.parent / "buttermilk" / "conf"

    conf1 = tmp_path / "config1"
    conf1.mkdir()
    shutil.copytree(src_conf, conf1 / "conf", dirs_exist_ok=True)

    conf2 = tmp_path / "config2"
    conf2.mkdir()
    shutil.copytree(src_conf, conf2 / "conf", dirs_exist_ok=True)

    # Act: Create sessions with different configs
    monkeypatch.chdir(conf1)
    bm1 = await init_async(job="session1", project_name="buttermilk", config_dir="conf")

    monkeypatch.chdir(conf2)
    bm2 = await init_async(job="session2", project_name="buttermilk", config_dir="conf")

    # Assert: Both should be initialized successfully with different session IDs
    assert bm1.session_info.job == "session1"
    assert bm2.session_info.job == "session2"
    assert bm1.session_info.session_id != bm2.session_info.session_id
