"""Test the fixed BM singleton pattern."""

from buttermilk import get_bm, set_bm
from buttermilk._core.bm_init import BM, create_session_bm


def test_get_bm_after_set():
    """Test that get_bm returns the instance after set_bm."""
    # Create a test instance using the new pattern
    test_instance = create_session_bm(
        name="test",
        job="test_job",
        platform="test",
        save_dir_base="/tmp/test_singleton",
        cloud_manager=None,
        secret_manager=None,
        llms_instance=None,
        logger_cfg=None
    )

    # Set it as the singleton
    set_bm(test_instance)

    # Get it back
    retrieved_instance = get_bm()

    # Verify it's the same instance
    assert retrieved_instance is test_instance
    assert retrieved_instance.session_info.name == "test"
    assert retrieved_instance.session_info.job == "test_job"


def test_import_singleton_from_different_modules():
    """Test that importing from different modules gets the same instance."""
    # Create a test instance using the new pattern
    test_instance = create_session_bm(
        name="test2",
        job="test_job2",
        platform="test",
        save_dir_base="/tmp/test_singleton2",
        cloud_manager=None,
        secret_manager=None,
        llms_instance=None,
        logger_cfg=None
    )

    # Set it as the singleton
    set_bm(test_instance)

    # Define a function that simulates importing from another module
    def import_from_another_module():
        # This imports get_bm fresh in this scope
        from buttermilk import get_bm as another_get_bm
        return another_get_bm()

    # Get the instance through the simulated import
    instance_from_other_module = import_from_another_module()

    # Verify it's the same instance
    assert instance_from_other_module is test_instance
    assert instance_from_other_module.session_info.name == "test2"


def test_deferred_import_function():
    """Test that the deferred import function works as expected."""
    # Create a test instance using the new pattern
    test_instance = create_session_bm(
        name="test3",
        job="test_job3",
        platform="test",
        save_dir_base="/tmp/test_singleton3",
        cloud_manager=None,
        secret_manager=None,
        llms_instance=None,
        logger_cfg=None
    )

    # Set it as the singleton
    set_bm(test_instance)

    # Define a function that simulates the deferred import pattern
    def get_bm_deferred():
        from buttermilk import get_bm as _get_bm
        return _get_bm()

    # Get the instance through the deferred import
    deferred_instance = get_bm_deferred()

    # Verify it's the same instance
    assert deferred_instance is test_instance
    assert deferred_instance.session_info.name == "test3"
