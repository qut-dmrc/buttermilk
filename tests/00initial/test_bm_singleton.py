"""Test the BM singleton pattern."""

import hydra

from buttermilk import (
    logger,  # noqa
)
from buttermilk._core.bm_init import BM, create_session_bm
from buttermilk._core.dmrc import get_bm


def test_conf(real_bm):
    """Test that the test configuration is loaded correctly."""
    # Test the actual nested configuration structure
    assert real_bm is not None, "BM instance should not be None"
    assert real_bm.session_info.job == "testing", "BM instance job should be 'testing'"
    assert real_bm.session_info.name == "buttermilk"


def test_singleton_instance(real_bm):
    """Test that singleton access returns the same instance, but new sessions create new instances."""
    # Get singleton instance should return the same BM
    from buttermilk import bm
    bm_direct = bm
    assert bm_direct is real_bm, "bm should be the same singleton instance"

    # But hydra.utils.instantiate creates new session-scoped instances (new architecture)
    bm_new_session = hydra.utils.instantiate(real_bm)
    assert bm_new_session is not None, "New BM instance should not be None"
    assert bm_new_session.session_info.job == "testing", "New BM instance job should be 'testing'"

    # New sessions have different session_ids (this is the intended behavior)
    assert bm_new_session is not real_bm, "New instantiation creates new session-scoped instance"
    assert bm_new_session.session_info.session_id != real_bm.session_info.session_id, "Different sessions have different IDs"

    # But both should have the same basic configuration
    assert bm_new_session.session_info.name == real_bm.session_info.name, "New session should match config"
    assert bm_new_session.session_info.job == real_bm.session_info.job, "New session should match config"
    assert bm_direct.session_info.name == real_bm.session_info.name, "Singleton should match config"
    assert bm_direct.session_info.job == real_bm.session_info.job, "Singleton should match config"


def test_session_scoped_instances(real_bm):
    """Test that creating new session-scoped instances works correctly."""
    # Initial values
    assert real_bm.session_info.job == "testing", "Initial job should be 'testing'"

    # Creating new session-scoped instances should work (new architecture)
    new_session = BM(session_info={"name": "test-project", "job": "new_task"})

    # Verify new session has different ID but works correctly
    assert new_session.session_info.job == "new_task", "New session should have new job"
    assert new_session.session_info.name == "test-project", "New session should have new name"
    assert new_session.session_info.session_id != real_bm.session_info.session_id, "Different sessions have different IDs"

    # Original singleton should be unchanged
    assert real_bm.session_info.job == "testing", "Original singleton should be unchanged"


def test_singleton_between_modules(real_bm):
    """Test that BM stays a singleton when accessed from different module functions."""
    # First initialize BM
    bm1 = real_bm
    assert real_bm.session_info.job == "testing"

    # Now import a module that will access BM (this simulates another module using BM)
    # We'll use a function for simplicity
    def second_module_access():
        """Function simulating another module accessing BM."""

        from buttermilk import bm
        return bm

    bm2 = second_module_access()

    # Both should be the same instance
    assert bm1 is bm2, "BM should be the same instance across different module functions"

    # Properties should be the same (using session_info)
    assert bm2.session_info.name == "buttermilk", "Property 'name' should be maintained across modules"
    assert bm2.session_info.job == "testing", "Property 'job' should be maintained across modules"
    assert bm2.session_info.session_id == bm1.session_info.session_id, "Property 'session_id' should be maintained across modules"


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
        logger_cfg=None,
    )
    from buttermilk._core.dmrc import set_bm
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
        logger_cfg=None,
    )

    # Set it as the singleton
    from buttermilk._core.dmrc import set_bm
    set_bm(test_instance)

    # Define a function that simulates importing from another module
    def import_from_another_module():
        # This imports get_bm fresh in this scope
        from buttermilk import bm as another_bm

        return another_bm

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
        logger_cfg=None,
    )

    # Set it as the singleton
    from buttermilk._core.dmrc import set_bm
    set_bm(test_instance)

    # Define a function that simulates the deferred import pattern
    def get_bm_deferred():
        from buttermilk import bm as _bm

        return _bm

    # Get the instance through the deferred import
    deferred_instance = get_bm_deferred()

    # Verify it's the same instance
    assert deferred_instance is test_instance
    assert deferred_instance.session_info.name == "test3"
