"""Test the BM singleton pattern."""

import hydra

"""Tests for the BM singleton implementation."""


from buttermilk import (
    BM,  # Removed logger import here
    get_bm,  # Import get_bm
)


def test_conf(conf):
    """Test that the test configuration is loaded correctly."""
    # Test the actual nested configuration structure
    assert conf.bm.session_info.job == "testing"
    assert conf.bm.session_info.name == "buttermilk"


def test_bm_instance(bm):
    # After BM instantiation, session_info is available
    assert bm.session_info.job == "testing"
    assert bm.session_info.name == "buttermilk"


def test_instantiate_hydra(conf):
    """Test that Hydra instantiation works with backward compatibility."""
    bm = hydra.utils.instantiate(conf.bm)
    assert bm is not None, "BM instance should not be None"
    assert bm.session_info.job == "testing", "BM instance job should be 'testing'"


def test_initialize_bm(conf):
    """Initialize BM with the provided configuration."""
    # Initialize BM with the test configuration
    bm = hydra.utils.instantiate(conf.bm)
    assert bm is not None, "BM instance should not be None"
    assert bm.session_info.job == "testing", "BM instance job should be 'testing'"


def test_singleton_instance(bm, conf):
    """Test that singleton access returns the same instance, but new sessions create new instances."""
    # Get singleton instance should return the same BM
    bm_direct = get_bm()  # Use get_bm() to access the singleton
    assert bm_direct is bm, "get_bm() should return the same singleton instance"

    # But hydra.utils.instantiate creates new session-scoped instances (new architecture)
    bm_new_session = hydra.utils.instantiate(conf.bm)
    assert bm_new_session is not None, "New BM instance should not be None"
    assert bm_new_session.session_info.job == "testing", "New BM instance job should be 'testing'"
    
    # New sessions have different session_ids (this is the intended behavior)
    assert bm_new_session is not bm, "New instantiation creates new session-scoped instance"
    assert bm_new_session.session_info.session_id != bm.session_info.session_id, "Different sessions have different IDs"

    # But both should have the same basic configuration
    assert bm_new_session.session_info.name == conf.bm.session_info.name, "New session should match config"
    assert bm_new_session.session_info.job == conf.bm.session_info.job, "New session should match config"
    assert bm_direct.session_info.name == conf.bm.session_info.name, "Singleton should match config"
    assert bm_direct.session_info.job == conf.bm.session_info.job, "Singleton should match config"


def test_session_scoped_instances(conf, bm):
    """Test that creating new session-scoped instances works correctly."""
    # Initial values
    assert bm.session_info.job == "testing", "Initial job should be 'testing'"

    # Creating new session-scoped instances should work (new architecture)
    new_session = BM(session_info={"name": "test-project", "job": "new_task"})
    
    # Verify new session has different ID but works correctly
    assert new_session.session_info.job == "new_task", "New session should have new job"
    assert new_session.session_info.name == "test-project", "New session should have new name"
    assert new_session.session_info.session_id != bm.session_info.session_id, "Different sessions have different IDs"
    
    # Original singleton should be unchanged
    assert bm.session_info.job == "testing", "Original singleton should be unchanged"


def test_singleton_between_modules(bm):
    """Test that BM stays a singleton when accessed from different module functions."""
    # First initialize BM
    bm1 = bm
    assert bm.session_info.job == "testing"

    # Now import a module that will access BM (this simulates another module using BM)
    # We'll use a function for simplicity
    def second_module_access():
        """Function simulating another module accessing BM."""
        from buttermilk import get_bm  # Correct import
        from buttermilk._core.log import logger  # noqa

        return get_bm()  # Use get_bm()

    bm2 = second_module_access()

    # Both should be the same instance
    assert bm1 is bm2, "BM should be the same instance across different module functions"

    # Properties should be the same (using session_info)
    assert bm2.session_info.name == "buttermilk", "Property 'name' should be maintained across modules"
    assert bm2.session_info.job == "testing", "Property 'job' should be maintained across modules"
    assert bm2.session_info.session_id == bm1.session_info.session_id, "Property 'session_id' should be maintained across modules"


def test_session_info_backward_compatibility():
    """Test that session_info gets automatically converted to session_info."""
    # Create BM with old session_info format - should show deprecation warning
    old_config = {
        "session_info": {
            "name": "test-project",
            "job": "test-task"
        }
    }
    
    bm = BM(**old_config)
    
    # Verify conversion worked
    assert bm.session_info.name == "test-project"
    assert bm.session_info.job == "test-task"
    assert bm.session_info.session_id.startswith("session-")
    
    # Test new format still works
    new_config = {
        "session_info": {
            "name": "test-project-2",
            "job": "test-task-2"
        }
    }
    
    bm2 = BM(**new_config)
    assert bm2.session_info.name == "test-project-2"
    assert bm2.session_info.job == "test-task-2"
