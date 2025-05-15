"""Tests for the BM singleton implementation."""

import pytest
from buttermilk.bm import BM, get_bm


@pytest.fixture
def initialize_bm():
    """Initialize BM with test values."""
    # This simulates what happens in cli.py when Hydra initializes BM
    bm = BM(
        name="test_name",
        job="test_job",
        run_id="test_run_id",
        platform="test"
    )
    return bm


def test_singleton_instance(initialize_bm):
    """Test that all ways of accessing BM return the same instance."""
    bm_init = initialize_bm
    
    # Access BM directly
    bm_direct = BM()
    
    # Access BM via get_bm()
    bm_via_getter = get_bm()
    
    # Verify all instances are the same
    assert bm_init is bm_direct, "Direct BM() access should return the same instance as initialization"
    assert bm_init is bm_via_getter, "get_bm() should return the same instance as initialization"
    assert bm_direct is bm_via_getter, "Different access methods should return the same instance"


def test_singleton_state_maintained(initialize_bm):
    """Test that BM properties are maintained across different access methods."""
    bm_init = initialize_bm
    
    # Access BM directly
    bm_direct = BM()
    
    # Access BM via get_bm()
    bm_via_getter = get_bm()
    
    # Check property values are maintained
    assert bm_direct.name == "test_name", "Property 'name' should be maintained"
    assert bm_direct.job == "test_job", "Property 'job' should be maintained"
    assert bm_direct.run_id == "test_run_id", "Property 'run_id' should be maintained"
    
    assert bm_via_getter.name == "test_name", "Property 'name' should be maintained"
    assert bm_via_getter.job == "test_job", "Property 'job' should be maintained"
    assert bm_via_getter.run_id == "test_run_id", "Property 'run_id' should be maintained"


def test_singleton_with_kwargs_update(initialize_bm):
    """Test that creating a singleton with kwargs updates existing attributes."""
    # Initial values
    bm_init = initialize_bm
    assert bm_init.name == "test_name", "Initial name should be 'test_name'"
    
    # Create a new instance with different name, should update the existing singleton
    bm_updated = BM(name="new_name")
    
    # Both should be the same instance
    assert bm_init is bm_updated, "Should be the same instance"
    
    # Property should be updated on both
    assert bm_init.name == "new_name", "Property should be updated on original instance"
    assert bm_updated.name == "new_name", "Property should be updated on new instance"
    
    # But other properties should remain unchanged
    assert bm_updated.job == "test_job", "Other properties should remain unchanged"
    assert bm_updated.run_id == "test_run_id", "Other properties should remain unchanged"


def test_singleton_between_modules():
    """Test that BM stays a singleton when accessed from different module functions."""
    # First initialize BM 
    bm1 = BM(
        name="module_test",
        job="module_job",
        run_id="module_run_id"
    )
    
    # Now import a module that will access BM (this simulates another module using BM)
    # We'll use a fixture for simplicity
    def second_module_access():
        """Function simulating another module accessing BM."""
        return get_bm()
    
    bm2 = second_module_access()
    
    # Both should be the same instance
    assert bm1 is bm2, "BM should be the same instance across different module functions"
    
    # Properties should be the same
    assert bm2.name == "module_test", "Property 'name' should be maintained across modules"
    assert bm2.job == "module_job", "Property 'job' should be maintained across modules"
    assert bm2.run_id == "module_run_id", "Property 'run_id' should be maintained across modules"
