"""Test the BM singleton pattern."""


from buttermilk.bm import get_bm, initialize_bm

"""Tests for the BM singleton implementation."""


import pytest

from buttermilk.bm import BM


def test_conf(conf):
    """Test that the test configuration is loaded correctly."""
    assert conf.job == "testing"
    assert conf.bm.job == "testing"
    assert conf.name == "development"
    assert conf.bm.name == "development"


def test_initialize_bm(conf):
    """Initialize BM with the provided configuration."""
    # Initialize BM with the test configuration
    bm = initialize_bm(conf.bm)
    assert bm is not None, "BM instance should not be None"
    assert bm.job == "testing", "BM instance job should be 'testing'"


def test_initialize_bm_with_kwargs(conf):
    bm = initialize_bm({**conf.bm})
    assert bm is not None, "BM instance should not be None"
    assert bm.job == "testing", "Initial job should be 'testing'"


def test_singleton_instance(conf):
    """Test that all ways of accessing BM return the same instance."""
    bm_init = initialize_bm(conf.bm)
    assert bm_init is not None, "BM instance should not be None"
    assert bm_init.job == "testing", "BM instance job should be 'testing'"

    # Access BM directly
    bm_direct = BM()

    # Access BM via get_bm()
    bm_via_getter = get_bm()

    # Verify all instances are the same
    assert bm_init is bm_direct, "Direct BM() access should return the same instance as initialization"
    assert bm_init is bm_via_getter, "get_bm() should return the same instance as initialization"
    assert bm_direct is bm_via_getter, "Different access methods should return the same instance"

    # Check property values are maintained
    assert bm_direct.name == conf.name, "Property 'name' should be maintained"
    assert bm_direct.job == conf.job, "Property 'job' should be maintained"
    assert bm_direct.run_id == bm_via_getter.run_id, "Property 'run_id' should be maintained"

    assert bm_via_getter.name == conf.name, "Property 'name' should be maintained"
    assert bm_via_getter.job == conf.job, "Property 'job' should be maintained"


def test_singleton_with_kwargs_update_fails(conf, bm):
    """Test that creating a singleton with kwargs updates existing attributes."""
    # Initial values
    assert bm.job == "testing", "Initial job should be 'testing'"

    # Creating a new instance should fail
    with pytest.raises(RuntimeError) as excinfo:
        bm_updated = initialize_bm({**conf.bm, "job": "new_job"})


def test_singleton_between_modules(bm):
    """Test that BM stays a singleton when accessed from different module functions."""
    # First initialize BM
    bm1 = bm
    assert bm.job == "testing"

    # Now import a module that will access BM (this simulates another module using BM)
    # We'll use a fixture for simplicity
    def second_module_access():
        """Function simulating another module accessing BM."""
        return get_bm()

    bm2 = second_module_access()

    # Both should be the same instance
    assert bm1 is bm2, "BM should be the same instance across different module functions"

    # Properties should be the same
    assert bm2.name == "development", "Property 'name' should be maintained across modules"
    assert bm2.job == "testing", "Property 'job' should be maintained across modules"
    assert bm2.run_id == bm1.run_id, "Property 'run_id' should be maintained across modules"


def test_get_bm_returns_same_instance():
    """Test that get_bm returns the same instance every time."""
    bm1 = get_bm()
    bm2 = get_bm()
    assert bm1 is bm2, "get_bm() should return the same instance every time"
