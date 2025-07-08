"""Test the BM singleton pattern."""

import hydra

"""Tests for the BM singleton implementation."""

import pytest

from buttermilk._core import BM  # Removed logger import here
from buttermilk._core.dmrc import get_bm  # Import get_bm
from buttermilk._core.exceptions import FatalError


def test_conf(conf):
    """Test that the test configuration is loaded correctly."""
    assert conf.job == "testing"
    assert conf.bm.job == "testing"
    assert conf.name == "buttermilk"
    assert conf.bm.name == "buttermilk"


def test_bm_instance(bm):
    assert bm.job == "testing"
    assert bm.name == "buttermilk"


def test_instantiate_hydra(conf):
    """I'm not sure whether this should work."""
    bm = hydra.utils.instantiate(conf.bm)
    assert bm is not None, "BM instance should not be None"
    assert bm.job == "testing", "BM instance job should be 'testing'"


def test_initialize_bm(conf):
    """Initialize BM with the provided configuration."""
    # Initialize BM with the test configuration
    bm = hydra.utils.instantiate(conf.bm)
    assert bm is not None, "BM instance should not be None"
    assert bm.job == "testing", "BM instance job should be 'testing'"


def test_singleton_instance(bm, conf):
    """Test that all ways of accessing BM return the same instance."""
    # Removed unnecessary logger import
    bm_direct = get_bm() # Use get_bm() to access the singleton
    bm_init = hydra.utils.instantiate(conf.bm)
    assert bm_init is not None, "BM instance should not be None"
    assert bm_init.job == "testing", "BM instance job should be 'testing'"

    # Verify all instances are the same
    assert bm_init is bm, "Different access methods should return the same instance"
    assert bm_direct is bm, "Different access methods should return the same instance"

    # Check property values are maintained
    assert bm_init.run_id == bm.run_id, "Property 'run_id' should be maintained"
    assert bm_direct.run_id == bm.run_id, "Property 'run_id' should be maintained"

    assert bm_init.name == conf.name, "Property 'name' should be maintained"
    assert bm_init.job == conf.job, "Property 'job' should be maintained"
    assert bm_direct.name == conf.name, "Property 'name' should be maintained"
    assert bm_direct.job == conf.job, "Property 'job' should be maintained"


def test_singleton_with_kwargs_update_fails(conf, bm):
    """Test that creating a singleton with kwargs updates existing attributes."""
    # Initial values
    assert bm.job == "testing", "Initial job should be 'testing'"

    # Creating a new instance should fail
    with pytest.raises(FatalError):
        # Attempt to instantiate BM directly after it's already set
        # This should raise FatalError due to the singleton pattern
        BM(**{**conf.bm, "job": "new_job"})


def test_singleton_between_modules(bm):
    """Test that BM stays a singleton when accessed from different module functions."""
    # First initialize BM
    bm1 = bm
    assert bm.job == "testing"

    # Now import a module that will access BM (this simulates another module using BM)
    # We'll use a function for simplicity
    def second_module_access():
        """Function simulating another module accessing BM."""
        from buttermilk._core.dmrc import get_bm  # Correct import
        from buttermilk._core.log import logger  # noqa
        return get_bm() # Use get_bm()

    bm2 = second_module_access()

    # Both should be the same instance
    assert bm1 is bm2, "BM should be the same instance across different module functions"

    # Properties should be the same
    assert bm2.name == "buttermilk", "Property 'name' should be maintained across modules"
    assert bm2.job == "testing", "Property 'job' should be maintained across modules"
    assert bm2.run_id == bm1.run_id, "Property 'run_id' should be maintained across modules"
