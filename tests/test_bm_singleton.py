"""Test the BM singleton pattern."""


from buttermilk.bm import get_bm, initialize_bm

"""Tests for the BM singleton implementation."""


from buttermilk.bm import BM


def test_conf(conf):
    """Test that the test configuration is loaded correctly."""
    assert conf.job == "testing"
    assert conf.bm.job == "testing"
    assert conf.name == "development"
    assert conf.bm.name == "development"


def test_singleton_instance(bm):
    """Test that all ways of accessing BM return the same instance."""
    # Access BM directly
    bm_direct = BM()

    # Access BM via get_bm()
    bm_via_getter = get_bm()

    # Verify all instances are the same
    assert bm is bm_direct, "Direct BM() access should return the same instance as initialization"
    assert bm is bm_via_getter, "get_bm() should return the same instance as initialization"
    assert bm_direct is bm_via_getter, "Different access methods should return the same instance"


def test_singleton_state_maintained(conf):
    """Test that BM properties are maintained across different access methods."""
    # Access BM directly
    bm_direct = BM()

    # Access BM via get_bm()
    bm_via_getter = get_bm()

    # Check property values are maintained
    assert bm_direct.name == conf.name, "Property 'name' should be maintained"
    assert bm_direct.job == conf.job, "Property 'job' should be maintained"
    assert bm_direct.run_id == bm_via_getter.run_id, "Property 'run_id' should be maintained"

    assert bm_via_getter.name == conf.name, "Property 'name' should be maintained"
    assert bm_via_getter.job == conf.job, "Property 'job' should be maintained"


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
        run_id="module_run_id",
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


def test_get_bm_returns_same_instance():
    """Test that get_bm returns the same instance every time."""
    bm1 = get_bm()
    bm2 = get_bm()
    assert bm1 is bm2, "get_bm() should return the same instance every time"


def test_initialize_bm_updates_existing_instance(conf):
    """Test that initialize_bm updates the existing instance."""
    # First call get_bm to ensure an instance exists
    bm1 = get_bm()

    # Now initialize it with some values
    bm2 = initialize_bm(conf.bm)

    # Check that we got the same instance back
    assert bm1 is bm2, "initialize_bm should return the same instance as get_bm"

    # Check that the values were updated
    assert bm1.name == "test_name"
    assert bm1.job == "test_job"
    assert bm1.run_id == "test_run_id"
    assert bm1.platform == "test"

    # Get a new reference and check it has the same values
    bm3 = get_bm()
    assert bm3.name == "test_name"
    assert bm3.job == "test_job"
    assert bm3.run_id == "test_run_id"
    assert bm3.platform == "test"


def test_initialize_bm_can_be_called_multiple_times(conf):
    """Test that initialize_bm can be called multiple times to update the instance."""
    # First initialize with some values
    bm1 = initialize_bm(conf.bm)

    # Now initialize again with different values
    bm2 = initialize_bm(dict(
        name="test_name2",
        job="test_job2",
        run_id="test_run_id2",
        platform="test2",
    ))

    # Check that we got the same instance back
    assert bm1 is bm2, "initialize_bm should return the same instance when called multiple times"

    # Check that the values were updated
    assert bm1.name == "test_name2"
    assert bm1.job == "test_job2"
    assert bm1.run_id == "test_run_id2"
    assert bm1.platform == "test2"
