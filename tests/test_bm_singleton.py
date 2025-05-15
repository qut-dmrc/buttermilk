"""Test the BM singleton pattern."""


from buttermilk.bm import get_bm, initialize_bm


def test_get_bm_returns_same_instance():
    """Test that get_bm returns the same instance every time."""
    bm1 = get_bm()
    bm2 = get_bm()
    assert bm1 is bm2, "get_bm() should return the same instance every time"


def test_initialize_bm_updates_existing_instance():
    """Test that initialize_bm updates the existing instance."""
    # First call get_bm to ensure an instance exists
    bm1 = get_bm()

    # Now initialize it with some values
    bm2 = initialize_bm(
        name="test_name",
        job="test_job",
        run_id="test_run_id",
        platform="test",
    )

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


def test_initialize_bm_can_be_called_multiple_times():
    """Test that initialize_bm can be called multiple times to update the instance."""
    # First initialize with some values
    bm1 = initialize_bm(
        name="test_name1",
        job="test_job1",
        run_id="test_run_id1",
        platform="test1",
    )

    # Now initialize again with different values
    bm2 = initialize_bm(
        name="test_name2",
        job="test_job2",
        run_id="test_run_id2",
        platform="test2",
    )

    # Check that we got the same instance back
    assert bm1 is bm2, "initialize_bm should return the same instance when called multiple times"

    # Check that the values were updated
    assert bm1.name == "test_name2"
    assert bm1.job == "test_job2"
    assert bm1.run_id == "test_run_id2"
    assert bm1.platform == "test2"
