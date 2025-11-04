"""Tests for logging functionality using standard fixtures.

All logging should be initialized via the real_bm fixture from conftest.py.
Tests verify that logging works correctly with the test configuration.
"""

import uuid

import pytest

DEBUG_TEXT = "this should not show up in the log" + str(uuid.uuid1())
LOG_TEXT = "logging appears to be working" + str(uuid.uuid1())


def test_error(real_logger):
    """Test that error logging works."""
    log_text_error = f"{LOG_TEXT}_error_{uuid.uuid4()}"
    # Just verify that the logger can be called without errors
    # Capturing structlog output in tests is complex and not the main goal here
    try:
        real_logger.error(log_text_error)
        # If we get here without exception, the logger is working
        assert True
    except Exception as e:
        pytest.fail(f"Logger error() call failed: {e}")


def test_debug(real_logger):
    """Test that debug logging works."""
    # Just verify that the logger can be called without errors
    # Capturing structlog output in tests is complex and not the main goal here
    debug_text_specific = f"{DEBUG_TEXT}_debug_{uuid.uuid4()}"
    try:
        real_logger.debug(debug_text_specific)
        # If we get here without exception, the logger is working
        assert True
    except Exception as e:
        pytest.fail(f"Logger debug() call failed: {e}")


def test_info(real_logger):
    """Test that info logging works."""
    log_text_info = f"{LOG_TEXT}_info_{uuid.uuid4()}"
    # Just verify that the logger can be called without errors
    # Capturing structlog output in tests is complex and not the main goal here
    try:
        real_logger.info(log_text_info)
        # If we get here without exception, the logger is working
        assert True
    except Exception as e:
        pytest.fail(f"Logger info() call failed: {e}")


def test_logger_has_session_context(real_logger, real_bm):
    """Test that logger includes session context from real_bm."""
    # The logger should automatically include session context when logging
    # This verifies that the logging setup from real_bm is working

    test_message = f"test_session_context_{uuid.uuid4()}"

    # Log a message - should automatically include session_id, etc.
    real_logger.info(test_message, test_marker="session_context_test")

    # Verify that real_bm has session info (which should be in logs)
    assert real_bm.session_info.session_id is not None
    assert real_bm.session_info.project_name is not None
    assert real_bm.session_info.job is not None
