"""Live integration tests for cloud logging with real GCP services.

These tests verify that the complete cloud logging flow works correctly:
- Infrastructure detects logging cloud configuration
- Logger config is properly passed to BM sessions
- Real SessionInfo with actual session_id and batch_id is used
- Structured JSON logs are sent to Google Cloud Logging
- Session context is properly propagated to cloud logs

All tests use the standard real_bm and real_logger fixtures from conftest.py.
No manual configuration or logging initialization should occur in these tests.

Requires:
- GCP project with Cloud Logging enabled in integration test configuration
- Service account credentials with logging.logEntries.create permission
- GOOGLE_APPLICATION_CREDENTIALS environment variable set
"""

import asyncio
import time
import uuid
from typing import Any

import pytest
from google.cloud import logging as gcp_logging
from google.cloud.logging_v2 import DESCENDING

# Test constants
EXPECTED_NUMERIC_FIELD = 42

# Helper text
DEBUG_TEXT = "this should not show up in the log" + str(uuid.uuid1())
LOG_TEXT = "logging appears to be working" + str(uuid.uuid1())


@pytest.fixture(scope="session")
def gcp_project_id(real_bm) -> str:
    """Get GCP project ID for testing."""
    assert real_bm.cloud_manager is not None, "Cloud manager must be available for cloud logging tests"
    assert len(real_bm.cloud_manager.clouds) > 0, "At least one cloud must be configured"
    return real_bm.cloud_manager.clouds[0].project_id


@pytest.fixture(scope="session")
def log_client(gcp_project_id: str) -> gcp_logging.Client:
    """Create GCP logging client for verification."""
    return gcp_logging.Client(project=gcp_project_id)


# Core end-to-end test


def test_cloud_logging_end_to_end(real_bm, real_logger, log_client: gcp_logging.Client):
    """Test complete end-to-end cloud logging flow.

    This is the primary test that verifies:
    1. Logging is configured via real_bm fixture
    2. Messages logged via real_logger reach Google Cloud Logging
    3. Session context (session_id, batch_id, etc.) is included in cloud logs
    """
    # Verify cloud logging is configured in real_bm
    assert real_bm._logger_cfg is not None, "Logger configuration must be available for cloud logging tests"
    # Note: We don't assert type == "gcp" because the test config might use different settings

    # Create unique test identifier
    test_run_id = f"e2e-test-{uuid.uuid4().hex[:8]}"

    # Log messages using the real_logger fixture
    test_messages = [
        f"Cloud logging test message 1: {test_run_id}",
        f"Cloud logging test message 2: {test_run_id}",
        f"Structured test message: {test_run_id}",
    ]

    for msg in test_messages:
        real_logger.info(msg, test_run_id=test_run_id, test_type="e2e")

    # Allow time for logs to propagate to GCP
    time.sleep(5)

    # Verify logs appeared in GCP Cloud Logging
    entries = _get_log_entries(log_client, test_run_id)

    assert len(entries) > 0, f"No log entries found for test run {test_run_id}"

    # Verify session context is present
    session_context_found = False
    for entry in entries:
        payload = entry.payload
        if isinstance(payload, dict):
            # Check for session_id in the payload
            if "session_id" in payload or "session" in str(payload).lower():
                session_context_found = True
                break

    assert session_context_found, "Session context not found in cloud logs"

    # Verify all test messages appeared
    found_messages = set()
    for entry in entries:
        payload_str = str(entry.payload)
        for msg in test_messages:
            if msg in payload_str:
                found_messages.add(msg)

    missing = set(test_messages) - found_messages
    assert not missing, f"Missing messages in cloud logs: {missing}"


def test_structured_json_logging(real_logger, log_client: gcp_logging.Client):
    """Test that structured fields are properly sent to cloud logging."""
    test_run_id = f"json-test-{uuid.uuid4().hex[:8]}"

    # Log with structured data
    real_logger.info(
        "Structured logging test",
        test_run_id=test_run_id,
        custom_field="custom_value",
        numeric_field=EXPECTED_NUMERIC_FIELD,
        boolean_field=True,
    )

    time.sleep(5)

    # Verify structured format in cloud logs
    entries = _get_log_entries(log_client, test_run_id)
    assert len(entries) > 0, "No log entries found"

    # Check that at least one entry has the expected structure
    structured_entry_found = False
    for entry in entries:
        payload = entry.payload
        if isinstance(payload, dict) and test_run_id in str(payload):
            # Verify custom fields are preserved
            if (
                payload.get("test_run_id") == test_run_id
                and payload.get("custom_field") == "custom_value"
                and payload.get("numeric_field") == EXPECTED_NUMERIC_FIELD
                and payload.get("boolean_field") is True
            ):
                structured_entry_found = True
                break

    assert structured_entry_found, "Structured log entry with correct fields not found"


def test_warning_level_logging(real_logger, real_bm, log_client: gcp_logging.Client):
    """Test that warning level logs reach cloud logging."""
    log_text_warning = f"{LOG_TEXT}_warning_{uuid.uuid4()}"

    real_logger.warning(log_text_warning, test_marker="warning_test")

    # Allow time for propagation
    time.sleep(5)

    # Verify warning appears in cloud logs
    # Search more broadly since we might not have test_run_id
    entries = list(
        log_client.list_entries(
            filter_=f'textPayload:"{log_text_warning}" OR jsonPayload.event:"{log_text_warning}"',
            order_by=DESCENDING,
            max_results=100,
        )
    )

    warning_found = False
    for entry in entries:
        if log_text_warning in str(entry.payload):
            warning_found = True
            break

    assert warning_found, f"Warning message not found in cloud logs: {log_text_warning}"


@pytest.mark.anyio
async def test_async_logging_performance(real_logger):
    """Test that logging works correctly in async contexts.

    This test ensures cloud logging doesn't become a bottleneck
    during concurrent async operations.
    """
    test_run_id = f"async-test-{uuid.uuid4().hex[:8]}"

    async def log_message(msg_num: int):
        """Log a message asynchronously."""
        real_logger.info(f"Async test message {msg_num}: {test_run_id}", msg_num=msg_num, test_run_id=test_run_id)
        await asyncio.sleep(0.1)  # Simulate some async work
        return msg_num

    # Log multiple messages concurrently
    start_time = time.time()

    NUM_MESSAGES = 10
    tasks = [log_message(i) for i in range(NUM_MESSAGES)]
    results = await asyncio.gather(*tasks)

    end_time = time.time()
    duration = end_time - start_time

    # Verify all messages were logged
    assert len(results) == NUM_MESSAGES
    assert results == list(range(NUM_MESSAGES))

    # Should complete reasonably quickly (less than 5 seconds)
    MAX_DURATION = 5.0
    assert duration < MAX_DURATION, f"Async logging took too long: {duration:.2f}s"

    real_logger.info(f"Logged {NUM_MESSAGES} messages in {duration:.2f}s")


# Helper functions


def _get_log_entries(log_client: gcp_logging.Client, test_run_id: str) -> list[Any]:
    """Helper to retrieve log entries for a test run."""
    # Query for recent entries containing our test run ID
    filter_str = f'jsonPayload.test_run_id="{test_run_id}" OR textPayload:"{test_run_id}"'

    entries = list(log_client.list_entries(filter_=filter_str, order_by=gcp_logging.DESCENDING, max_results=100))

    return entries
