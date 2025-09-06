"""Live integration tests for cloud logging with real GCP services.

These tests verify that the complete cloud logging flow works correctly:
- Infrastructure detects logging cloud configuration
- Logger config is properly passed to BM sessions
- Real SessionInfo with actual session_id and batch_id is used
- Structured JSON logs are sent to Google Cloud Logging
- Session context is properly propagated to cloud logs

Requires:
- GCP project with Cloud Logging enabled in integration test configuration
- Service account credentials with logging.logEntries.create permission
- GOOGLE_APPLICATION_CREDENTIALS environment variable set
"""

import asyncio
import json
import os
import time
import uuid
from typing import Any
from unittest.mock import patch

import pytest
from google.cloud import logging as gcp_logging

from buttermilk._core.bm_init import BM
from buttermilk._core.infrastructure import InfrastructureManager
from buttermilk._core.log import logger

# Test constant
EXPECTED_NUMERIC_FIELD = 42

# Helper functions


@pytest.fixture(scope="session")
def gcp_project_id(bm) -> str:
    """Get GCP project ID for testing."""
    return bm.cloud_manager.clouds[0].project_id


@pytest.fixture(scope="session")
def log_client(gcp_project_id: str) -> gcp_logging.Client:
    """Create GCP logging client for verification."""
    return gcp_logging.Client(project=gcp_project_id)


# Test functions using integration fixtures


def test_session_cloud_logging_end_to_end(bm: BM, infrastructure: InfrastructureManager, log_client: gcp_logging.Client, tmp_path):
    """Test complete end-to-end cloud logging flow with real session context."""

    # Integration test must fail if cloud logging not properly configured
    assert bm._logger_cfg is not None, "Logger configuration must be available for integration tests"
    assert bm._logger_cfg.type == "gcp", "GCP cloud logging must be configured for integration tests"

    # Create a unique test identifier for log verification
    test_run_id = f"test-{uuid.uuid4().hex[:8]}"

    # Create additional BM session for testing (using the same infrastructure)
    test_session = infrastructure.create_session_bm(
        name=f"cloud-logging-test-{test_run_id}",
        job="integration-testing",
        batch_id=f"batch-{test_run_id}",
        platform="pytest",
        save_dir_base=str(tmp_path),
    )

    # Verify BM session has cloud logging configured
    assert test_session._logger_cfg is not None
    assert test_session._logger_cfg.type == "gcp"
    assert test_session._cloud_manager is not None

    # Generate test log messages with session context
    test_messages = [
        f"Test message 1 from session {test_session.session_info.session_id}",
        f"Test message 2 with batch {test_session.session_info.batch_id}",
        f"Test structured message {test_run_id}",
    ]

    # Log messages using the configured logger
    for msg in test_messages:
        logger.info(msg, test_run_id=test_run_id)

    # Allow time for logs to propagate to GCP
    time.sleep(5)

    # Verify logs appeared in GCP Cloud Logging
    _verify_logs_in_gcp(log_client, test_run_id, test_session.session_info.session_id, test_session.session_info.batch_id, test_messages)


def test_multiple_sessions_isolated_logging(bm: BM, infrastructure: InfrastructureManager, log_client: gcp_logging.Client, tmp_path):
    """Test that multiple BM sessions have isolated but proper cloud logging."""

    # Integration test must fail if cloud logging not properly configured
    assert bm._logger_cfg is not None, "Logger configuration must be available for integration tests"
    assert bm._logger_cfg.type == "gcp", "GCP cloud logging must be configured for integration tests"

    test_run_id = f"multi-test-{uuid.uuid4().hex[:8]}"

    # Create two separate BM sessions
    session1 = infrastructure.create_session_bm(
        name=f"session1-{test_run_id}", job="multi-session-test", platform="pytest", save_dir_base=str(tmp_path / "session1")
    )

    session2 = infrastructure.create_session_bm(
        name=f"session2-{test_run_id}", job="multi-session-test", platform="pytest", save_dir_base=str(tmp_path / "session2")
    )

    # Both should have cloud logging configured
    assert session1._logger_cfg is not None
    assert session2._logger_cfg is not None
    assert session1.session_info.session_id != session2.session_info.session_id

    # Log from each session
    logger.info(f"Message from session 1: {test_run_id}", session_marker="session1")
    logger.info(f"Message from session 2: {test_run_id}", session_marker="session2")

    time.sleep(5)

    # Verify both sessions' logs appear with correct session context
    entries = _get_log_entries(log_client, test_run_id)

    session1_entries = [e for e in entries if "session1" in str(e.payload)]
    session2_entries = [e for e in entries if "session2" in str(e.payload)]

    assert len(session1_entries) > 0, "Session 1 logs not found"
    assert len(session2_entries) > 0, "Session 2 logs not found"


def test_structured_json_format_consistency(bm: BM, infrastructure: InfrastructureManager, log_client: gcp_logging.Client, tmp_path):
    """Test that cloud logs use consistent structured JSON format."""

    # Integration test must fail if cloud logging not properly configured
    assert bm._logger_cfg is not None, "Logger configuration must be available for integration tests"
    assert bm._logger_cfg.type == "gcp", "GCP cloud logging must be configured for integration tests"

    test_run_id = f"json-test-{uuid.uuid4().hex[:8]}"

    # Create a test session to enable proper cloud logging context
    infrastructure.create_session_bm(
        name=f"json-format-test-{test_run_id}", job="json-format-testing", platform="pytest", save_dir_base=str(tmp_path)
    )

    # Log structured data
    logger.info(
        "Structured test message", test_run_id=test_run_id, custom_field="custom_value", numeric_field=EXPECTED_NUMERIC_FIELD, boolean_field=True
    )

    time.sleep(5)

    # Verify structured format in cloud logs
    entries = _get_log_entries(log_client, test_run_id)
    assert len(entries) > 0, "No log entries found"

    # Check that log entries contain expected structured data
    for entry in entries:
        payload = entry.payload
        if isinstance(payload, dict):
            # Verify JSON structure
            assert "timestamp" in payload or "ts" in payload
            assert "level" in payload
            assert "event" in payload or "message" in payload

            # Verify custom fields are preserved
            if test_run_id in str(payload):
                assert payload.get("test_run_id") == test_run_id
                assert payload.get("custom_field") == "custom_value"
                assert payload.get("numeric_field") == EXPECTED_NUMERIC_FIELD
                assert payload.get("boolean_field") is True


def test_cloud_logging_error_handling(bm: BM, infrastructure: InfrastructureManager, tmp_path):
    """Test that cloud logging setup failures are handled gracefully."""

    # Integration test must fail if cloud logging not properly configured
    assert bm._logger_cfg is not None, "Logger configuration must be available for integration tests"
    assert bm._logger_cfg.type == "gcp", "GCP cloud logging must be configured for integration tests"

    test_run_id = f"error-test-{uuid.uuid4().hex[:8]}"

    # Mock CloudLoggingHandler to fail during setup
    with patch("google.cloud.logging_v2.handlers.CloudLoggingHandler") as mock_handler:
        mock_handler.side_effect = Exception("Simulated GCP failure")

        # BM session creation should still succeed
        test_session = infrastructure.create_session_bm(
            name=f"error-test-{test_run_id}", job="error-handling-test", platform="pytest", save_dir_base=str(tmp_path)
        )

        # Session should still be functional
        assert test_session.session_info.session_id is not None
        assert test_session._logger_cfg is not None  # Config still passed

        # Logging should still work (local only)
        logger.info(f"Test message after cloud logging failure: {test_run_id}")


# Helper functions


def _get_log_entries(log_client: gcp_logging.Client, test_run_id: str) -> list[Any]:
    """Helper to retrieve log entries for a test run."""
    # Query for recent entries containing our test run ID
    filter_str = f'jsonPayload.test_run_id="{test_run_id}" OR textPayload:"{test_run_id}"'

    entries = list(log_client.list_entries(filter_=filter_str, order_by=gcp_logging.DESCENDING, max_results=100))

    return entries


def _verify_logs_in_gcp(log_client: gcp_logging.Client, test_run_id: str, session_id: str, batch_id: str | None, expected_messages: list[str]):
    """Verify that expected log messages appear in GCP with correct session context."""

    entries = _get_log_entries(log_client, test_run_id)

    assert len(entries) > 0, f"No log entries found for test run {test_run_id}"

    # Verify session context is present in logs
    session_context_found = False
    for entry in entries:
        payload = entry.payload
        if isinstance(payload, dict):
            # Check for session context in JSON payload
            if (
                payload.get("session_id")
                and session_id[:12] in payload.get("session_id", "")
                or payload.get("batch_id")
                and batch_id
                and batch_id[:12] in payload.get("batch_id", "")
            ):
                session_context_found = True
                break
        elif isinstance(payload, str):
            # Check for session context in text payload
            if session_id[:12] in payload or (batch_id and batch_id[:12] in payload):
                session_context_found = True
                break

    assert session_context_found, f"Session context not found in logs. Session: {session_id}, Batch: {batch_id}"

    # Verify expected messages are present
    found_messages = set()
    for entry in entries:
        payload_str = str(entry.payload)
        for msg in expected_messages:
            if msg in payload_str:
                found_messages.add(msg)

    missing_messages = set(expected_messages) - found_messages
    assert not missing_messages, f"Missing expected messages in cloud logs: {missing_messages}"


@pytest.mark.integration
@pytest.mark.anyio
async def test_async_cloud_logging_performance(infrastructure: InfrastructureManager):
    """Test cloud logging performance under concurrent session creation."""

    # This test ensures cloud logging doesn't become a bottleneck
    # when multiple sessions are created rapidly

    test_run_id = f"perf-test-{uuid.uuid4().hex[:8]}"

    async def create_and_log_session(session_num: int):
        """Create a session and log a message."""
        test_session = infrastructure.create_session_bm(name=f"perf-session-{session_num}", job=f"perf-test-{test_run_id}", platform="pytest-async")

        logger.info(f"Performance test message from session {session_num}: {test_run_id}")
        return test_session.session_info.session_id

    # Create multiple sessions concurrently
    start_time = time.time()

    NUM_SESSIONS = 10
    tasks = [create_and_log_session(i) for i in range(NUM_SESSIONS)]
    session_ids = await asyncio.gather(*tasks)

    end_time = time.time()
    duration = end_time - start_time

    # Verify all sessions were created successfully
    assert len(session_ids) == NUM_SESSIONS
    assert len(set(session_ids)) == NUM_SESSIONS  # All unique

    # Performance should be reasonable (less than 30 seconds for sessions)
    MAX_DURATION = 30.0
    assert duration < MAX_DURATION, f"Session creation took too long: {duration:.2f}s"

    logger.info(f"Created {NUM_SESSIONS} sessions with cloud logging in {duration:.2f}s")
