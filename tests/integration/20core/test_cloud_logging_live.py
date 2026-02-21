"""Integration tests for cloud logging configuration and functionality.

These tests verify that cloud logging is properly configured and works correctly
in async contexts. We test our code's behavior, not external GCP service availability.

All tests use the standard real_bm and real_logger fixtures from conftest.py.
"""

import asyncio
import time
import uuid

import pytest


def test_cloud_logging_configured(real_bm):
    """Verify cloud logging is properly configured via real_bm fixture."""
    # Verify logger configuration is available
    assert real_bm._logger_cfg is not None, "Logger configuration must be available"

    # Verify cloud manager is configured
    assert real_bm.cloud_manager is not None, "Cloud manager must be available"
<<<<<<< HEAD
    assert len(real_bm.cloud_manager.clouds) > 0, "At least one cloud must be configured"
=======
    assert len(real_bm.cloud_manager.clouds) > 0, (
        "At least one cloud must be configured"
    )
>>>>>>> origin/stable

    # Verify first cloud has required fields
    cloud = real_bm.cloud_manager.clouds[0]
    assert cloud.project_id is not None, "Cloud project_id must be set"
    assert cloud.location is not None, "Cloud location must be set"


def test_logger_accepts_structured_data(real_logger):
    """Verify logger can handle structured JSON fields."""
    test_run_id = f"struct-test-{uuid.uuid4().hex[:8]}"

    # Log with various structured data types - should not raise
    real_logger.info(
        "Structured logging test",
        test_run_id=test_run_id,
        custom_field="custom_value",
        numeric_field=42,
        boolean_field=True,
        dict_field={"nested": "data"},
    )

    # Log warning with structured data - should not raise
    real_logger.warning("Warning with context", test_id=test_run_id, severity="medium")


@pytest.mark.anyio
async def test_async_logging_performance(real_logger):
    """Verify logging works correctly in async contexts without blocking.

    This test ensures cloud logging doesn't become a bottleneck
    during concurrent async operations.
    """
    test_run_id = f"async-test-{uuid.uuid4().hex[:8]}"

    async def log_message(msg_num: int):
        """Log a message asynchronously."""
        real_logger.info(
            f"Async test message {msg_num}: {test_run_id}",
            msg_num=msg_num,
            test_run_id=test_run_id,
        )
        await asyncio.sleep(0.01)  # Simulate some async work
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

    # Should complete reasonably quickly (less than 1 second with 0.01s sleeps)
    MAX_DURATION = 2.0
    assert duration < MAX_DURATION, f"Async logging took too long: {duration:.2f}s"

    real_logger.info(f"Logged {NUM_MESSAGES} messages in {duration:.2f}s")
