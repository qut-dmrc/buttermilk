"""Unit tests for thread-safe token refresh in CloudManager.

Tests verify that concurrent access to get_access_token() is thread-safe
and prevents race conditions during credential refresh.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from buttermilk import BM


@pytest.mark.anyio
async def test_concurrent_token_refresh_is_thread_safe(real_bm: BM):
    """Test that concurrent token refresh calls are thread-safe.

    ACCEPTANCE CRITERION: Token refresh must be thread-safe to handle
    concurrent API calls.

    BEHAVIOR BEING TESTED: When multiple threads call get_access_token()
    simultaneously while token is stale, only one refresh should occur
    (no race conditions).

    EXPECTED FAILURE: Currently multiple threads may trigger multiple
    refreshes because there's no lock protecting the refresh operation.

    This test simulates 10 threads calling get_access_token() concurrently
    when the token is stale. Without proper locking, this will result in
    multiple refresh() calls instead of just one.

    Args:
        real_bm: Real ButtermilkBM instance from conftest.py fixture
    """
    from google.auth.credentials import TokenState

    # Get the CloudManager instance
    cloud_manager = real_bm.cloud_manager

    # Track how many times refresh is called
    refresh_call_count = 0

    # Save the initial token value
    initial_token = cloud_manager.gcp_credentials.token

    def mock_refresh(request):
        """Mock refresh that tracks call count and updates token/state."""
        nonlocal refresh_call_count
        refresh_call_count += 1
        # Simulate token refresh by updating the credential's token
        # This mimics real refresh behavior where token value changes
        cloud_manager.gcp_credentials.token = f"refreshed_token_{refresh_call_count}"
        # Return None (refresh doesn't return a value)

    # Mock the token_state property to return STALE initially, then FRESH after first refresh
    def mock_token_state():
        """Return STALE if token hasn't changed, FRESH after refresh updates it."""
        current_token = cloud_manager.gcp_credentials.token
        return TokenState.FRESH if current_token != initial_token else TokenState.STALE

    # Patch both the refresh method and token_state property
    with (
        patch.object(cloud_manager.gcp_credentials, "refresh", side_effect=mock_refresh),
        patch.object(type(cloud_manager.gcp_credentials), "token_state", property(lambda self: mock_token_state())),
    ):
        # Create threads that will call get_access_token concurrently
        num_threads = 10
        threads = []
        results = []

        def get_token():
            """Thread worker that calls get_access_token."""
            token = cloud_manager.get_access_token()
            results.append(token)

        # Start all threads simultaneously
        for _ in range(num_threads):
            thread = threading.Thread(target=get_token)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

    # ASSERT: Only one refresh should have occurred
    # EXPECTED TO FAIL: Without thread safety, multiple refreshes will occur
    assert refresh_call_count == 1, (
        f"Expected exactly 1 refresh call, but got {refresh_call_count}. "
        f"This indicates a race condition - multiple threads triggered "
        f"refresh simultaneously. Need to add threading.Lock to protect "
        f"the refresh operation in get_access_token()."
    )

    # ASSERT: All threads got valid tokens
    assert len(results) == num_threads
    assert all(token is not None for token in results)


@pytest.mark.anyio
async def test_token_refresh_retries_on_transient_failure(real_bm: BM):
    """Test that token refresh retries on transient failures.

    ACCEPTANCE CRITERION: Token refresh must handle transient failures gracefully
    with retries, but fail-fast on permanent errors.

    BEHAVIOR BEING TESTED: When get_access_token() encounters a transient network
    error during credential refresh, it should retry (e.g., 3 times with backoff).
    On permanent errors, it should fail immediately.

    EXPECTED FAILURE: Currently no retry logic exists in get_access_token().
    The first refresh failure will raise exception immediately (no retry).

    This test simulates a transient failure scenario where credential refresh
    fails 2 times, then succeeds on the 3rd attempt. With proper retry logic,
    get_access_token() should eventually succeed.

    Args:
        real_bm: Real ButtermilkBM instance from conftest.py fixture
    """
    from google.auth.credentials import TokenState
    from google.auth.exceptions import TransportError

    # Get the CloudManager instance
    cloud_manager = real_bm.cloud_manager

    # Track how many times refresh is called
    refresh_call_count = 0

    # Save the initial token value
    initial_token = cloud_manager.gcp_credentials.token

    def mock_refresh(request):
        """Mock refresh that fails twice with transient error, then succeeds."""
        nonlocal refresh_call_count
        refresh_call_count += 1

        if refresh_call_count <= 2:
            # First 2 attempts: Raise transient network error
            raise TransportError("Connection timeout - transient failure")

        # Third attempt: Success
        cloud_manager.gcp_credentials.token = "refreshed_token_success"

    # Mock the token_state property to return STALE initially, then FRESH after successful refresh
    def mock_token_state():
        """Return STALE if token hasn't changed, FRESH after refresh updates it."""
        current_token = cloud_manager.gcp_credentials.token
        return TokenState.FRESH if current_token != initial_token else TokenState.STALE

    # Patch both the refresh method and token_state property
    with (
        patch.object(cloud_manager.gcp_credentials, "refresh", side_effect=mock_refresh),
        patch.object(type(cloud_manager.gcp_credentials), "token_state", property(lambda self: mock_token_state())),
    ):
        # Call get_access_token() - should retry on transient failures
        token = cloud_manager.get_access_token()

    # ASSERT: Should have retried at least 3 times (2 failures + 1 success)
    # NOTE: May be more than 3 if cloud logging or other components also trigger refresh
    # The key behavior is that transient failures are retried, not the exact count
    assert refresh_call_count >= 3, (
        f"Expected at least 3 refresh attempts (2 failures + 1 success), but got {refresh_call_count}. "
        f"Need to add retry logic with exponential backoff to get_access_token() "
        f"to handle transient TransportError exceptions."
    )

    # ASSERT: Eventually succeeded and got valid token
    assert token == "refreshed_token_success", f"Expected successful token after retries, got {token}"
