"""Unit tests for thread-safe token refresh in CloudManager.

Tests verify that concurrent access to get_access_token() is thread-safe
and prevents race conditions during credential refresh.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

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
        return None

    # Mock the token_state property to return STALE initially, then FRESH after first refresh
    def mock_token_state():
        """Return STALE if token hasn't changed, FRESH after refresh updates it."""
        current_token = cloud_manager.gcp_credentials.token
        return TokenState.FRESH if current_token != initial_token else TokenState.STALE

    # Patch both the refresh method and token_state property
    with patch.object(
        cloud_manager.gcp_credentials,
        'refresh',
        side_effect=mock_refresh
    ), patch.object(
        type(cloud_manager.gcp_credentials),
        'token_state',
        property(lambda self: mock_token_state())
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
async def test_litellm_wrapper_accepts_token_provider():
    """Test that LiteLLMWrapper accepts a token_provider callable.

    ACCEPTANCE CRITERION: LiteLLMWrapper must accept a token_provider callable
    that returns fresh tokens on each API call.

    BEHAVIOR BEING TESTED: LiteLLMWrapper should have a `token_provider` field
    that accepts a callable. When provided, this callable should be invoked to
    get fresh tokens instead of using a static `extra_headers`.

    EXPECTED FAILURE: Currently LiteLLMWrapper doesn't have a token_provider
    field, so instantiation with this parameter will raise ValidationError.

    This test verifies that we can instantiate LiteLLMWrapper with a
    token_provider callable, which will be used later to refresh tokens
    dynamically on each API call.
    """
    from buttermilk._core.llms import LiteLLMWrapper
    from pydantic import ValidationError

    # Define a simple token provider callable
    def get_fresh_token() -> str:
        """Mock token provider that returns a fresh token."""
        return "fresh_token_123"

    # Create minimal required parameters
    model_info = {"family": "test", "structured_output": False}

    # EXPECTED TO FAIL: token_provider field doesn't exist yet
    # Pydantic allows extra fields by default but doesn't store them
    wrapper = LiteLLMWrapper(
        model="test-model",
        model_info=model_info,
        litellm_model_name="test/model",
        token_provider=get_fresh_token,  # This field doesn't exist yet
    )

    # ASSERT: Verify token_provider field exists and is set correctly
    # This will FAIL because LiteLLMWrapper doesn't have token_provider field yet
    assert hasattr(wrapper, "token_provider"), (
        "LiteLLMWrapper missing token_provider field. "
        "Need to add: token_provider: Callable[[], str] | None = Field(default=None) "
        "to LiteLLMWrapper class definition."
    )

    assert wrapper.token_provider == get_fresh_token, (
        f"token_provider was not set correctly. "
        f"Expected {get_fresh_token}, got {getattr(wrapper, 'token_provider', None)}"
    )
