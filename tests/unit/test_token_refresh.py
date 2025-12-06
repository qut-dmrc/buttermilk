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


@pytest.mark.anyio
async def test_litellm_create_uses_token_provider_for_auth():
    """Test that LiteLLMWrapper.create() calls token_provider dynamically.

    ACCEPTANCE CRITERION: LiteLLMWrapper.create() must call token_provider()
    dynamically for each API call, not use a static token.

    BEHAVIOR BEING TESTED: When create() is called, if token_provider is set,
    it should be invoked to get a fresh token and that token should be used
    in the Authorization header passed to litellm.acompletion.

    EXPECTED FAILURE: Currently create() uses self.extra_headers statically
    at line 1415, so token_provider is never called. This test will fail
    because token_provider_call_count remains 0.

    This test mocks litellm.acompletion to avoid API costs and tracks:
    1. How many times token_provider is called
    2. That the token from token_provider appears in the Authorization header
    3. That calling create() twice results in two token_provider calls
    """
    from buttermilk._core.llms import LiteLLMWrapper
    from autogen_core.models import UserMessage

    # Track token_provider calls
    token_provider_call_count = 0

    def mock_token_provider() -> str:
        """Mock token provider that returns incrementing tokens."""
        nonlocal token_provider_call_count
        token_provider_call_count += 1
        return f"fresh_token_{token_provider_call_count}"

    # Mock litellm.acompletion to capture the headers passed
    captured_headers = []

    async def mock_acompletion(**kwargs):
        """Mock acompletion that captures headers."""
        # Capture the extra_headers if provided
        if "extra_headers" in kwargs:
            captured_headers.append(kwargs["extra_headers"])
        # Return a minimal valid response structure that matches litellm format
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message = Mock(content="test response", tool_calls=None)
        mock_choice.finish_reason = "stop"  # Must be a valid finish_reason
        mock_response.choices = [mock_choice]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_response.cached = False  # Boolean, not Mock
        mock_response.model = "test-model"
        return mock_response

    # Patch litellm.acompletion where it's imported in llms.py
    with patch("buttermilk._core.llms.acompletion", new=mock_acompletion):
        # Create LiteLLMWrapper with token_provider
        # Must create inside patch context so it doesn't try to validate with real litellm
        model_info = {"family": "test", "structured_output": False}
        wrapper = LiteLLMWrapper(
            model="test-model",
            model_info=model_info,
            litellm_model_name="openai/test-model",  # Use valid litellm format
            token_provider=mock_token_provider,
            base_url="https://test.openai.azure.com",  # Need base_url to trigger extra_headers path
            api_key="fake-api-key",  # Provide api_key to avoid auth errors
        )

        # Call create() first time
        messages = [UserMessage(content="test message", source="user")]
        await wrapper.create(messages=messages)

        # Call create() second time
        await wrapper.create(messages=messages)

    # ASSERT: token_provider should have been called twice (once per create() call)
    # EXPECTED TO FAIL: Currently token_provider is never called because
    # create() uses self.extra_headers statically instead of calling token_provider
    assert token_provider_call_count == 2, (
        f"Expected token_provider to be called 2 times (once per create() call), "
        f"but it was called {token_provider_call_count} times. "
        f"Need to modify create() to call token_provider() dynamically instead of "
        f"using self.extra_headers statically."
    )

    # ASSERT: The token from token_provider should appear in the Authorization header
    assert len(captured_headers) == 2, "Should have captured headers from both create() calls"

    # First call should use fresh_token_1
    assert "Authorization" in captured_headers[0], "Authorization header missing from first call"
    assert "fresh_token_1" in captured_headers[0]["Authorization"], (
        f"Expected fresh_token_1 in Authorization header, "
        f"got: {captured_headers[0].get('Authorization')}"
    )

    # Second call should use fresh_token_2
    assert "Authorization" in captured_headers[1], "Authorization header missing from second call"
    assert "fresh_token_2" in captured_headers[1]["Authorization"], (
        f"Expected fresh_token_2 in Authorization header, "
        f"got: {captured_headers[1].get('Authorization')}"
    )
