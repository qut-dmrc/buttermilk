"""Configuration for integration tests.

This file automatically applies markers to all tests in the integration/ directory.
"""

import inspect

import pytest

# Apply markers to all tests in this directory
pytestmark = [
    pytest.mark.anyio,
    pytest.mark.integration,
    pytest.mark.live_api,  # Since these are more properly "live tests"
]


# Ensure async tests get the anyio marker automatically,
# while sync tests run normally without any interference.
def pytest_collection_modifyitems(items):
    for item in items:
        # Check if the test function is async
        if inspect.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.anyio)
        # Apply all other markers defined in pytestmark
        for marker in pytestmark:
            item.add_marker(marker)


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
