"""Configuration for integration tests.

This file automatically applies markers to all tests in the integration/ directory.
"""

import pytest

# Apply markers to all tests in this directory
pytestmark = [
    pytest.mark.integration,
    pytest.mark.endtoend,  # Since these are more properly "live tests"
]


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
