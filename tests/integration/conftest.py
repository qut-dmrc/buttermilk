"""Configuration for integration tests.

This file automatically applies markers to all tests in the integration/ directory.
"""

import pytest

# Apply markers to all tests in this directory
pytestmark = [
    pytest.mark.integration,
]


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
