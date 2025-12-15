"""Configuration for end-to-end tests.

This file automatically applies markers to all tests in the directory.
"""

import pytest

# Apply markers to all tests in this directory
# End-to-end tests need longer timeout (240s) than unit tests (60s default)
pytestmark = [pytest.mark.endtoend, pytest.mark.timeout(240)]
