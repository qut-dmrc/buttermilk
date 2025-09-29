"""Configuration for end-to-end tests.

This file automatically applies markers to all tests in the directory.
"""

import pytest

# Apply markers to all tests in this directory
pytestmark = pytest.mark.endtoend

