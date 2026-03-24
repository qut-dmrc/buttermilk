"""Integration tests for LLMs infrastructure functionality.

These tests verify that LLMs are properly configured and accessible.
"""

import pytest

pytestmark = pytest.mark.slow

from buttermilk import BM


def test_config_llms(real_bm: BM):
    """Test that LLMs are properly configured and accessible."""
    models = real_bm.llms
    assert models
    # Verify we can access at least one model
    assert len(models) > 0
