"""Integration tests for LLMs infrastructure functionality.

These tests verify that LLMs are properly configured and accessible.
"""

import pytest
from buttermilk import BM


def test_config_llms(bm: BM):
    """Test that LLMs are properly configured and accessible."""
    models = bm.llms
    assert models
    # Verify we can access at least one model
    assert len(models) > 0