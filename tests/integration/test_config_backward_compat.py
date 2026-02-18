"""Test backward compatibility for ButtermilkConfig properties.

This module tests that deprecated/moved config properties still work
for backward compatibility with existing code.
"""

import pytest


def test_pipeline_property_backward_compatibility(real_bm):
    """Test that cfg.pipeline fails fast when run.pipeline is None.

    ARRANGE: Use real ButtermilkConfig from testing.yaml via real_bm fixture
    ACT: Access cfg.pipeline property when run.pipeline is None
    ASSERT: Raises AttributeError with helpful message

    This test verifies fail-fast behavior - previously pipeline was at root level,
    now it's under run config. The cfg.pipeline property should fail explicitly
    when run.pipeline is None (not set for current mode) rather than returning None.

    When run.pipeline has a value (mode=pipeline), cfg.pipeline should return it.
    When run.pipeline is None, cfg.pipeline should raise AttributeError with guidance
    to use pipelines.{pipeline_name} instead.
    """
    # ARRANGE: Get real config from fixture
    cfg = real_bm.cfg

    # ASSERT: testing.yaml uses mode=console, so run.pipeline is None
    assert cfg.run.pipeline is None, "testing.yaml should have run.pipeline=None"

    # ACT & ASSERT: Accessing cfg.pipeline should raise AttributeError
    with pytest.raises(
        AttributeError,
        match=r"run\.pipeline is None.*pipelines\.\{pipeline_name\}",
    ):
        _ = cfg.pipeline
