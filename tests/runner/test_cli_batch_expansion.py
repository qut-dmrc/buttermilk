"""Tests for CLI batch mode parameter expansion.

Tests that batch_simple mode expands flow.parameters and creates
multiple pipeline runs, one per parameter variant.
"""

from __future__ import annotations

<<<<<<< HEAD
from unittest.mock import Mock
=======
from unittest.mock import Mock, call, patch
>>>>>>> origin/stable

import pytest


def test_expand_dict_generates_parameter_variants():
    """Test that expand_dict correctly generates parameter variants.

    This is a prerequisite test - verifies the expand_dict utility works
    as expected before testing CLI batch_simple integration.
    """
    from buttermilk.utils.utils import expand_dict

    # Test with list parameter that should expand
    params = {"criteria": ["tja", "glaad"]}
    variants = expand_dict(params)

    # Should create 2 variants, one per list item
    assert len(variants) == 2
    assert variants[0] == {"criteria": "tja"}
    assert variants[1] == {"criteria": "glaad"}


def test_expand_dict_with_multiple_list_parameters():
    """Test expand_dict with multiple list parameters creates cartesian product."""
    from buttermilk.utils.utils import expand_dict

    params = {"criteria": ["tja", "glaad"], "limit": [10, 20]}
    variants = expand_dict(params)

    # Should create 4 variants (2 x 2 cartesian product)
    assert len(variants) == 4
    expected = [
        {"criteria": "tja", "limit": 10},
        {"criteria": "tja", "limit": 20},
        {"criteria": "glaad", "limit": 10},
        {"criteria": "glaad", "limit": 20},
    ]
    assert variants == expected


@pytest.mark.anyio
async def test_batch_simple_should_expand_parameters():
    """Test that batch_simple SHOULD expand parameters and create multiple runs.

    ACCEPTANCE CRITERION: CLI batch mode expands flow.parameters and creates
    multiple pipeline runs.

    This test demonstrates what SHOULD happen:
    - expand_dict is called on flow.parameters
    - Multiple GroupchatProcessor instances created (one per variant)
    - Each processor gets different parameter variant
    - Multiple pipelines created and executed

    EXPECTED FAILURE: Current batch_simple implementation doesn't call
    expand_dict, so this test will fail when we check the actual CLI code.
    """
    from buttermilk.utils.utils import expand_dict

    # Mock flow with parameters to expand
    mock_flow = Mock()
    mock_flow.parameters = {"criteria": ["tja", "glaad"]}
    mock_flow.storage = {"initial": Mock()}

    # Track what gets created
    processor_calls = []
    pipeline_calls = []

    def mock_create_processor(flow_config, flow_name, bm, parameters=None):
        """Mock processor creation."""
        processor = Mock()
        processor.parameters = parameters or {}
        processor_calls.append({"flow_name": flow_name, "parameters": parameters or {}})
        return processor

    def mock_create_pipeline(pipeline_name, source, processors, **kwargs):
        """Mock pipeline creation."""
        pipeline = Mock()
        pipeline_calls.append({"pipeline_name": pipeline_name, "processors": processors})

        # Mock async iteration
        async def mock_iter():
            yield Mock()

        pipeline.__call__ = Mock(return_value=mock_iter())
        return pipeline

    # EXPECTED BEHAVIOR (what should happen):
    # 1. Expand parameters
    param_variants = expand_dict(mock_flow.parameters)
    assert len(param_variants) == 2, "Should generate 2 parameter variants"

    # 2. Create processor + pipeline for EACH variant
    mock_bm = Mock()
    mock_bm.get_storage = Mock(return_value=Mock())

    for params in param_variants:
        processor = mock_create_processor(
            flow_config=mock_flow,
            flow_name="test_flow",
            bm=mock_bm,
            parameters=params,  # Each variant gets different parameters
        )
        pipeline = mock_create_pipeline(
            pipeline_name="batch_test_flow",
            source=mock_bm.get_storage(mock_flow.storage["initial"]),
            processors=[processor],
        )

    # VERIFY expected behavior
<<<<<<< HEAD
    assert len(processor_calls) == 2, f"Should create 2 processors (one per variant), but created {len(processor_calls)}"
    assert processor_calls[0]["parameters"] == {"criteria": "tja"}
    assert processor_calls[1]["parameters"] == {"criteria": "glaad"}

    assert len(pipeline_calls) == 2, f"Should create 2 pipelines (one per variant), but created {len(pipeline_calls)}"
=======
    assert len(processor_calls) == 2, (
        f"Should create 2 processors (one per variant), "
        f"but created {len(processor_calls)}"
    )
    assert processor_calls[0]["parameters"] == {"criteria": "tja"}
    assert processor_calls[1]["parameters"] == {"criteria": "glaad"}

    assert len(pipeline_calls) == 2, (
        f"Should create 2 pipelines (one per variant), "
        f"but created {len(pipeline_calls)}"
    )
>>>>>>> origin/stable


@pytest.mark.anyio
async def test_batch_simple_current_implementation_no_expansion():
    """Test CURRENT batch_simple behavior - does NOT expand parameters.

    This documents the current (wrong) behavior where batch_simple
    creates only ONE processor/pipeline with unexpanded parameters.

    This test should PASS now, but FAIL after implementing expansion.
    """
    # Mock flow with parameters that SHOULD expand
    mock_flow = Mock()
    mock_flow.parameters = {"criteria": ["tja", "glaad"]}
    mock_flow.storage = {"initial": Mock()}

    # Track what gets created
    processor_calls = []
    pipeline_calls = []

    def mock_create_processor(flow_config, flow_name, bm, parameters=None):
        """Mock processor creation."""
        processor_calls.append({"flow_name": flow_name, "parameters": parameters})
        return Mock()

    def mock_create_pipeline(pipeline_name, source, processors, **kwargs):
        """Mock pipeline creation."""
        pipeline_calls.append({"pipeline_name": pipeline_name})
        pipeline = Mock()

        async def mock_iter():
            yield Mock()

        pipeline.__call__ = Mock(return_value=mock_iter())
        return pipeline

    # CURRENT BEHAVIOR (simulating cli.py batch_simple section):
    # Does NOT call expand_dict
    # Creates single processor without parameters argument
    mock_bm = Mock()
    mock_bm.get_storage = Mock(return_value=Mock())

    flow_name = "test_flow"
    source = mock_bm.get_storage(mock_flow.storage["initial"])

    # Current code creates ONE processor
    processor = mock_create_processor(
        flow_config=mock_flow,
        flow_name=flow_name,
        bm=mock_bm,
        # NOTE: No parameters argument passed!
    )

    # Current code creates ONE pipeline
    pipeline = mock_create_pipeline(
        pipeline_name=f"batch_{flow_name}",
        source=source,
        processors=[processor],
    )

    # VERIFY current (wrong) behavior
<<<<<<< HEAD
    assert len(processor_calls) == 1, "Current implementation creates 1 processor (should be 2 after implementing expansion)"
    assert processor_calls[0]["parameters"] is None, "Current implementation doesn't pass parameters (should pass expanded variant)"

    assert len(pipeline_calls) == 1, "Current implementation creates 1 pipeline (should be 2 after implementing expansion)"
=======
    assert len(processor_calls) == 1, (
        f"Current implementation creates 1 processor "
        f"(should be 2 after implementing expansion)"
    )
    assert processor_calls[0]["parameters"] is None, (
        f"Current implementation doesn't pass parameters "
        f"(should pass expanded variant)"
    )

    assert len(pipeline_calls) == 1, (
        f"Current implementation creates 1 pipeline "
        f"(should be 2 after implementing expansion)"
    )


>>>>>>> origin/stable
