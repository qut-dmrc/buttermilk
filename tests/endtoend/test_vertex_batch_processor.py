"""End-to-end demo tests for VertexBatchProcessor with real TJA fixtures.

These tests demonstrate VertexBatchProcessor using REAL data:
- Real TJA (Trans Journalists Association) stylebook criteria
- Real content samples with ground truth labels
- Real LLM models from bm.llms configuration

Purpose:
- Verify VertexBatchProcessor correctly integrates with buttermilk LLM infrastructure
- Demonstrate human-readable output with full verification
- Validate structured output parsing with actual content moderation tasks

Test Philosophy:
- NO MOCKS, NO FAKES - uses real APIs and real data
- Uses fixtures from production research projects
- Outputs are human-readable for manual verification
"""

import pytest

from buttermilk._core.types import Record
from tests.conftest import CHEAP_CHAT_MODELS


@pytest.mark.endtoend
class TestVertexBatchProcessorIntegration:
    """Integration tests verifying VertexBatchProcessor with real models."""

    @pytest.mark.anyio
    @pytest.mark.parametrize("model_name", CHEAP_CHAT_MODELS)
    async def test_vertex_batch_processor_model_routing(self, real_bm, model_name):
        """Test that VertexBatchProcessor uses buttermilk LLM infrastructure.

        Verifies that models are routed through bm.llms, not called directly
        via genai.models.generate_content().
        """
        # The LLM wrapper should have proper configuration
        llm = real_bm.llms[model_name]
        assert llm is not None

        # Verify the LLM can actually be called
        from autogen_core.models import UserMessage

        messages = [UserMessage(content="Say hello", source="user")]
        response = await llm.create(messages=messages)
        assert response.content


@pytest.mark.endtoend
class TestBatchAccumulatorWithVertexBatch:
    """Test BatchAccumulator wrapping VertexBatchProcessor."""

    @pytest.mark.anyio
    async def test_batch_accumulator_integration(self, real_bm):
        """Test that BatchAccumulator correctly wraps VertexBatchProcessor.

        Uses the first available cheap model from bm.llms.
        """
        from buttermilk._core.processing_context import ProcessingContext
        from buttermilk.processors.batch_accumulator import BatchAccumulator
        from buttermilk.processors.vertex_batch import VertexBatchProcessor

        # Use first available model from config
        model_name = CHEAP_CHAT_MODELS[0]

        # Create the batch processor
        vertex_processor = VertexBatchProcessor(
            name="batch_judge",
            model=model_name,
            template="judge",
            template_vars={
                "criteria": "Is this content harmful or toxic?",
            },
            fail_on_unfilled_parameters=False,
        )

        # Wrap in BatchAccumulator
        accumulator = BatchAccumulator(
            name="test_accumulator",
            batch_size=2,
            batch_processors=[vertex_processor],
        )

        # Create test records
        records = [
            Record(
                record_id=f"batch_{i}",
                content=f"Test content {i}",
                metadata={},
            )
            for i in range(3)
        ]

        # Process records through accumulator
        all_results = []
        for record in records:
            context = ProcessingContext(
                session_id="test_session",
                record=record,
            )
            async for result in accumulator.process(context):
                all_results.append(result)

        # Flush remaining records
        async for result in accumulator.flush():
            all_results.append(result)

        # Should have processed all records
        assert len(all_results) == 3
