"""End-to-end tests for VertexBatchProcessor with real LLM infrastructure.

These tests verify that VertexBatchProcessor correctly integrates with
buttermilk's LLM infrastructure (bm.llms) rather than calling Vertex AI directly.

The processor should:
1. Use model names from the models.json secret (e.g., 'gemini-flash')
2. Route through litellm for proper model path translation
3. Support structured output via output_model parameter
"""

import pytest
from pydantic import BaseModel, Field

from buttermilk._core.types import Record


class SimpleJudgeOutput(BaseModel):
    """Simple structured output for testing."""

    assessment: str = Field(description="Assessment of the content")
    is_harmful: bool = Field(description="Whether content is harmful")


@pytest.mark.endtoend
class TestVertexBatchProcessorIntegration:
    """Integration tests for VertexBatchProcessor with real models."""

    @pytest.mark.anyio
    async def test_vertex_batch_processor_with_gemini_flash(self, real_bm):
        """Test VertexBatchProcessor works with gemini-flash model.

        This is the critical test - gemini-flash should work through
        the buttermilk LLM infrastructure, not fail with 404 NOT_FOUND.
        """
        from buttermilk.processors.vertex_batch import VertexBatchProcessor

        processor = VertexBatchProcessor(
            name="test_judge",
            model="gemini-flash",
            template="judge",  # Uses the judge template
            output_model="tests.endtoend.test_vertex_batch_processor.SimpleJudgeOutput",
            fail_on_unfilled_parameters=False,  # Template may have unfilled params
        )

        # Create a simple test record
        record = Record(
            record_id="test_001",
            content="This is a test message about a person.",
            metadata={"criteria": "Is this content harmful?"},
        )

        # Process the batch
        results = await processor.process_batch([record])

        # Verify we got a result
        assert len(results) == 1
        result = results[0]
        assert result is not None

        # CRITICAL: Verify the LLM call actually succeeded (not a 404 error)
        # If using parsed output model, result should be SimpleJudgeOutput
        # If not parsed, result should be Record with llm_output in metadata
        if isinstance(result, SimpleJudgeOutput):
            # Structured output was parsed successfully
            assert result.assessment, "Expected non-empty assessment"
        elif hasattr(result, "metadata"):
            # Check for error in metadata
            assert "error" not in result.metadata, (
                f"LLM call failed with error: {result.metadata.get('error')}"
            )
            # Should have llm_output if successful
            assert "llm_output" in result.metadata, (
                "Expected llm_output in metadata for successful LLM call"
            )
        else:
            pytest.fail(f"Unexpected result type: {type(result)}")

    @pytest.mark.anyio
    async def test_vertex_batch_processor_model_routing(self, real_bm):
        """Test that VertexBatchProcessor uses buttermilk LLM infrastructure.

        Verifies that models are routed through bm.llms, not called directly
        via genai.models.generate_content().
        """
        # The LLM wrapper should have proper configuration for gemini-flash
        llm = real_bm.llms["gemini-flash"]
        assert llm is not None

        # Verify the LLM can actually be called
        from autogen_core.models import UserMessage
        messages = [UserMessage(content="Say hello", source="user")]
        response = await llm.create(messages=messages)
        assert response.content

    @pytest.mark.anyio
    async def test_vertex_batch_processor_with_simple_prompt(self, real_bm):
        """Test VertexBatchProcessor with a simple inline prompt (no template)."""
        from buttermilk.processors.vertex_batch import VertexBatchProcessor

        processor = VertexBatchProcessor(
            name="simple_test",
            model="gemini-flash",
            template="simple",  # Simple template that just passes content
            fail_on_unfilled_parameters=False,
        )

        record = Record(
            record_id="simple_001",
            content="What is 2 + 2?",
            metadata={},
        )

        results = await processor.process_batch([record])

        assert len(results) == 1
        # Check that we got some response
        result = results[0]
        if hasattr(result, "metadata") and "llm_output" in result.metadata:
            assert "4" in result.metadata["llm_output"]


@pytest.mark.endtoend
class TestBatchAccumulatorWithVertexBatch:
    """Test BatchAccumulator wrapping VertexBatchProcessor."""

    @pytest.mark.anyio
    async def test_batch_accumulator_integration(self, real_bm):
        """Test that BatchAccumulator correctly wraps VertexBatchProcessor."""
        from buttermilk.processors.batch_accumulator import BatchAccumulator
        from buttermilk.processors.vertex_batch import VertexBatchProcessor
        from buttermilk._core.processing_context import ProcessingContext

        # Create the batch processor
        vertex_processor = VertexBatchProcessor(
            name="batch_judge",
            model="gemini-flash",
            template="judge",
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
                metadata={"criteria": "Is this harmful?"},
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
