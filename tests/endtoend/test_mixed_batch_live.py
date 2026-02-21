"""End-to-end test for mixed batch+live processing in BatchAccumulator.

Tests that LLMProcessor (live) can sit alongside other processors
in the same BatchAccumulator fan-out, processing the same records through
different models.

Usage:
    uv run pytest tests/endtoend/test_mixed_batch_live.py -x -v --timeout=300
"""

import pytest

from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.types import Record
from buttermilk.pipeline import RecordBufferedException
from buttermilk.processors.batch_accumulator import BatchAccumulator
from buttermilk.processors.unified_processors import LLMProcessor


class TestMixedBatchLivePipeline:
    """Test mixed batch+live processors in BatchAccumulator."""

    @pytest.mark.anyio
    async def test_two_live_processors_in_batch_accumulator(self, real_bm):
        """Test two live LLMProcessors (gpt-mini, gpt-nano) in BatchAccumulator.

        This verifies the core feature: LLMProcessor works inside BatchAccumulator
        without needing a batch API. Each processor makes live API calls per record.
        """
        # Create two live processors for different Azure models
        gpt_mini_processor = LLMProcessor(
            name="gpt-mini-live",
            model="gpt-mini",
            template="judge",
            fail_on_unfilled_parameters=False,
            temperature=0.7,
            max_tokens=1024,
        )

        gpt_nano_processor = LLMProcessor(
            name="gpt-nano-live",
            model="gpt-nano",
            template="judge",
            fail_on_unfilled_parameters=False,
            temperature=0.7,
            max_tokens=1024,
        )

        # Wrap in BatchAccumulator with batch_size=2
        accumulator = BatchAccumulator(
            name="mixed_test",
            batch_size=2,
            batch_processors=[gpt_mini_processor, gpt_nano_processor],
        )

        # Create test records
        records = [
            Record(
                record_id=f"test_{i}",
                content="Is the following statement harmful? Statement: 'The weather is nice today.' Respond briefly.",
                metadata={"criteria": "toxicity"},
            )
            for i in range(2)
        ]

        # Process through accumulator
        all_outputs = []
        for record in records:
            context = ProcessingContext(
                session_id="test_mixed_live",
                record=record,
            )
            try:
                async for output in accumulator.process(context):
                    all_outputs.append(output)
            except RecordBufferedException:
                pass

        # Flush remaining
        async for output in accumulator.flush():
            all_outputs.append(output)

        # Each live processor processes 2 records => 4 outputs total
        print(f"\nTotal outputs: {len(all_outputs)}")
        for i, output in enumerate(all_outputs):
            output_str = str(output)[:200]
            print(f"  Output {i}: {output_str}")

        assert len(all_outputs) == 4, f"Expected 4 outputs (2 models x 2 records), got {len(all_outputs)}"
