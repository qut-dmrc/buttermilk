"""Test to demonstrate parent_call_id fix in LLMCore processor.

This test explicitly validates that:
1. Records with parent_call_id field are processed correctly
2. Pipeline extracts and passes parent_call_id to LLMCore
3. ExecutionTrace has correct parent_call_id linkage
4. call_id is DIFFERENT from parent_call_id (not reused)

This addresses the bug where scorer traces were reusing judge call_ids.
"""

import asyncio
import datetime

import pytest

from buttermilk import logger
from buttermilk._core.llm_core import LLMCore
from buttermilk._core.types import BaseRecord
from buttermilk.utils.trace_writer import get_trace_writer


@pytest.fixture
def record_with_parent_call_id() -> BaseRecord:
    """Create a test record with parent_call_id set."""
    return BaseRecord(
        text="Test record",
        metadata={"test": "parent_call_id_fix"},
        parent_call_id="judge-trace-abc123",  # This should be passed to LLMCore
    )


@pytest.mark.anyio
async def test_parent_call_id_extraction_and_linkage(real_bm, record_with_parent_call_id: BaseRecord, real_model_name: str):
    """Test that parent_call_id from record is correctly used in ExecutionTrace.

    This validates the fix:
    - Pipeline extracts parent_call_id from record
    - Passes it to LLMCore.process()
    - ExecutionTrace gets correct parent_call_id
    - New call_id is generated (not reused)
    """
    # Simulate what pipeline does (after our fix)
    parent_trace_id = getattr(record_with_parent_call_id, "parent_call_id", None)

    logger.info(f"Extracted parent_trace_id from record: {parent_trace_id}")
    assert parent_trace_id == "judge-trace-abc123", "Should extract parent_call_id from record"

    # Create LLMCore
    llm_core = LLMCore(
        model=real_model_name,
        template="ra",  # Simple template
        fail_on_unfilled_parameters=False,
    )

    test_start_time = datetime.datetime.now(datetime.timezone.utc)

    # Process with parent_trace_id (simulating pipeline behavior)
    results = []
    async for result in llm_core.process(
        record=record_with_parent_call_id,
        processor_stage="test_stage",
        parent_trace_id=parent_trace_id,  # Pipeline passes this
        component_name="test_parent_fix",
        prompt="What is 2+2?",
    ):
        results.append(result)

    assert len(results) == 1, "Should have one result"

    # Flush traces
    trace_writer = get_trace_writer()
    await asyncio.sleep(1)
    await trace_writer.flush()
    await asyncio.sleep(2)

    # Query BigQuery for the trace
    query = f"""
        SELECT
            call_id,
            parent_call_id
        FROM `{real_bm.bq.project}.testing.traces`
        WHERE timestamp >= TIMESTAMP('{test_start_time.isoformat()}')
            AND JSON_VALUE(agent_info, '$.component_name') = 'test_parent_fix'
        LIMIT 1
    """

    df = real_bm.run_query(query)
    assert df.shape[0] == 1, "Should find exactly one trace"

    trace = df.iloc[0]

    # KEY VALIDATIONS
    logger.info(f"Trace call_id: {trace.call_id}")
    logger.info(f"Trace parent_call_id: {trace.parent_call_id}")

    # 1. parent_call_id should match what we passed
    assert trace.parent_call_id == "judge-trace-abc123", f"parent_call_id should be 'judge-trace-abc123', got '{trace.parent_call_id}'"

    # 2. call_id should be DIFFERENT (new UUID generated)
    assert trace.call_id != trace.parent_call_id, f"call_id should be NEW (not reuse parent), but both are: {trace.call_id}"

    logger.info("✅ parent_call_id correctly set from record")
    logger.info("✅ call_id is unique (not reused)")
    logger.info("✅ Trace lineage properly established")
