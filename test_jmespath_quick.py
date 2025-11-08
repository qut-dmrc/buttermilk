"""Quick test to verify JMESPathTransform works."""

import asyncio

from buttermilk._core.types import BaseRecord
from buttermilk.processors.jmespath_transform import JMESPathTransform


async def main():
    # Test 1: Simple field extraction
    print("Test 1: Simple field extraction")
    record = BaseRecord(
        record_id="test_1", metadata={"outputs": {"result": "test_value"}}
    )

    processor = JMESPathTransform(mappings={"answer": "metadata.outputs.result"})

    async for result in processor.process(record, processor_stage="transform"):
        print(f"  Has answer attribute: {hasattr(result, 'answer')}")
        if hasattr(result, "answer"):
            print(f"  Answer value: {result.answer}")
        print(f"  Metadata preserved: {result.metadata == record.metadata}")

    # Test 2: Nested object construction
    print("\nTest 2: Nested object construction")
    record2 = BaseRecord(
        record_id="trace_1",
        metadata={
            "agent_info": {"agent_id": "agent_123"},
            "outputs": {"conclusion": "test conclusion"},
            "call_id": "call_456",
        },
    )

    processor2 = JMESPathTransform(
        mappings={
            "answers": "{agent_id: metadata.agent_info.agent_id, result: metadata.outputs, answer_id: metadata.call_id}"
        }
    )

    async for result in processor2.process(record2, processor_stage="transform"):
        print(f"  Has answers attribute: {hasattr(result, 'answers')}")
        if hasattr(result, "answers"):
            print(f"  Answers type: {type(result.answers)}")
            print(f"  Answers value: {result.answers}")

    # Test 3: Missing field
    print("\nTest 3: Missing field handling")
    record3 = BaseRecord(record_id="test_3", metadata={"other_field": "value"})

    processor3 = JMESPathTransform(mappings={"value": "metadata.nonexistent.field"})

    async for result in processor3.process(record3, processor_stage="transform"):
        print(f"  Has value attribute: {hasattr(result, 'value')}")
        print(f"  Metadata preserved: {result.metadata == record3.metadata}")

    print("\nAll quick tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
