"""Example of using standalone trace context for batch processing.

This demonstrates how to use the StandaloneTraceContext for scripts and batch
processes that run outside of an orchestrator context.
"""

import asyncio
from buttermilk import buttermilk as bm
from buttermilk._core.standalone_trace import create_standalone_trace
from buttermilk._core.contract import AgentInput
from buttermilk.agents.llm import LLMAgent


async def process_batch_with_trace():
    """Example batch processing with trace context."""
    
    # Initialize buttermilk if needed
    # bm = BM(...)
    
    # Create an agent for testing
    agent = LLMAgent(
        agent_id="test_agent",
        agent_name="Test Agent",
        description="A test agent for standalone tracing",
        parameters={"model": "gpt-4o-mini"},
    )
    
    # Create a standalone trace context
    async with create_standalone_trace(
        "batch_processing_example",
        batch_size=3,
        job_type="test"
    ) as trace:
        
        # Process multiple items in the trace context
        items = ["Item 1", "Item 2", "Item 3"]
        
        for i, item in enumerate(items):
            # Create agent input with parent trace
            agent_input = AgentInput(
                inputs={"item": item},
                parent_call_id=trace.get_call_id(),  # Link to parent trace
            )
            
            # Agent will automatically nest its trace under the parent
            result = await agent.invoke(agent_input)
            print(f"Processed {item}: {result}")
        
        print(f"Batch processing complete. Trace ID: {trace.get_trace_id()}")


async def process_with_manual_context():
    """Example of manually managing trace context."""
    from buttermilk._core.standalone_trace import StandaloneTraceContext
    
    # Create context manually
    context = StandaloneTraceContext("manual_process", {"custom": "attribute"})
    
    async with context:
        # Your processing logic here
        print(f"Processing with trace ID: {context.get_trace_id()}")
        
        # If you have agents or other traced operations, pass context.trace_call
        # as the parent_call parameter


if __name__ == "__main__":
    # Run the example
    asyncio.run(process_batch_with_trace())