"""Flow streaming API functionality.

This module handles streaming API requests to flows, using the unified RunRequest model.
"""

from collections.abc import AsyncGenerator

from buttermilk._core import logger
from buttermilk._core.log import logger
from buttermilk._core.types import RunRequest  # Import RunRequest


async def flow_stream(
    flow,
    run_request: RunRequest, # Changed parameter name and type hint
    return_json=True,
) -> AsyncGenerator[str, None]:
    """Stream a flow execution.

    Args:
        flow: Flow to execute
        run_request: The request object (expected to be RunRequest) # Updated description
        return_json: Whether to return JSON or objects

    Returns:
        AsyncGenerator yielding results

    """
    # Removed legacy FlowRequest handling and local Job creation

    # Run the flow directly with RunRequest
    async for result in flow.run_flows(run_request=run_request): # Pass run_request directly
        if result:
            # Assuming result is AgentTrace or similar with outputs and agent_info
            agent_name = getattr(getattr(result, "agent_info", None), "name", "unknown")
            if not getattr(result, "outputs", None):
                 logger.info(
                    f"No data to return from flow step {agent_name} (completed successfully).",
                )
                # raise StopAsyncIteration

            if return_json:
                # Assuming result has model_dump_json method (like AgentTrace)
                yield result.model_dump_json()
            else:
                yield result

    # Assuming flow object has a source attribute
    flow_source = getattr(flow, "source", "unknown")
    logger.info(f"Finished flow {flow_source}.") # Access source from flow object
    return
