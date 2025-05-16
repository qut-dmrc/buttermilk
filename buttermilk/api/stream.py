"""Flow streaming API functionality.

This module handles streaming API requests to flows, using the unified RunRequest model.
"""

from collections.abc import AsyncGenerator

from pydantic import (
    BaseModel,
    Field,
)

from buttermilk._core import logger
from buttermilk._core.log import logger
from buttermilk._core.types import Record
from buttermilk.utils.media import download_and_convert


class Job(BaseModel):
    """Legacy job class for compatibility.
    
    Will be removed once all code is migrated to use RunRequest directly.
    """

    # Basic identifiers and metadata
    flow: str
    parameters: dict = Field(default_factory=dict)
    inputs: dict = Field(default_factory=dict)
    record: Record | None = None


async def flow_stream(
    flow,
    flow_request,
    return_json=True,
) -> AsyncGenerator[str, None]:
    """Stream a flow execution.
    
    Args:
        flow: Flow to execute
        flow_request: Request object (FlowRequest for compatibility, or RunRequest)
        return_json: Whether to return JSON or objects
        
    Returns:
        AsyncGenerator yielding results

    """
    # Convert FlowRequest to RunRequest if needed
    if hasattr(flow_request, "to_job"):
        # Legacy FlowRequest conversion
        job = flow_request.to_job()

        # First step, fetch the record if we need to.
        if not job.record and job.inputs:
            if record_id := job.inputs.pop("record_id", None):
                job.record = flow.get_record(record_id=record_id)
            else:
                job.record = await download_and_convert(**job.inputs)

    else:
        # New RunRequest model
        run_request = flow_request

        # Create job from RunRequest (temporary conversion until all code
        # is updated to use RunRequest directly)
        job = Job(
            flow=run_request.flow,
            parameters=run_request.parameters,
        )

        # Set record if available
        if run_request.record_id:
            job.record = flow.get_record(record_id=run_request.record_id)
        elif run_request.uri:
            job.record = await download_and_convert(uri=run_request.uri)
        elif len(run_request.records) > 0:
            job.record = run_request.records[0]

    # Run the flow
    async for result in flow.run_flows(job=job):
        if result:
            if not result.outputs:
                logger.info(
                    f"No data to return from flow step {result.agent_info.get('name')} (completed successfully).",
                )
                # raise StopAsyncIteration

            if return_json:
                yield result.model_dump_json()
            else:
                yield result

    logger.info(f"Finished flow {flow.source}.")
    return
