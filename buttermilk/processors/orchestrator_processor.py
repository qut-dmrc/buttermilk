"""Orchestrator processor for pipeline integration.

This module provides a processor that wraps Buttermilk orchestrators (complex
multi-agent flows) as pipeline processors, enabling them to be used within
the PipelineOrchestrator framework.

IMPORTANT: Caching is permanently disabled for orchestrator processors because
orchestrators are inherently non-deterministic (LLM calls, multi-agent
conversations produce different results each time).
"""

from typing import Any, AsyncGenerator

from pydantic import BaseModel, ConfigDict, Field

from buttermilk import logger
from buttermilk._core.contract import ExecutionTrace
from buttermilk._core.orchestrator import OrchestratorProtocol
from buttermilk._core.types import BaseRecord, RunRequest
from buttermilk.runner.flowrunner import OrchestratorFactory


class OrchestratorProcessor(BaseModel):
    """Wraps an Orchestrator as a Pipeline Processor.

    Allows complex multi-agent flows to be used as pipeline processing steps.
    Each input record is processed by a fresh orchestrator instance to ensure
    complete state isolation between records.

    Caching is permanently disabled since orchestrators are non-deterministic
    (LLM calls, multi-agent conversations produce different results each time).

    Example usage in pipeline config:
        ```yaml
        processors:
          - _target_: buttermilk.processors.orchestrator_processor.OrchestratorProcessor
            flow_config: ${run.flows.my_flow}
            flow_name: my_flow
        ```

    Example usage in code:
        ```python
        processor = OrchestratorProcessor(
            flow_config=flows["my_flow"],
            flow_name="my_flow",
        )

        pipeline = PipelineOrchestrator(
            pipeline_name="batch_my_flow",
            source=source,
            processors=[processor],
            enable_record_cache=False,  # Redundant but explicit
        )
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # CRITICAL: Always skip pipeline caching - orchestrators are non-deterministic
    skip_cache: bool = Field(
        default=True,
        description="Always skip pipeline caching for orchestrators (non-deterministic)",
    )

    # Flow configuration
    flow_config: OrchestratorProtocol = Field(
        ...,
        description="Flow configuration (OrchestratorProtocol) defining the orchestrator",
    )
    flow_name: str = Field(
        ...,
        description="Name of the flow to execute",
    )

    # Optional: parameters to pass to orchestrator
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters to pass to orchestrator RunRequest",
    )

    # Optional: collect ExecutionTrace outputs from orchestrator
    collect_traces: bool = Field(
        default=True,
        description="Whether to collect ExecutionTrace outputs and attach to record metadata",
    )

    # Optional: BM instance for session-scoped observability
    bm: Any | None = Field(
        default=None,
        exclude=True,
        description="Optional session-scoped BM instance for observability isolation",
    )

    async def process(
        self,
        record: BaseRecord,
        *,
        processor_stage: str,
        parent_trace_id: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Execute orchestrator on a record and yield the enriched result.

        Creates a fresh orchestrator instance for each record to ensure complete
        state isolation. Orchestrator outputs (ExecutionTrace objects) are collected
        via callback and attached to the record's metadata.

        Args:
            record: Input record to process
            processor_stage: Pipeline stage identifier (for tracing)
            parent_trace_id: Parent trace ID for distributed tracing
            **kwargs: Additional arguments (ignored)

        Yields:
            BaseRecord: The input record enriched with orchestrator outputs in metadata
        """
        record_id = getattr(record, "record_id", "unknown")

        logger.debug(
            "OrchestratorProcessor starting",
            record_id=record_id,
            flow_name=self.flow_name,
            processor_stage=processor_stage,
        )

        # Create fresh orchestrator for each record (state isolation)
        orchestrator = OrchestratorFactory.create_orchestrator(
            self.flow_config, self.flow_name
        )

        # Inject session-scoped BM if available
        if self.bm is not None:
            orchestrator.set_bm(self.bm)

        # Collect ExecutionTrace outputs via callback
        traces: list[ExecutionTrace] = []

        async def collect_callback(message: Any) -> None:
            """Callback to collect ExecutionTrace outputs from orchestrator."""
            if isinstance(message, ExecutionTrace):
                traces.append(message)

        # Create RunRequest from record
        run_request = RunRequest(
            flow=self.flow_name,
            inputs={
                "record_id": record_id,
                "record": record.model_dump() if hasattr(record, "model_dump") else record,
            },
            parameters=self.parameters,
            callback_to_ui=collect_callback if self.collect_traces else None,
        )

        # Add parent trace ID if available
        if parent_trace_id:
            run_request.inputs["parent_call_id"] = parent_trace_id

        try:
            # Run orchestrator (returns None, results flow through callback)
            await orchestrator.run(request=run_request)

            # Build outputs from collected traces
            outputs = []
            for trace in traces:
                if trace.outputs is not None:
                    outputs.append(trace.outputs)

            # Enrich record metadata with orchestrator results
            enriched_metadata = {
                **(record.metadata if record.metadata else {}),
                processor_stage: {
                    "status": "processed",
                    "flow_name": self.flow_name,
                    "trace_count": len(traces),
                    "outputs": outputs,
                },
            }

            logger.debug(
                "OrchestratorProcessor completed",
                record_id=record_id,
                flow_name=self.flow_name,
                trace_count=len(traces),
                output_count=len(outputs),
            )

            yield record.model_copy(update={"metadata": enriched_metadata})

        except Exception as e:
            logger.error(
                "OrchestratorProcessor failed",
                record_id=record_id,
                flow_name=self.flow_name,
                error=str(e),
            )
            # Re-raise to let pipeline handle the error
            raise

    async def finalize_processing(self) -> bool:
        """Optional cleanup after pipeline completes.

        OrchestratorProcessor doesn't hold persistent state, so no cleanup needed.

        Returns:
            bool: Always True
        """
        return True
