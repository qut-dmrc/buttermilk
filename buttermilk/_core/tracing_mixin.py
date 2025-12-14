"""Unified trace emission functionality for all processor types.

This module provides TracingMixin, a mixin class that encapsulates shared
tracing infrastructure for all processors:
- LLMCore
- ClassifierCore
- ToxicityClassifierCore

The mixin provides:
- Lazy trace_writer property for BigQuery persistence
- Standardized agent_info and metadata building
- Success and error trace emission helpers
- Duration calculation utilities

Design: Mixin pattern enables sharing trace logic across both plain classes
(ProcessorCore) and Pydantic models (ToxicityClassifierCore).
"""

import time
import uuid
from typing import TYPE_CHECKING, Any, Optional

from buttermilk import logger

if TYPE_CHECKING:
    from buttermilk._core.contract import ExecutionTrace
    from buttermilk._core.types import BaseRecord


class TracingMixin:
    """Mixin providing trace emission functionality for all processor types.

    This mixin extracts common trace infrastructure from ProcessorCore,
    allowing it to be shared with ToxicityClassifierCore (which inherits
    from Pydantic BaseModel) without code duplication.

    Classes using this mixin should:
    1. Have a `parameters` attribute (dict of processor configuration)
    2. Initialize `_trace_writer = None` in their __init__

    Example:
        ```python
        class MyProcessor(TracingMixin):
            def __init__(self):
                self.parameters = {}
                self._trace_writer = None

            async def process(self, record):
                start_time = time.time()
                try:
                    result = await self._do_work(record)
                    await self._emit_success_trace(
                        record=record,
                        outputs=result,
                        processor_stage="my_stage",
                        parent_trace_id=None,
                        duration_ms=(time.time() - start_time) * 1000,
                    )
                except Exception as e:
                    await self._emit_error_trace(
                        record=record,
                        error=e,
                        processor_stage="my_stage",
                        parent_trace_id=None,
                        duration_ms=(time.time() - start_time) * 1000,
                    )
                    raise
        ```
    """

    # Expected attributes from the using class
    parameters: dict[str, Any]
    _trace_writer: Any

    @property
    def trace_writer(self) -> Any:
        """Lazy-load trace writer for BigQuery persistence.

        Returns:
            TraceWriter instance or None if unavailable
        """
        if self._trace_writer is None:
            try:
                from buttermilk.utils.trace_writer import get_trace_writer

                self._trace_writer = get_trace_writer()
            except Exception as e:
                logger.warning(f"Failed to initialize trace writer: {e}")
                self._trace_writer = None
        return self._trace_writer

    async def _emit_trace(self, execution_trace: "ExecutionTrace") -> None:
        """Emit execution trace with error handling.

        Args:
            execution_trace: The ExecutionTrace to persist
        """
        if self.trace_writer:
            try:
                await self.trace_writer.add(execution_trace)
            except Exception as e:
                logger.warning(f"Failed to emit trace: {e}")

    def _build_agent_info(
        self,
        processor_stage: str,
        execution_type: str = "processing",
        extra_config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build standardized agent_info dict for ExecutionTrace.

        Args:
            processor_stage: Pipeline stage identifier
            execution_type: Type of execution (e.g., "classification", "llm_processing")
            extra_config: Additional config to include

        Returns:
            Standardized agent_info dictionary
        """
        config = {**getattr(self, "parameters", {})}
        if extra_config:
            config.update(extra_config)

        return {
            "component_name": self.__class__.__name__,
            "execution_type": execution_type,
            "processor_stage": processor_stage,
            "config": config,
        }

    def _build_trace_metadata(
        self,
        record: Optional["BaseRecord"],
        duration_ms: float,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build standardized metadata dict for ExecutionTrace.

        Args:
            record: Input record (for extracting input metadata)
            duration_ms: Processing duration in milliseconds
            extra_metadata: Additional metadata to include

        Returns:
            Standardized metadata dictionary with input metadata and duration
        """
        metadata: dict[str, Any] = {"duration_ms": duration_ms}

        # Include input metadata under 'input' key if available
        if record is not None and hasattr(record, "metadata") and record.metadata:
            metadata["input"] = record.metadata

        if extra_metadata:
            metadata.update(extra_metadata)

        return metadata

    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID.

        Returns:
            UUID string for trace correlation
        """
        return str(uuid.uuid4())

    async def _emit_success_trace(
        self,
        record: "BaseRecord",
        outputs: Any,
        processor_stage: str,
        parent_trace_id: Optional[str],
        duration_ms: float,
        messages: Optional[list] = None,
        inputs: Optional[dict[str, Any]] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
        execution_type: str = "processing",
        trace_id: Optional[str] = None,
        extra_parameters: Optional[dict[str, Any]] = None,
    ) -> str:
        """Emit a success execution trace.

        This is the primary method for emitting traces after successful processing.
        It handles all the boilerplate of constructing ExecutionTrace objects.

        Args:
            record: Input record
            outputs: Processing outputs (any serializable type)
            processor_stage: Pipeline stage identifier
            parent_trace_id: Parent trace ID for correlation
            duration_ms: Processing duration in milliseconds
            messages: Optional message history (LLM messages, prompts, etc.)
            inputs: Optional template inputs/variables
            extra_metadata: Additional metadata to merge
            execution_type: Type of execution (default: "processing")
            trace_id: Optional pre-generated trace ID (generates new one if None)
            extra_parameters: Additional parameters to merge with self.parameters

        Returns:
            trace_id: The generated or provided trace ID for correlation
        """
        from buttermilk._core.contract import ExecutionTrace

        trace_id = trace_id or self._generate_trace_id()

        # Merge parameters
        parameters = {**getattr(self, "parameters", {})}
        if extra_parameters:
            parameters.update(extra_parameters)

        execution_trace = ExecutionTrace(
            call_id=trace_id,
            agent_info=self._build_agent_info(processor_stage, execution_type),
            inputs=inputs,
            outputs=outputs,
            messages=messages,
            parameters=parameters,
            metadata=self._build_trace_metadata(record, duration_ms, extra_metadata),
            parent_call_id=parent_trace_id,
            record=record,
        )

        await self._emit_trace(execution_trace)
        return trace_id

    async def _emit_error_trace(
        self,
        record: Optional["BaseRecord"],
        error: Exception,
        processor_stage: str,
        parent_trace_id: Optional[str],
        duration_ms: float,
        inputs: Optional[dict[str, Any]] = None,
        execution_type: str = "processing",
        extra_parameters: Optional[dict[str, Any]] = None,
    ) -> None:
        """Emit an error execution trace.

        This method captures error information in a standardized trace format.
        Always call this when processing fails to ensure error visibility.

        Args:
            record: Input record (may be None for early failures)
            error: The exception that occurred
            processor_stage: Pipeline stage identifier
            parent_trace_id: Parent trace ID for correlation
            duration_ms: Processing duration in milliseconds
            inputs: Optional template inputs/variables
            execution_type: Type of execution (default: "processing")
            extra_parameters: Additional parameters to merge with self.parameters
        """
        from buttermilk._core.contract import ExecutionTrace

        # Merge parameters
        parameters = {**getattr(self, "parameters", {})}
        if extra_parameters:
            parameters.update(extra_parameters)

        error_trace = ExecutionTrace(
            agent_info=self._build_agent_info(processor_stage, execution_type),
            inputs=inputs,
            error={
                "event": str(error),
                "details": {"error_type": type(error).__name__},
            },
            parameters=parameters,
            metadata=self._build_trace_metadata(record, duration_ms),
            parent_call_id=parent_trace_id,
            record=record,
        )

        await self._emit_trace(error_trace)


def calculate_duration_ms(start_time: float) -> float:
    """Calculate duration in milliseconds from a start time.

    Args:
        start_time: Start time from time.time()

    Returns:
        Duration in milliseconds
    """
    return (time.time() - start_time) * 1000
