"""Shared base class for all pipeline processors.

This module provides ProcessorCore, a minimal base class that encapsulates
common functionality shared between ClassifierCore, LLMCore, and
ToxicityClassifierCore.

The design enables:
- Consistent tracing across all processor types
- Shared infrastructure (trace_writer, error handling)
- Unified Processor protocol implementation patterns
"""

import time
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Optional

from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, PrivateAttr, computed_field

from buttermilk import logger
from buttermilk._core.contract import ExecutionTrace
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.types import BaseRecord


class ProcessorCore(ABC, BaseModel):
    """Minimal shared base for all pipeline processors.

    Provides:
    - Lazy trace_writer property for BigQuery persistence
    - Common trace emission patterns
    - Processor protocol scaffolding

    Subclasses (ClassifierCore, LLMCore, ToxicityClassifierCore) implement
    their specific processing logic while inheriting common infrastructure.

    Example:
        ```python
        class MyProcessor(ProcessorCore):
            async def process(
                self,
                record: BaseRecord,
                *,
                processor_stage: str,
                **kwargs,
            ) -> AsyncGenerator[BaseRecord, None]:
                # Process record
                result = await self._do_processing(record)

                # Emit trace
                await self._emit_trace(...)

                yield enriched_record
        ```
    """

    model_config = ConfigDict(
        extra="forbid",  # Strict - no unknown fields
        arbitrary_types_allowed=True,
        frozen=False,
    )

    # Private attributes (not included in serialization)
    _trace_writer: Any = PrivateAttr(default=None)

    @computed_field
    @property
    def parameters(self) -> dict[str, Any]:
        """Return processor config for tracing (JSON-serializable)."""
        excluded = {"client", "credentials", "tokenizer", "parameters"}
        return {
            k: v
            for k, v in self.model_dump(exclude_none=True, exclude={"parameters"}).items()
            if k not in excluded
        }

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

    async def _emit_trace(self, execution_trace: ExecutionTrace) -> None:
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
    ) -> dict[str, Any]:
        """Build standardized agent_info dict for ExecutionTrace.

        Args:
            processor_stage: Pipeline stage identifier
            execution_type: Type of execution (e.g., "classification", "llm_processing")

        Returns:
            Standardized agent_info dictionary
        """
        return {
            "component_name": self.__class__.__name__,
            "processor_class": self.__class__.__name__,
            "execution_type": execution_type,
            "processor_stage": processor_stage,
        }

    def _build_trace_metadata(
        self,
        record: Optional[BaseRecord],
        duration_ms: float,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build standardized metadata dict for ExecutionTrace.

        Args:
            record: Input record (for extracting input metadata)
            duration_ms: Processing duration in milliseconds
            extra_metadata: Additional metadata to include

        Returns:
            Standardized metadata dictionary
        """
        metadata: dict[str, Any] = {"duration_ms": duration_ms}

        # Include input metadata under 'input' key if available
        if record is not None and hasattr(record, "metadata") and record.metadata:
            metadata["input"] = record.metadata

        if extra_metadata:
            metadata.update(extra_metadata)

        return metadata

    async def _emit_success_trace(
        self,
        record: BaseRecord,
        outputs: Any,
        processor_stage: str,
        parent_trace_id: Optional[str],
        duration_ms: float,
        messages: Optional[list] = None,
        inputs: Optional[dict[str, Any]] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
        execution_type: str = "processing",
        trace_id: Optional[str] = None,
        parameters: Optional[dict[str, Any]] = None,
        component_name: Optional[str] = None,
    ) -> str:
        """Emit a success execution trace.

        Args:
            record: Input record
            outputs: Processing outputs
            processor_stage: Pipeline stage identifier
            parent_trace_id: Parent trace ID for correlation
            duration_ms: Processing duration in milliseconds
            messages: Optional message history
            inputs: Optional template inputs/variables
            extra_metadata: Additional metadata
            execution_type: Type of execution
            trace_id: Optional trace ID (generated if not provided)
            parameters: Optional parameters override (defaults to self.parameters)
            component_name: Optional component name (defaults to class name)

        Returns:
            trace_id: The trace ID used
        """
        import uuid

        if trace_id is None:
            trace_id = str(uuid.uuid4())

        # Build agent_info
        agent_info = self._build_agent_info(processor_stage, execution_type)
        if component_name:
            agent_info["component_name"] = component_name

        execution_trace = ExecutionTrace(
            call_id=trace_id,
            agent_info=agent_info,
            inputs=inputs,
            outputs=outputs,
            messages=messages,
            parameters=parameters if parameters is not None else self.parameters,
            metadata=self._build_trace_metadata(record, duration_ms, extra_metadata),
            parent_call_id=parent_trace_id,
            record=record,
        )

        await self._emit_trace(execution_trace)
        return trace_id

    async def _emit_error_trace(
        self,
        record: Optional[BaseRecord],
        error: Exception,
        processor_stage: str,
        parent_trace_id: Optional[str],
        duration_ms: float,
        inputs: Optional[dict[str, Any]] = None,
        execution_type: str = "processing",
        parameters: Optional[dict[str, Any]] = None,
        component_name: Optional[str] = None,
    ) -> None:
        """Emit an error execution trace.

        Args:
            record: Input record (may be None for early failures)
            error: The exception that occurred
            processor_stage: Pipeline stage identifier
            parent_trace_id: Parent trace ID for correlation
            duration_ms: Processing duration in milliseconds
            inputs: Optional template inputs/variables
            execution_type: Type of execution
            parameters: Optional parameters override (defaults to self.parameters)
            component_name: Optional component name (defaults to class name)
        """
        # Build agent_info
        agent_info = self._build_agent_info(processor_stage, execution_type)
        if component_name:
            agent_info["component_name"] = component_name

        error_trace = ExecutionTrace(
            agent_info=agent_info,
            inputs=inputs,
            error={
                "event": str(error),
                "details": {"error_type": type(error).__name__},
            },
            parameters=parameters if parameters is not None else self.parameters,
            metadata=self._build_trace_metadata(record, duration_ms),
            parent_call_id=parent_trace_id,
            record=record,
        )

        await self._emit_trace(error_trace)

    @abstractmethod
    async def process(
        self,
        record: BaseRecord,
        *,
        processor_stage: str,
        parent_trace_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Process a BaseRecord and yield zero or more output records.

        This is the Processor protocol method that subclasses must implement.

        Args:
            record: Input BaseRecord to process
            processor_stage: Unique stage identifier for tracing
            parent_trace_id: Optional trace ID for distributed tracing
            **kwargs: Additional arguments

        Yields:
            BaseRecord: Enriched record(s) with processing results

        Raises:
            ProcessingError: If processing fails
        """
        raise NotImplementedError("Subclasses must implement process()")
        yield  # Make this a generator
