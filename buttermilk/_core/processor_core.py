"""Base class for all pipeline processors.

This module provides ProcessorCore, the unified base class for all processors.
It combines:
- OpenTelemetry span wrapping (automatic for all processors)
- ExecutionTrace emission helpers (opt-in for LLM/batch processors)
- Hydra config compatibility (Pydantic models with _target_)

The design enables consistent observability across all processor types.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Optional

from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field

from buttermilk import logger
from buttermilk._core.contract import ExecutionTrace
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.types import BaseRecord
from buttermilk.pipeline import RecordBufferedException


class ObservabilityMixin(BaseModel):
    """Mixin providing observability features (tracing, logging)."""

    # Common fields
    name: str | None = Field(default=None, description="Processor instance name for tracing")
    enabled: bool = Field(default=True, description="Whether this processor is active")

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        frozen=False,
    )

    _trace_writer: Any = PrivateAttr(default=None)

    @property
    def processor_type(self) -> str:
        """Return processor type name for tracing. Defaults to class name."""
        return self.__class__.__name__

    @computed_field
    @property
    def config_dict(self) -> dict[str, Any]:
        """Return processor config for tracing (JSON-serializable)."""
        excluded = {"client", "credentials", "tokenizer", "config_dict"}
        return {k: v for k, v in self.model_dump(exclude_none=True, exclude={"config_dict"}).items() if k not in excluded}

    # ─────────────────────────────────────────────────────────────────────────
    # ExecutionTrace Helpers (opt-in for LLM/batch processors)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def trace_writer(self) -> Any:
        """Lazy-load trace writer for BigQuery persistence."""
        if self._trace_writer is None:
            try:
                from buttermilk.utils.trace_writer import get_trace_writer

                self._trace_writer = get_trace_writer()
            except Exception as e:
                logger.warning(f"Failed to initialize trace writer: {e}")
                self._trace_writer = None
        return self._trace_writer

    async def _emit_trace(self, execution_trace: ExecutionTrace) -> None:
        """Emit execution trace with error handling."""
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
        """Build standardized agent_info dict."""
        return {
            "component_name": self.name or self.__class__.__name__,
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
        """Build standardized metadata dict."""
        metadata: dict[str, Any] = {"duration_ms": duration_ms}
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
        component_name: Optional[str] = None,
    ) -> str:
        """Emit a success execution trace."""
        import uuid

        if trace_id is None:
            trace_id = str(uuid.uuid4())

        agent_info = self._build_agent_info(processor_stage, execution_type)
        if component_name:
            agent_info["component_name"] = component_name

        execution_trace = ExecutionTrace(
            call_id=trace_id,
            agent_info=agent_info,
            inputs=inputs,
            outputs=outputs,
            messages=messages,
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
        component_name: Optional[str] = None,
    ) -> None:
        """Emit an error execution trace."""
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
            metadata=self._build_trace_metadata(record, duration_ms),
            parent_call_id=parent_trace_id,
            record=record,
        )

        await self._emit_trace(error_trace)


class ProcessorCore(ObservabilityMixin, ABC):
    """Base class for single-record pipeline processors.

    Implements Processor protocol with OTEL tracing.
    Supports typed data flow: processors can yield Any type, not just BaseRecord.
    """

    async def process(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[Any, None]:
        """Process with OTEL span wrapping.

        Creates an OpenTelemetry span for this processor execution,
        then delegates to _process_record().

        Args:
            context: ProcessingContext with record and session state

        Yields:
            Any: Output objects from _process_record(). Typically BaseRecord,
                 but can be any type for typed data flow.
        """
        tracer = trace.get_tracer("buttermilk.processor")
        parent_context = trace.set_span_in_context(context.span) if context.span else None
        processor_name = self.name or self.processor_type
        record_id = getattr(context.record, "record_id", None) or str(type(context.record).__name__)

        with tracer.start_as_current_span(
            f"processor.{self.processor_type}",
            context=parent_context,
            attributes={
                "processor.name": processor_name,
                "processor.type": self.processor_type,
                "record.id": record_id,
            },
        ) as span:
            try:
                async for output in self._process_record(context):
                    yield output
            except RecordBufferedException:
                # Buffered records are not failures - re-raise without error logging
                # The pipeline handler will log at DEBUG level
                raise
            except Exception as e:
                span.record_exception(e)
                logger.error(
                    f"Processor {processor_name} failed: {e}",
                    processor=processor_name,
                    record_id=record_id,
                    error=str(e),
                )
                raise

    @abstractmethod
    async def _process_record(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[Any, None]:
        """Concrete processing logic. Must be implemented by subclasses.

        Args:
            context: ProcessingContext with record and session state

        Yields:
            Any: Zero or more output objects (typically BaseRecord, but can be any type)
        """
        raise NotImplementedError("Subclasses must implement _process_record")
        yield

    async def flush(self) -> AsyncGenerator[Any, None]:
        """Flush any buffered records after source exhaustion.

        Default implementation yields nothing. Override in processors
        that buffer records (like BatchAccumulator).

        Yields:
            Any: Any remaining buffered records/objects after processing.
        """
        return
        yield  # Make this a generator


class BatchProcessorCore(ObservabilityMixin, ABC):
    """Base class for batch pipeline processors.

    Implements batch processing with OTEL tracing.
    """

    async def process_batch(
        self,
        records: list[BaseRecord],
    ) -> list[BaseRecord]:
        """Process batch with OTEL span wrapping.

        Args:
            records: List of records to process

        Returns:
            List of processed records
        """
        tracer = trace.get_tracer("buttermilk.processor")
        processor_name = self.name or self.processor_type

        attributes = {
            "processor.name": processor_name,
            "processor.type": self.processor_type,
            "batch.size": len(records),
        }

        with tracer.start_as_current_span(
            f"batch_processor.{self.processor_type}",
            attributes=attributes,
        ) as span:
            try:
                # Delegate to concrete implementation
                return await self._process_batch(records)
            except Exception as e:
                span.record_exception(e)
                logger.error(
                    f"Batch Processor {processor_name} failed: {e}",
                    processor=processor_name,
                    batch_size=len(records),
                    error=str(e),
                )
                raise

    @abstractmethod
    async def _process_batch(
        self,
        records: list[BaseRecord],
    ) -> list[BaseRecord]:
        """Concrete batch processing logic.

        Args:
            records: List of records to process

        Returns:
            List of processed records
        """
        raise NotImplementedError("Subclasses must implement _process_batch")
