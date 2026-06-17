"""Base class for all pipeline processors.

This module provides ProcessorCore, the unified base class for all processors.
It combines:
- OpenTelemetry span wrapping (automatic for all processors)
- ExecutionTrace emission helpers (opt-in for LLM/batch processors)
- Hydra config compatibility (Pydantic models with _target_)

The design enables consistent observability across all processor types.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import jmespath  # type: ignore[import-untyped]  # jmespath: types-jmespath stubs not installed
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field

from buttermilk import logger
from buttermilk._core.contract import ExecutionTrace
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.messages import LLMMessage
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.types import BaseRecord
from buttermilk.pipeline import RecordBufferedException


@dataclass
class TraceParams:
    """Parameters for emitting execution traces.

    Groups the arguments for _emit_success_trace / _emit_error_trace
    to keep method signatures concise.
    """

    processor_stage: str
    parent_trace_id: str | None
    duration_ms: float
    execution_type: str = "processing"
    component_name: str | None = None
    messages: list[LLMMessage] | None = None
    inputs: dict[str, Any] | None = None
    extra_metadata: dict[str, Any] | None = None
    trace_id: str | None = None
    # Multi-step provenance: stamped into agent_info so tja.traces is scorable by
    # role+index, not only template_hash (Proposal v4).
    step: str | None = None
    agent_id: str | None = None


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

    @computed_field  # type: ignore[prop-decorator]  # pydantic computed_field over property; mypy does not model this pattern
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
        record: BaseRecord | None,
        duration_ms: float,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build standardized metadata dict."""
        metadata: dict[str, Any] = {"duration_ms": duration_ms}
        if record is not None and hasattr(record, "metadata") and record.metadata:
            metadata["input"] = record.metadata
        if extra_metadata:
            metadata.update(extra_metadata)
        return metadata

    def _stamp_identity(self, agent_info: dict[str, Any], tp: TraceParams) -> None:
        """Stamp multi-step role/agent identity into agent_info for scorability."""
        if tp.component_name:
            agent_info["component_name"] = tp.component_name
        if tp.step is not None:
            agent_info["step"] = tp.step
        if tp.agent_id is not None:
            agent_info["agent_id"] = tp.agent_id

    async def _emit_success_trace(
        self,
        record: BaseRecord,
        outputs: Any,
        tp: TraceParams,
    ) -> ExecutionTrace:
        """Emit a success execution trace and return it (for StepResult projection).

        Args:
            record: The record being processed.
            outputs: The processing outputs.
            tp: Trace parameters (stage, timing, metadata, etc.).

        Returns:
            The emitted ExecutionTrace, so callers can project a StepResult from the
            same source (single-projection guarantee, Proposal v4).
        """
        import uuid

        trace_id = tp.trace_id or str(uuid.uuid4())

        agent_info = self._build_agent_info(tp.processor_stage, tp.execution_type)
        self._stamp_identity(agent_info, tp)

        execution_trace = ExecutionTrace(
            call_id=trace_id,
            agent_info=agent_info,
            inputs=tp.inputs,
            outputs=outputs,
            messages=tp.messages or [],
            metadata=self._build_trace_metadata(record, tp.duration_ms, tp.extra_metadata),
            parent_call_id=tp.parent_trace_id,
            record=record,
        )

        await self._emit_trace(execution_trace)
        return execution_trace

    async def _emit_error_trace(
        self,
        record: BaseRecord | None,
        error: Exception,
        tp: TraceParams,
    ) -> ExecutionTrace:
        """Emit an error execution trace and return it (for StepResult projection).

        Args:
            record: The record being processed (may be None).
            error: The exception that occurred.
            tp: Trace parameters (stage, timing, metadata, etc.).

        Returns:
            The emitted error ExecutionTrace.
        """
        agent_info = self._build_agent_info(tp.processor_stage, tp.execution_type)
        self._stamp_identity(agent_info, tp)

        # Omit call_id when no trace_id so ExecutionTrace's default_factory supplies one.
        call_id_kwarg: dict[str, Any] = {"call_id": tp.trace_id} if tp.trace_id else {}
        error_trace = ExecutionTrace(
            **call_id_kwarg,
            agent_info=agent_info,
            inputs=tp.inputs,
            error={
                "event": str(error),
                "details": {"error_type": type(error).__name__},
            },
            metadata=self._build_trace_metadata(record, tp.duration_ms, tp.extra_metadata),
            parent_call_id=tp.parent_trace_id,
            record=record,
        )

        await self._emit_trace(error_trace)
        return error_trace

    @staticmethod
    def _append_history(record: BaseRecord, *step_results: Any) -> dict[str, Any]:
        """Return updated metadata with one or more StepResults appended to ``history``.

        ``history`` is the single, ordered accumulator of per-step outputs on the record
        (Proposal v4). Entries are stored as plain dicts (``StepResult.model_dump()``) so
        JMESPath over the dumped record envelope sees a uniform shape regardless of how the
        record is later serialized. The typed ``StepResult`` enforces the shape at
        construction time; this helper never mutates the input record's metadata in place.

        NB: ``agent_id`` follows two conventions that legitimately coexist in one history
        list — positional ``"<step>#<index>"`` for synthetic producers (LLMProcessor,
        FanInLLMProcessor) and the real agent identity for GroupchatProcessor. Both are
        opaque labels; downstream selectors filter by ``step`` first (``index`` is unique
        only within a step), so the two conventions never need to be reconciled.
        """
        existing = list(record.metadata.get("history", [])) if record.metadata else []
        for sr in step_results:
            existing.append(sr.model_dump())
        return {
            **(record.metadata if record.metadata else {}),
            "history": existing,
        }


class ProcessorCore(ObservabilityMixin, ABC):
    """Base class for single-record pipeline processors.

    Implements Processor protocol with OTEL tracing.
    Supports typed data flow: processors can yield Any type, not just BaseRecord.

    The ``inputs`` field provides opt-in JMESPath-based resolution of per-record
    values from the record envelope.  Keys that match processor config fields
    (e.g. ``model``, ``template``) override those fields for the current record;
    all other keys are injected as template variables.

    Example YAML config::

        steps:
          - processor: LLMProcessor
            model: gpt-4o            # default
            template: analyze_text   # default
            inputs:
              model: record.metadata.model        # per-record override
              template: record.metadata.template
              country: record.metadata.country     # template variable
    """

    inputs: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "JMESPath mappings from record fields to processor inputs. "
            "Paths are evaluated against {'record': record.model_dump()}. "
            "E.g. {'answers': 'record.metadata.history[?step==`judge` && error==null]'}. "
            "Inputs are REQUIRED by default: a path that resolves to None or an empty "
            "list raises ProcessingError (fail-loud, research integrity). Declare keys "
            "that may legitimately be absent in `optional_inputs`."
        ),
    )

    optional_inputs: list[str] = Field(
        default_factory=list,
        description=(
            "Input keys that are OPTIONAL: silently dropped when their JMESPath yields "
            "None or an empty list, instead of raising. All other inputs are required."
        ),
    )

    def _optional_input_keys(self) -> set[str]:
        """Keys exempt from fail-loud resolution. Subclasses may widen this set."""
        return set(self.optional_inputs)

    def _resolve_inputs(self, context: ProcessingContext) -> dict[str, Any]:
        """Resolve ``inputs`` from the record via JMESPath, fail-loud on missing required inputs.

        A required input whose JMESPath expression resolves to ``None`` OR an empty list
        ``[]`` raises :class:`ProcessingError`. The empty-list case is essential: a no-match
        JMESPath filter (e.g. a panel selector that matched nothing) returns ``[]``, not
        ``None`` — silently keeping it would let a missing/short panel flow into a template,
        a validity threat. Optional inputs (declared in ``optional_inputs``) are dropped
        silently when missing so callers fall back to defaults.

        Returns a dict of resolved values (only keys whose JMESPath expression matched).
        """
        if not self.inputs:
            return {}

        envelope: dict[str, Any]
        if hasattr(context.record, "model_dump"):
            envelope = {"record": context.record.model_dump()}
        else:
            envelope = {"record": context.record}

        optional = self._optional_input_keys()
        record_id = getattr(context.record, "record_id", None) or type(context.record).__name__

        resolved: dict[str, Any] = {}
        for name, path in self.inputs.items():
            value = jmespath.search(path, envelope)
            if value is None or value == []:
                if name in optional:
                    continue
                raise ProcessingError(
                    f"Required input '{name}' (JMESPath '{path}') for "
                    f"{self.__class__.__name__} resolved to missing (None or empty list) "
                    f"on record '{record_id}'. If this input may legitimately be absent, "
                    f"declare it in `optional_inputs`."
                )
            resolved[name] = value
        return resolved

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
    Batch processors receive ProcessingContexts (which carry variant_params)
    and return BaseRecords.
    """

    async def process_batch(
        self,
        contexts: list[ProcessingContext],
    ) -> list[BaseRecord]:
        """Process batch with OTEL span wrapping.

        Args:
            contexts: List of ProcessingContext objects to process

        Returns:
            List of processed records
        """
        tracer = trace.get_tracer("buttermilk.processor")
        processor_name = self.name or self.processor_type

        attributes: dict[str, str | int] = {
            "processor.name": processor_name,
            "processor.type": self.processor_type,
            "batch.size": len(contexts),
        }

        with tracer.start_as_current_span(
            f"batch_processor.{self.processor_type}",
            attributes=attributes,
        ) as span:
            try:
                # Delegate to concrete implementation
                return await self._process_batch(contexts)
            except Exception as e:
                span.record_exception(e)
                logger.error(
                    f"Batch Processor {processor_name} failed: {e}",
                    processor=processor_name,
                    batch_size=len(contexts),
                    error=str(e),
                )
                raise

    @abstractmethod
    async def _process_batch(
        self,
        contexts: list[ProcessingContext],
    ) -> list[BaseRecord]:
        """Concrete batch processing logic.

        Args:
            contexts: List of ProcessingContext objects to process

        Returns:
            List of processed records
        """
        raise NotImplementedError("Subclasses must implement _process_batch")
