"""Core LLM functionality shared between agents and processors.

This module provides the LLMCore class that encapsulates the essential
LLM operations (template rendering, LLM calling, tracing) that can be
reused across different contexts - both in Agent-based flows and in
pipeline processors.

LLMCore extends ProcessorCore to share common infrastructure (trace_writer,
tracing patterns) with ClassifierCore and ToxicityClassifierCore.

The design intentionally avoids Agent-specific concepts to maintain
flexibility while preserving full observability through metadata tracking.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional, Self

import pydantic
from autogen_core import CancellationToken
from autogen_core.models import LLMMessage
from opentelemetry import trace
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from buttermilk import bm, logger
from buttermilk._core.contract import ErrorEvent
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.processor_core import ObservabilityMixin

if TYPE_CHECKING:
    from buttermilk._core.llms import CreateResult, ModelOutput
from buttermilk._core.types import BaseRecord
from buttermilk.utils.templating import make_messages, render_template
from buttermilk.utils.utils import scrub_serializable
from buttermilk.utils.validators import import_class_from_path


class LLMResult(BaseModel):
    """Lightweight result structure for LLM operations.

    This provides a simple, standardized format for LLM results that
    can be used in both agent and pipeline contexts.
    """

    content: Any = Field(..., description="The LLM output - string or parsed object")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Usage, pricing, model info")
    trace_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for correlation",
    )
    template_metadata: dict[str, Any] = Field(default_factory=dict, description="Template name, hash, etc")
    messages: list[LLMMessage] = Field(default_factory=list, description="Messages exchanged with LLM")
    error: str | ErrorEvent | None = Field(None, description="Error message if processing failed")
    resolved_inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="All resolved inputs used for template rendering",
    )


class LLMCore(ObservabilityMixin):
    """Core LLM functionality shared between agents and processors.

    Extends ProcessorCore to share common infrastructure with ClassifierCore
    and ToxicityClassifierCore. Extracts the essential LLM operations from
    LLMAgent, making them reusable in different contexts while maintaining
    observability and traceability.

    Key responsibilities:
    - Template loading and rendering
    - LLM API calls with retry logic
    - Lightweight tracing without agent concepts
    - Metadata tracking for observability
    """

    model_config = {"extra": "forbid"}

    # Required fields
    model: str
    template: str

    # Optional fields with defaults
    output_model: str | type[BaseModel] | None = None  # String path or class for config
    tools: list[Any] = Field(default_factory=list)
    fail_on_unfilled_parameters: bool = True
    fail_on_unfilled_parameters: bool = True
    human_in_loop: bool = False  # Whether to require human approval before LLM calls

    # LLM inference parameters (all optional)
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop_sequences: list[str] | None = None
    seed: int | None = None

    # Template variables - user-defined variables for Jinja2 rendering
    # Separate from LLM config to maintain strict validation on config fields
    template_vars: dict[str, Any] = Field(
        default_factory=dict,
        description="User-defined template variables (e.g., criteria, instructions)",
    )

    # Private attrs for runtime objects
    _resolved_output_model: type[BaseModel] | None = PrivateAttr(default=None)
    _template_metadata: dict[str, Any] = PrivateAttr(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_output_model(cls, data: Any) -> Any:
        """Convert class to string path for storage."""
        if isinstance(data, dict) and "output_model" in data:
            output_model = data["output_model"]
            # If it's a class, convert to string path for storage
            if isinstance(output_model, type) and issubclass(output_model, BaseModel):
                data["output_model"] = f"{output_model.__module__}.{output_model.__name__}"
        return data

    @model_validator(mode="after")
    def _resolve_config(self) -> Self:
        """Resolve output_model string to class."""
        if isinstance(self.output_model, str) and self.output_model:
            try:
                self._resolved_output_model = import_class_from_path(self.output_model, expected_base_class=pydantic.BaseModel)
            except (ImportError, AttributeError, ValueError) as e:
                raise ValueError(f"Failed to resolve output_model '{self.output_model}': {e}")
        elif isinstance(self.output_model, type) and issubclass(self.output_model, BaseModel):
            # If it's already a class (shouldn't happen after before validator, but handle it)
            self._resolved_output_model = self.output_model
        return self

    @property
    def template_metadata(self) -> dict[str, Any]:
        """Backward compatibility property for template_metadata."""
        return self._template_metadata

    async def process(
        self,
        record: Any = BaseRecord,
        *,
        processor_stage: str,
        parent_trace_id: Optional[str] = None,
        component_name: str = "LLMCore",
        cancellation_token: Optional[CancellationToken] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Unified LLM processing method for Pipeline operations.

        Args:
            inputs: BaseRecord object
            parent_trace_id: Optional parent trace ID for correlation
            component_name: Name of the component using this (for tracing)
            cancellation_token: Optional token for cancelling LLM calls
            **kwargs: Additional input variables passed as keyword arguments

        Yields:
            BaseRecord: The LLM output directly (typed model or string)

        Raises:
            ProcessingError: If processing fails (fail-fast semantics)
        """
        start_time = time.time()
        tracer = trace.get_tracer("buttermilk.llm_core")

        # Build span attributes
        span_attributes = {
            "llm.model": self.model,
            "llm.template": self.template,
            "component.name": component_name,
            "processor.stage": processor_stage,
        }
        if parent_trace_id:
            span_attributes["parent_trace_id"] = parent_trace_id

        with tracer.start_as_current_span("llm_core.unified_process", attributes=span_attributes) as span:
            result = None  # Initialize for error handling
            try:
                # Build template_vars: if kwargs provided, merge with record fields
                # Record content is separately handled by make_messages at {{ record }} placeholders
                # Exclude computed hashes to prevent duplication in trace
                template_vars_derived_from_record = False
                if kwargs:
                    # Merge record fields with kwargs to create template_vars
                    template_vars = {
                        **(record.model_dump(exclude={"record_hash", "ground_truth_hash"}) if record and hasattr(record, "model_dump") else {}),
                        **kwargs,
                    }
                    template_vars_derived_from_record = bool(record)
                else:
                    template_vars = None

                result = await self.process_with_llm(
                    template_vars=template_vars,
                    record=record,
                    parent_trace_id=parent_trace_id,
                    cancellation_token=cancellation_token,
                    _template_vars_derived_from_record=template_vars_derived_from_record,
                )
                # Create ExecutionTrace for observability
                duration_ms = (time.time() - start_time) * 1000

                # Get model configuration for complete traceability
                # model_configs contains temperature, api_version, safety_settings, etc. from models.json
                model_configs = {}
                if self.model in bm.llms.connections:
                    llm_config = bm.llms.connections[self.model]
                    model_configs = llm_config.configs.copy() if llm_config.configs else {}

                # Add model config to extra_metadata for traceability
                # (model, template, temperature etc. are static config, not template vars)
                config_metadata = {
                    "llm_config": {
                        "model": self.model,
                        "template": self.template,
                        **model_configs,
                    }
                }
                combined_metadata = {**result.metadata, **config_metadata}

                # Emit success trace using inherited helper
                # inputs contains template variables (criteria, instructions, etc.)
                # Static config (model, template) is in metadata.llm_config
                await self._emit_success_trace(
                    record=record,
                    outputs=result.content,
                    processor_stage=processor_stage,
                    parent_trace_id=parent_trace_id,
                    duration_ms=duration_ms,
                    messages=result.messages,
                    inputs=result.resolved_inputs if result.resolved_inputs else kwargs,
                    extra_metadata=combined_metadata,
                    execution_type="llm_processing",
                    trace_id=result.trace_id,
                    component_name=component_name,
                )

                span.set_status(trace.Status(trace.StatusCode.OK))

                span.set_status(trace.Status(trace.StatusCode.OK))

                # Yield the content directly
                yield result.content

            except ProcessingError as e:
                # Create error trace with same structure as success trace
                duration_ms = (time.time() - start_time) * 1000

                # Emit error trace using inherited helper
                # inputs contains template variables (criteria, instructions, etc.)
                # Use kwargs as fallback if result was never assigned
                inputs_for_trace = kwargs
                if result is not None and result.resolved_inputs:
                    inputs_for_trace = result.resolved_inputs
                await self._emit_error_trace(
                    record=record,
                    error=e,
                    processor_stage=processor_stage,
                    parent_trace_id=parent_trace_id,
                    duration_ms=duration_ms,
                    inputs=inputs_for_trace,
                    execution_type="llm_processing",
                    component_name=component_name,
                )

                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

            except Exception as e:
                logger.error(f"Unexpected error in LLMCore.process: {e}")
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise ProcessingError(f"LLMCore processing failed: {e}") from e

    async def process_with_llm(
        self,
        template_vars: dict[str, Any] | None = None,
        *,
        record: Optional[BaseRecord] = None,
        context: Optional[list[LLMMessage]] = None,
        parent_trace_id: Optional[str] = None,
        cancellation_token: Optional[CancellationToken] = None,
        _template_vars_derived_from_record: bool = False,
    ) -> LLMResult:
        """Process through template rendering and LLM calling.

        This is the main entry point for both agents and processors.

        Args:
            template_vars: Variables to fill Jinja2 template placeholders.
            record: Optional record for {{ record }} placeholders. Handled by
                   make_messages() which calls record.as_message() to insert content.

            context: Optional conversation history for message context injection.
            parent_trace_id: Optional parent trace ID for correlation.
            cancellation_token: Optional token for cancelling LLM calls.

        Returns:
            LLMResult with the processed output and metadata.

        Example:
            result = await llm_core.process_with_llm(
                template_vars={"question": "What is 2+2?"},
                record=document_record,  # Inserted at {{ record }} placeholder
                context=conversation_history,
            )
        """
        # Lazy import to avoid loading litellm at module load time
        from buttermilk._core.llms import ModelOutput

        tracer = trace.get_tracer("buttermilk.llm_core")
        result = LLMResult(content=None, error=None)

        # Build span attributes
        span_attributes = {
            "llm.model": self.model,
            "llm.template": self.template,
        }
        if parent_trace_id:
            span_attributes["parent_trace_id"] = parent_trace_id

        with tracer.start_as_current_span("llm_core.process", attributes=span_attributes) as span:
            try:
                # === NORMALIZE INPUTS ===
                # template_vars are passed explicitly; record is handled separately
                # by make_messages() which inserts it at {{ record }} placeholders
                if template_vars is None:
                    template_vars = {}

                # Ensure context is always a list
                if context is None:
                    context = []
                elif not isinstance(context, list):
                    context = [context]

                # Detect potential record mismatch
                # Check if template_vars was passed as a kwarg (nested dict case from process() method)
                if template_vars is not None and record is not None:
                    # If template_vars dict contains a 'template_vars' key, user passed it through process()
                    actual_template_vars = template_vars.get("template_vars", template_vars)
                    tv_text = actual_template_vars.get("text") if isinstance(actual_template_vars, dict) else None
                    record_text = getattr(record, "text", None)
                    if tv_text and record_text and tv_text != record_text:
                        logger.warning(
                            "Record mismatch detected: template_vars.text differs from record.text. This may indicate data integrity issues."
                        )

                # Store resolved inputs for traceability
                # Avoid duplication: if template_vars was derived from record, strip BULKY
                # content fields but keep lightweight identifiers (record_id, dataset_name, etc.)
                # Full record is in trace.record
                if _template_vars_derived_from_record and record:
                    # Only strip bulky content fields - keep identifiers for quick reference
                    bulky_fields = {"text", "content", "metadata", "images", "attachments", "embedding"}
                    template_vars_for_trace = {k: v for k, v in template_vars.items() if k not in bulky_fields}
                else:
                    template_vars_for_trace = template_vars

                # Flatten template_vars directly into inputs (no wrapper)
                # Record data lives ONLY in trace.record, not duplicated in inputs
                # Include both config-time template_vars and runtime template_vars
                result.resolved_inputs = {
                    **self.template_vars,  # Config template vars (criteria, instructions, etc.)
                    **template_vars_for_trace,  # Runtime template vars (override config)
                    "context": context,  # Conversation history (reserved key)
                }

                # Store this for later use in trace emission
                result.metadata["_template_vars_from_record"] = _template_vars_derived_from_record

                # Fill template
                llm_messages = await self._fill_template(template_vars, record=record, context=context)

                # Store template metadata (without hash - hash goes to hashes dict)
                result.metadata["template"] = {
                    "template_name": self._template_metadata.get("template_name"),
                    "unfilled_vars": self._template_metadata.get("unfilled_vars", []),
                }

                # Consolidate all hashes in metadata.hashes
                result.metadata["hashes"] = {
                    "template_hash": self._template_metadata.get("template_hash"),
                }
                if record is not None:
                    result.metadata["hashes"]["record_hash"] = record.record_hash
                    if hasattr(record, "ground_truth_hash") and record.ground_truth_hash:
                        result.metadata["hashes"]["ground_truth_hash"] = record.ground_truth_hash
                    # Store record_id separately (not a hash)
                    result.metadata["record_id"] = record.record_id

                # Call LLM
                llm_result = await self._call_llm_with_trace(
                    messages=llm_messages,
                    cancellation_token=cancellation_token,
                    parent_trace_id=parent_trace_id,
                )

                # Check for errors in LLM result
                if isinstance(llm_result, ModelOutput) and llm_result.error_message:
                    raise ProcessingError(llm_result.error_message)

                # Extract content based on output type
                if self._resolved_output_model and isinstance(llm_result, ModelOutput):
                    result.content = llm_result.parsed_object
                else:
                    result.content = llm_result.content

                # Store messages (input prompts + LLM response)
                from autogen_core.models import AssistantMessage

                result.messages = llm_messages.copy()
                # Add the assistant's response as a message
                if result.content:
                    # Convert content to string, handling BaseModel via model_dump_json
                    if isinstance(result.content, str):
                        content_str = result.content
                    elif hasattr(result.content, "model_dump_json"):
                        content_str = result.content.model_dump_json()
                    else:
                        content_str = str(result.content)

                    result.messages.append(AssistantMessage(content=content_str, source=self.model))

                # Collect metadata (preserve existing template metadata)
                # Model name comes from LLM wrapper (actual from API or config as fallback)
                model_name = self.model  # Default to config name
                if isinstance(llm_result, ModelOutput) and hasattr(llm_result, "metadata"):
                    # Use model from wrapper (already contains actual API model or fallback)
                    model_name = llm_result.metadata.get("model", self.model)

                result.metadata = {
                    **result.metadata,  # Keep template metadata added earlier
                    "model": model_name,  # Actual model from API or config name as fallback
                    "finish_reason": llm_result.finish_reason,
                    "usage": llm_result.usage,
                }

                # Add pricing if available
                if isinstance(llm_result, ModelOutput) and hasattr(llm_result, "metadata"):
                    if "pricing" in llm_result.metadata:
                        result.metadata["pricing"] = llm_result.metadata["pricing"]

                # Ensure all metadata is serializable
                result.metadata = scrub_serializable(result.metadata)

                span.set_status(trace.Status(trace.StatusCode.OK))

            except ProcessingError as e:
                # Don't log here - let the final handler log once
                result.error = str(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

            except Exception as e:
                logger.error(f"Unexpected error in LLMCore: {e}")
                result.error = f"Unexpected error: {e}"
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise ProcessingError(f"LLMCore processing failed: {e}") from e

        return result

    async def _fill_template(
        self,
        template_vars: dict[str, Any],
        *,
        record: BaseRecord = None,
        context: list[LLMMessage] | None = None,
    ) -> list[LLMMessage]:
        """Render the template with provided data.

        Args:
            template_vars: Runtime variables to fill template placeholders.
            record: Optional record for render_or_include placeholders.
            context: Optional conversation history for context injection.

        Template variable precedence (later overrides earlier):
        1. self.template_vars (from LLMCore config, set at init)
        2. template_vars argument (runtime variables from caller)
        """
        if not self.template:
            raise ProcessingError("'template' is required but not specified")

        logger.debug(f"LLMCore: Using template '{self.template}'")

        # Render template using shared utility (handles merging and fail-on-unfilled)
        result = render_template(
            template=self.template,
            template_vars=template_vars,
            base_template_vars=self.template_vars,
            fail_on_unfilled=self.fail_on_unfilled_parameters,
        )

        # Convert to LLM messages
        try:
            llm_messages, processed_placeholders = make_messages(
                local_template=result.rendered, record=record, context=context
            )
        except Exception as e:
            raise ProcessingError(f"Failed to create messages from template '{self.template}'") from e

        # Update unfilled vars (remove any processed placeholders)
        unfilled_vars = set(result.unfilled_vars) - processed_placeholders
        if unfilled_vars:
            logger.warning(f"Template has unfilled parameters: {unfilled_vars}")

        # Store template metadata
        self._template_metadata = {
            "template_name": result.template_name,
            "template_hash": result.template_hash,
            "unfilled_vars": list(unfilled_vars) if unfilled_vars else [],
        }

        logger.debug(f"Template '{result.template_name}' rendered into {len(llm_messages)} messages")
        return llm_messages

    async def _call_llm_with_trace(
        self,
        messages: list[LLMMessage],
        cancellation_token: Optional[CancellationToken],
        parent_trace_id: Optional[str],
    ) -> CreateResult | ModelOutput:
        """Call the LLM with lightweight tracing.

        This provides the core LLM calling logic with observability
        but without agent-specific concepts.
        """

        tracer = trace.get_tracer("buttermilk.llm_core")

        # Build span attributes
        span_attributes = {
            "llm.model": self.model,
            "llm.message_count": len(messages),
            "llm.has_tools": len(self.tools) > 0,
            "llm.has_schema": self._resolved_output_model is not None,
        }
        if parent_trace_id:
            span_attributes["parent_trace_id"] = parent_trace_id

        with tracer.start_as_current_span("llm_core.call_llm", attributes=span_attributes) as span:
            try:
                # Get LLM client from global BM instance
                model_client = bm.llms.get_autogen_chat_client(self.model)

                logger.debug(
                    f"LLMCore: Calling {self.model} with {len(messages)} messages, {len(self.tools)} tools, schema={self._resolved_output_model}",
                    model=self.model,
                    message_count=len(messages),
                    tool_count=len(self.tools),
                    schema=self._resolved_output_model if self._resolved_output_model else None,
                )

                # Make the actual LLM call, respecting API concurrency limits
                from buttermilk._core.context import ApiSemaphoreContext

                async with ApiSemaphoreContext():
                    result = await model_client.call_chat(
                        messages=messages,
                        tools_list=self.tools,
                        cancellation_token=cancellation_token,
                        schema=self._resolved_output_model,
                    )

                # Record token usage in span if available
                if hasattr(result, "usage") and result.usage:
                    prompt_tokens = getattr(result.usage, "prompt_tokens", None)
                    completion_tokens = getattr(result.usage, "completion_tokens", None)

                    if prompt_tokens is not None:
                        span.set_attribute("llm.usage.prompt_tokens", prompt_tokens)
                    if completion_tokens is not None:
                        span.set_attribute("llm.usage.completion_tokens", completion_tokens)

                    # Calculate total tokens from prompt + completion
                    if prompt_tokens is not None and completion_tokens is not None:
                        total = prompt_tokens + completion_tokens
                        span.set_attribute("llm.usage.total_tokens", total)

                span.set_status(trace.Status(trace.StatusCode.OK))
                return result

            except Exception as e:
                # Don't log here - let the final handler log once
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise ProcessingError(f"LLM call to '{self.model}' failed: {e}") from e
