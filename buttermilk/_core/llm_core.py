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

import time
import uuid
from typing import Any, AsyncGenerator, Optional

import pydantic
from autogen_core import CancellationToken
from autogen_core.models import LLMMessage
from autogen_core.tools import Tool
from opentelemetry import trace
from pydantic import BaseModel, Field

from buttermilk import bm, logger
from buttermilk._core.contract import ErrorEvent, ExecutionTrace
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.llms import CreateResult, ModelOutput
from buttermilk._core.processor_core import ProcessorCore
from buttermilk._core.types import BaseRecord
from buttermilk.utils.templating import load_template, make_messages
from buttermilk.utils.utils import clean_empty_values, scrub_serializable
from buttermilk.utils.validators import import_class_from_path


class LLMResult(BaseModel):
    """Lightweight result structure for LLM operations.

    This provides a simple, standardized format for LLM results that
    can be used in both agent and pipeline contexts.
    """

    content: Any = Field(..., description="The LLM output - string or parsed object")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Usage, pricing, model info"
    )
    trace_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for correlation",
    )
    template_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Template name, hash, etc"
    )
    messages: list[LLMMessage] = Field(
        default_factory=list, description="Messages exchanged with LLM"
    )
    error: str | ErrorEvent | None = Field(
        None, description="Error message if processing failed"
    )
    resolved_inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="All resolved inputs used for template rendering",
    )


class LLMCore(ProcessorCore):
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

    def __init__(  # noqa: PLR0913
        self,
        model: str,
        template: str,
        output_model: Optional[type[pydantic.BaseModel] | str] = None,
        tools: Optional[list[Tool]] = None,
        fail_on_unfilled_parameters: bool = True,
        output_col: str = "output",
        **kwargs: Any,
    ):
        """Initialize the LLM core with configuration.

        Args:
            parameters: Configuration dict containing at minimum:
                - model: Name of the LLM model to use
                - template: Name of the template to render
                - temperature: Optional temperature setting
                - fail_on_unfilled_parameters: Whether to fail on missing template vars
            output_model: Optional Pydantic model for structured output (class or string path)
            tools: Optional list of tools the LLM can use
        """
        # CRITICAL: Include model and template in parameters for trace writing
        # Issue #280: Traces need these fields for observability/analysis
        # Initialize ProcessorCore with model and template in kwargs
        super().__init__(model=model, template=template, **kwargs)

        # Resolve output_model if it's a string
        if isinstance(output_model, str):
            try:
                self.output_model = import_class_from_path(
                    output_model, expected_base_class=pydantic.BaseModel
                )
            except (ImportError, AttributeError, ValueError) as e:
                raise ProcessingError(
                    f"Failed to resolve output_model '{output_model}': {e}"
                )
        else:
            self.output_model = output_model

        self.tools = tools or []
        self.output_col = output_col

        # Extract commonly used parameters
        self.model = model
        self.template = template
        self._fail_on_unfilled_parameters = fail_on_unfilled_parameters

        # Template metadata for tracking
        self.template_metadata: dict[str, Any] = {}

        # Initialize trace writer (lazy loading)
        self._trace_writer = None

    @property
    def trace_writer(self) -> Any:
        """Lazy load trace writer."""
        if self._trace_writer is None:
            from buttermilk.utils.trace_writer import get_trace_writer

            self._trace_writer = get_trace_writer()
        return self._trace_writer

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
            BaseRecord: Enriched record with LLM output in self.output_col and metadata

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

        with tracer.start_as_current_span(
            "llm_core.unified_process", attributes=span_attributes
        ) as span:
            try:
                # Build template_vars: if kwargs provided, merge with record fields
                # Otherwise let process_with_llm derive from record
                template_vars = (
                    {**(record.model_dump() if record and hasattr(record, "model_dump") else {}), **kwargs} if kwargs else None
                )

                result = await self.process_with_llm(
                    template_vars=template_vars,
                    record=record,
                    parent_trace_id=parent_trace_id,
                    cancellation_token=cancellation_token,
                )
                # Create ExecutionTrace for observability
                duration_ms = (time.time() - start_time) * 1000

                # Get model configuration for complete traceability
                # model_configs contains temperature, api_version, safety_settings, etc. from models.json
                model_configs = {}
                if self.model in bm.llms.connections:
                    llm_config = bm.llms.connections[self.model]
                    model_configs = llm_config.configs.copy() if llm_config.configs else {}

                # Build trace parameters: LLMCore params + model configs (no duplication)
                # - self.parameters has: model (alias), template, and any runtime kwargs
                # - model_configs has: temperature, api_version, safety_settings, etc.
                trace_parameters = {**self.parameters, **model_configs}

                # Build metadata: merge input metadata (from record) with output metadata
                # Input metadata comes from record.metadata if available
                input_metadata = {}
                if record is not None and hasattr(record, "metadata") and record.metadata:
                    input_metadata = {"input": record.metadata}

                execution_trace = ExecutionTrace(
                    call_id=result.trace_id,
                    agent_info={
                        "component_name": component_name,
                        "execution_type": "llm_processing",
                        "processor_stage": processor_stage,
                    },
                    # inputs = template variables only (record/context stored separately)
                    inputs=result.resolved_inputs if result.resolved_inputs else kwargs,
                    outputs=result.content,
                    messages=result.messages,
                    parameters=trace_parameters,
                    # metadata merges: input metadata (under 'input' key) + output metadata + duration
                    metadata={
                        **input_metadata,
                        **result.metadata,
                        "duration_ms": duration_ms,
                    },
                    parent_call_id=parent_trace_id,
                    record=record,
                )

                # Emit trace if trace writer is available
                if hasattr(self, "trace_writer") and self.trace_writer:
                    try:
                        await self.trace_writer.add(execution_trace)
                    except Exception as e:
                        logger.warning(f"Failed to emit trace: {e}")

                span.set_status(trace.Status(trace.StatusCode.OK))

                enriched_record = record.model_copy(
                    update={
                        self.output_col: result.content,
                        "metadata": {
                            **record.metadata,
                            processor_stage: result.metadata,
                        },
                    }
                )
                yield enriched_record

            except ProcessingError as e:
                # Create error trace with same structure as success trace
                duration_ms = (time.time() - start_time) * 1000

                # Get model configs for error trace too
                error_model_configs = {}
                if self.model in bm.llms.connections:
                    error_llm_config = bm.llms.connections[self.model]
                    error_model_configs = error_llm_config.configs.copy() if error_llm_config.configs else {}

                # Build error metadata with input metadata if available
                error_input_metadata = {}
                if record is not None and hasattr(record, "metadata") and record.metadata:
                    error_input_metadata = {"input": record.metadata}

                error_trace = ExecutionTrace(
                    agent_info={
                        "component_name": component_name,
                        "execution_type": "llm_processing",
                        "processor_stage": processor_stage,
                    },
                    # inputs = template variables only (kwargs passed to process)
                    inputs=kwargs if kwargs else None,
                    parameters={**self.parameters, **error_model_configs},
                    error={
                        "event": str(e),
                        "details": {"error_type": type(e).__name__},
                    },
                    metadata={
                        **error_input_metadata,
                        "duration_ms": duration_ms,
                    },
                    parent_call_id=parent_trace_id,
                    record=record,
                )

                # Emit error trace
                if hasattr(self, "trace_writer") and self.trace_writer:
                    try:
                        await self.trace_writer.add(error_trace)
                    except Exception as te:
                        logger.warning(f"Failed to emit error trace: {te}")

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
    ) -> LLMResult:
        """Process through template rendering and LLM calling.

        This is the main entry point for both agents and processors.

        Args:
            template_vars: Variables to fill Jinja2 template placeholders.
                          If None and record is provided, uses record.model_dump().
            record: Optional record for {{ render_or_include(record) }} placeholders.
                   NOT used as template variables unless template_vars is None.
            context: Optional conversation history for message context injection.
            parent_trace_id: Optional parent trace ID for correlation.
            cancellation_token: Optional token for cancelling LLM calls.

        Returns:
            LLMResult with the processed output and metadata.

        Example - Agent mode (explicit template vars):
            result = await llm_core.process_with_llm(
                template_vars={"question": "What is 2+2?"},
                record=document_record,
                context=conversation_history,
            )

        Example - Processor mode (template vars from record):
            result = await llm_core.process_with_llm(
                template_vars=None,  # Will use record.model_dump()
                record=enriched_record,
            )
        """
        tracer = trace.get_tracer("buttermilk.llm_core")
        result = LLMResult(content=None, error=None)

        # Build span attributes
        span_attributes = {
            "llm.model": self.model,
            "llm.template": self.template,
        }
        if parent_trace_id:
            span_attributes["parent_trace_id"] = parent_trace_id

        with tracer.start_as_current_span(
            "llm_core.process", attributes=span_attributes
        ) as span:
            try:
                # === NORMALIZE INPUTS ===
                # If template_vars not provided, derive from record
                if template_vars is None:
                    template_vars = record.model_dump() if record else {}

                # Ensure context is always a list
                if context is None:
                    context = []
                elif not isinstance(context, list):
                    context = [context]

                # Store resolved inputs for traceability
                result.resolved_inputs = {
                    "template_vars": template_vars,
                    "record": record,
                    "context": context,
                }

                # Fill template
                llm_messages = await self._fill_template(
                    template_vars, record=record, context=context
                )

                # Store template metadata
                result.metadata["template"] = self.template_metadata

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
                if self.output_model and isinstance(llm_result, ModelOutput):
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

                    result.messages.append(
                        AssistantMessage(content=content_str, source=self.model)
                    )

                # Collect metadata (preserve existing template metadata)
                # Model name comes from LLM wrapper (actual from API or config as fallback)
                model_name = self.model  # Default to config name
                if isinstance(llm_result, ModelOutput) and hasattr(
                    llm_result, "metadata"
                ):
                    # Use model from wrapper (already contains actual API model or fallback)
                    model_name = llm_result.metadata.get("model", self.model)

                result.metadata = {
                    **result.metadata,  # Keep template metadata added earlier
                    "model": model_name,  # Actual model from API or config name as fallback
                    "finish_reason": llm_result.finish_reason,
                    "usage": llm_result.usage,
                }

                # Add pricing if available
                if isinstance(llm_result, ModelOutput) and hasattr(
                    llm_result, "metadata"
                ):
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
            template_vars: Dictionary of variables to fill template placeholders.
            record: Optional record for render_or_include placeholders.
            context: Optional conversation history for context injection.
        """
        template_name = self.template
        if not template_name:
            raise ProcessingError("'template' is required but not specified")

        logger.debug(f"LLMCore: Using template '{template_name}'")

        # Clean and prepare template variables
        filtered_vars = clean_empty_values(template_vars) if template_vars else {}

        # Load and render template
        rendered_template_str, unfilled_vars, template_hash = load_template(
            template=template_name,
            parameters=self.parameters,
            untrusted_inputs=filtered_vars,
        )

        try:
            llm_messages, processed_placeholders = make_messages(
                local_template=rendered_template_str, record=record, context=context
            )
        except Exception as e:
            raise ProcessingError(
                f"Failed to create messages from template '{template_name}'"
            ) from e

        unfilled_vars -= processed_placeholders

        # Check for missing variables
        if unfilled_vars and self._fail_on_unfilled_parameters:
            raise ProcessingError(
                f"Template '{template_name}' has unfilled parameters: {', '.join(sorted(unfilled_vars))}"
            )
        elif unfilled_vars:
            logger.warning(f"Template has unfilled parameters: {unfilled_vars}")

        # Store template metadata
        self.template_metadata = {
            "template_name": template_name,
            "template_hash": template_hash,
            "unfilled_vars": list(unfilled_vars) if unfilled_vars else [],
        }

        logger.debug(
            f"Template '{template_name}' rendered into {len(llm_messages)} messages"
        )
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
            "llm.has_schema": self.output_model is not None,
        }
        if parent_trace_id:
            span_attributes["parent_trace_id"] = parent_trace_id

        with tracer.start_as_current_span(
            "llm_core.call_llm", attributes=span_attributes
        ) as span:
            try:
                # Get LLM client from global BM instance
                model_client = bm.llms.get_autogen_chat_client(self.model)

                logger.debug(
                    f"LLMCore: Calling {self.model} with {len(messages)} messages, {len(self.tools)} tools, schema={self.output_model}",
                    model=self.model,
                    message_count=len(messages),
                    tool_count=len(self.tools),
                    schema=self.output_model if self.output_model else None,
                )

                # Make the actual LLM call, respecting API concurrency limits
                from buttermilk._core.context import ApiSemaphoreContext

                async with ApiSemaphoreContext():
                    result = await model_client.call_chat(
                        messages=messages,
                        tools_list=self.tools,
                        cancellation_token=cancellation_token,
                        schema=self.output_model,
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
