"""Core LLM functionality shared between agents and processors.

This module provides the LLMCore class that encapsulates the essential
LLM operations (template rendering, LLM calling, tracing) that can be
reused across different contexts - both in Agent-based flows and in
pipeline processors.

The design intentionally avoids Agent-specific concepts to maintain
flexibility while preserving full observability through metadata tracking.
"""

from typing import Any, Optional
import uuid

import pydantic
from autogen_core import CancellationToken
from autogen_core.models import LLMMessage
from autogen_core.tools import Tool
from opentelemetry import trace
from pydantic import BaseModel, Field

from buttermilk import logger
from buttermilk._core.exceptions import ProcessingError
from buttermilk._core.llms import CreateResult, ModelOutput
from buttermilk._core.types import BaseRecord
from buttermilk.utils.templating import load_template, make_messages
from buttermilk.utils.utils import clean_empty_values


class LLMResult(BaseModel):
    """Lightweight result structure for LLM operations.

    This provides a simple, standardized format for LLM results that
    can be used in both agent and pipeline contexts.
    """

    content: Any = Field(..., description="The LLM output - string or parsed object")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Usage, pricing, model info")
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for correlation")
    template_metadata: dict[str, Any] = Field(default_factory=dict, description="Template name, hash, etc")
    error: Optional[str] = Field(None, description="Error message if processing failed")


class LLMCore:
    """Core LLM functionality shared between agents and processors.

    This class extracts the essential LLM operations from LLMAgent,
    making them reusable in different contexts while maintaining
    observability and traceability.

    Key responsibilities:
    - Template loading and rendering
    - LLM API calls with retry logic
    - Lightweight tracing without agent concepts
    - Metadata tracking for observability
    """

    def __init__(
        self,
        parameters: dict[str, Any],
        output_model: Optional[type[pydantic.BaseModel]] = None,
        tools: Optional[list[Tool]] = None
    ):
        """Initialize the LLM core with configuration.

        Args:
            parameters: Configuration dict containing at minimum:
                - model: Name of the LLM model to use
                - template: Name of the template to render
                - temperature: Optional temperature setting
                - fail_on_unfilled_parameters: Whether to fail on missing template vars
            output_model: Optional Pydantic model for structured output
            tools: Optional list of tools the LLM can use
        """
        self.parameters = parameters
        self.output_model = output_model
        self.tools = tools or []

        # Extract commonly used parameters
        self._model = parameters.get("model")
        self._template = parameters.get("template")
        self._fail_on_unfilled_parameters = parameters.get("fail_on_unfilled_parameters", True)

        if not self._model:
            raise ValueError("'model' is required in parameters")
        if not self._template:
            raise ValueError("'template' is required in parameters")

        # Template metadata for tracking
        self._template_metadata: dict[str, Any] = {}

    async def process_with_llm(
        self,
        inputs: dict[str, Any],
        context: Optional[list[LLMMessage]] = None,
        records: Optional[list[BaseRecord]] = None,
        parent_trace_id: Optional[str] = None,
        cancellation_token: Optional[CancellationToken] = None
    ) -> LLMResult:
        """Process inputs through template rendering and LLM calling.

        This is the main entry point that orchestrates the full LLM workflow.

        Args:
            inputs: Input data to inject into the template
            context: Optional conversation history
            records: Optional records to include in template rendering
            parent_trace_id: Optional parent trace ID for correlation
            cancellation_token: Optional token for cancelling LLM calls

        Returns:
            LLMResult with the processed output and metadata
        """
        tracer = trace.get_tracer("buttermilk.llm_core")
        result = LLMResult(content=None)

        # Build span attributes, filtering out None values
        span_attributes = {
            "llm.model": self._model,
            "llm.template": self._template,
        }
        if parent_trace_id:
            span_attributes["parent_trace_id"] = parent_trace_id

        with tracer.start_as_current_span(
            "llm_core.process",
            attributes=span_attributes
        ) as span:
            try:
                # Fill template
                llm_messages = await self._fill_template(
                    inputs=inputs,
                    context=context or [],
                    records=records or []
                )

                # Store template metadata
                result.template_metadata = self._template_metadata

                # Call LLM
                llm_result = await self._call_llm_with_trace(
                    messages=llm_messages,
                    cancellation_token=cancellation_token,
                    parent_trace_id=parent_trace_id
                )

                # Extract content based on output type
                if self.output_model and isinstance(llm_result, ModelOutput):
                    result.content = llm_result.parsed_object
                else:
                    result.content = llm_result.content

                # Collect metadata
                result.metadata = {
                    "model": self._model,
                    "finish_reason": llm_result.finish_reason,
                    "usage": llm_result.usage,
                }

                # Add pricing if available
                if isinstance(llm_result, ModelOutput) and hasattr(llm_result, "metadata"):
                    if "pricing" in llm_result.metadata:
                        result.metadata["pricing"] = llm_result.metadata["pricing"]

                span.set_status(trace.Status(trace.StatusCode.OK))

            except ProcessingError as e:
                logger.error(f"LLMCore processing error: {e}")
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
        inputs: dict[str, Any],
        context: list[LLMMessage],
        records: list[BaseRecord]
    ) -> list[LLMMessage]:
        """Render the template with provided data.

        Extracted from LLMAgent._fill_template with minor adaptations
        for the more generic interface.
        """
        template_name = self._template
        if not template_name:
            raise ProcessingError("'template' is required but not specified")

        logger.debug(f"LLMCore: Using template '{template_name}'")

        # Clean and prepare inputs
        filtered_inputs = clean_empty_values(inputs).copy() if inputs else {}

        # Check for duplicate prompts in context
        if context and "prompt" in filtered_inputs:
            from autogen_core.models import UserMessage
            last_msg = context[-1]
            if isinstance(last_msg, UserMessage) and last_msg.content == filtered_inputs["prompt"]:
                logger.debug("Removing duplicate prompt from inputs")
                del filtered_inputs["prompt"]

        # Load and render template
        rendered_template_str, unfilled_vars, template_hash = load_template(
            template=template_name,
            parameters=self.parameters,
            untrusted_inputs=filtered_inputs
        )

        try:
            llm_messages = make_messages(
                local_template=rendered_template_str,
                context=context,
                records=records
            )

            # Remove from unfilled if provided
            if context:
                unfilled_vars.discard("context")
            if records:
                unfilled_vars.discard("records")

        except Exception as e:
            raise ProcessingError(f"Failed to create messages from template '{template_name}'") from e

        # Check for missing variables
        if unfilled_vars and self._fail_on_unfilled_parameters:
            raise ProcessingError(
                f"Template '{template_name}' has unfilled parameters: {', '.join(sorted(unfilled_vars))}"
            )
        elif unfilled_vars:
            logger.warning(f"Template has unfilled parameters: {unfilled_vars}")

        # Store template metadata
        self._template_metadata = {
            "template_name": template_name,
            "template_hash": template_hash,
            "unfilled_vars": list(unfilled_vars) if unfilled_vars else []
        }

        logger.debug(f"Template '{template_name}' rendered into {len(llm_messages)} messages")
        return llm_messages

    async def _call_llm_with_trace(
        self,
        messages: list[LLMMessage],
        cancellation_token: Optional[CancellationToken],
        parent_trace_id: Optional[str]
    ) -> CreateResult | ModelOutput:
        """Call the LLM with lightweight tracing.

        This provides the core LLM calling logic with observability
        but without agent-specific concepts.
        """
        from buttermilk import get_bm

        tracer = trace.get_tracer("buttermilk.llm_core")

        # Build span attributes
        span_attributes = {
            "llm.model": self._model,
            "llm.message_count": len(messages),
            "llm.has_tools": len(self.tools) > 0,
            "llm.has_schema": self.output_model is not None,
        }
        if parent_trace_id:
            span_attributes["parent_trace_id"] = parent_trace_id

        with tracer.start_as_current_span(
            "llm_core.call_llm",
            attributes=span_attributes
        ) as span:
            try:
                # Get LLM client from global BM instance
                bm = get_bm()
                model_client = bm.llms.get_autogen_chat_client(self._model)

                logger.debug(
                    f"LLMCore: Calling {self._model} with {len(messages)} messages, "
                    f"{len(self.tools)} tools, schema={self.output_model}"
                )

                # Make the actual LLM call
                result = await model_client.call_chat(
                    messages=messages,
                    tools_list=self.tools,
                    cancellation_token=cancellation_token,
                    schema=self.output_model
                )

                # Record token usage in span if available
                if hasattr(result, "usage") and result.usage:
                    if hasattr(result.usage, "prompt_tokens"):
                        span.set_attribute("llm.usage.prompt_tokens", result.usage.prompt_tokens)
                    if hasattr(result.usage, "completion_tokens"):
                        span.set_attribute("llm.usage.completion_tokens", result.usage.completion_tokens)
                    # Calculate total tokens from prompt + completion
                    if hasattr(result.usage, "prompt_tokens") and hasattr(result.usage, "completion_tokens"):
                        total = result.usage.prompt_tokens + result.usage.completion_tokens
                        span.set_attribute("llm.usage.total_tokens", total)

                span.set_status(trace.Status(trace.StatusCode.OK))
                return result

            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise ProcessingError(f"LLM call to '{self._model}' failed: {e}") from e