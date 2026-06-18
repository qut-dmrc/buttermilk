"""Manages Language Model (LLM) configurations, clients, and interactions.

This module provides structures for defining LLM configurations (`LLMConfig`),
managing different LLM providers and their clients (`LLMs`), and wrapping chat
completion clients with additional functionality such as retry logic
(`LiteLLMWrapper`).

It uses LiteLLM as the unified interface for interacting with various LLM APIs
and provides a consistent interface for agents within the Buttermilk framework.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import random
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any

# LiteLLM is lazy-loaded to speed up import time (~3s savings)
# Use _get_litellm() and _get_acompletion() instead of direct imports
_litellm_module = None
_litellm_initialized = False


def _init_litellm():
    """Initialize litellm with logging suppression. Called once on first use."""
    global _litellm_module, _litellm_initialized
    if _litellm_initialized:
        return

    try:
        import litellm

        _litellm_module = litellm

        # Suppress litellm logging - we handle errors via retry wrapper
        litellm.suppress_debug_info = True

        # Suppress all LiteLLM loggers including internal workers
        logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
        logging.getLogger("litellm").setLevel(logging.CRITICAL)

        # Suppress asyncio logging for LiteLLM's internal background tasks
        litellm_logger = logging.getLogger("litellm.litellm_core_utils.logging_worker")
        litellm_logger.setLevel(logging.CRITICAL)
        litellm_logger.propagate = False

        _litellm_initialized = True
    except ImportError:
        _litellm_module = None
        _litellm_initialized = True


def _get_litellm():
    """Get the litellm module, initializing if needed."""
    _init_litellm()
    return _litellm_module


def _get_acompletion():
    """Get litellm.acompletion function."""
    litellm = _get_litellm()
    if litellm is None:
        return None
    return litellm.acompletion


def _litellm_available() -> bool:
    """Check if litellm is available."""
    return _get_litellm() is not None


# Module-level __getattr__ for lazy attribute access (PEP 562)
# This allows LITELLM_AVAILABLE to be imported without triggering litellm import
# until the attribute is actually accessed
def __getattr__(name: str):
    if name == "LITELLM_AVAILABLE":
        return _litellm_available()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Buttermilk types
# from google import genai  # Google Generative AI library (unused in current implementation)
from pydantic import (
    BaseModel,  # Pydantic models for configuration
    ConfigDict,
    Field,
    field_validator,
)

from buttermilk import logger
from buttermilk._core.constants import CONFIG_CACHE_FILENAME, cache, get_base_cache_dir  # Models cache constants
from buttermilk._core.exceptions import ContentBlockedError, ProcessingError  # Custom Buttermilk exceptions
from buttermilk._core.messages import (
    AssistantMessage,
    CreateResult,
    FunctionCall,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    ModelInfo,
    RequestUsage,
)
from buttermilk._core.tool_types import CancellationToken, Tool, ToolSchema
from buttermilk.utils.pricing import calculate_token_cost, extract_cached_tokens  # Token cost calculation


def litellm_provider_segment(litellm_model: str) -> str:
    """Return the leading litellm provider segment of a full model name.

    Examples:
        "vertex_ai/gemini-3-flash-preview" -> "vertex_ai"
        "vertex_ai/claude-sonnet-4-6"      -> "vertex_ai"
        "azure/gpt-5-mini"                 -> "azure"
        "azure_ai/grok-4-1-fast-non-reasoning" -> "azure_ai"
        "anthropic/claude-3-5-sonnet"      -> "anthropic"
        "gpt-4o" (no prefix)               -> "" (treated as bare OpenAI)

    The full provider-prefixed litellm name is the single source of truth for
    routing/auth/format; there is no ClientType enum any more.
    """
    if not litellm_model or "/" not in litellm_model:
        return ""
    return litellm_model.split("/", 1)[0]


class ModelParameters(BaseModel):
    """Inference parameters for LLM API calls.

    Provides a standardized interface for common inference parameters across
    different LLM providers. Allows provider-specific parameters via extra fields.

    All parameters are optional (None by default) to allow selective overrides
    when merging configurations.

    Attributes:
        temperature: Sampling temperature (0.0-2.0). Higher values make output
            more random, lower values more deterministic.
        max_tokens: Maximum number of tokens to generate. Must be positive.
        top_p: Nucleus sampling threshold (0.0-1.0). Alternative to temperature.
        top_k: Top-k sampling limit. Only the k most likely tokens are considered.
        frequency_penalty: Penalty for token frequency (-2.0 to 2.0). Positive
            values discourage repetition.
        presence_penalty: Penalty for token presence (-2.0 to 2.0). Positive
            values encourage topic diversity.
        stop_sequences: List of strings that will stop generation when encountered.
        seed: Random seed for deterministic sampling (if supported by provider).
    """

    model_config = ConfigDict(extra="allow")

    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, gt=0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    top_k: int | None = Field(None, gt=0)
    frequency_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    stop_sequences: list[str] | None = None
    seed: int | None = None

    def merge_with(self, other: ModelParameters | dict | None) -> ModelParameters:
        """Merge with another ModelParameters instance or dict.

        Non-None values from 'other' take precedence over self's values.
        This allows layering configurations where more specific configs
        override more general ones.

        Args:
            other: ModelParameters instance, dict, or None to merge with.
                If None, returns a copy of self.

        Returns:
            New ModelParameters instance with merged values.

        Example:
            base = ModelParameters(temperature=0.7, max_tokens=1000)
            override = ModelParameters(temperature=0.9)
            merged = base.merge_with(override)
            # Result: temperature=0.9, max_tokens=1000
        """
        if other is None:
            return self.model_copy(deep=True)

        # Convert dict to ModelParameters if needed
        if isinstance(other, dict):
            other = ModelParameters(**other)

        # Start with self's values
        merged_data = self.model_dump()

        # Override with other's non-None values
        other_data = other.model_dump()
        for key, value in other_data.items():
            if value is not None:
                merged_data[key] = value

        return ModelParameters(**merged_data)

    def to_api_params(self) -> dict[str, Any]:
        """Convert to API parameter dictionary.

        Returns dictionary containing only non-None values, suitable for
        passing to LLM API calls. Maps stop_sequences to 'stop' key for
        API compatibility.

        Returns:
            Dictionary with non-None parameter values, ready for API calls.

        Example:
            params = ModelParameters(temperature=0.7, max_tokens=1000)
            api_params = params.to_api_params()
            # Result: {'temperature': 0.7, 'max_tokens': 1000}
        """
        result: dict[str, Any] = {}

        # Get all fields including extras
        all_data = self.model_dump()

        for key, value in all_data.items():
            if value is not None:
                # Map stop_sequences to 'stop' for API compatibility
                if key == "stop_sequences":
                    result["stop"] = value
                else:
                    result[key] = value

        return result


class LLMConfig(BaseModel):
    """Configuration for a specific Language Model (LLM).

    Routing is driven entirely by the full provider-prefixed litellm model name
    (`litellm_model`, e.g. "vertex_ai/gemini-3-flash-preview", "azure/gpt-5-mini",
    "azure_ai/grok-4-1-fast-non-reasoning", "vertex_ai/claude-sonnet-4-6"). There is
    no longer a `ClientType` enum: litellm + ambient GCP ADC handle auth/transport,
    and the few genuine non-string-derivable branches (batch SDK + message-format
    selection) are derived from the litellm provider segment.

    Attributes:
        litellm_model (str): The full provider-prefixed litellm model identifier.
            Required. This is the single source of truth for provider routing.
        api_key (str | None): The API key for the provider, if needed. None when
            auth is ambient (e.g. Vertex via Application Default Credentials).
        base_url (str | None): Custom API endpoint (e.g. Azure / Azure AI / self-hosted).
        region (str | None): Vertex AI location, passed straight through as
            litellm's `vertex_location`. Required per-entry for region-pinned models
            (gemini-3.x need "global"); litellm defaults to us-central1 otherwise.
        api_version (str | None): Azure OpenAI api-version (used by the batch SDK path).
        model_info (ModelInfo): Model metadata (family, context size, etc.).
        configs (dict): Extra options. `configs["model"]` is the display/registry id.
        parameters (ModelParameters): Default inference parameters.
    """

    litellm_model: str = Field(
        ...,
        description="Full provider-prefixed litellm model identifier (e.g. 'vertex_ai/gemini-3-flash-preview'). Single source of truth for routing.",
    )
    api_key: str | None = Field(
        default=None,
        description="API key to use for this model (None when auth is ambient, e.g. Vertex ADC)",
    )
    base_url: str | None = Field(default=None, description="Custom URL to call")
    region: str | None = Field(
        default=None,
        description="Vertex AI location, passed through as litellm vertex_location (e.g. 'global', 'us-east5')",
    )
    api_version: str | None = Field(default=None, description="Azure OpenAI api-version (batch SDK path)")

    model_info: ModelInfo = Field(..., description="Model metadata (family, context size, etc.)")
    configs: dict = Field(default_factory=dict, description="Options to pass to the LiteLLM client")
    parameters: ModelParameters = Field(
        default_factory=ModelParameters,
        description="Default inference parameters (temperature, max_tokens, etc.)",
    )

    @property
    def provider_segment(self) -> str:
        """Leading litellm provider segment of `litellm_model` (e.g. 'vertex_ai', 'azure', 'azure_ai')."""
        return litellm_provider_segment(self.litellm_model)

    @field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters(cls, v: Any) -> ModelParameters:
        """Validate and convert parameters field to ModelParameters.

        Args:
            v: The input value to validate (dict, ModelParameters, or None)

        Returns:
            ModelParameters: The validated parameters instance

        Raises:
            ValueError: If parameters is not a dict or ModelParameters instance

        """
        if v is None:
            return ModelParameters()
        if isinstance(v, ModelParameters):
            return v
        if isinstance(v, dict):
            return ModelParameters(**v)
        raise ValueError(f"parameters must be a dict or ModelParameters, got {type(v)}")


# Generate with:
# ```sh
# cat .cache/buttermilk/models.json | jq "keys[]"
# ```
"""A predefined list of chat model identifiers available within the Buttermilk setup."""
CHAT_MODELS = [
    "google/gemini-3.1-pro-preview",
    "google/gemini-3-flash-preview",
    "google/gemini-3.5-flash",
    "google/gemini-3.1-flash-lite",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4o",
    "meta/llama-4-maverick-17b-128e-instruct-maas",
    "claude-sonnet-4-6",
    "deepseek-ai/deepseek-v3.2-maas",
]

"""A predefined list of identifiers for cost-effective chat models."""
CHEAP_CHAT_MODELS = [
    "google/gemini-3.1-flash-lite",
    "google/gemini-3.5-flash",
    "meta/llama-4-maverick-17b-128e-instruct-maas",
    "gpt-5-nano",
    "claude-haiku-4-5@20251001",
]


class ModelOutput(CreateResult):
    """Extends `CreateResult` with structured output parsing and pricing.

    Adds a parsed_object field to hold a Pydantic model instance when
    the LLM returns structured JSON output that can be parsed.

    This class is also used to encapsulate error responses from LLM calls,
    providing additional context about the error that occurred.

    Attributes:
        parsed_object (BaseModel | None): The Pydantic model instance hydrated from
            the LLM's JSON content. None if no structured output or parsing failed.
        error_message (str): A descriptive message about the error that occurred.
        error_code (int | None): An optional error code associated with the error.
            Can be None if no specific code is provided.
        raw_response (Any | None): The raw response from the LLM, if available.
        metadata (dict[str, Any]): Metadata including pricing information.

    """

    parsed_object: BaseModel | None = Field(
        default=None,
        description="The Pydantic model instance hydrated from LLM's JSON or structured output.",
    )
    error_message: str | None = Field(default=None, description="Descriptive message about the error")
    error_code: int | None = Field(default=None, description="Optional error code associated with the error")
    raw_response: Any | None = Field(default=None, description="Raw response from the LLM, if available")
    tool_outputs: list[FunctionExecutionResult] | None = Field(default=None, description="Tool outputs if any were executed")
    tool_calls: list[FunctionCall] | None = Field(default=None, description="Tool calls made by the LLM, if any")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata including pricing information")


def _validate_schema_constraints(schema: type[BaseModel], model_info: Any) -> None:
    """Validate that the schema meets provider requirements (e.g. OpenAI/Azure extra='forbid')."""
    # Normalize model_info to get family
    family = ""
    if isinstance(model_info, dict):
        family = model_info.get("family", "")
    else:
        # Assuming object with attributes (ModelInfo)
        family = getattr(model_info, "family", "")

    family = str(family).lower()

    # Check for OpenAI/Azure families
    # These providers strictly require additionalProperties: false in JSON schema
    if "gpt" in family or "openai" in family or "azure" in family:
        # Check Pydantic config for extra='forbid'
        config = getattr(schema, "model_config", {})
        extra = config.get("extra")

        if extra != "forbid":
            raise ValueError(
                f"Model family '{family}' requires structured output schemas to have "
                "extra='forbid'. Please add `model_config = ConfigDict(extra='forbid')` to your Pydantic model."
            )


def _convert_xml_params_to_json(xml_text: str) -> str:
    """Convert Claude's XML-like parameter format to JSON.

    Claude sometimes returns tool arguments in XML format like:
        <parameter name="field1">value1</parameter>
        <parameter name="field2">["item1", "item2"]</parameter>

    This converts it to proper JSON:
        {"field1": "value1", "field2": ["item1", "item2"]}
    """
    import json
    import re

    result = {}
    # Match <parameter name="key">value</parameter>
    pattern = r'<parameter\s+name="([^"]+)">(.*?)</parameter>'

    for match in re.finditer(pattern, xml_text, re.DOTALL):
        key = match.group(1)
        value_str = match.group(2).strip()

        # Try to parse the value as JSON (for lists, dicts, bools, numbers)
        try:
            value = json.loads(value_str)
        except json.JSONDecodeError:
            # If not valid JSON, use as string
            value = value_str

        result[key] = value

    return json.dumps(result)


async def _parse_structured_output(  # noqa: PLR0912
    content: str | dict | BaseModel,
    schema: type[BaseModel],
) -> type[BaseModel]:
    """Parse LLM response into structured output using the provided schema.

    Args:
        content: text or object to parse and/or validate into the schema
        schema: The Pydantic model to validate against

    Returns:
        Validated Pydantic model instance of the specified schema type.

    Raises:
        ProcessingError: If parsing or validation fails

    """
    parsed_object = None
    if isinstance(content, str):
        # Local dynamic import to avoid cycles and linter complaints about import location
        _mod = importlib.import_module("buttermilk.utils.json_parser")
        ChatParser = _mod.ChatParser
        simple_clean_llm_json_text = _mod.simple_clean_llm_json_text

        # Try to parse as strict JSON first to preserve types (avoid coercion)
        logger.debug(f"Attempting to parse string response into {schema.__name__}")
        text = simple_clean_llm_json_text(content)

        if parsed_object is None:
            try:
                parsed_object = schema.model_validate_json(text)
            except Exception:
                parsed_object = None

        if parsed_object is None:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    parsed_object = parsed
            except Exception:
                parsed_object = None

        if parsed_object is None:
            # Fallback to tolerant parser for messy outputs
            try:
                parser = ChatParser()
                parsed_object = parser.parse(text)
                # ChatParser returns error dict on failure instead of raising
                if isinstance(parsed_object, dict) and "error" in parsed_object and "response" in parsed_object:
                    raise ProcessingError(
                        f"Failed to parse LLM response as JSON: {parsed_object['error']}. Raw response: {parsed_object['response'][:200]}..."
                    )
            except ProcessingError:
                raise
            except Exception as parse_error:
                raise ProcessingError(
                    f"Failed to parse LLM response into required schema {schema.__name__}: {parse_error}",
                ) from parse_error
    else:
        parsed_object = content

    # Validate the parsed object against the schema
    try:
        parsed_object = schema.model_validate(parsed_object)
    except Exception as parse_error:
        raise ProcessingError(
            f"Failed to parse LLM response into required schema {schema.__name__}: {parse_error}",
        ) from parse_error

    if parsed_object is None:
        raise ProcessingError(
            f"Structured output of type {schema.__name__} required but parsing failed",
        )

    return parsed_object


# =============================================================================
# LiteLLM Integration - Message Format Converters
# =============================================================================


def _add_anthropic_cache_control(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add an ephemeral cache_control breakpoint to the last system message.

    Anthropic requires explicit cache_control breakpoints — without them nothing
    is cached. This marks the stable system prefix as cacheable by placing a
    ``{"type": "ephemeral"}`` breakpoint on the final system-role message, which
    is the natural stable/variable boundary in a buttermilk prompt.

    LiteLLM translates ``cache_control`` blocks to the Anthropic API format for
    both ``client_type=anthropic`` and ``client_type=anthropic_vertex``.
    """
    last_system_idx = None
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            last_system_idx = i

    if last_system_idx is None:
        return messages

    result = list(messages)
    msg = result[last_system_idx]
    content = msg["content"]

    if isinstance(content, str):
        result[last_system_idx] = {
            **msg,
            "content": [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}],
        }
    elif isinstance(content, list) and content:
        new_content = list(content)
        last_block = new_content[-1]
        if isinstance(last_block, dict) and "cache_control" not in last_block:
            new_content[-1] = {**last_block, "cache_control": {"type": "ephemeral"}}
        result[last_system_idx] = {**msg, "content": new_content}

    return result


def to_litellm_messages(messages: Sequence[LLMMessage]) -> list[dict[str, Any]]:
    """Convert LLMMessage objects to LiteLLM message format.

    Args:
        messages: Sequence of LLMMessage objects

    Returns:
        List of dicts in LiteLLM format
    """
    litellm_messages = []

    for msg in messages:
        # Determine message role from type
        msg_type = type(msg).__name__

        if msg_type == "SystemMessage":
            litellm_messages.append({"role": "system", "content": msg.content})
        elif msg_type == "UserMessage":
            litellm_messages.append({"role": "user", "content": msg.content})
        elif msg_type == "AssistantMessage":
            # Handle tool calls in assistant messages
            if isinstance(msg.content, list) and all(isinstance(c, FunctionCall) for c in msg.content):
                # Convert FunctionCall objects to tool_calls format
                tool_calls = []
                for fc in msg.content:
                    tool_calls.append(
                        {
                            "id": fc.id,
                            "type": "function",
                            "function": {"name": fc.name, "arguments": fc.arguments},
                        }
                    )
                litellm_messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
            else:
                # Regular text response
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                litellm_messages.append({"role": "assistant", "content": content})
        elif msg_type == "FunctionExecutionResultMessage":
            # Convert tool results to tool message format
            for result in msg.content:
                litellm_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "name": result.name,
                        "content": result.content,
                    }
                )
        else:
            # Fallback for unknown message types
            logger.warning(f"Unknown message type {msg_type}, treating as user message")
            litellm_messages.append({"role": "user", "content": str(msg.content)})

    return litellm_messages


def litellm_to_model_output(response: Any, usage: Any, model: str, schema: type[BaseModel] | None = None) -> ModelOutput:
    """Convert LiteLLM response to ModelOutput.

    Args:
        response: LiteLLM response object or dict
        usage: Usage information from LiteLLM
        model: Model name used (our shorthand)
        schema: Optional Pydantic schema for structured output

    Returns:
        ModelOutput (always returns ModelOutput to preserve pricing metadata)
    """

    # Extract content from response
    content: str | list[FunctionCall]
    thought: str | None = None
    if hasattr(response, "choices") and response.choices and len(response.choices) > 0:
        choice = response.choices[0]
        message = choice.message if hasattr(choice, "message") else choice
        raw_finish_reason = choice.finish_reason if hasattr(choice, "finish_reason") else "stop"
        finish_reason_map = {
            "tool_calls": "function_calls",
            "tool_use": "function_calls",  # Some providers use this
        }
        finish_reason = finish_reason_map.get(raw_finish_reason, raw_finish_reason)

        # Vertex/Gemini reasoning models can return either a null message OR a
        # non-null message whose `content` is None when hidden reasoning tokens
        # consume the entire output budget (finish_reason=length, small max_tokens).
        # Guarding only `message is None` (the legacy compat-shim shape) misses the
        # native `vertex_ai/` shape where `message` is present but `message.content`
        # is None, which previously flowed into ModelOutput(content=None) and raised
        # a pydantic ValidationError out of create(). Coerce both shapes to "".
        message_content = getattr(message, "content", None) if message is not None else None
        if message is None or (message_content is None and not getattr(message, "tool_calls", None)):
            content = ""
        else:
            # Check for tool calls (check both existence and non-empty list)
            tool_calls_attr = getattr(message, "tool_calls", None)
            if tool_calls_attr is not None and isinstance(tool_calls_attr, list) and len(tool_calls_attr) > 0:
                # Convert to FunctionCall objects
                tool_calls: list[FunctionCall] = []
                for tc in tool_calls_attr:
                    tool_calls.append(FunctionCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments))
                content = tool_calls
            else:
                # Regular text content
                content = message.content if hasattr(message, "content") else str(message)
                # litellm 1.83.0 already surfaces provider reasoning via
                # `message.reasoning_content` and strips <think>/<thinking>/
                # <budget:thinking> from `content` on its own parse path
                # (litellm_core_utils/prompt_templates/common_utils.py:1294-1316).
                # We only need to ROUTE that into buttermilk's `thought` field.
                # The inline-<think> stripping fallback is KEPT for the
                # DeepSeek-R1-via-Vertex-MAAS edge, which emits chain-of-thought
                # inline in `content` rather than as structured reasoning_content
                # (UNVERIFIED whether litellm 1.83.0 covers that specific MAAS path,
                # so retained fail-safe). structured reasoning takes precedence.
                if isinstance(content, str):
                    parser = importlib.import_module("buttermilk.utils.json_parser").ChatParser()
                    content = parser.extract_reasoning(
                        content,
                        structured_reasoning=getattr(message, "reasoning_content", None),
                    )
                    thought = parser.thought
    else:
        # Fallback for unexpected response format
        content = str(response)
        finish_reason = "stop"

    # Create RequestUsage object
    if hasattr(usage, "prompt_tokens"):
        request_usage = RequestUsage(
            prompt_tokens=usage.prompt_tokens or 0,
            completion_tokens=usage.completion_tokens or 0,
            cached_tokens=extract_cached_tokens(usage),
        )
    else:
        request_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    # Check if content is cached (some providers support this)
    cached = getattr(response, "cached", False)

    # Extract actual model name from response
    # Prefer actual model from API (e.g., "gemini-2.0-flash-exp")
    # Fall back to our shorthand if API doesn't provide it (e.g., "gemini25flash")
    model_name = getattr(response, "model", model)

    # Always return ModelOutput to preserve pricing metadata
    result = ModelOutput(
        content=content,
        finish_reason=finish_reason,
        usage=request_usage,
        cached=cached,
        thought=thought,
        parsed_object=None,  # Will be parsed by caller if needed
    )

    # Store model name directly (actual from API or fallback to config name)
    result.metadata["model"] = model_name

    return result


# =============================================================================
# LiteLLM Wrapper
# =============================================================================


class LiteLLMWrapper(BaseModel):
    """Wraps LiteLLM to provide a unified LLM interface.

    This class uses LiteLLM as the underlying chat completion client,
    providing a consistent interface with full support for:
    - Structured output (Pydantic schemas)
    - Tool/function calling
    - Retry logic
    - Pricing calculation
    - Observability (Weave tracing)

    Attributes:
        model: Model name in LiteLLM format (e.g., "gpt-4", "azure/gpt-4")
        model_info: ModelInfo metadata about the model
        litellm_model_name: Resolved model name for LiteLLM
        api_key: API key for the provider (if needed)
        base_url: Custom base URL (if needed)
        default_parameters: Default inference parameters (temperature, max_tokens, etc.)
    """

    model: str = Field(..., description="Model name in LiteLLM format")
    model_info: dict[str, Any] = Field(..., description="Model metadata")
    litellm_model_name: str = Field(..., description="Resolved model name for LiteLLM")
    api_key: str | None = Field(default=None, description="API key for the provider")
    base_url: str | None = Field(default=None, description="Custom base URL")
    vertex_location: str | None = Field(default=None, description="GCP region for Vertex AI providers (litellm vertex_location)")
    client_type: str | None = Field(default=None, description="Client type identifier (e.g., 'anthropic', 'vertex_ai')")
    default_parameters: ModelParameters = Field(
        default_factory=ModelParameters,
        description="Default inference parameters (temperature, max_tokens, etc.)",
    )

    # Retry configuration
    cooldown_seconds: float = 0.5
    max_retries: int = 3
    min_wait_seconds: float = 5.0
    max_wait_seconds: float = 60.0
    jitter_seconds: float = 5.0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: Any):
        """Initialize LiteLLMWrapper."""
        super().__init__(**data)

        if not _litellm_available():
            raise ImportError("LiteLLM is not installed. Please install it with: pip install litellm")

    async def _execute_with_retry(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute a litellm call with exponential-backoff retries.

        litellm classifies which exceptions are transient via its typed exception
        hierarchy (RateLimitError, Timeout, APIConnectionError, InternalServerError,
        ServiceUnavailableError, …); we retry exactly those types instead of the old
        brittle string/urllib3 keyword matching.

        The ONE genuinely buttermilk-specific coupling kept here — the reason we do
        not just pass `num_retries` to litellm and be done — is: do NOT retry when the
        root cause is an ``AttributeError``. litellm wraps a deterministic null-message
        parse (reasoning model, finish_reason=length, exhausted budget) as a transient-
        looking InternalServerError; retrying it just burns the budget again. litellm's
        own `num_retries`/`RetryPolicy` cannot express "retry InternalServerError EXCEPT
        when __cause__ is AttributeError", so we own the retry loop for this case.
        """
        litellm_mod = _get_litellm()
        candidate_types = (
            [
                getattr(litellm_mod, name, None)
                for name in ("RateLimitError", "Timeout", "APIConnectionError", "InternalServerError", "ServiceUnavailableError")
            ]
            if litellm_mod is not None
            else [TimeoutError, ConnectionError]
        )
        # Keep only real exception classes — guards against a partially-mocked litellm
        # module (test doubles) whose exception attributes are not types.
        retryable_types: tuple[type[BaseException], ...] = tuple(t for t in candidate_types if isinstance(t, type) and issubclass(t, BaseException))

        last_exception: Exception | None = None
        wait_time = self.min_wait_seconds

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    await asyncio.sleep(self.cooldown_seconds)
                return await func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                is_retryable = isinstance(e, retryable_types)

                # AttributeError as the root cause means a deterministic parse failure
                # (litellm wraps null-message parse as InternalServerError) — never retry.
                if is_retryable:
                    cause = e.__cause__ or e.__context__
                    if isinstance(cause, AttributeError):
                        is_retryable = False

                if attempt < self.max_retries and is_retryable:
                    jitter = random.uniform(-self.jitter_seconds, self.jitter_seconds)
                    actual_wait = min(wait_time + jitter, self.max_wait_seconds)
                    logger.warning(f"LiteLLM call failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. Retrying in {actual_wait:.1f}s...")
                    await asyncio.sleep(actual_wait)
                    wait_time *= 2  # Exponential backoff
                else:
                    raise

        raise last_exception or ProcessingError("LiteLLM call failed after all retries")

    async def create(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema] = [],
        schema: type[BaseModel] | None = None,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> CreateResult | ModelOutput:
        """Create a chat completion using LiteLLM.

        Args:
            messages: Sequence of LLMMessage objects
            tools: Optional sequence of tools the LLM can call
            schema: Optional Pydantic schema for structured output
            cancellation_token: Optional cancellation token (not used by LiteLLM)
            **kwargs: Additional arguments for LiteLLM

        Returns:
            CreateResult or ModelOutput
        """
        # Convert messages to LiteLLM format
        litellm_messages = to_litellm_messages(messages)

        # Anthropic requires explicit cache_control breakpoints for prompt caching.
        # Without them Claude caches nothing. Gemini/OpenAI cache implicitly from
        # the system-block position and do not need this.
        if self.client_type in ("anthropic", "anthropic_vertex"):
            litellm_messages = _add_anthropic_cache_control(litellm_messages)

        # Merge default parameters with runtime kwargs (runtime takes precedence)
        merged_params = self.default_parameters.to_api_params()
        merged_params.update(kwargs)

        # The full provider-prefixed litellm model name is the single source of truth
        # for routing; litellm picks the provider/transport from its prefix (vertex_ai/,
        # azure/, azure_ai/, anthropic/, gemini/, …). Vertex auth is ambient via GCP ADC.
        litellm_params = {
            "model": self.litellm_model_name,
            "messages": litellm_messages,
            **merged_params,
        }

        # Add API key if provided
        if self.api_key:
            litellm_params["api_key"] = self.api_key

        # Add base URL if provided
        if self.base_url:
            litellm_params["base_url"] = self.base_url

        # Add Vertex AI location if provided. Vertex auth/project come from ambient ADC;
        # only the location is per-model (gemini-3.x need "global"). litellm region
        # precedence: vertex_location kwarg -> litellm.vertex_location -> VERTEXAI_LOCATION.
        if self.vertex_location:
            litellm_params["vertex_location"] = self.vertex_location

        # Handle structured output via response_format or tool calling fallback
        # LiteLLM uses json_schema format for providers that support structured output
        # For providers without native structured output (e.g., Anthropic on Vertex),
        # we fall back to using a fake tool to get structured output
        # See: https://docs.litellm.ai/docs/completion/json_mode
        structured_output_enabled = self.model_info.get("structured_output", False)
        function_calling_enabled = self.model_info.get("function_calling", False)
        used_fake_schema_tool = False
        fake_tool_name = None

        logger.debug(
            f"LiteLLMWrapper: schema={schema}, structured_output_enabled={structured_output_enabled}, "
            f"function_calling_enabled={function_calling_enabled}, model_info={self.model_info}"
        )

        if schema and structured_output_enabled:
            # Native structured output: pass the RAW pydantic JSON schema straight to
            # litellm's response_format. litellm 1.83.0 runs the provider-specific
            # transforms itself (_build_vertex_schema does $ref expansion + enum/type
            # coercion for Vertex/Gemini; type_to_response_format_param for OpenAI/Azure;
            # Anthropic-on-Vertex via the tool transform). Live-verified 2026-06-22 that
            # litellm accepts our raw nested schemas (JudgeReasons, QualScore with $defs +
            # Literal enums) and returns valid parsed JSON for vertex_ai/gemini@global,
            # vertex_ai/claude, and azure_ai — so buttermilk's own $ref/required/enum
            # massaging here is redundant and was removed.
            _validate_schema_constraints(schema, self.model_info)
            schema_dict = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema.schema()

            litellm_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__ if hasattr(schema, "__name__") else "response_schema",
                    "strict": True,
                    "schema": schema_dict,
                },
            }
            logger.debug(f"LiteLLMWrapper: Using native response_format with json_schema for {schema.__name__}")

        elif schema and function_calling_enabled and not tools:
            # No native structured output, but function calling available and no tools provided
            # Use a fake tool to get structured output
            schema_dict = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema.schema()
            fake_tool_name = f"create_{schema.__name__.lower()}"
            litellm_params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": fake_tool_name,
                        "description": f"Create a {schema.__name__} object with the specified fields",
                        "parameters": schema_dict,
                    },
                }
            ]
            # Force the model to use this tool
            litellm_params["tool_choice"] = {"type": "function", "function": {"name": fake_tool_name}}
            used_fake_schema_tool = True
            logger.debug(f"LiteLLMWrapper: Using fake tool '{fake_tool_name}' for structured output (no native support)")

        # Log the final litellm_params for debugging
        logger.debug(
            f"LiteLLMWrapper: Final params (keys): {list(litellm_params.keys())}",
            has_response_format=litellm_params.get("response_format") is not None,
        )

        # Handle tools
        if tools:
            # Convert Tool objects to LiteLLM format
            litellm_tools = []
            for tool in tools:
                if hasattr(tool, "schema"):
                    tool_schema = tool.schema
                    # Type check: tool_schema can be dict (ToolSchema) or object with attributes
                    if isinstance(tool_schema, dict):
                        name = tool_schema.get("name", getattr(tool, "name", ""))
                        description = tool_schema.get("description", "")
                        parameters = tool_schema.get("parameters", {})
                    else:
                        # Handle Tool objects with attribute access
                        name = getattr(tool_schema, "name", getattr(tool, "name", ""))
                        description = getattr(tool_schema, "description", "")
                        parameters = getattr(tool_schema, "parameters", {})

                    litellm_tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": name,
                                "description": description,
                                "parameters": parameters,
                            },
                        }
                    )

            if litellm_tools:
                litellm_params["tools"] = litellm_tools

        # Disable LiteLLM's internal retry logic since we handle retries ourselves
        # LiteLLM defaults to num_retries=None (no retries), but some providers
        # may have their own defaults. We explicitly set num_retries=0 to ensure
        # we control all retry behavior via our own _execute_with_retry wrapper.
        #
        # Note: HuggingFace models don't support max_retries and will log a warning
        # if it's passed. Since we handle retries ourselves, we skip num_retries
        # for HuggingFace to avoid the spurious warning.
        is_huggingface = self.litellm_model_name.startswith("huggingface/") if self.litellm_model_name else False
        if not is_huggingface:
            litellm_params["num_retries"] = 0

        # Execute with retry logic
        async def _call_litellm() -> Any:
            acompletion = _get_acompletion()
            return await acompletion(**litellm_params)

        try:
            response = await self._execute_with_retry(_call_litellm)
        except Exception as e:
            # Check if it's a ContentPolicyViolationError (lazy check since litellm is lazy-loaded)
            litellm_mod = _get_litellm()
            if litellm_mod is not None and isinstance(e, litellm_mod.ContentPolicyViolationError):
                # Handle content moderation errors from OpenAI/Azure without traceback
                error_msg = f"Content blocked by provider safety filter: {e!s}"
                logger.error(error_msg)
                raise ContentBlockedError(
                    message=error_msg,
                    filter_result={},  # LiteLLM doesn't provide detailed filter results
                ) from e
            # litellm wraps AttributeError (null message parse) as InternalServerError when
            # finish_reason=length and a reasoning model exhausted its token budget —
            # surface as a clean truncated ModelOutput instead of a fake 500
            if litellm_mod is not None and isinstance(e, litellm_mod.InternalServerError):
                cause = e.__cause__ or e.__context__
                if isinstance(cause, AttributeError):
                    error_str = str(e)
                    logger.warning(
                        "LiteLLM InternalServerError from null message parse "
                        f"(finish_reason=length, insufficient max_tokens for reasoning model): {error_str}"
                    )
                    return ModelOutput(
                        content="",
                        finish_reason="length",
                        usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
                        cached=False,
                        error_message=error_str,
                    )
            # Generic LiteLLM error
            error_msg = f"LiteLLM call failed: {e}"
            raise ProcessingError(error_msg) from e

        # Calculate pricing from the full response (lets litellm.completion_cost do
        # the per-token arithmetic / cache-read discount itself).
        usage = response.usage if hasattr(response, "usage") else None
        pricing_metadata = self._calculate_pricing(usage, response=response)

        # Handle fake tool response - extract arguments as the structured content
        if used_fake_schema_tool and fake_tool_name:
            # Check if the response contains a tool call
            choices = getattr(response, "choices", [])
            if choices:
                choice = choices[0]
                message = choice.message
                tool_calls = getattr(message, "tool_calls", None)
                if tool_calls and len(tool_calls) > 0:
                    tool_call = tool_calls[0]
                    if tool_call.function.name == fake_tool_name:
                        # Extract the arguments as the structured content
                        arguments_json = tool_call.function.arguments
                        logger.debug(f"LiteLLMWrapper: Extracted fake tool arguments: {arguments_json[:200]}...")

                        # Handle Claude's XML-like parameter format
                        # Claude sometimes returns: <parameter name="field">value</parameter>
                        # instead of proper JSON
                        if "<parameter" in arguments_json:
                            arguments_json = _convert_xml_params_to_json(arguments_json)
                            logger.debug(f"LiteLLMWrapper: Converted XML params to JSON: {arguments_json[:200]}...")

                        # Replace the message content with the tool arguments
                        # and clear tool_calls so litellm_to_model_output treats it as text
                        message.content = arguments_json
                        message.tool_calls = None  # Clear tool calls
                        choice.finish_reason = "stop"  # Set finish_reason to stop
                    else:
                        logger.warning(f"LiteLLMWrapper: Expected fake tool '{fake_tool_name}', got '{tool_call.function.name}'")

        # Convert response to ModelOutput format
        result = litellm_to_model_output(response, usage, self.litellm_model_name, schema)

        # Add pricing metadata (result is always ModelOutput now)
        # Preserve actual_model that was set in litellm_to_model_output()
        result.metadata["pricing"] = pricing_metadata

        # Parse structured output if schema was provided
        if schema:
            try:
                parsed = await _parse_structured_output(result.content, schema)
                result.parsed_object = parsed
            except Exception as e:
                result.error_message = f"Failed to parse structured output: {e.args}"
                result.parsed_object = None

        return result

    async def call_chat(
        self,
        messages: list[LLMMessage],
        cancellation_token: CancellationToken | None,
        *,
        tools_list: Sequence[Tool | ToolSchema] = [],
        schema: type[BaseModel] | None = None,
        intercept_tools: bool = False,
    ) -> CreateResult | ModelOutput:
        """Manage chat interaction with tool execution.

        Args:
            messages: List of LLMMessage objects (mutable)
            cancellation_token: Optional cancellation token
            tools_list: Optional sequence of tools
            schema: Optional Pydantic schema
            intercept_tools: If True, return tool calls without executing

        Returns:
            CreateResult or ModelOutput
        """
        # Step 1: Initial call with tools (no schema)
        try:
            create_result = await self.create(
                messages=messages,
                tools=tools_list,
                cancellation_token=cancellation_token,
                schema=None if tools_list else schema,  # Only use schema if no tools
            )
        except Exception as e:
            raise ProcessingError(f"Failed to query LLM: {e}") from e

        # Extract pricing from initial call
        # Use `or 0` pattern to handle both missing keys AND explicit None values
        initial_pricing = create_result.metadata.get("pricing", {}) if hasattr(create_result, "metadata") else {}
        aggregated_pricing = {
            "prompt_tokens": initial_pricing.get("prompt_tokens") or 0,
            "completion_tokens": initial_pricing.get("completion_tokens") or 0,
            "cached_tokens": initial_pricing.get("cached_tokens") or 0,
            "total_cost": initial_pricing.get("total_cost") or 0.0,
        }

        # Step 2: Handle tool calls if present
        if isinstance(create_result.content, list) and all(isinstance(c, FunctionCall) for c in create_result.content):
            tool_calls: list[FunctionCall] = create_result.content

            if intercept_tools:
                logger.debug(f"Intercepting {len(tool_calls)} tool calls without execution")
                return create_result

            # Add assistant message with tool calls to history
            assistant_msg = AssistantMessage(content=tool_calls, source="assistant")
            messages.append(assistant_msg)

            try:
                # Execute tools
                tool_outputs = await self._execute_tools(
                    calls=tool_calls,
                    tools_list=tools_list,
                    cancellation_token=cancellation_token,
                )
                tool_result_messages = FunctionExecutionResultMessage(content=tool_outputs)
                messages.append(tool_result_messages)
            except Exception as e:
                raise ProcessingError(f"Failed to execute tools: {e}") from e

            # Step 3: Synthesis call with schema and tools
            # Note: tools must be passed to maintain context for Anthropic models
            # when conversation history contains tool calls and results
            try:
                synthesis_result = await self.create(
                    messages=messages,
                    tools=tools_list,
                    cancellation_token=cancellation_token,
                    schema=schema,
                )

                # Aggregate pricing
                if hasattr(synthesis_result, "metadata") and "pricing" in synthesis_result.metadata:
                    synthesis_pricing = synthesis_result.metadata["pricing"]
                    aggregated_pricing["prompt_tokens"] += synthesis_pricing.get("prompt_tokens") or 0
                    aggregated_pricing["completion_tokens"] += synthesis_pricing.get("completion_tokens") or 0
                    aggregated_pricing["cached_tokens"] += synthesis_pricing.get("cached_tokens") or 0
                    aggregated_pricing["total_cost"] += synthesis_pricing.get("total_cost") or 0.0
                    synthesis_result.metadata["pricing"] = aggregated_pricing

                return synthesis_result
            except Exception as e:
                raise ProcessingError(f"Failed to synthesize after tool execution: {e}") from e

        # Return original result if no tool calls
        return create_result

    async def _execute_tools(
        self,
        calls: list[FunctionCall],
        tools_list: Sequence[Tool | ToolSchema],
        cancellation_token: CancellationToken | None,
    ) -> list[FunctionExecutionResult]:
        """Execute tools from function calls."""
        tasks = []
        for call in calls:
            # Find the tool by name
            tool = next((t for t in tools_list if t.name == call.name), None)
            if tool is None:
                raise ProcessingError(f"Tool '{call.name}' requested by LLM not found in provided tools list.")

            tasks.append(self._call_tool(call, tool, cancellation_token))

        return await asyncio.gather(*tasks)

    async def _call_tool(
        self,
        call: FunctionCall,
        tool: Tool,
        cancellation_token: CancellationToken | None,
    ) -> FunctionExecutionResult:
        """Execute a single tool call."""
        arguments = json.loads(call.arguments)
        arguments.update(arguments.pop("kwargs", {}))

        ct: CancellationToken = cancellation_token or CancellationToken()
        result = await tool.run_json(arguments, ct)

        return FunctionExecutionResult(
            call_id=call.id,
            name=tool.name,
            content=tool.return_value_as_string(result),
        )

    def _calculate_pricing(self, usage: Any, response: Any = None) -> dict[str, Any]:
        """Calculate pricing information from a litellm response.

        The total cost is computed by litellm.completion_cost() directly on the
        response — litellm already applies the model's per-token rates and the
        cache-read discount (live-verified 2026-06-22 across vertex_ai/gemini,
        vertex_ai/claude and azure_ai). We only fall back to buttermilk's
        calculate_token_cost (which carries the _simple_model_resolution name
        remapping) when completion_cost() is unavailable or raises — e.g. a
        litellm model name it cannot price.
        """
        if usage is None:
            logger.warning("LLM response had no usage data - using 0 tokens for pricing")
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_cost": 0.0,
            }

        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        cached_tokens = extract_cached_tokens(usage)

        total_cost: float | None = None
        litellm_mod = _get_litellm()
        if response is not None and litellm_mod is not None:
            try:
                total_cost = float(litellm_mod.completion_cost(completion_response=response))
            except Exception as e:
                logger.debug(f"litellm.completion_cost failed for {self.litellm_model_name}: {e}; falling back to calculate_token_cost")
                total_cost = None

        if total_cost is None:
            # Fallback path: buttermilk name-remapping + cost_per_token with cache discount.
            _, _, total_cost = calculate_token_cost(
                model=self.litellm_model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
            )

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "total_cost": total_cost,
        }


class LLMs(BaseModel):
    """Manages a collection of LLM configurations and their instantiated clients.

    This class serves as a central registry for all LLMs defined in the
    Buttermilk configuration. It can:
    1.  Store multiple `LLMConfig` entries, keyed by a connection name.
    2.  On demand, instantiate and cache `LiteLLMWrapper` clients for these
        configurations using `get_client`.
    3.  Provide convenient attribute-style access (e.g., `llms.my_gpt_model`)
        and item-style access (e.g., `llms["my_gpt_model"]`) to these clients.

    Attributes:
        connections (dict[str, LLMConfig]): A dictionary where keys are
            connection names (e.g., "azure_prod_gpt4") and values are
            `LLMConfig` objects detailing the configuration for that LLM.
        cached_clients (dict[str, LiteLLMWrapper]): A cache for instantiated
            `LiteLLMWrapper` clients. This is populated on-demand when a client
            is first requested. Not meant to be set directly by users.
        model_config (ConfigDict): Pydantic model configuration.
            - `use_enum_values`: True - Ensures enum members are used for validation/serialization.

    """

    connections: dict[str, LLMConfig] = Field(
        default_factory=dict,
        description="A dictionary where keys are connection names and values are LLMConfig objects.",
    )
    model_parameters: dict[str, ModelParameters | dict] = Field(
        default_factory=dict,
        description="Per-model parameter overrides from YAML config (model_name -> parameters)",
    )
    cached_clients: dict[str, LiteLLMWrapper] = Field(
        default_factory=dict,
        description="Cache for instantiated LiteLLMWrapper clients. Populated on demand.",
        exclude=True,
    )
    model_registry_cache: dict[str, Any] = Field(
        default_factory=dict,
        description="Cached model registry loaded from models.json",
        exclude=True,  # Exclude from model dump as it's runtime state
    )

    model_config = ConfigDict(use_enum_values=True)

    @property
    def all_model_names(self) -> Enum:
        """Provides an Enum of all configured LLM connection names.

        Returns:
            Enum: An Enum where members are the keys from the `connections` dictionary.

        """
        return Enum("AllModelNames", {name: name for name in self.connections.keys()})

    def _load_model_registry(self, force: bool = False) -> dict[str, Any]:
        """Load (and cache) the models.json registry. Tolerates // comment lines."""
        if self.model_registry_cache and not force:
            return self.model_registry_cache

        # Use centralized cache directory
        cache_dir = get_base_cache_dir() / cache.MODELS
        models_json_path = cache_dir / CONFIG_CACHE_FILENAME
        try:
            text = models_json_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.debug(f"Model registry not found at {models_json_path}")
            self.model_registry_cache = {}
            return self.model_registry_cache
        except Exception as e:
            logger.warning(f"Failed reading model registry {models_json_path}: {e}")
            self.model_registry_cache = {}
            return self.model_registry_cache

        cleaned_lines: list[str] = []
        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("//"):
                continue
            cleaned_lines.append(line)
        cleaned = "\n".join(cleaned_lines)

        try:
            self.model_registry_cache = json.loads(cleaned)
        except Exception as e:
            logger.warning(f"Failed parsing model registry {models_json_path}: {e}")
            self.model_registry_cache = {}
        return self.model_registry_cache

    def get_merged_parameters(self, name: str) -> ModelParameters:
        """Get merged parameters for a model.

        Merges parameters from multiple sources in order of precedence:
        1. LLMConfig.parameters (from models.json) - lowest priority
        2. LLMs.model_parameters (from YAML config) - highest priority

        Args:
            name: Model connection name

        Returns:
            ModelParameters: Merged parameters for the model
        """
        if name not in self.connections:
            return ModelParameters()

        config = self.connections[name]

        # Start with parameters from LLMConfig (models.json)
        base_params = config.parameters

        # Override with YAML model_parameters if present
        yaml_params = self.model_parameters.get(name)
        if yaml_params:
            if isinstance(yaml_params, dict):
                yaml_params = ModelParameters(**yaml_params)
            return base_params.merge_with(yaml_params)

        return base_params

    def get_client(self, name: str) -> LiteLLMWrapper:
        """Gets or creates an LLM wrapper for the configuration specified by `name`.

        Routing is entirely driven by the full provider-prefixed `litellm_model`
        name on the config. litellm + ambient GCP ADC handle auth/transport; the
        only per-model knob we pass through is `region` -> litellm `vertex_location`.

        Args:
            name: The connection name of the LLM configuration (must be a key
                in `self.connections`).

        Returns:
            LiteLLMWrapper: The instantiated wrapper with `.create()` and `.call_chat()` methods.

        Raises:
            AttributeError: If `name` is not found in `self.connections`.
            ImportError: If LiteLLM is not available.
            ProcessingError: If the model is a classification-only provider (zentropi).
        """
        # Check cache first
        if name in self.cached_clients:
            return self.cached_clients[name]

        if name not in self.connections:
            raise AttributeError(f"LLM configuration named '{name}' not found in connections.")

        config = self.connections[name]
        model_name = config.configs.get("model") or name

        # Zentropi is a classification API, not an LLM - use ZentropiClassifier agent instead.
        # Guard off the litellm provider segment instead of a client_type enum.
        if config.provider_segment == "zentropi":
            raise ProcessingError(
                "Zentropi models cannot be used as LLM clients. Use buttermilk.agents.ZentropiClassifier for classification tasks instead."
            )

        # Get merged parameters for this model
        merged_params = self.get_merged_parameters(name)

        logger.debug(f"Creating LiteLLMWrapper for model '{name}' with litellm_model '{config.litellm_model}'")

        # vertex_location: per-model region for Vertex providers. gemini-3.x require
        # "global"; litellm otherwise defaults to us-central1. Project + auth are
        # ambient via GCP ADC (set up in the BM startup loop), so we pass nothing else.
        vertex_location: str | None = config.region
        if config.provider_segment == "vertex_ai" and not vertex_location:
            vertex_location = "global"

        wrapped_client = LiteLLMWrapper(
            model=model_name,
            model_info=config.model_info,
            litellm_model_name=config.litellm_model,
            api_key=config.api_key,
            base_url=config.base_url,
            vertex_location=vertex_location,
            client_type=config.provider_segment,
            default_parameters=merged_params,
        )

        self.cached_clients[name] = wrapped_client
        return wrapped_client

    def __getattr__(self, __name: str) -> LiteLLMWrapper:
        """Provides attribute-style access to LLM clients (e.g., `llms.my_model`)."""
        if __name not in self.connections:
            raise AttributeError(f"No LLM configuration found for '{__name}'. Available: {list(self.connections.keys())}")
        return self.get_client(__name)

    def __getitem__(self, __name: str) -> LiteLLMWrapper:
        """Provides item-style access to LLM clients (e.g., `llms["my_model"]`)."""
        return self.__getattr__(__name)
