"""Manages Language Model (LLM) configurations, clients, and interactions.

This module provides structures for defining LLM configurations (`LLMConfig`),
managing different LLM providers and their clients (`LLMs`, `LLMClient`), and
wrapping chat completion clients with additional functionality such as rate
limiting and retry logic (`LiteLLMWrapper`).

It uses LiteLLM as the unified interface for interacting with various LLM APIs
and provides a consistent interface for agents within the Buttermilk framework.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import random
import socket
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any

import urllib3.exceptions

# Core LLM library imports
from google.auth.exceptions import TransportError as GoogleAuthTransportError

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


# Native Buttermilk types (replaces autogen_core.models and autogen_core.tools)
from buttermilk._core.tool_types import CancellationToken, FunctionCall, Tool, ToolSchema
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

# from google import genai  # Google Generative AI library (unused in current implementation)
from pydantic import (
    BaseModel,  # Pydantic models for configuration
    ConfigDict,
    Field,
    field_validator,
)

from buttermilk import bm, logger

# ToolOutput import removed - using autogen's FunctionExecutionResult directly
from buttermilk._core.constants import CONFIG_CACHE_FILENAME, cache, get_base_cache_dir  # Models cache constants
from buttermilk._core.exceptions import ContentBlockedError, ProcessingError  # Custom Buttermilk exceptions
from buttermilk._core.json_schema import make_all_properties_required, resolve_json_schema_refs  # Schema $ref resolution for Azure compatibility
from buttermilk.utils.pricing import calculate_token_cost  # Token cost calculation


class ClientType(Enum):
    """Enumeration of supported LLM client types.

    Used to categorize LLM providers or services.

    Attributes:
        OPENAI: OpenAI platform.
        GEMINI: Google Generative AI platform (e.g., Gemini API).
        GEMINI_VERTEX: Gemini client on vertex platform.
        VERTEX_OPENAI: Google Vertex AI platform with OpenAI-compatible endpoint (legacy).
        LLAMA_VERTEX: Llama models on Vertex AI via native LiteLLM support.
        DEEPSEEK_VERTEX: DeepSeek models on Vertex AI via native LiteLLM support.
        ANTHROPIC: Anthropic platform (e.g., Claude models).
        ANTHROPIC_VERTEX: Anthropic models hosted on Google Vertex AI.
        llama: Llama models (often self-hosted or via specific providers).
        AZURE: Microsoft Azure AI platform (e.g., Azure OpenAI).

    """

    OPENAI = "openai"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    ANTHROPIC_VERTEX = "anthropic_vertex"
    GEMINI = "gemini"
    GEMINI_VERTEX = "gemini_vertex"
    VERTEX_OPENAI = "vertex_openai"  # OpenAI-compatible endpoint on Vertex (legacy)
    LLAMA_VERTEX = "llama_vertex"  # Llama models on Vertex AI via native LiteLLM
    DEEPSEEK_VERTEX = "deepseek_vertex"  # DeepSeek models on Vertex AI via native LiteLLM
    MISTRAL_VERTEX = "mistral_vertex"  # Mistral models on Vertex AI via native LiteLLM
    HUGGINGFACE = "huggingface"  # HuggingFace Inference API (serverless or dedicated)
    ZENTROPI = "zentropi"  # Zentropi toxicity/content moderation API


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

    Defines the client type, API key, custom base URL, model-specific information,
    and any additional configurations required by the LLM client.

    Attributes:
        client_type (ClientType): The type of client to instantiate (e.g., "openai",
            "anthropic", "gemini_vertex"). This determines which LiteLLM provider to use.
        api_key (str | None): The API key required for authenticating with the
            LLM provider. Can be None if authentication is handled differently
            (e.g., via environment variables or instance metadata).
        base_url (str | None): A custom base URL for the API endpoint, if
            different from the provider's default (e.g., for Azure OpenAI or
            self-hosted models).
        model_info (ModelInfo): A ModelInfo object containing detailed metadata
            about the model, such as its family, context window size,
            support for structured output, etc.
        configs (dict): A dictionary for additional options or configurations
            to pass to the LiteLLM client.
        litellm_model (str | None): An optional explicit litellm model identifier.
            If provided, this will be used instead of automatic resolution from
            client_type and model info. Useful for models that need specific naming
            for litellm pricing calculations.
        parameters (ModelParameters): Default inference parameters (temperature,
            max_tokens, etc.) for this model. Defaults to empty ModelParameters
            instance. Can be specified as a dict which will be converted to
            ModelParameters during validation.

    """

    client_type: ClientType = Field(
        description="Type of client to instantiate (determines which LiteLLM provider to use)",
    )
    api_key: str | None = Field(
        default=None,
        description="API key to use for this model",
    )
    base_url: str | None = Field(default=None, description="Custom URL to call")

    model_info: ModelInfo = Field(..., description="Model metadata (family, context size, etc.)")
    configs: dict = Field(default_factory=dict, description="Options to pass to the LiteLLM client")
    litellm_model: str | None = Field(default=None, description="Explicit litellm model identifier override")
    parameters: ModelParameters = Field(
        default_factory=ModelParameters,
        description="Default inference parameters (temperature, max_tokens, etc.)",
    )

    @field_validator("client_type", mode="before")
    @classmethod
    def validate_client_type(cls, v: Any) -> ClientType:
        """Validate client_type and convert string values to ClientType enum.

        Args:
            v: The input value to validate (typically from models.json)

        Returns:
            ClientType: The validated enum value

        Raises:
            ValueError: If the client_type string is not supported

        """
        if isinstance(v, ClientType):
            return v
        if isinstance(v, str):
            # Try to match the string value to an enum member
            try:
                return ClientType(v)
            except ValueError:
                # If direct value match fails, try case-insensitive matching
                for client_type in ClientType:
                    if client_type.value.lower() == v.lower():
                        return client_type
                # If no match found, raise a descriptive error
                supported_values = [ct.value for ct in ClientType]
                raise ValueError(
                    f"Unsupported client_type '{v}'. Supported values are: {supported_values}",
                )
        raise ValueError(f"client_type must be a string or ClientType enum, got {type(v)}")

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
CHAT_MODELS = ["google/gemini-3.1-pro-preview", "google/gemini-3-flash-preview", "google/gemini-3.5-flash", "google/gemini-3.1-flash-lite", "gpt-5-mini", "gpt-5-nano", "gpt-4o", "meta/llama-4-maverick-17b-128e-instruct-maas", "claude-sonnet-4-5@20250929", "deepseek-ai/deepseek-v3.2-maas"]

"""A predefined list of identifiers for cost-effective chat models."""
CHEAP_CHAT_MODELS = [
    "google/gemini-3.1-flash-lite",
    "google/gemini-3.5-flash",
    "meta/llama-4-maverick-17b-128e-instruct-maas",
    "gpt-5-nano",
    "claude-haiku-4-5@20251001",
]


class LLMClient(BaseModel):
    """Represents an instantiated LLM client along with its connection and parameters.

    This model is used to store and pass around active LLM client instances.

    Attributes:
        client (Any): The actual instantiated LLM client object (e.g., an instance
            of `OpenAIChatCompletionClient`, `AsyncAnthropicVertex`, etc.).
        connection (str): The connection identifier (from `LLMConfig.connection`)
            associated with this client.
        parameters (dict): A dictionary of parameters that were used to configure
            this client instance, or default parameters for its use.
            Defaults to an empty dict.

    """

    client: Any  # The actual LLM client object (e.g., OpenAIChatCompletionClient)
    connection: str  # Identifier for the connection type (e.g., "azure_gpt4")
    parameters: dict = Field(default_factory=dict)  # Parameters for this client


class ModelOutput(CreateResult):
    """Extends Autogen's `CreateResult` with structured output parsing and pricing.

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
        # Assuming object with attributes (Autogen ModelInfo)
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


def autogen_to_litellm_messages(messages: Sequence[LLMMessage]) -> list[dict[str, Any]]:
    """Convert Autogen LLMMessage objects to LiteLLM message format.

    Args:
        messages: Sequence of Autogen LLMMessage objects

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


def litellm_to_autogen_result(response: Any, usage: Any, model: str, schema: type[BaseModel] | None = None) -> ModelOutput:
    """Convert LiteLLM response to Autogen ModelOutput.

    Args:
        response: LiteLLM response object or dict
        usage: Usage information from LiteLLM
        model: Model name used (our shorthand)
        schema: Optional Pydantic schema for structured output

    Returns:
        ModelOutput compatible with Autogen interface (always returns ModelOutput to preserve pricing metadata)
    """

    # Extract content from response
    content: str | list[FunctionCall]
    thought: str | None = None
    if hasattr(response, "choices") and response.choices and len(response.choices) > 0:
        choice = response.choices[0]
        message = choice.message if hasattr(choice, "message") else choice

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
            # Route the response through ChatParser to (a) strip inline
            # <think>...</think> blocks emitted by DeepSeek-R1 via Vertex MAAS
            # and (b) capture any reasoning into the proper `thought` field.
            # Providers that emit structured `reasoning_content` (DeepSeek
            # reasoner, OpenAI o-series, Anthropic extended thinking, Gemini
            # thinking) take precedence over inline-extracted text.
            if isinstance(content, str):
                parser = importlib.import_module("buttermilk.utils.json_parser").ChatParser()
                content = parser.extract_reasoning(
                    content,
                    structured_reasoning=getattr(message, "reasoning_content", None),
                )
                thought = parser.thought

        raw_finish_reason = choice.finish_reason if hasattr(choice, "finish_reason") else "stop"
        # Map LiteLLM finish_reason to Autogen values
        # LiteLLM uses "tool_calls", Autogen uses "function_calls"
        finish_reason_map = {
            "tool_calls": "function_calls",
            "tool_use": "function_calls",  # Some providers use this
        }
        finish_reason = finish_reason_map.get(raw_finish_reason, raw_finish_reason)
    else:
        # Fallback for unexpected response format
        content = str(response)
        finish_reason = "stop"

    # Create RequestUsage object
    if hasattr(usage, "prompt_tokens"):
        request_usage = RequestUsage(
            prompt_tokens=usage.prompt_tokens or 0,
            completion_tokens=usage.completion_tokens or 0,
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
# LiteLLM Wrapper - Drop-in Replacement for AutoGenWrapper
# =============================================================================


class LiteLLMWrapper(BaseModel):
    """Wraps LiteLLM to provide the same interface as AutoGenWrapper.

    This class provides a drop-in replacement for AutoGenWrapper that uses
    LiteLLM instead of Autogen's ChatCompletionClient. It maintains full
    compatibility with:
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
    extra_headers: dict[str, str] | None = Field(default=None, description="Extra headers for the API request (e.g., Authorization)")
    token_provider: Callable[[], str] | None = Field(default=None, description="Optional callable that returns an authentication token")
    vertex_project: str | None = Field(default=None, description="GCP project ID for Vertex AI providers")
    vertex_location: str | None = Field(default=None, description="GCP region for Vertex AI providers")
    default_parameters: ModelParameters = Field(
        default_factory=ModelParameters,
        description="Default inference parameters (temperature, max_tokens, etc.)",
    )

    # Retry configuration (matching AutoGenWrapper)
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
        """Execute a function with exponential backoff retry logic.

        Mirrors RetryWrapper behavior for consistency.
        """
        last_exception: Exception | None = None
        wait_time = self.min_wait_seconds

        for attempt in range(self.max_retries + 1):
            try:
                # Add cooldown before each attempt (except first)
                if attempt > 0:
                    await asyncio.sleep(self.cooldown_seconds)

                # Execute the function
                result = await func(*args, **kwargs)
                return result

            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()

                # Check if this is a retryable error (by type or message)
                retryable_types = (
                    TimeoutError,
                    ConnectionError,
                    ConnectionResetError,
                    ConnectionAbortedError,
                    socket.gaierror,
                    urllib3.exceptions.ProtocolError,
                    urllib3.exceptions.TimeoutError,
                    urllib3.exceptions.NameResolutionError,
                    urllib3.exceptions.NewConnectionError,
                    GoogleAuthTransportError,
                )

                is_retryable = isinstance(e, retryable_types) or any(
                    keyword in error_msg for keyword in ["rate limit", "timeout", "503", "429", "502", "500", "name resolution"]
                )

                if attempt < self.max_retries and is_retryable:
                    # Calculate wait time with jitter
                    jitter = random.uniform(-self.jitter_seconds, self.jitter_seconds)
                    actual_wait = min(wait_time + jitter, self.max_wait_seconds)

                    logger.warning(f"LiteLLM call failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. Retrying in {actual_wait:.1f}s...")

                    await asyncio.sleep(actual_wait)
                    wait_time *= 2  # Exponential backoff
                else:
                    # Not retryable or out of retries
                    raise

        # Should not reach here, but just in case
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

        This method provides the same interface as AutoGenWrapper.create().

        Args:
            messages: Sequence of Autogen LLMMessage objects
            tools: Optional sequence of tools the LLM can call
            schema: Optional Pydantic schema for structured output
            cancellation_token: Optional cancellation token (not used by LiteLLM)
            **kwargs: Additional arguments for LiteLLM

        Returns:
            CreateResult or ModelOutput
        """
        # Convert messages to LiteLLM format
        litellm_messages = autogen_to_litellm_messages(messages)

        # Merge default parameters with runtime kwargs (runtime takes precedence)
        merged_params = self.default_parameters.to_api_params()
        merged_params.update(kwargs)

        # Build LiteLLM parameters
        # Determine the model name for the API call:
        # - For vertex_openai (custom GCP endpoint with extra_headers), use openai/<model> format
        # - For other providers (Azure, Anthropic, etc.), use litellm_model_name with proper prefix
        # Note: self.litellm_model_name is used for pricing lookups (may have different prefix)
        if self.base_url and self.extra_headers:
            # Vertex OpenAI-compatible endpoint - use openai/ prefix for OpenAI API format
            api_model = f"openai/{self.model}"
        else:
            # Standard provider (Azure, Anthropic, etc.) - use the resolved litellm model name
            api_model = self.litellm_model_name

        litellm_params = {
            "model": api_model,
            "messages": litellm_messages,
            **merged_params,
        }

        # Add API key if provided
        if self.api_key:
            litellm_params["api_key"] = self.api_key

        # Add base URL if provided
        if self.base_url:
            litellm_params["base_url"] = self.base_url

        # Get fresh token from token_provider if configured
        headers_to_add = {}
        if self.token_provider:
            fresh_token = self.token_provider()
            headers_to_add["Authorization"] = f"Bearer {fresh_token}"

        # Add extra headers if provided (e.g., Authorization for GCP)
        if self.extra_headers:
            # Merge token_provider headers with extra_headers
            merged_headers = {**self.extra_headers, **headers_to_add}
            litellm_params["extra_headers"] = merged_headers
        elif headers_to_add:
            litellm_params["extra_headers"] = headers_to_add

        # Add Vertex AI configuration if provided (for anthropic_vertex, gemini_vertex)
        if self.vertex_project:
            litellm_params["vertex_project"] = self.vertex_project
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
            # Native structured output supported - use response_format
            _validate_schema_constraints(schema, self.model_info)
            schema_dict = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema.schema()

            # Azure and Vertex AI models require $ref to be resolved inline
            # Vertex AI (including Llama) may not fully support $defs in JSON schemas
            is_azure = self.litellm_model_name and self.litellm_model_name.startswith("azure/")
            is_vertex = self.litellm_model_name and "vertex" in self.litellm_model_name.lower()
            if is_azure or is_vertex:
                logger.debug(f"LiteLLMWrapper: Resolving $ref in schema for {self.litellm_model_name}")
                schema_dict = resolve_json_schema_refs(schema_dict)
                schema_dict = make_all_properties_required(schema_dict)

            # Vertex AI requires enum values to be strings, not integers
            # Convert integer enums to string enums in the schema
            def convert_enum_values_to_strings(obj: Any) -> Any:
                """Recursively convert integer enum values to strings for Vertex AI compatibility."""
                if isinstance(obj, dict):
                    # Check if this is an enum property
                    if "enum" in obj and isinstance(obj["enum"], list):
                        obj["enum"] = [str(v) for v in obj["enum"]]
                    # Recursively process nested objects
                    return {k: convert_enum_values_to_strings(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert_enum_values_to_strings(item) for item in obj]
                return obj

            # Apply conversion for Vertex AI models (gemini, etc.)
            if self.litellm_model_name and ("gemini" in self.litellm_model_name.lower() or "vertex" in self.litellm_model_name.lower()):
                schema_dict = convert_enum_values_to_strings(schema_dict)

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
            # Use a fake tool to get structured output (same approach as AutoGenWrapper)
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
            # Convert Autogen Tool objects to LiteLLM format
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
        is_huggingface = api_model.startswith("huggingface/") if api_model else False
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
            litellm = _get_litellm()
            if litellm is not None and isinstance(e, litellm.ContentPolicyViolationError):
                # Handle content moderation errors from OpenAI/Azure without traceback
                error_msg = f"Content blocked by provider safety filter: {e!s}"
                logger.error(error_msg)
                raise ContentBlockedError(
                    message=error_msg,
                    filter_result={},  # LiteLLM doesn't provide detailed filter results
                ) from e
            # Generic LiteLLM error
            error_msg = f"LiteLLM call failed: {e}"
            raise ProcessingError(error_msg) from e

        # Calculate pricing from usage
        usage = response.usage if hasattr(response, "usage") else None
        pricing_metadata = self._calculate_pricing(usage)

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
                        # and clear tool_calls so litellm_to_autogen_result treats it as text
                        message.content = arguments_json
                        message.tool_calls = None  # Clear tool calls
                        choice.finish_reason = "stop"  # Set finish_reason to stop
                    else:
                        logger.warning(f"LiteLLMWrapper: Expected fake tool '{fake_tool_name}', got '{tool_call.function.name}'")

        # Convert response to Autogen format (always returns ModelOutput now)
        result = litellm_to_autogen_result(response, usage, self.litellm_model_name, schema)

        # Add pricing metadata (result is always ModelOutput now)
        # Preserve actual_model that was set in litellm_to_autogen_result()
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
        """Manage chat interaction with tool execution (matching AutoGenWrapper interface).

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
                # Execute tools (reuse logic from AutoGenWrapper)
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
        """Execute tools (reuse AutoGenWrapper implementation)."""
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

    def _calculate_pricing(self, usage: Any) -> dict[str, Any]:
        """Calculate pricing information from usage data."""
        if usage is None:
            logger.warning("LLM response had no usage data - using 0 tokens for pricing")
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_cost": 0.0,
            }

        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        # Use existing pricing calculation utility
        prompt_tokens, completion_tokens, total_cost = calculate_token_cost(
            model=self.litellm_model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
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

    @staticmethod
    def _provider_prefix_for_client_type(client_type: str) -> str:
        """Normalize internal client_type to a litellm provider prefix."""
        # Mapping of internal client types to litellm provider prefixes
        prefix_map = {
            "azure": "azure",
            "openai": "openai",
            "gemini": "gemini",  # litellm uses 'gemini' for Gemini API
            "gemini_vertex": "gemini",  # vertex-hosted Gemini still routes differently upstream
            "huggingface": "huggingface",
            "vertex_openai": "vertex_ai",  # For litellm pricing, Vertex models need vertex_ai prefix
            "llama_vertex": "vertex_ai",  # Llama on Vertex via native LiteLLM support
            "deepseek_vertex": "vertex_ai",  # DeepSeek on Vertex via native LiteLLM support
            "mistral_vertex": "vertex_ai",  # Mistral on Vertex via native LiteLLM support
            "anthropic_vertex": "vertex_ai",  # Anthropic-on-Vertex
            "anthropic": "anthropic",
            "zentropi": "zentropi",  # Zentropi custom API
        }
        return prefix_map.get(client_type, client_type)  # fallback / extension

    @staticmethod
    def _is_already_litellm_identifier(model_name: str, registry: dict[str, Any] | None = None) -> bool:
        """Heuristic: treat as already-qualified if first segment is a known provider and not an internal key."""
        # Known litellm provider prefixes
        known_litellm_providers = {
            "azure",
            "openai",
            "gemini",
            "huggingface",
            "vertex_ai",
            "anthropic",
        }

        if "/" not in model_name:
            return False
        first = model_name.split("/", 1)[0]
        keys = set(registry.keys()) if registry else set()
        return first in known_litellm_providers and model_name not in keys

    @staticmethod
    def lookup_litellm_model_name(model_name: str, client_type: str = "") -> str | None:
        """Resolve an internal model key to a litellm-compatible identifier.

        This function provides robust model name resolution by:
        1. Determining the correct provider prefix for the client_type
        2. Extracting the base model name (stripping any existing prefixes if needed)
        3. Applying intelligent mapping for known model variations
        4. Constructing the final litellm-compatible identifier

        Args:
            model_name: The model name from config (e.g., "google/gemini-2.5-flash")
            client_type: The client type (e.g., "vertex_openai", "gemini_vertex")

        Returns:
            Properly formatted litellm model identifier
        """
        if not model_name:
            return model_name

        # Get the correct provider prefix for this client type
        expected_prefix = LLMs._provider_prefix_for_client_type(client_type)

        # Handle special cases and extract base model name
        base_model = LLMs._extract_base_model_name(model_name, client_type)

        # For native API client types (gemini, anthropic), bare model names work
        # For openai client_type, always add the openai/ prefix — LiteLLM needs it
        # for provider routing, especially with custom base_url (e.g., grok on Azure AI)
        if client_type in {
            "gemini",
            "anthropic",
        } and not model_name.startswith(expected_prefix + "/"):
            return model_name

        # If the model name already has the correct prefix, return as-is
        if model_name.startswith(expected_prefix + "/"):
            return model_name

        # Construct the final litellm identifier
        return f"{expected_prefix}/{base_model}"

    @staticmethod
    def _extract_base_model_name(model_name: str, client_type: str) -> str:
        """Extract the base model name, handling various prefix patterns.

        Examples:
        - "google/gemini-2.5-flash" -> "gemini-2.5-flash" (strip google/ for litellm compatibility)
        - "gemini-2.5-flash" -> "gemini-2.5-flash"
        - "claude-sonnet-4@20250514" -> "claude-sonnet-4@20250514"
        """
        # Handle known model name patterns and client type combinations

        # For vertex_openai (legacy), strip google/ prefix from Gemini models for litellm pricing
        # e.g., "google/gemini-2.5-flash" -> "gemini-2.5-flash" (litellm expects vertex_ai/gemini-2.5-flash)
        # But preserve meta/ prefix for Llama models (litellm expects vertex_ai/meta/llama-*)
        if client_type == "vertex_openai":
            if model_name.startswith("google/"):
                return model_name[len("google/") :]  # Strip google/ prefix
            return model_name  # Keep other prefixes (e.g., meta/llama-*)

        # For llama_vertex, preserve the meta/ prefix for native LiteLLM support
        # LiteLLM expects model names like "meta/llama-4-maverick-17b-128e-instruct-maas"
        # and will construct the full model name as "vertex_ai/meta/llama-..."
        if client_type == "llama_vertex":
            # Keep meta/ prefix - litellm needs it for proper routing
            return model_name

        # For deepseek_vertex, preserve the deepseek-ai/ prefix for native LiteLLM support
        # LiteLLM expects model names like "deepseek-ai/deepseek-r1-0528-maas"
        # and will construct the full model name as "vertex_ai/deepseek-ai/deepseek-r1-..."
        if client_type == "deepseek_vertex":
            # Keep deepseek-ai/ prefix - litellm needs it for proper routing
            return model_name

        # For mistral_vertex, strip the mistralai/ prefix - LiteLLM adds it as the publisher
        if client_type == "mistral_vertex":
            if model_name.startswith("mistralai/"):
                return model_name[len("mistralai/") :]
            return model_name

        # For anthropic_vertex clients with provider-specific models, preserve format
        if client_type == "anthropic_vertex" and "/" in model_name:
            return model_name

        # For other cases, strip common provider prefixes if they don't match client type
        if "/" in model_name:
            prefix, base = model_name.split("/", 1)

            # If the existing prefix matches what we expect, keep the base
            expected_prefix = LLMs._provider_prefix_for_client_type(client_type)
            if prefix == expected_prefix:
                return base
            # Keep the full name as-is for cross-provider compatibility
            return model_name

        # No prefix found, return as-is
        return model_name

    def get_client(self, name: str) -> LiteLLMWrapper:
        """Gets or creates an LLM wrapper for the configuration specified by `name`.

        If a client for the given name already exists in the cache,
        it is returned. Otherwise, a new LiteLLMWrapper is instantiated based on the
        `LLMConfig` found in `connections`, cached, and returned.

        Args:
            name: The connection name of the LLM configuration (must be a key
                in `self.connections`).

        Returns:
            LiteLLMWrapper: The instantiated wrapper with `.create()` and `.call_chat()` methods.

        Raises:
            AttributeError: If `name` is not found in `self.connections`.
            ImportError: If LiteLLM is not available.
            ValueError: If essential configuration like GCP credentials for Vertex
                are missing.

        """
        # Check cache first
        if name in self.cached_clients:
            return self.cached_clients[name]

        if name not in self.connections:
            raise AttributeError(f"LLM configuration named '{name}' not found in connections.")

        config = self.connections[name]
        model_name = config.configs.get("model")

        # Resolve litellm model name: explicit override > lookup > raw model_name
        resolved_litellm = config.litellm_model or self.lookup_litellm_model_name(model_name or name, config.client_type.value) or model_name

        # Zentropi is a classification API, not an LLM - use ZentropiClassifier agent instead
        if config.client_type == ClientType.ZENTROPI:
            raise ProcessingError(
                "Zentropi models cannot be used as LLM clients. Use buttermilk.agents.ZentropiClassifier for classification tasks instead."
            )

        # Get merged parameters for this model
        merged_params = self.get_merged_parameters(name)

        logger.debug(f"Creating LiteLLMWrapper for model '{name}' with provider '{config.client_type.value}'")

        # Prepare provider-specific configuration
        extra_headers: dict[str, str] | None = None
        api_key = config.api_key
        vertex_project: str | None = None
        vertex_location: str | None = None
        # Default to config base_url, but some providers override this
        effective_base_url: str | None = config.base_url

        if config.client_type == ClientType.VERTEX_OPENAI:
            # Vertex OpenAI-compatible endpoints require GCP auth
            if not bm.gcp_credentials:
                raise ValueError("GCP credentials not available for Vertex AI.")
            gcp_token = bm.get_gcp_access_token()
            extra_headers = {"Authorization": f"Bearer {gcp_token}"}
            # LiteLLM requires an api_key, use placeholder for Vertex
            api_key = api_key or "vertex-gcp-auth"

        elif config.client_type == ClientType.ANTHROPIC_VERTEX:
            # Anthropic on Vertex requires project/location
            vertex_project = config.configs.get("project_id")
            vertex_location = config.configs.get("region")
            if not vertex_project or not vertex_location:
                raise ValueError("project_id and region are required for Anthropic Vertex AI.")

        elif config.client_type == ClientType.GEMINI_VERTEX:
            # Gemini on Vertex - uses standard Vertex AI auth
            vertex_project = config.configs.get("project_id")
            vertex_location = config.configs.get("region")

        elif config.client_type in (ClientType.LLAMA_VERTEX, ClientType.DEEPSEEK_VERTEX, ClientType.MISTRAL_VERTEX):
            # Vertex AI MaaS models via native LiteLLM support (Llama, DeepSeek, etc.)
            # LiteLLM handles auth and endpoint construction - no base_url needed
            if not bm.gcp_credentials:
                raise ValueError("GCP credentials not available for Vertex AI.")
            vertex_project = config.configs.get("project_id")
            vertex_location = config.configs.get("region")
            if not vertex_project or not vertex_location:
                provider = config.client_type.value
                raise ValueError(f"project_id and region are required for {provider}.")
            # Don't pass base_url - let LiteLLM construct the correct endpoint
            effective_base_url = None

        # Determine if token_provider is needed for this provider
        token_provider = None
        if config.client_type in (
            ClientType.VERTEX_OPENAI,
            ClientType.GEMINI_VERTEX,
            ClientType.LLAMA_VERTEX,
            ClientType.DEEPSEEK_VERTEX,
            ClientType.MISTRAL_VERTEX,
        ):
            # Vertex models need GCP token refresh
            def get_vertex_token() -> str:
                return bm.get_gcp_access_token()

            token_provider = get_vertex_token

        wrapped_client = LiteLLMWrapper(
            model=model_name,
            model_info=config.model_info,
            litellm_model_name=resolved_litellm,
            api_key=api_key,
            base_url=effective_base_url,
            extra_headers=extra_headers,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            default_parameters=merged_params,
            token_provider=token_provider,
        )

        self.cached_clients[name] = wrapped_client
        return wrapped_client

    # Alias for backwards compatibility
    get_autogen_chat_client = get_client

    def __getattr__(self, __name: str) -> LiteLLMWrapper:
        """Provides attribute-style access to LLM clients (e.g., `llms.my_model`)."""
        if __name not in self.connections:
            raise AttributeError(f"No LLM configuration found for '{__name}'. Available: {list(self.connections.keys())}")
        return self.get_client(__name)

    def __getitem__(self, __name: str) -> LiteLLMWrapper:
        """Provides item-style access to LLM clients (e.g., `llms["my_model"]`)."""
        return self.__getattr__(__name)
