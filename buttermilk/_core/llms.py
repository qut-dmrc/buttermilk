"""Manages Language Model (LLM) configurations, clients, and interactions.

This module provides structures for defining LLM configurations (`LLMConfig`),
managing different LLM providers and their clients (`LLMs`, `LLMClient`), and
wrapping chat completion clients (like those from Autogen) with additional
functionality such as rate limiting and retry logic (`AutoGenWrapper`).

It aims to abstract the complexities of interacting with various LLM APIs
and provide a consistent interface for agents within the Buttermilk framework.
"""

import asyncio
import importlib
import inspect
import json
import random
from collections.abc import Sequence
from enum import Enum
from typing import Any, Callable, TypeVar

# Core LLM library imports - these are required dependencies
from anthropic import AsyncAnthropicVertex

# LiteLLM imports
try:
    from litellm import acompletion

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    acompletion = None

# Autogen library imports - these are required dependencies
from autogen_core import CancellationToken, FunctionCall  # Autogen core types
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    CreateResult,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    ModelInfo,
)
from autogen_core.tools import Tool  # Autogen tool handling
from autogen_core.tools import BaseTool, ToolSchema
from autogen_ext.models.anthropic import AnthropicChatCompletionClient  # Autogen Anthropic client
from autogen_ext.models.openai import (  # Autogen OpenAI clients
    AzureOpenAIChatCompletionClient,
    OpenAIChatCompletionClient,
)
# from google import genai  # Google Generative AI library (unused in current implementation)
from pydantic import BaseModel  # Pydantic models for configuration
from pydantic import ConfigDict, Field, field_validator

from buttermilk import bm, logger
# ToolOutput import removed - using autogen's FunctionExecutionResult directly
from buttermilk._core.constants import (  # Models cache constants
    CONFIG_CACHE_FILENAME,
    cache,
    get_base_cache_dir,
)
from buttermilk._core.exceptions import ProcessingError  # Custom Buttermilk exceptions
from buttermilk.utils.pricing import calculate_token_cost  # Token cost calculation

from .retry import RetryWrapper  # Retry logic wrapper

_ = "ChatCompletionClient"  # Placeholder for type checking if needed


class ClientType(Enum):
    """Enumeration of supported LLM client types.

    Used to categorize LLM providers or services.

    Attributes:
        OPENAI: OpenAI platform.
        GEMINI: Google Generative AI platform (e.g., Gemini API).
        GEMINI_VERTEX: Gemini client on vertex platform.
        VERTEX_OPENAI: Google Vertex AI platform with OpenAI-compatible endpoint.
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
    VERTEX_OPENAI = "vertex_openai"  # OpenAI-compatible endpoint on Vertex
    HUGGINGFACE = "huggingface"  # HuggingFace Inference API (serverless or dedicated)


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

    def merge_with(self, other: "ModelParameters | dict | None") -> "ModelParameters":
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
            "anthropic", "gemini_vertex"). This determines which client class to use.
        api_key (str | None): The API key required for authenticating with the
            LLM provider. Can be None if authentication is handled differently
            (e.g., via environment variables or instance metadata).
        base_url (str | None): A custom base URL for the API endpoint, if
            different from the provider's default (e.g., for Azure OpenAI or
            self-hosted models).
        model_info (ModelInfo): An Autogen `ModelInfo` object containing detailed
            metadata about the model, such as its family, context window size,
            support for structured output, etc.
        configs (dict): A dictionary for additional options or configurations
            to pass directly to the constructor of the LLM client.
        litellm_model (str | None): An optional explicit litellm model identifier.
            If provided, this will be used instead of automatic resolution from
            client_type and model info. Useful for models that need specific naming
            for litellm pricing calculations.
        use_litellm (bool): If True, use LiteLLMWrapper instead of AutoGenWrapper.
            Defaults to False for backward compatibility. Enable this to use LiteLLM's
            unified interface for provider-agnostic LLM calls.
        parameters (ModelParameters): Default inference parameters (temperature,
            max_tokens, etc.) for this model. Defaults to empty ModelParameters
            instance. Can be specified as a dict which will be converted to
            ModelParameters during validation.

    """

    client_type: ClientType = Field(
        description="Type of client to instantiate (determines which client class to use)",
    )
    api_key: str | None = Field(
        default=None,
        description="API key to use for this model",
    )
    base_url: str | None = Field(default=None, description="Custom URL to call")

    model_info: ModelInfo = Field(
        ..., description="Model metadata (family, context size, etc.)"
    )
    configs: dict = Field(
        default_factory=dict, description="Options to pass to the constructor"
    )
    litellm_model: str | None = Field(
        default=None, description="Explicit litellm model identifier override"
    )
    use_litellm: bool = Field(
        default=False,
        description="Use LiteLLMWrapper instead of AutoGenWrapper (default: False for backward compatibility)",
    )
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
        raise ValueError(
            f"client_type must be a string or ClientType enum, got {type(v)}"
        )

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
    "gemini-pro",
    "gemini-flash",
    "gemini-flash-lite",
    "gpt5mini",
    "gpt5nano",
    "gpt-4o",
    "llama4maverick",
    "claude45sonnet",
    "gpt-oss-safeguard-20b",
    "gpt-oss-safeguard-120b",
]

"""A predefined list of identifiers for cost-effective chat models."""
CHEAP_CHAT_MODELS = [
    "gemini-flash",
    "gemini-flash-lite",
    "gpt5nano",
    "claude45haiku",
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


T_ChatClient = TypeVar("T_ChatClient", bound=ChatCompletionClient)
"""Type variable for generic Autogen ChatCompletionClient."""


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
    error_message: str | None = Field(
        default=None, description="Descriptive message about the error"
    )
    error_code: int | None = Field(
        default=None, description="Optional error code associated with the error"
    )
    raw_response: Any | None = Field(
        default=None, description="Raw response from the LLM, if available"
    )
    tool_outputs: list[FunctionExecutionResult] | None = Field(
        default=None, description="Tool outputs if any were executed"
    )
    tool_calls: list[FunctionCall] | None = Field(
        default=None, description="Tool calls made by the LLM, if any"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata including pricing information"
    )


class AutoGenWrapper(BaseModel):
    """Wraps an Autogen `ChatCompletionClient` to add rate limiting and robust retry logic.

    This class enhances Autogen clients by:
    1.  Implementing retry mechanisms (via composition with `RetryWrapper`) to
        handle transient API failures, rate limit errors, etc.
    2.  Potentially adding rate limiting capabilities (though semaphore usage is
        commented out in the provided code, it's a common pattern for such wrappers).
    3.  Simplifying the interface for making chat completion requests, including
        handling of structured output (JSON mode or Pydantic schema parsing) and
        tool/function calling.

    Attributes:
        client_factory (Callable[[], ChatCompletionClient]): Factory function that creates
            fresh client instances with current credentials/tokens.
        model_info (ModelInfo): Metadata about the model being wrapped, used to
            determine capabilities like structured output support.

    """

    client_factory: Callable[[], ChatCompletionClient] = Field(
        ..., description="Factory function for creating fresh client instances."
    )
    model_info: ModelInfo = Field(
        ..., description="Model metadata (family, context size, etc.)"
    )
    litellm_model_name: str = Field(
        default=None, description="Resolved litellm model name for pricing"
    )
    default_parameters: ModelParameters = Field(
        default_factory=ModelParameters,
        description="Default inference parameters (temperature, max_tokens, etc.)",
    )

    # Retry configuration (copied from RetryWrapper)
    cooldown_seconds: float = 0.5
    max_retries: int = 3
    min_wait_seconds: float = 5.0
    max_wait_seconds: float = 60.0
    jitter_seconds: float = 5.0

    model_config = {"arbitrary_types_allowed": True}

    def _get_fresh_client(self) -> ChatCompletionClient:
        """Get a fresh client instance with current credentials/tokens."""
        return self.client_factory()

    def _get_retry_wrapper(self) -> RetryWrapper:
        """Create RetryWrapper with fresh client instance."""
        fresh_client = self._get_fresh_client()
        return RetryWrapper(
            client=fresh_client,
            cooldown_seconds=self.cooldown_seconds,
            max_retries=self.max_retries,
            min_wait_seconds=self.min_wait_seconds,
            max_wait_seconds=self.max_wait_seconds,
            jitter_seconds=self.jitter_seconds,
        )

    async def create(  # noqa: PLR0912 - acceptable branching to normalize diverse provider results
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema] = [],
        schema: type[BaseModel] | None = None,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> CreateResult | ModelOutput:
        """Creates a chat completion using the wrapped client, with retry and structured output handling.

            This method attempts to make a chat completion call. It determines if
        structured output via a Pydantic schema should be requested based on the
        `schema` argument and `model_info`. It then uses the retry logic from
        `RetryWrapper` to execute the call.

            Args:
                messages: A sequence of `LLMMessage` objects representing the
                    conversation history.
                tools: An optional sequence of `Tool` objects that
                    the LLM can call.
                schema: An optional Pydantic `BaseModel` subclass. If provided and
                    the model supports structured output (`model_info.structured_output`),
                    the LLM will be instructed to generate output matching this schema.
                cancellation_token: An optional `CancellationToken` for aborting the request.
                **kwargs: Additional keyword arguments to pass to the underlying
                    client's `create` method.

            Returns:
                CreateResult | ModelOutput: The result from the LLM. If a schema was provided,
                    returns ModelOutput with parsed_object field containing the Pydantic instance.

            Raises:
                ProcessingError: If the LLM returns an empty/invalid response, unexpected
                    tool response types are received, or any post-call normalization/parsing fails.

        """
        parsed_object = None
        tool_calls: list[FunctionCall] | None = []

        is_valid_schema_type = (
            schema is not None
            and inspect.isclass(schema)
            and issubclass(schema, BaseModel)
            and schema
            is not BaseModel  # Ensure it's a specific subclass, not BaseModel itself
        )

        # Merge default parameters with runtime kwargs (runtime takes precedence)
        merged_params = self.default_parameters.to_api_params()
        merged_params.update(kwargs)

        # Build call kwargs, omitting json_output when tools are provided
        create_call_kwargs: dict[str, Any] = {
            "tools": tools,
            "cancellation_token": cancellation_token,
            "extra_create_args": merged_params,
        }

        # If caller requested a schema and didn't provide tools, choose best path per model capability
        used_fake_schema_tool = False
        fake_schema_tool = None
        if is_valid_schema_type:
            if self.model_info.get("structured_output", False):
                # Native structured output supported
                create_call_kwargs["json_output"] = schema  # type: ignore[arg-type]
            elif self.model_info.get("function_calling", True) and not tools:
                # No native structured output, but tool calling available and not used: use a fake tool

                class PydanticModelTool(BaseTool[BaseModel, BaseModel]):
                    """A tool that creates instances of a Pydantic model."""

                    def __init__(self, model: type[BaseModel]):
                        super().__init__(
                            args_type=model,
                            return_type=model,
                            name=f"create_{model.__name__.lower()}",
                            description=f"Create a {model.__name__} object with the specified fields",
                        )
                        self._model = model

                    async def run(
                        self, args: BaseModel, cancellation_token: CancellationToken
                    ) -> BaseModel:
                        # Ensure the provided args match the expected model type
                        if not isinstance(args, self._model):
                            raise ProcessingError(
                                f"PydanticModelTool expected {self._model.__name__}, got {type(args).__name__}",
                            )
                        return args

                fake_schema_tool = PydanticModelTool(schema)
                create_call_kwargs["tools"] = [fake_schema_tool]
                used_fake_schema_tool = True

        # Defensive check: autogen_ext has a bug where it crashes on empty messages
        # See: autogen_ext/models/anthropic/_anthropic_client.py:546
        # messages[-1] access without checking if list is empty
        if not messages or len(messages) == 0:
            raise ProcessingError(
                "Cannot call LLM with empty messages list (autogen_ext bug workaround)"
            )

        try:
            # Get retry wrapper with fresh client and current credentials/tokens
            retry_wrapper = self._get_retry_wrapper()
            create_result = await retry_wrapper._execute_with_retry(
                retry_wrapper.client.create,  # The method to call
                messages,  # Positional arguments for client.create
                **create_call_kwargs,  # Keyword arguments for client.create
            )

        except Exception as e:  # Wrap other exceptions
            import traceback
            error_msg = f"Error during LLM call: {e!s}\nTraceback: {traceback.format_exc()}"
            logger.error(error_msg)
            raise ProcessingError(f"Error during LLM call: {e!s}") from e

        # Calculate pricing from usage data (defensive check for None)
        usage = getattr(create_result, "usage", None)
        pricing_metadata = self._calculate_pricing(usage)

        # Extract actual model name from response if available (some providers return this)
        # Prefer actual model from API, fallback to our litellm_model_name
        actual_model_name = getattr(create_result, "model", self.litellm_model_name)

        # Now that we've made the LLM call and received a response, from
        # this point on, any errors we encounter will return a CreateResult or ModelOutput object
        # so that we can still finish tracing properly and log the received output.
        try:
            # First, check if the response content is empty
            if not create_result.content:
                raise ProcessingError("Empty response content from LLM.")
            if (
                isinstance(create_result.content, str)
                and not create_result.content.strip()
            ):
                raise ProcessingError("Empty string response from LLM.")

            # Next, check if content is a list and if all items are FunctionCall (valid tool call scenario)
            if isinstance(create_result.content, list):
                if all(
                    isinstance(item, FunctionCall) for item in create_result.content
                ):
                    if tools and not used_fake_schema_tool:
                        # If we have tools and didn't use a fake schema tool, return the tool calls with pricing
                        metadata = {
                            "pricing": pricing_metadata,
                            "model": actual_model_name,
                        }
                        return ModelOutput(
                            content=create_result.content,
                            finish_reason=create_result.finish_reason,
                            usage=create_result.usage,
                            thought=getattr(create_result, "thought", None),
                            cached=create_result.cached,
                            tool_calls=create_result.content,
                            metadata=metadata,
                        )
                    elif used_fake_schema_tool:
                        # If we used a fake schema tool, parse the tool call
                        # we don't log this fake tool as a tool call -- leave tool_calls empty.
                        tool_calls = None
                        if (
                            len(create_result.content) == 1
                            and fake_schema_tool
                            and create_result.content[0].name == fake_schema_tool.name
                        ):
                            parsed_object = json.loads(
                                create_result.content[0].arguments
                            )
                            create_result.content = json.dumps(parsed_object)
                        else:
                            raise ProcessingError(
                                "Malformed tool call response from LLM (expected fake schema tool call).",
                                create_result.content,
                            )
                    else:
                        raise ProcessingError(
                            "Malformed tool call response from LLM.",
                            create_result.content,
                        )
                else:
                    # If we have a list but not all items are FunctionCall, or the fake tool didn't fit, raise an error
                    raise ProcessingError(
                        "Unexpected response type from LLM when expecting tool calls or text.",
                        create_result.content,
                    )

            # If we get back a pydantic model or dict, we have to normalize so
            # that .content is always a string (or tool calls).
            if hasattr(create_result.content, "model_dump"):  # Pydantic BaseModel
                parsed_object = create_result.content
                create_result.content = parsed_object.model_dump_json()
            elif isinstance(create_result.content, dict):
                # If the content is a dict, convert it to a JSON string
                parsed_object = create_result.content
                create_result.content = json.dumps(create_result.content)

            # Handle schema parsing if requested
            if schema and is_valid_schema_type:
                # This will fail fast if the content cannot be parsed into the schema
                # Parse the content (which is now always a string) with the schema
                schema_parsed_object = await self._parse_structured_output(
                    create_result.content, schema
                )
                metadata = {
                    "pricing": pricing_metadata,
                    "model": actual_model_name,
                }
                return ModelOutput(
                    content=create_result.content,
                    finish_reason=create_result.finish_reason,
                    usage=create_result.usage,
                    thought=getattr(create_result, "thought", None),
                    parsed_object=schema_parsed_object,
                    cached=create_result.cached,
                    metadata=metadata,
                )

            metadata = {
                "pricing": pricing_metadata,
                "model": actual_model_name,
            }
            result = ModelOutput(
                content=create_result.content,
                finish_reason=create_result.finish_reason,
                usage=create_result.usage,
                thought=getattr(create_result, "thought", None),
                cached=create_result.cached,
                parsed_object=parsed_object,
                tool_calls=tool_calls,
                metadata=metadata,
            )
        except Exception as e:
            metadata = {
                "pricing": pricing_metadata,
                "model": actual_model_name,
            }
            result = ModelOutput(
                content=create_result.content,
                finish_reason=create_result.finish_reason,
                usage=create_result.usage,
                thought=getattr(create_result, "thought", None),
                cached=create_result.cached,
                parsed_object=None,  # Always None on error to prevent malformed BaseModel objects
                tool_calls=tool_calls,
                metadata=metadata,
            )
            result.error_message = f"LLM call failed: {e!s}"
            result.error_code = getattr(e, "code", None)  # Use code if available
            result.raw_response = (
                create_result.content
            )  # Store the raw response for debugging

        return result

    async def call_chat(  # noqa: PLR0913
        self,
        messages: list[LLMMessage],  # Made mutable for extending with tool results
        cancellation_token: CancellationToken | None,
        *,
        tools_list: Sequence[Tool | ToolSchema] = [],
        schema: type[BaseModel] | None = None,
        intercept_tools: bool = False,
    ) -> CreateResult | ModelOutput:
        """Manages a chat interaction with a single tool execution pass followed by optional synthesis.

        This method sends an initial set of messages to the LLM with tools (no schema).
        If the LLM responds with tool calls, executes them and makes a synthesis call
        with schema only (no tools). This avoids model limitations where tools and
        schemas cannot be used simultaneously.

        Args:
            messages: A list of `LLMMessage` objects forming the conversation.
                This list will be mutated if tool calls occur.
            cancellation_token: A `CancellationToken` for the operation.
            tools_list: An optional sequence of `Tool` or `ToolSchema` objects
                available for the LLM to call.
            schema: An optional Pydantic `BaseModel` subclass for structured output.
            intercept_tools: If True, return FunctionCall objects without executing them.
                This is useful for agents that need to handle tool calls specially.

        Returns:
            CreateResult | ModelOutput: The final result from the LLM.
                Returns ModelOutput if schema was provided or if synthesis was performed.

        """
        # Step 1: Initial call
        try:
            create_result = await self.create(
                messages=messages,
                tools=tools_list,
                cancellation_token=cancellation_token,
                schema=schema,
            )
        except Exception as e:
            # The call failed
            raise ProcessingError(f"Failed to query LLM: {e!s}") from e

        # Extract pricing from initial call
        # Use `or 0` pattern to handle both missing keys AND explicit None values
        initial_pricing = (
            create_result.metadata.get("pricing", {})
            if hasattr(create_result, "metadata")
            else {}
        )
        aggregated_pricing = {
            "prompt_tokens": initial_pricing.get("prompt_tokens") or 0,
            "completion_tokens": initial_pricing.get("completion_tokens") or 0,
            "total_cost": initial_pricing.get("total_cost") or 0.0,
        }

        # Step 2: Handle tool calls if present
        if isinstance(create_result.content, list) and all(
            isinstance(c, FunctionCall) for c in create_result.content
        ):
            tool_calls: list[FunctionCall] = create_result.content

            if intercept_tools:
                logger.debug(
                    f"Intercepting {len(tool_calls)} tool calls without execution"
                )
                return create_result

            # Add the assistant message with tool calls to the history
            assistant_msg = AssistantMessage(content=tool_calls, source="assistant")
            messages += [assistant_msg]

            try:
                tool_outputs = await self._execute_tools(
                    calls=tool_calls,
                    tools_list=tools_list,
                    cancellation_token=cancellation_token,
                )
                # Tool results are already FunctionExecutionResult objects
                tool_result_messages = FunctionExecutionResultMessage(
                    content=tool_outputs
                )
                messages += [tool_result_messages]
            except Exception as e:
                # Fail-fast: surface tool execution failures immediately
                raise ProcessingError(f"Failed to execute tools: {e!s}") from e

            # Step 3: Synthesis call with schema only (no tools)
            try:
                synthesis_result = await self.create(
                    messages=messages,
                    tools=[],  # No tools on synthesis call
                    cancellation_token=cancellation_token,
                    schema=schema,  # Apply schema for structured output
                )

                # Aggregate pricing from synthesis call
                if (
                    hasattr(synthesis_result, "metadata")
                    and "pricing" in synthesis_result.metadata
                ):
                    synthesis_pricing = synthesis_result.metadata["pricing"]
                    aggregated_pricing["prompt_tokens"] += (
                        synthesis_pricing.get("prompt_tokens") or 0
                    )
                    aggregated_pricing["completion_tokens"] += (
                        synthesis_pricing.get("completion_tokens") or 0
                    )
                    aggregated_pricing["total_cost"] += (
                        synthesis_pricing.get("total_cost") or 0.0
                    )
                    synthesis_result.metadata["pricing"] = aggregated_pricing

                return synthesis_result
            except Exception as e:
                raise ProcessingError(
                    f"Failed to synthesize after tool execution: {e!s}"
                ) from e

        # Return the original result
        return create_result

    async def _call_tool(  # noqa: D401
        self,
        call: FunctionCall,
        tool: Tool,
        cancellation_token: CancellationToken | None,
    ) -> FunctionExecutionResult:
        """Executes a single tool call and returns the result.

        Args:
            call: The FunctionCall from the LLM
            tool: The Tool object that matches call.name
            cancellation_token: Optional cancellation token

        Returns:
            FunctionExecutionResult ready to be sent back to the LLM

        """
        arguments = json.loads(call.arguments)
        arguments.update(
            arguments.pop("kwargs", {})
        )  # Merge 'kwargs' into arguments if present

        # Execute the tool
        ct: CancellationToken = cancellation_token or CancellationToken()
        result = await tool.run_json(arguments, ct)

        # Return autogen's native type directly
        return FunctionExecutionResult(
            call_id=call.id,
            name=tool.name,
            content=tool.return_value_as_string(result),
        )

    async def _execute_tools(
        self,
        calls: list[FunctionCall],
        tools_list: Sequence[Tool | ToolSchema],
        cancellation_token: CancellationToken | None,
    ) -> list[FunctionExecutionResult]:
        """Executes a list of tool calls concurrently.

        Args:
            calls: List of FunctionCall objects from the LLM
            tools_list: List of available Tool objects
            cancellation_token: Optional cancellation token

        Returns:
            List of FunctionExecutionResult objects

        """
        tasks = []
        for call in calls:
            # Find the tool by name
            tool = next((t for t in tools_list if t.name == call.name), None)
            if tool is None:
                raise ProcessingError(
                    f"Tool '{call.name}' requested by LLM not found in provided tools list."
                )

            tasks.append(self._call_tool(call, tool, cancellation_token))

        # Execute all tool calls concurrently
        return await asyncio.gather(*tasks)

    def _calculate_pricing(self, usage: Any) -> dict[str, Any]:
        """Calculate pricing information from usage data.

        Args:
            usage: RequestUsage object or None

        Returns:
            Dictionary with pricing information
        """
        if usage is None:
            logger.warning(
                "LLM response had no usage data - using 0 tokens for pricing"
            )
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": 0.0}

        # Extract tokens from usage object
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        # Calculate cost using the utility function with resolved litellm model name
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

    @staticmethod
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
            ChatParser = getattr(_mod, "ChatParser")
            simple_clean_llm_json_text = getattr(_mod, "simple_clean_llm_json_text")

            # Try to parse as strict JSON first to preserve types (avoid coercion)
            logger.debug(
                f"AutoGenWrapper: Attempting to parse string response into {schema.__name__}"
            )
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
                except Exception as parse_error:
                    raise ProcessingError(
                        f"AutoGenWrapper failed to parse LLM response into required schema {schema.__name__}: {parse_error}",
                    ) from parse_error
        else:
            parsed_object = content

        # Validate the parsed object against the schema
        try:
            parsed_object = schema.model_validate(parsed_object)
        except Exception as parse_error:
            raise ProcessingError(
                f"AutoGenWrapper failed to parse LLM response into required schema {schema.__name__}: {parse_error}",
            ) from parse_error

        if parsed_object is None:
            raise ProcessingError(
                f"AutoGenWrapper requires structured output of type {schema.__name__} but parsing failed",
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
            if isinstance(msg.content, list) and all(
                isinstance(c, FunctionCall) for c in msg.content
            ):
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
                litellm_messages.append(
                    {"role": "assistant", "content": None, "tool_calls": tool_calls}
                )
            else:
                # Regular text response
                content = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
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


def litellm_to_autogen_result(
    response: Any, usage: Any, model: str, schema: type[BaseModel] | None = None
) -> ModelOutput:
    """Convert LiteLLM response to Autogen ModelOutput.

    Args:
        response: LiteLLM response object or dict
        usage: Usage information from LiteLLM
        model: Model name used (our shorthand)
        schema: Optional Pydantic schema for structured output

    Returns:
        ModelOutput compatible with Autogen interface (always returns ModelOutput to preserve pricing metadata)
    """
    from autogen_core.models import RequestUsage

    # Extract content from response
    content: str | list[FunctionCall]
    if hasattr(response, "choices") and response.choices and len(response.choices) > 0:
        choice = response.choices[0]
        message = choice.message if hasattr(choice, "message") else choice

        # Check for tool calls (check both existence and non-empty list)
        tool_calls_attr = getattr(message, "tool_calls", None)
        if (
            tool_calls_attr is not None
            and isinstance(tool_calls_attr, list)
            and len(tool_calls_attr) > 0
        ):
            # Convert to FunctionCall objects
            tool_calls: list[FunctionCall] = []
            for tc in tool_calls_attr:
                tool_calls.append(
                    FunctionCall(
                        id=tc.id, name=tc.function.name, arguments=tc.function.arguments
                    )
                )
            content = tool_calls
        else:
            # Regular text content
            content = message.content if hasattr(message, "content") else str(message)

        finish_reason = (
            choice.finish_reason if hasattr(choice, "finish_reason") else "stop"
        )
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

        if not LITELLM_AVAILABLE:
            raise ImportError(
                "LiteLLM is not installed. Please install it with: pip install litellm"
            )

    async def _execute_with_retry(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
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

                # Check if this is a retryable error
                is_retryable = any(
                    keyword in error_msg
                    for keyword in ["rate limit", "timeout", "503", "429", "502", "500"]
                )

                if attempt < self.max_retries and is_retryable:
                    # Calculate wait time with jitter
                    jitter = random.uniform(-self.jitter_seconds, self.jitter_seconds)
                    actual_wait = min(wait_time + jitter, self.max_wait_seconds)

                    logger.warning(
                        f"LiteLLM call failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {actual_wait:.1f}s..."
                    )

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

        # Handle structured output via response_format
        if schema and self.model_info.get("structured_output", False):
            litellm_params["response_format"] = {
                "type": "json_object",
                "schema": schema.model_json_schema()
                if hasattr(schema, "model_json_schema")
                else schema.schema(),
            }

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

        # Execute with retry logic
        async def _call_litellm() -> Any:
            return await acompletion(**litellm_params)

        try:
            response = await self._execute_with_retry(_call_litellm)
        except Exception as e:
            error_msg = f"LiteLLM call failed: {e}"
            raise ProcessingError(error_msg) from e

        # Calculate pricing from usage
        usage = response.usage if hasattr(response, "usage") else None
        pricing_metadata = self._calculate_pricing(usage)

        # Convert response to Autogen format (always returns ModelOutput now)
        result = litellm_to_autogen_result(
            response, usage, self.litellm_model_name, schema
        )

        # Add pricing metadata (result is always ModelOutput now)
        # Preserve actual_model that was set in litellm_to_autogen_result()
        result.metadata["pricing"] = pricing_metadata

        # Parse structured output if schema was provided
        if schema:
            try:
                parsed = await AutoGenWrapper._parse_structured_output(
                    result.content, schema
                )
                result.parsed_object = parsed
            except Exception as e:
                result.error_message = f"Failed to parse structured output: {e}"
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
        initial_pricing = (
            create_result.metadata.get("pricing", {})
            if hasattr(create_result, "metadata")
            else {}
        )
        aggregated_pricing = {
            "prompt_tokens": initial_pricing.get("prompt_tokens") or 0,
            "completion_tokens": initial_pricing.get("completion_tokens") or 0,
            "total_cost": initial_pricing.get("total_cost") or 0.0,
        }

        # Step 2: Handle tool calls if present
        if isinstance(create_result.content, list) and all(
            isinstance(c, FunctionCall) for c in create_result.content
        ):
            tool_calls: list[FunctionCall] = create_result.content

            if intercept_tools:
                logger.debug(
                    f"Intercepting {len(tool_calls)} tool calls without execution"
                )
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
                tool_result_messages = FunctionExecutionResultMessage(
                    content=tool_outputs
                )
                messages.append(tool_result_messages)
            except Exception as e:
                raise ProcessingError(f"Failed to execute tools: {e}") from e

            # Step 3: Synthesis call with schema only (no tools)
            try:
                synthesis_result = await self.create(
                    messages=messages,
                    tools=[],
                    cancellation_token=cancellation_token,
                    schema=schema,
                )

                # Aggregate pricing
                if (
                    hasattr(synthesis_result, "metadata")
                    and "pricing" in synthesis_result.metadata
                ):
                    synthesis_pricing = synthesis_result.metadata["pricing"]
                    aggregated_pricing["prompt_tokens"] += (
                        synthesis_pricing.get("prompt_tokens") or 0
                    )
                    aggregated_pricing["completion_tokens"] += (
                        synthesis_pricing.get("completion_tokens") or 0
                    )
                    aggregated_pricing["total_cost"] += (
                        synthesis_pricing.get("total_cost") or 0.0
                    )
                    synthesis_result.metadata["pricing"] = aggregated_pricing

                return synthesis_result
            except Exception as e:
                raise ProcessingError(
                    f"Failed to synthesize after tool execution: {e}"
                ) from e

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
                raise ProcessingError(
                    f"Tool '{call.name}' requested by LLM not found in provided tools list."
                )

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
            logger.warning(
                "LLM response had no usage data - using 0 tokens for pricing"
            )
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_cost": 0.0}

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
    2.  On demand, instantiate and cache `AutoGenWrapper` clients for these
        configurations using `get_autogen_chat_client`.
    3.  Provide convenient attribute-style access (e.g., `llms.my_gpt_model`)
        and item-style access (e.g., `llms["my_gpt_model"]`) to these clients.

    Attributes:
        connections (dict[str, LLMConfig]): A dictionary where keys are
            connection names (e.g., "azure_prod_gpt4") and values are
            `LLMConfig` objects detailing the configuration for that LLM.
        autogen_models (dict[str, AutoGenWrapper]): A cache for instantiated
            `AutoGenWrapper` clients. This is populated on-demand when a client
            is first requested. Not meant to be set directly by users.
        model_config (ConfigDict): Pydantic model configuration.
            - `use_enum_values`: True - Ensures enum members are used for validation/serialization.

    """

    connections: dict[str, LLMConfig] = Field(
        default_factory=dict,  # Changed from list to dict factory
        description="A dictionary where keys are connection names and values are LLMConfig objects.",
    )
    default_wrapper: str = Field(
        default="autogen",
        description="Default LLM wrapper type (autogen or litellm). Used when config.use_litellm is None.",
    )
    model_parameters: dict[str, ModelParameters | dict] = Field(
        default_factory=dict,
        description="Per-model parameter overrides from YAML config (model_name -> parameters)",
    )
    autogen_models: dict[str, AutoGenWrapper] = Field(
        default_factory=dict,  # For caching instantiated clients
        description="Cache for instantiated AutoGenWrapper clients. Populated on demand.",
        exclude=True,  # Exclude from model dump as it's runtime state
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
            "vertex_openai": "vertex_ai",  # Vertex OpenAI-compatible
            "anthropic_vertex": "vertex_ai",  # Anthropic-on-Vertex
            "anthropic": "anthropic",
        }
        return prefix_map.get(client_type, client_type)  # fallback / extension

    @staticmethod
    def _is_already_litellm_identifier(
        model_name: str, registry: dict[str, Any] | None = None
    ) -> bool:
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

        # For certain client types, we need the model name as-is (already has correct prefix)
        if client_type in {
            "gemini",
            "openai",
            "anthropic",
        } and not model_name.startswith(expected_prefix + "/"):
            # These often use bare model names without provider prefix
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

        # For vertex_openai client with google/ models, strip the google/ prefix for litellm compatibility
        if client_type == "vertex_openai" and model_name.startswith("google/"):
            return model_name[7:]  # Strip "google/" prefix

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

    def get_autogen_chat_client(self, name: str) -> AutoGenWrapper | LiteLLMWrapper:  # noqa: PLR0912 - branching per client type
        """Gets or creates an LLM wrapper for the configuration specified by `name`.

        If a client for the given name already exists in the `autogen_models` cache,
        it is returned. Otherwise, a new client is instantiated based on the
        `LLMConfig` found in `connections`, wrapped with either `AutoGenWrapper`
        or `LiteLLMWrapper` (depending on `use_litellm` flag), cached, and returned.

        Args:
            name: The connection name of the LLM configuration (must be a key
                in `self.connections`).

        Returns:
            AutoGenWrapper | LiteLLMWrapper: The instantiated wrapper. Both types
                provide the same interface (`.create()`, `.call_chat()`).

        Raises:
            AttributeError: If `name` is not found in `self.connections`.
            ImportError: If necessary client libraries are not available.
            ValueError: If essential configuration like GCP credentials for Vertex
                are missing.

        """
        # Check cache first
        if name in self.autogen_models:
            return self.autogen_models[name]

        if name not in self.connections:
            raise AttributeError(
                f"LLM configuration named '{name}' not found in connections."
            )

        config = self.connections[name]
        model_name = config.configs.get("model")
        # Prepare client parameters from configs
        client_params: dict[str, Any] = {
            "model": model_name,
            "api_key": config.api_key,
            **config.configs,
        }

        # Resolve litellm model name using internal method
        resolved_litellm = (
            self.lookup_litellm_model_name(model_name or name, config.client_type.value)
            or model_name
        )

        # Create client factory function based on config.client_type
        def create_client_factory() -> Callable[[], ChatCompletionClient]:
            """Create a factory function that returns fresh clients with current credentials."""

            # Define factory functions for each client type
            def _openai_factory() -> ChatCompletionClient:
                return OpenAIChatCompletionClient(
                    base_url=config.base_url
                    or "",  # Provide default empty string if None
                    model_info=config.model_info,
                    **client_params,
                )

            def _azure_factory() -> ChatCompletionClient:
                if not config.base_url:
                    raise ValueError("Azure endpoint URL is required for Azure client")
                return AzureOpenAIChatCompletionClient(
                    azure_endpoint=config.base_url,
                    model_info=config.model_info,
                    **client_params,
                )

            def _anthropic_factory() -> ChatCompletionClient:
                # Direct Anthropic API
                return AnthropicChatCompletionClient(**client_params)

            def _anthropic_vertex_factory() -> ChatCompletionClient:
                # Anthropic via Vertex AI
                if not bm.gcp_credentials:
                    raise ValueError(
                        "GCP credentials not available for Anthropic via Vertex AI."
                    )

                vertex_params = {
                    "region": config.configs.get("region"),
                    "project_id": config.configs.get("project_id"),
                    "credentials": bm.gcp_credentials,
                }
                vertex_params = {
                    k: v for k, v in vertex_params.items() if v is not None
                }

                try:
                    vertex_client = AsyncAnthropicVertex(**vertex_params)
                    # Remove api_key for Vertex auth
                    vertex_client_params = client_params.copy()
                    vertex_client_params.pop("api_key", None)

                    client = AnthropicChatCompletionClient(**vertex_client_params)
                    client._client = vertex_client  # type: ignore[attr-defined]
                    return client
                except Exception as e:
                    logger.error(
                        f"Error initializing Anthropic client for Vertex: {e!s}"
                    )
                    raise

            def _gemini_factory() -> ChatCompletionClient:
                # Google Generative AI (Gemini) API
                if not bm.gcp_credentials:
                    raise ValueError("GCP credentials not available for Gemini API.")
                return OpenAIChatCompletionClient(
                    model_info=config.model_info,
                    **client_params,
                )

            def _gemini_vertex_factory() -> ChatCompletionClient:
                vertex_params = client_params.copy()
                # Get fresh token on each client creation
                vertex_params["api_key"] = bm.get_gcp_access_token()
                return OpenAIChatCompletionClient(
                    base_url=config.base_url,
                    model_info=config.model_info,
                    **vertex_params,
                )

            def _vertex_openai_factory() -> ChatCompletionClient:
                # OpenAI-compatible endpoint on Vertex (for Llama, etc.)
                if not bm.gcp_credentials:
                    raise ValueError("GCP credentials not available for Vertex AI.")

                vertex_params = client_params.copy()

                # Set up OAuth2 bearer token authentication with fresh token
                headers = {
                    "Authorization": f"Bearer {bm.get_gcp_access_token()}",
                }

                # Dummy API key for OpenAI client validation
                if vertex_params.get("api_key") is None:
                    vertex_params["api_key"] = "dummy-key-for-vertex"

                vertex_params["default_headers"] = headers

                if not config.base_url:
                    raise ValueError("Base URL is required for Vertex OpenAI endpoint")
                return OpenAIChatCompletionClient(
                    base_url=config.base_url,
                    model_info=config.model_info,
                    **vertex_params,
                )

            # Map client types to their factory functions
            factory_map = {
                ClientType.OPENAI: _openai_factory,
                ClientType.AZURE: _azure_factory,
                ClientType.ANTHROPIC: _anthropic_factory,
                ClientType.ANTHROPIC_VERTEX: _anthropic_vertex_factory,
                ClientType.GEMINI: _gemini_factory,
                ClientType.GEMINI_VERTEX: _gemini_vertex_factory,
                ClientType.VERTEX_OPENAI: _vertex_openai_factory,
            }

            # Get the appropriate factory or raise error
            factory = factory_map.get(config.client_type)
            if factory is None:
                raise ProcessingError(f"Unsupported client_type: {config.client_type}")
            return factory

        # Choose wrapper type based on configuration
        # Per-model use_litellm takes precedence over global default_wrapper
        use_litellm = (
            config.use_litellm
            if config.use_litellm is not None
            else (self.default_wrapper == "litellm")
        )

        # Get merged parameters for this model
        merged_params = self.get_merged_parameters(name)

        if use_litellm:
            # Use LiteLLMWrapper for unified provider support
            logger.debug(
                f"Using LiteLLMWrapper for model '{name}' with provider '{config.client_type.value}'"
            )

            wrapped_client = LiteLLMWrapper(
                model=model_name,
                model_info=config.model_info,
                litellm_model_name=resolved_litellm,
                api_key=config.api_key,
                base_url=config.base_url,
                default_parameters=merged_params,
            )
        else:
            # Use AutoGenWrapper (existing behavior)
            logger.debug(
                f"Using AutoGenWrapper for model '{name}' with provider '{config.client_type.value}'"
            )
            client_factory = create_client_factory()
            wrapped_client = AutoGenWrapper(
                client_factory=client_factory,
                model_info=config.model_info,
                litellm_model_name=resolved_litellm,
                default_parameters=merged_params,
            )

        self.autogen_models[name] = wrapped_client
        return wrapped_client

    def __getattr__(self, __name: str) -> AutoGenWrapper | LiteLLMWrapper:
        """Provides attribute-style access to LLM clients (e.g., `llms.my_model`)."""
        if __name not in self.connections:
            raise AttributeError(
                f"No LLM configuration found for '{__name}'. Available: {list(self.connections.keys())}"
            )
        return self.get_autogen_chat_client(__name)

    def __getitem__(self, __name: str) -> AutoGenWrapper | LiteLLMWrapper:
        """Provides item-style access to LLM clients (e.g., `llms["my_model"]`)."""
        return self.__getattr__(__name)
