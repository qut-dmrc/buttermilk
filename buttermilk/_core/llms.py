"""Manages Language Model (LLM) configurations, clients, and interactions.

This module provides structures for defining LLM configurations (`LLMConfig`),
managing different LLM providers and their clients (`LLMs`, `LLMClient`), and
wrapping chat completion clients (like those from Autogen) with additional
functionality such as rate limiting and retry logic (`AutoGenWrapper`).

It aims to abstract the complexities of interacting with various LLM APIs
and provide a consistent interface for agents within the Buttermilk framework.
"""

import asyncio
import inspect
import json
from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

# Core LLM library imports - these are required dependencies
import openai
from anthropic import (
    AnthropicVertex,  # Client for Anthropic models on Vertex AI
    AsyncAnthropicVertex,
)

# Autogen library imports - these are required dependencies
from autogen_core import CancellationToken, FunctionCall  # Autogen core types
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelFamily,
    ModelInfo,
    FunctionExecutionResultMessage,
    FunctionExecutionResult,
)
from autogen_core.models import AssistantMessage, UserMessage
from autogen_core.tools import Tool  # Autogen tool handling
from autogen_ext.models.anthropic import AnthropicChatCompletionClient  # Autogen Anthropic client
from autogen_ext.models.openai import (  # Autogen OpenAI clients
    AzureOpenAIChatCompletionClient,
    OpenAIChatCompletionClient,
)
from autogen_ext.models.openai._transformation.registry import (
    _find_model_family,  # Helper for model family detection
)
from autogen_openaiext_client import GeminiChatCompletionClient  # Autogen Gemini client
from pydantic import BaseModel, ConfigDict, Field  # Pydantic models for configuration


# Use a function for deferred import to avoid circular references
def get_bm():
    """Get the BM singleton with delayed import to avoid circular references."""
    from buttermilk._core.dmrc import get_bm as _get_bm  # Actual import of get_bm
    return _get_bm()


# ToolOutput import removed - using autogen's FunctionExecutionResult directly
from buttermilk._core.exceptions import ProcessingError  # Custom Buttermilk exceptions
from buttermilk._core.log import logger  # Buttermilk logger

from .retry import RetryWrapper  # Retry logic wrapper

_ = "ChatCompletionClient"  # Placeholder for type checking if needed


class LLMConfig(BaseModel):
    """Configuration for a specific Language Model (LLM).

    Defines the connection details, model object to instantiate, API type,
    API key, custom base URL, model-specific information, and any additional
    configurations required by the LLM client.

    Attributes:
        connection (str): A descriptive identifier for the type of connection
            or provider (e.g., "AzureOpenAI", "VertexAI-Gemini", "OpenAI-GPT4").
            This helps in categorizing and managing different LLM setups.
        obj (str): The name of the specific model object or deployment to
            instantiate (e.g., "gpt-4-turbo", "gemini-1.5-pro", "claude-3-opus").
        api_type (str): The type of API the model uses (e.g., "openai",
            "vertex", "azure", "anthropic"). Defaults to "openai".
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

    """

    connection: str = Field(
        ...,
        description="Descriptive identifier for the type of connection used (e.g. Azure, Vertex)",
    )
    obj: str = Field(..., description="Name of the model object to instantiate")
    api_type: str = Field(
        description="Type of API to use (e.g. openai, vertex, azure)",
    )
    api_key: str | None = Field(
        default=None,
        description="API key to use for this model",
    )
    base_url: str | None = Field(default=None, description="Custom URL to call")

    model_info: ModelInfo
    configs: dict = Field(default_factory=dict, description="Options to pass to the constructor")


class MLPlatformTypes(Enum):
    """Enumeration of supported Machine Learning platform types.

    Used to categorize LLM providers or services.

    Attributes:
        openai: OpenAI platform.
        google_genai: Google Generative AI platform (e.g., Gemini API).
        google_vertexai: Google Vertex AI platform.
        anthropic: Anthropic platform (e.g., Claude models).
        llama: Llama models (often self-hosted or via specific providers).
        azure: Microsoft Azure AI platform (e.g., Azure OpenAI).

    """

    openai = "openai"
    google_genai = "google-genai"
    google_vertexai = "google-vertexai"
    anthropic = "anthropic"
    llama = "llama"
    azure = "azure"


# Generate with:
# ```sh
# cat .cache/buttermilk/models.json | jq "keys[]"
# ```
"""A predefined list of chat model identifiers available within the Buttermilk setup."""
CHATMODELS = [
    "gemini25pro",
    "gemini25flash",
    "gpt41nano",
    "gpt41mini",
    "gpt41",
    "o3mini",
    "llama4maverick",
    "llama32_90b",
    "llama33_70b",
    "opus",
    "haiku",
    "sonnet",
]

"""A predefined list of identifiers for cost-effective chat models."""
CHEAP_CHAT_MODELS = [
    "haiku",
    "gemini25flash",
    "o3mini",
    "gpt41mini",
]

MULTIMODAL_MODELS = ["gemini25pro", "llama4maverick", "gemini25flash", "gpt41", "llama32_90b"]
"""A predefined list of identifiers for multimodal models (supporting text, images, etc.)."""


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
    """Extends Autogen's `CreateResult` with structured output parsing.

    Adds a parsed_object field to hold a Pydantic model instance when
    the LLM returns structured JSON output that can be parsed.

    Attributes:
        parsed_object (BaseModel | None): The Pydantic model instance hydrated from
            the LLM's JSON content. None if no structured output or parsing failed.

    """

    parsed_object: BaseModel | None = Field(
        default=None,
        description="The Pydantic model instance hydrated from LLM's JSON or structured output.",
    )


class AutoGenWrapper(RetryWrapper):
    """Wraps an Autogen `ChatCompletionClient` to add rate limiting and robust retry logic.

    This class enhances Autogen clients by:
    1.  Implementing retry mechanisms (via inheritance from `RetryWrapper`) to
        handle transient API failures, rate limit errors, etc.
    2.  Potentially adding rate limiting capabilities (though semaphore usage is
        commented out in the provided code, it's a common pattern for such wrappers).
    3.  Simplifying the interface for making chat completion requests, including
        handling of structured output (JSON mode or Pydantic schema parsing) and
        tool/function calling.

    Attributes:
        client (ChatCompletionClient): The underlying Autogen chat completion client instance.
        model_info (ModelInfo): Metadata about the model being wrapped, used to
            determine capabilities like structured output support.

    """

    client: ChatCompletionClient
    model_info: ModelInfo

    async def create(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool] = [],
        schema: type[BaseModel] | None = None,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> CreateResult:
        """Creates a chat completion using the wrapped client, with retry and structured output handling.

        This method attempts to make a chat completion call. It determines if
        JSON mode or structured output via a Pydantic schema should be requested
        based on the `schema` argument and `model_info`. It then uses the
        retry logic from `RetryWrapper` to execute the call.

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
            CreateResult: The result from the LLM, potentially including tool calls
                or the final response after tool execution.

        Raises:
            ProcessingError: If the LLM returns an empty response or an unexpected
                tool response, or if any other error occurs during the LLM call
                after retries are exhausted.

        """
        try:
            is_valid_schema_type = (
                schema is not None
                and inspect.isclass(schema)
                and issubclass(schema, BaseModel)
                and schema is not BaseModel  # Ensure it's a specific subclass, not BaseModel itself
            )

            json_output_requested: bool | type[BaseModel] = False  # Default to no JSON mode
            if is_valid_schema_type and self.model_info.get("structured_output"):
                json_output_requested = schema  # type: ignore # Pass the schema for structured output

            # Some models don't support simultaneous tool calling and structured output
            # Check model family to determine capabilities
            model_family = self.model_info.get('family', ModelFamily.UNKNOWN)
            
            # Gemini models can't use tools with structured output (json_output with Pydantic model)
            # This is a known limitation documented in Gemini's API
            gemini_families = {ModelFamily.GEMINI_1_5_FLASH, ModelFamily.GEMINI_1_5_PRO, 
                             ModelFamily.GEMINI_2_0_FLASH, ModelFamily.GEMINI_2_5_PRO, 
                             ModelFamily.GEMINI_2_5_FLASH, 'gemini'}
            
            if model_family in gemini_families and json_output_requested and isinstance(json_output_requested, type):
                # For Gemini with Pydantic schema, we can't use tools
                tools_to_pass = []
            else:
                tools_to_pass = tools
            
            create_result = await self._execute_with_retry(
                self.client.create,  # The method to call
                messages,          # Positional arguments for self.client.create
                tools=tools_to_pass,
                json_output=json_output_requested,
                cancellation_token=cancellation_token,
                extra_create_args=kwargs,  # Keyword arguments for self.client.create
            )

            if not create_result.content:
                raise ProcessingError("Empty response content from LLM.")
            if isinstance(create_result.content, str) and not create_result.content.strip():
                raise ProcessingError("Empty string response from LLM.")
            # Check if content is a list and if all items are FunctionCall (valid tool call scenario)
            if isinstance(create_result.content, list) and \
               not all(isinstance(item, FunctionCall) for item in create_result.content):
                raise ProcessingError("Unexpected response type from LLM when expecting tool calls or text.", create_result.content)

        except ProcessingError:  # Re-raise known ProcessingErrors
            raise
        except Exception as e:  # Wrap other exceptions
            error_msg = f"Error during LLM call: {e!s}"
            raise ProcessingError(error_msg) from e

        return create_result  # type: ignore # Expect CreateResult or compatible

    async def call_chat(
        self,
        messages: list[LLMMessage],  # Made mutable for extending with tool results
        cancellation_token: CancellationToken | None,
        tools_list: Sequence[Tool] = [],
        schema: type[BaseModel] | None = None,
    ) -> CreateResult:
        """Manages a chat interaction, including potential tool calls and responses.

        This method sends an initial set of messages to the LLM. If the LLM
        responds with tool call requests, this method executes those tools,
        appends their results back to the message history, and sends the updated
        history back to the LLM to get a final response.

        Args:
            messages: A list of `LLMMessage` objects forming the conversation.
                This list will be mutated if tool calls occur.
            cancellation_token: A `CancellationToken` for the operation.
            tools_list: An optional sequence of `Tool` objects
                available for the LLM to call.
            schema: An optional Pydantic `BaseModel` subclass for structured output
                requests (if no tools are called).

        Returns:
            CreateResult: The final chat completion result from the LLM after
                any tool call cycles.

        """
        # If we have a schema (structured output), don't pass tools to avoid Gemini conflict
        # Models that support both will still work, models like Gemini will get either/or
        effective_tools = tools_list if not schema else []
        
        create_result = await self.create(
            messages=messages,
            tools=effective_tools,
            cancellation_token=cancellation_token,
            schema=schema,
        )

        # If the LLM responded with a request to call tools
        if isinstance(create_result.content, list) and all(isinstance(c, FunctionCall) for c in create_result.content):
            tool_calls: list[FunctionCall] = create_result.content

            # Add the assistant message with tool calls to the history
            assistant_msg = AssistantMessage(content=tool_calls, source="assistant")
            messages = messages + [assistant_msg]

            try:
                tool_outputs = await self._execute_tools(
                    calls=tool_calls,
                    tools_list=tools_list,
                    cancellation_token=cancellation_token,
                )
            except Exception as e:
                # If tool execution fails, we can log the error and return the original result
                logger.error(f"Error executing tools: {e!s}")
                raise ProcessingError(f"Failed to execute tools: {e!s}") from e

            # Tool results are already FunctionExecutionResult objects
            tool_result_messages = FunctionExecutionResultMessage(content=tool_outputs)
            messages = messages + [tool_result_messages]  # Append tool results to the message history

            try:
                # Call the LLM again with the tool results included in the history
                create_result = await self.create(
                    messages=messages,
                    cancellation_token=cancellation_token,
                    schema=schema,
                )
            except Exception as e:
                # If tool execution fails, we can log the error and return the original result
                logger.error(f"Error executing tools: {e!s}")
                raise ProcessingError(f"Failed to execute tools: {e!s}") from e

        return create_result  # type: ignore # Expect CreateResult

    async def _call_tool(
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
        arguments.update(arguments.pop("kwargs", {}))  # Merge 'kwargs' into arguments if present

        # Execute the tool
        result = await tool.run_json(arguments, cancellation_token)

        # Return autogen's native type directly
        return FunctionExecutionResult(
            call_id=call.id,
            name=tool.name,
            content=tool.return_value_as_string(result)
        )

    async def _execute_tools(
        self,
        calls: list[FunctionCall],
        tools_list: Sequence[Tool],
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
                raise ProcessingError(f"Tool '{call.name}' requested by LLM not found in provided tools list.")

            tasks.append(self._call_tool(call, tool, cancellation_token))

        # Execute all tool calls concurrently
        return await asyncio.gather(*tasks)


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
    autogen_models: dict[str, AutoGenWrapper] = Field(
        default_factory=dict,  # For caching instantiated clients
        description="Cache for instantiated AutoGenWrapper clients. Populated on demand.",
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

    def get_autogen_chat_client(self, name: str) -> AutoGenWrapper:
        """Gets or creates an `AutoGenWrapper` for the LLM configuration specified by `name`.

        If a client for the given name already exists in the `autogen_models` cache,
        it is returned. Otherwise, a new client is instantiated based on the
        `LLMConfig` found in `connections`, wrapped with `AutoGenWrapper`, cached,
        and then returned.

        Args:
            name: The connection name of the LLM configuration (must be a key
                in `self.connections`).

        Returns:
            AutoGenWrapper: The instantiated and wrapped Autogen chat completion client.

        Raises:
            AttributeError: If `name` is not found in `self.connections`.
            ImportError: If necessary client libraries (e.g., for Anthropic on Vertex)
                are not available.
            ValueError: If essential configuration like GCP credentials for Vertex
                are missing.

        """
        # Check cache first (though current implementation always creates new, which might be intended for some reason)
        # if name in self.autogen_models:
        #     return self.autogen_models[name]

        if name not in self.connections:
            raise AttributeError(f"LLM configuration named '{name}' not found in connections.")

        config = self.connections[name]
        client: ChatCompletionClient | None = None  # Initialize client as None

        client_params: dict[str, Any] = {
            "base_url": config.base_url,
            "model_info": config.model_info,  # Keep as ModelInfo object for type safety
            "api_key": config.api_key,
            **config.configs,  # Add other configs from LLMConfig.configs
        }

        family = config.model_info.family if hasattr(config.model_info, 'family') else ModelFamily.UNKNOWN
        # Ensure model family is correctly registered or handled for Autogen
        if _find_model_family("openai", family) == ModelFamily.UNKNOWN:  # Check against openai as it's a common base
            # This logic might need adjustment based on how Autogen handles various families.
            # For non-OpenAI models, Autogen might require specific client types or family settings.
            # Note: We should not modify the ModelInfo object - it's immutable
            logger.debug(f"Model family {family} not recognized by autogen, using as-is")

        api_type = config.api_type.lower()  # Normalize api_type

        if api_type == "azure":
            client_params["azure_endpoint"] = client_params.pop("base_url", None)  # Rename field for Azure client
            client = AzureOpenAIChatCompletionClient(**client_params)
        elif api_type in {"google", "google_genai"}:  # Assuming "google" might mean Gemini API
            # For GeminiChatCompletionClient, specific setup might be needed.
            # Current Autogen might use OpenAIChatCompletionClient as a wrapper for some non-OpenAI models if configured.
            # This part needs to align with how GeminiChatCompletionClient is expected to be used.
            # If GeminiChatCompletionClient is a direct wrapper:
            # client = GeminiChatCompletionClient(**client_params)
            # Or if it's accessed via OpenAIChatCompletionClient with specific base_url/api_type:
            client = OpenAIChatCompletionClient(**client_params)
        elif api_type in {"vertex", "google_vertexai"}:
            # Vertex AI often uses application default credentials or specific service account keys.
            # API key might be a token for Vertex.
            bm_instance = get_bm()  # Get Buttermilk global instance
            if not bm_instance.gcp_credentials:
                raise ValueError("GCP credentials not available in Buttermilk instance for Vertex AI.")
            client_params["api_key"] = bm_instance.gcp_credentials.token  # Use token for Vertex

            # Depending on the actual model (e.g., Gemini on Vertex, Claude on Vertex)
            # The client instantiation will differ.
            # For Gemini on Vertex via Autogen's OpenAI client compatibility:
            # client = OpenAIChatCompletionClient(**client_params)
            # For Anthropic Claude on Vertex:
            if "claude" in config.obj.lower():  # Simple check for Claude model name
                _vertex_params = {
                    "region": client_params.pop("region", None) or bm_instance.gcp_project_region,  # Get region
                    "project_id": client_params.pop("project_id", None) or bm_instance.gcp_project_id,  # Get project_id
                    "credentials": bm_instance.gcp_credentials,
                }
                # Remove None values to avoid passing them to AsyncAnthropicVertex
                _vertex_params = {k: v for k, v in _vertex_params.items() if v is not None}

                try:
                    _vertex_client = AsyncAnthropicVertex(**_vertex_params)
                    # Autogen's AnthropicChatCompletionClient needs to be initialized
                    # then its internal _client replaced.
                    anthropic_client_params = client_params.copy()
                    # Ensure 'model' is passed for Anthropic client
                    anthropic_client_params.setdefault("model", config.obj)
                    client = AnthropicChatCompletionClient(**anthropic_client_params)
                    client._client = _vertex_client  # type: ignore # Replace internal client
                except Exception as e:
                    logger.error(f"Error initializing Anthropic client for Vertex: {e!s}")
                    raise
            else:  # Default to OpenAI compatible client for other Vertex models (e.g., Gemini)
                 client = OpenAIChatCompletionClient(**client_params)

        elif api_type == "anthropic":  # Direct Anthropic API (not via Vertex)
            # This would use autogen_ext.models.anthropic.AnthropicChatCompletionClient directly
            # Ensure 'model' parameter is correctly passed from config.obj or client_params
            client_params.setdefault("model", config.obj)
            client = AnthropicChatCompletionClient(**client_params)
        else:  # Default to OpenAIChatCompletionClient for "openai" or unknown types
            client = OpenAIChatCompletionClient(**client_params)

        if client is None:  # Should not happen if logic is correct
            raise ProcessingError(f"Could not instantiate LLM client for '{name}' with api_type '{api_type}'.")

        # Wrap with AutoGenWrapper
        wrapped_client = AutoGenWrapper(client=client, model_info=client_params["model_info"])
        # self.autogen_models[name] = wrapped_client # Cache the client
        return wrapped_client  # Return a new instance as per original logic (or cached if uncommented)

    def __getattr__(self, __name: str) -> AutoGenWrapper:
        """Provides attribute-style access to LLM clients (e.g., `llms.my_model`)."""
        if __name not in self.connections:
            raise AttributeError(f"No LLM configuration found for '{__name}'. Available: {list(self.connections.keys())}")
        return self.get_autogen_chat_client(__name)

    def __getitem__(self, __name: str) -> AutoGenWrapper:
        """Provides item-style access to LLM clients (e.g., `llms["my_model"]`)."""
        return self.__getattr__(__name)
