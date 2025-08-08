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
from typing import Any, TypeVar

# Core LLM library imports - these are required dependencies
import weave
from anthropic import (
    AsyncAnthropicVertex,
)

# Autogen library imports - these are required dependencies
from autogen_core import CancellationToken, FunctionCall  # Autogen core types
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    CreateResult,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    ModelFamily,
    ModelInfo,
)
from autogen_core.tools import Tool  # Autogen tool handling
from autogen_ext.models.anthropic import AnthropicChatCompletionClient  # Autogen Anthropic client
from autogen_ext.models.openai import (  # Autogen OpenAI clients
    AzureOpenAIChatCompletionClient,
    OpenAIChatCompletionClient,
)
from google import genai  # Google Generative AI library
from pydantic import BaseModel, ConfigDict, Field, field_validator  # Pydantic models for configuration


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

    """

    client_type: ClientType = Field(
        description="Type of client to instantiate (determines which client class to use)",
    )
    api_key: str | None = Field(
        default=None,
        description="API key to use for this model",
    )
    base_url: str | None = Field(default=None, description="Custom URL to call")

    model_info: ModelInfo = Field(..., description="Model metadata (family, context size, etc.)")
    configs: dict = Field(default_factory=dict, description="Options to pass to the constructor")

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


# Generate with:
# ```sh
# cat .cache/buttermilk/models.json | jq "keys[]"
# ```
"""A predefined list of chat model identifiers available within the Buttermilk setup."""
CHAT_MODELS = [
    "gemini25flash",
    "gemini25pro",
    "gpt5mini",
    "gpt5nano",
    "gpt5chat",
    "llama4maverick",
    "opus",
    "sonnet",
]

"""A predefined list of identifiers for cost-effective chat models."""
CHEAP_CHAT_MODELS = [
    "gemini25flash",
    "o4mini",
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

    client: ChatCompletionClient = Field(..., description="The underlying Autogen client instance.")
    model_info: ModelInfo = Field(..., description="Model metadata (family, context size, etc.)")

    @weave.op
    async def create(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool] = [],
        schema: type[BaseModel] | None = None,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> CreateResult | ModelOutput:
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
            CreateResult | ModelOutput: The result from the LLM. If a schema was provided,
                returns ModelOutput with parsed_object field containing the Pydantic instance.

        Raises:
            ProcessingError: If the LLM returns an empty response or an unexpected
                tool response, or if any other error occurs during the LLM call
                after retries are exhausted. Also raised if schema parsing fails.

        """
        is_valid_schema_type = (
            schema is not None
            and inspect.isclass(schema)
            and issubclass(schema, BaseModel)
            and schema is not BaseModel  # Ensure it's a specific subclass, not BaseModel itself
        )

        json_output_requested: bool | type[BaseModel] = False  # Default to no JSON mode
        fake_schema_tool = None  # Will hold our fake tool if needed

        if is_valid_schema_type and self.model_info.get("structured_output"):
            json_output_requested = schema  # type: ignore # Pass the schema for structured output
        elif is_valid_schema_type and not self.model_info.get("structured_output") and not tools and self.model_info.get("function_calling", True):
            # Create a fake tool for models that support function calling but not structured output
            # This allows us to get structured output via tool calling
            from autogen_core.tools import BaseTool

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

                async def run(self, args: BaseModel, cancellation_token: CancellationToken) -> BaseModel:
                    # args is already validated as our model type by BaseTool
                    return args

            # Create the fake tool
            fake_schema_tool = PydanticModelTool(schema)

            # Add the fake tool to the tools list
            tools = [fake_schema_tool]

        # Some models don't support simultaneous tool calling and structured output
        # Check model family to determine capabilities
        model_family = self.model_info.get("family", ModelFamily.UNKNOWN)

        # Gemini models can't use tools with structured output (json_output with Pydantic model)
        # This is a known limitation documented in Gemini's API
        gemini_families = {
            ModelFamily.GEMINI_1_5_FLASH,
            ModelFamily.GEMINI_1_5_PRO,
            ModelFamily.GEMINI_2_0_FLASH,
            ModelFamily.GEMINI_2_5_PRO,
            ModelFamily.GEMINI_2_5_FLASH,
            "gemini",
        }

        # Some other models also have this limitation (discovered through testing)
        models_with_tool_schema_conflict = gemini_families | {"llama-4-maverick"}

        if tools and model_family in models_with_tool_schema_conflict and json_output_requested and isinstance(json_output_requested, type):
            # For models that can't handle tools + structured output together, we don't ask for a structured output
            json_output_requested = False

        try:
            create_result = await self._execute_with_retry(
                self.client.create,  # The method to call
                messages,  # Positional arguments for self.client.create
                tools=tools,
                json_output=json_output_requested,
                cancellation_token=cancellation_token,
                extra_create_args=kwargs,  # Keyword arguments for self.client.create
            )

        except Exception as e:  # Wrap other exceptions
            error_msg = f"Error during LLM call: {e!s}"
            raise ProcessingError(error_msg) from e

        if not create_result.content:
            raise ProcessingError("Empty response content from LLM.")
        if isinstance(create_result.content, str) and not create_result.content.strip():
            raise ProcessingError("Empty string response from LLM.")
        # Check if content is a list and if all items are FunctionCall (valid tool call scenario)
        if isinstance(create_result.content, list) and not all(isinstance(item, FunctionCall) for item in create_result.content):
            raise ProcessingError("Unexpected response type from LLM when expecting tool calls or text.", create_result.content)

        # Handle structured output parsing if schema was provided
        if schema and is_valid_schema_type and not (tools and not fake_schema_tool):
            # Only parse if: we have a schema AND (we created a fake tool OR no tools were provided)
            # Check if we used a fake tool and got a tool call response
            if fake_schema_tool and isinstance(create_result.content, list) and all(isinstance(c, FunctionCall) for c in create_result.content):
                # Extract the tool call and execute it to get the structured object
                tool_calls = create_result.content
                if len(tool_calls) == 1 and tool_calls[0].name == fake_schema_tool.name:
                    # Execute the fake tool to get the structured object
                    arguments = json.loads(tool_calls[0].arguments)

                    # Call the schema constructor directly with the arguments
                    try:
                        parsed_object = schema(**arguments)
                    except Exception as e:
                        raise ProcessingError(f"Failed to create {schema.__name__} from tool arguments: {e}")

                    # Return ModelOutput with the parsed object
                    return ModelOutput(
                        content=json.dumps(parsed_object.model_dump()),
                        finish_reason=create_result.finish_reason,
                        usage=create_result.usage,
                        thought=getattr(create_result, "thought", None),
                        parsed_object=parsed_object,
                        cached=create_result.cached,
                    )
            else:
                # Regular structured output parsing
                return await self._parse_structured_output(create_result, schema)

        return create_result  # type: ignore # Expect CreateResult or compatible

    @weave.op
    async def call_chat(
        self,
        messages: list[LLMMessage],  # Made mutable for extending with tool results
        cancellation_token: CancellationToken | None,
        tools_list: Sequence[Tool] = [],
        schema: type[BaseModel] | None = None,
        intercept_tools: bool = False,
    ) -> CreateResult | ModelOutput:
        """Manages a chat interaction, including potential tool calls and responses.

        This method sends an initial set of messages to the LLM. If the LLM
        responds with tool call requests, this method executes those tools
        (unless intercept_tools is True), appends their results back to the 
        message history, and sends the updated history back to the LLM to get 
        a final response.

        Args:
            messages: A list of `LLMMessage` objects forming the conversation.
                This list will be mutated if tool calls occur.
            cancellation_token: A `CancellationToken` for the operation.
            tools_list: An optional sequence of `Tool` objects
                available for the LLM to call.
            schema: An optional Pydantic `BaseModel` subclass for structured output
                requests (if no tools are called).
            intercept_tools: If True, return FunctionCall objects without executing them.
                This is useful for agents that need to handle tool calls specially.

        Returns:
            CreateResult | ModelOutput: The final result from the LLM after
                any tool call cycles. Returns ModelOutput if schema was provided.

        """
        # Pass tools as-is - the create() method will handle conflicts between tools and schema
        effective_tools = tools_list
        try:
            create_result = await self.create(
                messages=messages,
                tools=effective_tools,
                cancellation_token=cancellation_token,
                schema=schema,
            )
        except Exception as e:
            # The first call failed -- before we have executed any tools
            raise ProcessingError(f"Failed to query LLM: {e!s}") from e

        # If the LLM responded with a request to call tools
        if isinstance(create_result.content, list) and all(isinstance(c, FunctionCall) for c in create_result.content):
            tool_calls: list[FunctionCall] = create_result.content

            # If intercepting tools, return the result with tool calls without executing
            if intercept_tools:
                logger.debug(f"Intercepting {len(tool_calls)} tool calls without execution")
                return create_result

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
                raise ProcessingError(f"Failed to synthesise tool results: {e!s}") from e

        return create_result  # type: ignore # Expect CreateResult or ModelOutput

    @weave.op
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
            content=tool.return_value_as_string(result),
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

    async def _parse_structured_output(
        self,
        create_result: CreateResult,
        schema: type[BaseModel],
    ) -> ModelOutput:
        """Parse LLM response into structured output using the provided schema.

        Args:
            create_result: The raw result from the LLM
            schema: The Pydantic model to parse into

        Returns:
            ModelOutput with parsed_object field containing the Pydantic instance

        Raises:
            ProcessingError: If parsing fails

        """
        parsed_object = None

        # Check if the result already contains a parsed object of the correct type
        if isinstance(create_result, ModelOutput) and isinstance(create_result.parsed_object, schema):
            parsed_object = create_result.parsed_object
        elif isinstance(create_result.content, str):
            # Try to parse the string response
            logger.debug(f"AutoGenWrapper: Attempting to parse string response into {schema.__name__}")
            try:
                # Import locally to avoid circular dependencies
                from buttermilk.utils.json_parser import ChatParser

                parser = ChatParser()
                parsed_dict = parser.parse(create_result.content)
                parsed_object = schema(**parsed_dict) if isinstance(parsed_dict, dict) else None
                if parsed_object:
                    logger.debug(f"AutoGenWrapper: Successfully parsed response into {schema.__name__}")
            except Exception as parse_error:
                raise ProcessingError(
                    f"AutoGenWrapper failed to parse LLM response into required schema {schema.__name__}: {parse_error}",
                ) from parse_error
        elif hasattr(create_result.content, "model_dump"):
            # Already a Pydantic object, but might be wrong type
            if isinstance(create_result.content, schema):
                parsed_object = create_result.content
            else:
                logger.warning(
                    f"AutoGenWrapper: Response is {type(create_result.content).__name__}, "
                    f"expected {schema.__name__}",
                )

        # Create ModelOutput with the parsed object
        if parsed_object is None:
            raise ProcessingError(
                f"AutoGenWrapper requires structured output of type {schema.__name__} but parsing failed",
            )

        return ModelOutput(
            content=create_result.content,
            finish_reason=create_result.finish_reason,
            usage=create_result.usage,
            thought=getattr(create_result, "thought", None),
            parsed_object=parsed_object,
            cached=create_result.cached,
        )


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
        # Check cache first
        if name in self.autogen_models:
            return self.autogen_models[name]

        if name not in self.connections:
            raise AttributeError(f"LLM configuration named '{name}' not found in connections.")

        config = self.connections[name]

        # Optionally annotate resolved litellm identifier for downstream cost/accounting
        # Import here to avoid circular import issues
        from buttermilk.utils.model_registry import resolve_litellm_model_name
        resolved_litellm = resolve_litellm_model_name(name)
        # Expose for inspection (non-destructive; do not overwrite 'model')
        config.configs.setdefault("_resolved_litellm_model", resolved_litellm)

        # Prepare client parameters from configs
        client_params: dict[str, Any] = {
            "model": config.configs.get("model"),
            "api_key": config.api_key,
            **config.configs,
        }

        # Create client based on config.client_type - clean single branch per type
        if config.client_type == ClientType.OPENAI:
            client = OpenAIChatCompletionClient(
                base_url=config.base_url or "",  # Provide default empty string if None
                model_info=config.model_info,
                **client_params,
            )

        elif config.client_type == ClientType.AZURE:
            if not config.base_url:
                raise ValueError("Azure endpoint URL is required for Azure client")
            client = AzureOpenAIChatCompletionClient(
                azure_endpoint=config.base_url,
                model_info=config.model_info,
                **client_params,
            )

        elif config.client_type == ClientType.ANTHROPIC:
            # Direct Anthropic API
            client = AnthropicChatCompletionClient(**client_params)

        elif config.client_type == ClientType.ANTHROPIC_VERTEX:
            # Anthropic via Vertex AI
            bm_instance = get_bm()
            if not bm_instance.gcp_credentials:
                raise ValueError("GCP credentials not available for Anthropic via Vertex AI.")

            vertex_params = {
                "region": config.configs.get("region"),
                "project_id": config.configs.get("project_id"),
                "credentials": bm_instance.gcp_credentials,
            }
            vertex_params = {k: v for k, v in vertex_params.items() if v is not None}

            try:
                vertex_client = AsyncAnthropicVertex(**vertex_params)
                # Remove api_key for Vertex auth
                vertex_client_params = client_params.copy()
                vertex_client_params.pop("api_key", None)

                client = AnthropicChatCompletionClient(**vertex_client_params)
                client._client = vertex_client
            except Exception as e:
                logger.error(f"Error initializing Anthropic client for Vertex: {e!s}")
                raise

        elif config.client_type == ClientType.GEMINI:
            # Google Generative AI (Gemini) API
            bm_instance = get_bm()
            if not bm_instance.gcp_credentials:
                raise ValueError("GCP credentials not available for Gemini API.")
            client = OpenAIChatCompletionClient(
                model_info=config.model_info,
                **client_params,
            )
        elif config.client_type == ClientType.GEMINI_VERTEX:
            raise NotImplementedError(
                "Gemini native client for Vertex is not yet implemented. "
                "Please use the Gemini API or OpenAIChatCompletionClient with Vertex parameters.",
            )

            bm_instance = get_bm()
            if not bm_instance.gcp_credentials:
                raise ValueError("GCP credentials not available for Vertex AI.")
            vertex_params = {
                "region": config.configs.get("region"),
                "project_id": config.configs.get("project_id"),
                "credentials": bm_instance.gcp_credentials,
            }
            vertex_params = {k: v for k, v in vertex_params.items() if v is not None}
            gemini_client = genai.Client(  # not used yet, not compatible with autogen
                vertexai=True, **vertex_params,
            )

        elif config.client_type == ClientType.VERTEX_OPENAI:
            # OpenAI-compatible endpoint on Vertex (for Llama, etc.)
            bm_instance = get_bm()
            if not bm_instance.gcp_credentials:
                raise ValueError("GCP credentials not available for Vertex AI.")

            vertex_params = client_params.copy()

            # Set up OAuth2 bearer token authentication
            headers = {
                "Authorization": f"Bearer {bm_instance.get_gcp_access_token()}",
            }

            # Dummy API key for OpenAI client validation
            if vertex_params.get("api_key") is None:
                vertex_params["api_key"] = "dummy-key-for-vertex"

            vertex_params["default_headers"] = headers

            if not config.base_url:
                raise ValueError("Base URL is required for Vertex OpenAI endpoint")
            client = OpenAIChatCompletionClient(
                base_url=config.base_url,
                model_info=config.model_info,
                **vertex_params,
            )
        else:
            raise ProcessingError(f"Unsupported client_type: {config.client_type}")

        # Wrap with AutoGenWrapper and cache
        wrapped_client = AutoGenWrapper(client=client, model_info=config.model_info)
        self.autogen_models[name] = wrapped_client
        return wrapped_client

    def __getattr__(self, __name: str) -> AutoGenWrapper:
        """Provides attribute-style access to LLM clients (e.g., `llms.my_model`)."""
        if __name not in self.connections:
            raise AttributeError(f"No LLM configuration found for '{__name}'. Available: {list(self.connections.keys())}")
        return self.get_autogen_chat_client(__name)

    def __getitem__(self, __name: str) -> AutoGenWrapper:
        """Provides item-style access to LLM clients (e.g., `llms["my_model"]`)."""
        return self.__getattr__(__name)
