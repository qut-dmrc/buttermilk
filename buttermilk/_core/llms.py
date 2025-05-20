import asyncio
import inspect
import json
from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

from anthropic import (
    AnthropicVertex,
    AsyncAnthropicVertex,
)
from autogen_core import CancellationToken, FunctionCall
from autogen_core.models import ChatCompletionClient, CreateResult, LLMMessage, ModelFamily, ModelInfo
from autogen_core.tools import Tool, ToolSchema
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_ext.models.openai import (
    AzureOpenAIChatCompletionClient,
    OpenAIChatCompletionClient,
)
from autogen_ext.models.openai._transformation.registry import (
    _find_model_family,
)
from autogen_openaiext_client import GeminiChatCompletionClient
from langfuse.openai import openai  # OpenAI integration # noqa
from pydantic import BaseModel, ConfigDict, Field

from .log import logger


# Use a function for deferred import to avoid circular references
def get_bm():
    """Get the BM singleton with delayed import to avoid circular references."""
    from buttermilk._core.dmrc import get_bm as _get_bm
    return _get_bm()


from buttermilk._core.contract import ToolOutput
from buttermilk._core.exceptions import ProcessingError

from .retry import RetryWrapper

_ = "ChatCompletionClient"


if TYPE_CHECKING:
    _: list[Any] = [
        AzureOpenAIChatCompletionClient,
        OpenAIChatCompletionClient,
        GeminiChatCompletionClient,
        AnthropicVertex,
    ]


class LLMConfig(BaseModel):
    connection: str = Field(
        ...,
        description="Descriptive identifier for the type of connection used (e.g. Azure, Vertex)",
    )
    obj: str = Field(..., description="Name of the model object to instantiate")
    api_type: str = Field(
        default="openai",
        description="Type of API to use (e.g. openai, vertex, azure)",
    )
    api_key: str | None = Field(
        default=None,
        description="API key to use for this model",
    )
    base_url: str | None = Field(default=None, description="Custom URL to call")

    model_info: ModelInfo
    configs: dict = Field(default={}, description="Options to pass to the constructor")


class MLPlatformTypes(Enum):
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
    "haiku",
    "sonnet",
]
CHEAP_CHAT_MODELS = [
    "haiku",
    "gemini25flash",
    "o3mini",
    "gpt41mini",
    "llama31_8b",
    "haiku",
]
MULTIMODAL_MODELS = ["gemini25pro", "llama4maverick", "gemini25flash", "gpt41", "llama32_90b"]


class LLMClient(BaseModel):
    client: Any
    connection: str
    parameters: dict = {}


T_ChatClient = TypeVar("T_ChatClient", bound=ChatCompletionClient)


class ModelOutput(CreateResult):
    object: BaseModel | None = Field(default=None, description="The hydrated object from JSON or structured output")


class AutoGenWrapper(RetryWrapper):
    """Wraps any ChatCompletionClient and adds rate limiting via a semaphore
    plus robust retry logic for handling API failures.
    """

    client: ChatCompletionClient
    model_info: ModelInfo

    async def create(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema] = [],
        schema: type[BaseModel] | None = None,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> CreateResult | list[ToolOutput]:
        """Rate-limited version of the underlying client's create method with retries"""
        try:
            is_valid_schema_type = (
                schema is not None
                and inspect.isclass(schema)
                and issubclass(schema, BaseModel)
                and schema is not BaseModel
            )
            if is_valid_schema_type and self.model_info.get("structured_output", False):
                json_output = schema

            else:
                # By preference, pass a pydantic schema for structured output
                # Otherwise, set json_output to True if the model supports it

                # TODO: check if the word 'json' is in the system message or add a quick direction.
                json_output = self.model_info.get("json_output", False)

            # Don't use json_output if we're calling tools
            if tools:
                json_output = False

            # # --- Langfuse tracing ---
            # langfuse_context.update_current_observation(input=messages)

            # Use the retry logic
            create_result = await self._execute_with_retry(
                self.client.create,
                messages,
                tools=tools,
                json_output=json_output,
                cancellation_token=cancellation_token,
                extra_create_args=kwargs,
            )
            if not create_result.content:
                logger.error(error_msg := "Empty response from LLM")
                raise ProcessingError(error_msg)
            if isinstance(create_result.content, str) and not create_result.content.strip():
                logger.error(error_msg := "Empty response from LLM")
                raise ProcessingError(error_msg)
            if isinstance(create_result.content, list) and not all(isinstance(item, FunctionCall) for item in create_result.content):
                logger.error(error_msg := "Unexpected tool response from LLM")
                raise ProcessingError(error_msg, create_result.content)
        except Exception as e:
            error_msg = f"Error during LLM call: {e}"
            logger.error(error_msg)
            raise ProcessingError(error_msg)
        return create_result

    async def call_chat(
        self,
        messages,
        cancellation_token,
        tools_list=[],
        schema: type[BaseModel] | None = None,
    ) -> CreateResult:
        """Pass messages to the Chat LLM, run tools if required, and reflect."""
        create_result = await self.create(messages=messages, tools=tools_list, cancellation_token=cancellation_token, schema=schema)

        if isinstance(create_result.content, list):
            # Tool choices

            tool_outputs = await self._execute_tools(
                calls=create_result.content,
                tools_list=tools_list,
                cancellation_token=cancellation_token,
            )
            for tool_result in tool_outputs:
                messages.extend(tool_result.messages)
            create_result = await self.create(messages=messages, cancellation_token=cancellation_token)

        return create_result

    async def _call_tool(
        self,
        call: FunctionCall,
        tool,
        cancellation_token: CancellationToken | None,
    ) -> list[ToolOutput]:
        # Run the tool and capture the result.
        arguments = json.loads(call.arguments)
        results = await tool.run_json(arguments, cancellation_token)

        if not isinstance(results, list):
            results = [results]
        outputs = []
        for result in results:
            try:
                result.name = tool.name
                result.call_id = call.id
            except:
                pass
            outputs.append(result)
        return outputs

    async def _execute_tools(
        self,
        calls: list[FunctionCall],
        tools_list: list,
        cancellation_token: CancellationToken | None,
    ) -> list[ToolOutput]:
        """Execute the tools and return the results."""
        tasks = []
        for call in calls:
            # Find the tool by name.
            tool = next((tool for tool in tools_list if tool.name == call.name), None)
            assert tool is not None
            tasks.append(self._call_tool(call, tool, cancellation_token))

        # Execute the tool calls.
        results = await asyncio.gather(*tasks)

        # flatten results
        results = [record for result in results for record in result if record is not None]

        return results


class LLMs(BaseModel):
    connections: dict[str, LLMConfig] = Field(
        default=[],
        description="A dict of dicts each specifying connection information and parameters for an LLM.",
    )

    autogen_models: dict[str, AutoGenWrapper] = Field(
        default={},
        description="Holds the instantiated model objects",
    )

    model_config = ConfigDict(use_enum_values=True)

    @property
    def all_model_names(self) -> Enum:
        return Enum("AllModelNames", list(self.connections.keys()))

    def get_autogen_chat_client(self, name) -> AutoGenWrapper:
        from buttermilk._core.log import logger  # noqa

        client: ChatCompletionClient = None

        client_params = {}
        client_params["base_url"] = self.connections[name].base_url
        client_params["model_info"] = self.connections[name].model_info
        client_params["api_key"] = self.connections[name].api_key
        family = client_params["model_info"].get("family", ModelFamily.UNKNOWN)
        if _find_model_family("openai", family) == ModelFamily.UNKNOWN:
            # In future we may need to register a pipeline for models not explicitly
            # supported by Autogen. Ideally it's fixed upstream, but user beware.
            client_params["model_info"]["family"] = ModelFamily.UNKNOWN

        # add in model parameters from the config dict
        client_params.update(**self.connections[name].configs)
        if self.connections[name].api_type == "azure":
            client_params["azure_endpoint"] = client_params.pop("base_url")  # rename field
            client = AzureOpenAIChatCompletionClient(**client_params)
        elif self.connections[name].api_type == "google":
            client = OpenAIChatCompletionClient(  # GeminiChatCompletionClient(
                **client_params,
            )
        elif self.connections[name].api_type == "vertex":
            client_params["api_key"] = get_bm().gcp_credentials.token
            #             client = GeminiChatCompletionClient(**parameters)
            client = OpenAIChatCompletionClient(
                **client_params,
            )

        elif self.connections[name].api_type == "anthropic":
            # token = credentials.refresh(google.auth.transport.requests.Request())
            _vertex_params = {k: v for k, v in client_params.items() if k in ["region", "project_id"]}
            _vertex_params["credentials"] = get_bm().gcp_credentials
            _vertex_client = AsyncAnthropicVertex(**_vertex_params)
            client = AnthropicChatCompletionClient(**client_params)
            client._client = _vertex_client  # type: ignore # replace client with vertexai version
        else:
            client = OpenAIChatCompletionClient(**client_params)

        return AutoGenWrapper(client=client, model_info=client_params["model_info"])  # RETURN A NEW INSTANCE

    def __getattr__(self, __name: str) -> AutoGenWrapper:
        if __name not in self.connections:
            raise AttributeError
        return self.get_autogen_chat_client(__name)

    def __getitem__(self, __name: str):
        return self.__getattr__(__name)
