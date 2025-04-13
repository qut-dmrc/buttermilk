import asyncio
from enum import Enum
import json
from typing import TYPE_CHECKING, Any, Optional, Sequence, TypeVar
from autogen_core import CancellationToken, FunctionCall
import weave
from anthropic import (
    AnthropicVertex,
    AsyncAnthropicVertex,
)

from autogen_core.tools import FunctionTool, Tool, ToolSchema
from autogen_core.models import ChatCompletionClient, ModelFamily, ModelInfo
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_ext.models.openai import (
    AzureOpenAIChatCompletionClient,
    OpenAIChatCompletionClient,
)
from autogen_ext.models.openai._transformation.registry import (
    _find_model_family,
)
from autogen_openaiext_client import GeminiChatCompletionClient
from pydantic import BaseModel, ConfigDict, Field
from buttermilk import logger
from buttermilk._core.agent import ToolOutput
from buttermilk._core.exceptions import ProcessingError
from .retry import RetryWrapper

from autogen_core.models import CreateResult, LLMMessage, RequestUsage

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
    "gemini2flash",
    "gemini2flashlite",
    "gemini2flashthinking",
    "gemini2pro",
    "gpt4o",
    "o3-mini-high",
    "llama31_8b",
    "llama32_90b",
    "llama33_70b",
    "haiku",
    "sonnet",
]
CHEAP_CHAT_MODELS = [
    "haiku",
    "gemini2flash",
    "o3-mini-high",
    "llama31_8b",
    "gemini2flashthinking",
    "gemini2flashlite",
]
MULTIMODAL_MODELS = ["gemini15pro", "gpt4o", "llama32_90b", "gemini2pro"]


class LLMClient(BaseModel):
    client: Any
    connection: str
    params: dict = {}


T_ChatClient = TypeVar("T_ChatClient", bound=ChatCompletionClient)


class AutoGenWrapper(RetryWrapper):
    """Wraps any ChatCompletionClient and adds rate limiting via a semaphore
    plus robust retry logic for handling API failures.
    """

    client: ChatCompletionClient
    model_info: ModelInfo

    @weave.op
    async def create(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema] = [],
        schema: Optional[type[BaseModel]] = None,
        cancellation_token: Optional[CancellationToken] = None,
        **kwargs: Any,
    ) -> CreateResult:
        """Rate-limited version of the underlying client's create method with retries"""
        try:
            # Use the retry logic
            create_result = await self._execute_with_retry(
                self.client.create,
                messages,
                tools=tools,
                json_output=schema,
                cancellation_token=cancellation_token,
                extra_create_args=kwargs,
            )
            if not create_result.content:
                raise ProcessingError("Empty response from LLM")
            if isinstance(create_result.content, str) and not create_result.content.strip():
                raise ProcessingError("Empty response from LLM")
            if isinstance(create_result.content, list) and not all(isinstance(item, FunctionCall) for item in create_result.content):
                raise ProcessingError("Unexpected tool response from LLM", create_result.content)
            return create_result
        except Exception as e:
            error_msg = f"Error during LLM call: {e}"
            logger.warning(error_msg, exc_info=False)
            raise ProcessingError(error_msg)

    @weave.op
    async def call_chat(self, messages, tools_list, cancellation_token, reflect_on_tool_use: bool = True) -> CreateResult | list[ToolOutput] | None:
        create_result = await self.create(messages=messages, tools=tools_list, cancellation_token=cancellation_token)

        if isinstance(create_result.content, str):
            return create_result

        tool_outputs = await self._execute_tools(
            calls=create_result.content,
            cancellation_token=cancellation_token,
        )
        if not reflect_on_tool_use:
            return tool_outputs

        reflection_messages = messages.copy()
        for tool_result in tool_outputs:
            reflection_messages.extend(tool_result.messages)
        reflection_result = await self.create(messages=reflection_messages, cancellation_token=cancellation_token)
        return reflection_result

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
            result.call_id = call.id
            result.name = tool.name
            outputs.append(result)
        return outputs

    @weave.op
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
        from buttermilk.bm import bm

        if name in self.autogen_models:
            return self.autogen_models[name]

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
            client = AzureOpenAIChatCompletionClient(**client_params)
        elif self.connections[name].api_type == "google":

            client = OpenAIChatCompletionClient(  # GeminiChatCompletionClient(
                **client_params,
            )
        elif self.connections[name].api_type == "vertex":
            client_params["api_key"] = bm._gcp_credentials.token
            #             client = GeminiChatCompletionClient(**params)
            client = OpenAIChatCompletionClient(
                **client_params,
            )

        elif self.connections[name].api_type == "anthropic":
            # token = credentials.refresh(google.auth.transport.requests.Request())
            _vertex_params = {
                k: v for k, v in client_params.items() if k in ["region", "project_id"]
            }
            _vertex_params["credentials"] = bm._gcp_credentials
            _vertex_client = AsyncAnthropicVertex(**_vertex_params)
            client = AnthropicChatCompletionClient(**client_params)
            client._client = _vertex_client  # type: ignore # replace client with vertexai version
        else:
            client = OpenAIChatCompletionClient(**client_params)

        # Store for next time so that we only maintain one client
        self.autogen_models[name] = AutoGenWrapper(client=client, model_info=client_params["model_info"])

        return self.autogen_models[name]

    def __getattr__(self, __name: str) -> AutoGenWrapper:
        if __name in self.autogen_models:
            return self.autogen_models[__name]
        if __name not in self.connections:
            raise AttributeError
        return self.get_autogen_chat_client(__name)

    def __getitem__(self, __name: str):
        return self.__getattr__(__name)
