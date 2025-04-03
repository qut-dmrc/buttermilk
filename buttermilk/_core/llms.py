from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

from anthropic import (
    AnthropicVertex,
    AsyncAnthropicVertex,
)
from autogen_core.models import ChatCompletionClient
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_ext.models.openai import (
    AzureOpenAIChatCompletionClient,
    OpenAIChatCompletionClient,
)
from autogen_openaiext_client import GeminiChatCompletionClient
from pydantic import BaseModel, ConfigDict, Field

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

    model_info: dict = {}
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
    "gemini15pro",
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

    async def create(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Rate-limited version of the underlying client's create method with retries"""
        # Use the retry logic
        result = await self._execute_with_retry(
            self.client.create,
            *args,
            **kwargs,
        )

        return result

    async def agenerate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Rate-limited version of the underlying client's agenerate method with retries"""
        if hasattr(self.client, "agenerate"):
            # Use the retry logic with agenerate
            result = await self._execute_with_retry(
                self.client.agenerate,
                *args,
                **kwargs,
            )
        else:
            # Fallback to create with retry
            result = await self._execute_with_retry(
                self.client.create,
                *args,
                **kwargs,
            )

        return result


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

        params = self.connections[name].configs.copy()

        params["base_url"] = self.connections[name].base_url
        params["model_info"] = self.connections[name].model_info
        params["api_key"] = self.connections[name].api_key

        if self.connections[name].api_type == "azure":
            client = AzureOpenAIChatCompletionClient(**params)
        elif self.connections[name].api_type == "google":
            client = GeminiChatCompletionClient(**params)
        elif self.connections[name].api_type == "vertex":
            params["api_key"] = bm._gcp_credentials.token
            client = OpenAIChatCompletionClient(
                **params,
            )
        elif self.connections[name].api_type == "anthropic":
            # token = credentials.refresh(google.auth.transport.requests.Request())
            _vertex_params = {
                k: v for k, v in params.items() if k in ["region", "project_id"]
            }
            _vertex_params["credentials"] = bm._gcp_credentials
            _vertex_client = AsyncAnthropicVertex(**_vertex_params)
            client = AnthropicChatCompletionClient(**params)
            client._client = _vertex_client  # type: ignore # replace client with vertexai version
        else:
            client = OpenAIChatCompletionClient(**params)
        # from autogen_core.models import    UserMessage
        # test:  await client.create([UserMessage(content="hi", source="dev")])
        # Store for next time so that we only maintain one client
        self.autogen_models[name] = AutoGenWrapper(client=client)

        return self.autogen_models[name]

    def __getattr__(self, __name: str) -> AutoGenWrapper:
        if __name in self.autogen_models:
            return self.autogen_models[__name]
        if __name not in self.connections:
            raise AttributeError
        return self.get_autogen_chat_client(__name)

    def __getitem__(self, __name: str):
        return self.__getattr__(__name)
