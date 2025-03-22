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
from google.cloud import aiplatform
from langchain_google_vertexai import ChatVertexAI, HarmBlockThreshold, HarmCategory

from buttermilk._core.retry import RetryWrapper

_ = "ChatCompletionClient"

try:
    from langchain_anthropic import ChatAnthropic
except:
    pass
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    _ = [HarmBlockThreshold, HarmCategory]

# from ..libs.hf import Llama2ChatMod
# from ..libs.replicate import replicatellama3

# Set parameters that apply to all models here
global_langchain_configs = {
    # We want to handle retries ourselves, not through langchain
    "max_retries": 0,
}

MODEL_CLASSES = [
    ChatOpenAI,
    AzureChatOpenAI,
    AnthropicVertex,
    ChatVertexAI,
]


class LLMCapabilities(BaseModel):
    chat: bool = True
    image: bool = False
    video: bool = False
    audio: bool = False
    media_uri: bool = False
    expects_text_with_media: bool = False


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


VERTEX_SAFETY_SETTINGS_NONE = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}
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
    "llama31_405b-azure",
    "llama31_8b",
    "llama32_90b_azure",
    "llama33_70b",
    "haiku",
    "sonnet",
]
CHEAP_CHAT_MODELS = [
    "haiku",
    "gemini2flash",
    "o3-mini-high",
    "gemini2flashthinking",
    "gemini2flashlite",
]
MULTIMODAL_MODELS = ["gemini15pro", "gpt4o", "llama32_90b", "gemini2pro"]


def VertexNoFilter(*args, **kwargs):
    return ChatVertexAI(
        *args,
        **kwargs,
        safety_settings=VERTEX_SAFETY_SETTINGS_NONE,
        _raise_on_blocked=False,
    )


def LangChainAnthropicVertex(
    region: str,
    project_id: str,
    model_name: str,
    *args,
    **kwargs,
):
    llm = ChatAnthropic(model_name=model_name, *args, **kwargs)
    llm._client = AnthropicVertex(region=region, project_id=project_id)
    llm._async_client = AsyncAnthropicVertex(region=region, project_id=project_id)
    return llm


def VertexMAAS(
    model_name: str,
    *,
    project: str,
    location: str,
    staging_bucket: str,
    **kwargs,
):
    _ = aiplatform.init(
        project=project,
        location=location,
        staging_bucket=staging_bucket,
    )
    model = ChatVertexAI(
        model_name=model_name,
        project=project,
        location=location,
        **kwargs,
    )
    return model
    MODEL_LOCATION = "us-central1"
    MAAS_ENDPOINT = f"{MODEL_LOCATION}-aiplatform.googleapis.com"

    client = openai.OpenAI(
        base_url=f"https://{MAAS_ENDPOINT}/v1beta1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/openapi",
        api_key=credentials.token,
    )


def _Llama(*args, **kwargs):
    default_opts = {"content_formatter": CustomOpenAIChatContentFormatter()}
    default_opts.update(**kwargs)
    return AzureChatOpenAI(*args, **default_opts)


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

    langchain_models: dict[str, LLMClient] = Field(
        default={},
        description="Holds the instantiated model objects",
    )
    autogen_models: dict[str, AutoGenWrapper] = Field(
        default={},
        description="Holds the instantiated model objects",
    )

    class Config:
        use_enum_values = True

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
            client = OpenAIChatCompletionClient(**params)
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
