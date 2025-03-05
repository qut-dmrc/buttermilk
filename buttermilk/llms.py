from enum import Enum
from typing import TYPE_CHECKING, Any

from anthropic import AnthropicVertex, AsyncAnthropicVertex
from autogen import OpenAIWrapper
from autogen_core.models import ChatCompletionClient
from autogen_ext.models.openai import (
    AzureOpenAIChatCompletionClient,
    OpenAIChatCompletionClient,
)
from autogen_openaiext_client import GeminiChatCompletionClient
from google.cloud import aiplatform
from langchain_google_vertexai import ChatVertexAI, HarmBlockThreshold, HarmCategory

from buttermilk.utils.utils import scrub_keys

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
    model: str = Field(
        ...,
        description="The full identifier of this particular model (passed to constructor/API)",
    )
    connection: str = Field(
        ...,
        description="Descriptive identifier for the type of connection used (e.g. Azure, Vertex)",
    )
    obj: str = Field(..., description="Name of the model object to instantiate")
    api_type: str = Field(
        default="openai",
        description="Type of API to use (e.g. openai, vertex, azure)",
    )
    base_url: str | None = Field(default=None, description="Custom URL to call")

    model_info: dict = {}
    capabilities: LLMCapabilities = Field(
        default_factory=LLMCapabilities,
        description="Capabilities of the model (particularly multi-modal)",
    )
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
# cat .cache/buttermilk/models.json | jq ".[].name"
# ```
CHATMODELS = [
    "gemini15pro",
    "gemini15pro_safety",
    "gemini2flash",
    "gemini2flashlite",
    "gemini2flashthinking",
    "gemini2pro",
    "gpt4o",
    "haiku",
    "llama31_405b",
    "llama31_405b-azure",
    "llama31_8b",
    "llama31_8b_guard",
    "llama32_90b",
    "llama32_90b_azure",
    "llama32_90b_guard",
    "llama33_70b",
    "o3-mini-high",
    "sonnet",
    "sonnetanthropic",
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
    capabilities: LLMCapabilities
    connection: str
    params: dict = {}


class LLMs(BaseModel):
    connections: dict[str, LLMConfig] = Field(
        default=[],
        description="A dict of dicts each specifying connection information and parameters for an LLM.",
    )

    models: dict[str, LLMClient] = Field(
        default={},
        description="Holds the instantiated model objects",
    )

    class Config:
        use_enum_values = True

    @property
    def all_model_names(self) -> Enum:
        return Enum("AllModelNames", list(self.connections.keys()))

    def get_autogen_generic(self, name):
        params = self.connections[name].configs.copy()
        params["api_type"] = self.connections[name].api_type
        params["base_url"] = self.connections[name].base_url
        params["model_info"] = self.connections[name].model_info

        wrapper = OpenAIWrapper(config_list=[params])

        return wrapper

    def get_autogen_chat_client(self, name) -> ChatCompletionClient:
        # Let Autogen handle the correct arguments for different models
        wrapper = self.get_autogen_generic(name)

        # Add in any config keys autogen has removed
        # params = {k:v for k, v in self.connections[name].configs.items() if k not in params}
        params = self.connections[name].configs.copy()

        params["base_url"] = self.connections[name].base_url
        params["model_info"] = self.connections[name].model_info

        # params.update(**wrapper._config_list[0])

        if self.connections[name].api_type == "azure":
            client = AzureOpenAIChatCompletionClient(**params)
        elif self.connections[name].api_type == "google":
            client = GeminiChatCompletionClient(**params)
        else:
            client = OpenAIChatCompletionClient(**params)

        return client

    # def get_autogen_chat_client(self, name):
    #     params = self.connections[name].configs
    #     params["model_info"] = self.connections[name].model_info

    #     if self.connections[name].api_type == "azure":
    #         client = AzureOpenAIChatCompletionClient(**params)
    #     else:
    #         client = OpenAIChatCompletionClient(**params)
    #     return client

    def __getattr__(self, __name: str) -> LLMClient:
        if __name in self.models:
            return self.models[__name]
        if __name not in self.connections:
            raise AttributeError

        model_config = self.connections[__name]

        # Merge global with model specific configs
        params = dict(**global_langchain_configs)
        params.update(**model_config.configs)

        # Instantiate the model client object
        _llm = LLMClient(
            client=globals()[model_config.obj](
                **params,
            ),
            capabilities=model_config.capabilities,
            connection=scrub_keys(model_config.connection),
            params=scrub_keys(params),
        )
        self.models[__name] = _llm
        return self.models[__name]

    def __getitem__(self, __name: str):
        return self.__getattr__(__name)
