from enum import Enum
from typing import TYPE_CHECKING

from anthropic import AnthropicVertex, AsyncAnthropicVertex
from google.cloud import aiplatform

from langchain_google_vertexai import ChatVertexAI, HarmBlockThreshold, HarmCategory
try:
    from langchain_anthropic import ChatAnthropic
except:
    pass
from langchain_core.language_models.llms import LLM
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    _ = [HarmBlockThreshold, HarmCategory]

# from ..libs.hf import Llama2ChatMod
# from ..libs.replicate import replicatellama3

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

class LLMConfig(BaseModel):
    model: str = Field(..., description="The full identifier of this particular model (passed to constructor/API)")
    connection:  str = Field(..., description="Descriptive identifier for the type of connection used (e.g. Azure, Vertex)")
    obj: str = Field(..., description="Name of the model object to instantiate")
    capabilities: LLMCapabilities = Field(default_factory=LLMCapabilities, description="Capabilities of the model (particularly multi-modal)")
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
    "llama31_405b",
    "llama32_11b",
    "llama32_90b",
    "gpt4o",
    "sonnet",
    "haiku",
    "gemini15pro",
]
CHEAP_CHAT_MODELS = ["haiku", 
    "llama32_11b",]
MULTIMODAL_MODELS = ["gemini15pro", "gpt4o", "sonnet", "llama32_90b"]


def VertexNoFilter(*args, **kwargs):
    return ChatVertexAI(
        *args,
        **kwargs,
        safety_settings=VERTEX_SAFETY_SETTINGS_NONE,
        _raise_on_blocked=False,
        response_mime_type="application/json",
    )


def LangChainAnthropicVertex(
    region: str, project_id: str, model_name: str, *args, **kwargs
):
    llm = ChatAnthropic(model_name=model_name, *args, **kwargs)
    llm._client = AnthropicVertex(region=region, project_id=project_id)
    llm._async_client = AsyncAnthropicVertex(region=region, project_id=project_id)
    return llm


def VertexMAAS(
    model_name: str, *,
    project: str,
    location: str,
    staging_bucket: str,
    **kwargs,
):
    _ = aiplatform.init(project=project,
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


class LLMs(BaseModel):
    connections: dict[str, LLMConfig] = Field(default=[], description="A dict of dicts each specifying connection information and parameters for an LLM.")
    
    models: dict = Field(default={}, description="Holds the instantiated model objects")

    class Config:
        use_enum_values = True

    @property
    def all_model_names(self) -> Enum:
        return Enum("AllModelNames", list(self.connections.keys()))

    def __getattr__(self, __name: str) -> LLM:
        if __name in self.models:
            return self.models[__name]

        model_config = self.connections[__name]
        self.models[__name] = globals()[model_config.obj](
            **model_config.configs,
        )
        return self.models[__name]


    def __getitem__(self, __name: str):
        return self.__getattr__(__name)