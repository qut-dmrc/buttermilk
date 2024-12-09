from enum import Enum
from typing import TYPE_CHECKING

from anthropic import AnthropicVertex, AsyncAnthropicVertex

from langchain_google_vertexai import ChatVertexAI, HarmBlockThreshold, HarmCategory
try:
    from langchain_anthropic import ChatAnthropic
except:
    pass
from langchain_core.language_models.llms import LLM
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import BaseModel

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
    project: str,
    location: str,
    staging_bucket: str,
    model_name: str,
    *args,
    **kwargs,
):
    _ = vertexai.init(
        project=project,
        location=location,
        staging_bucket=staging_bucket,
    )
    model = ChatVertexAI(
        model_name=model_name,
        api_endpoint="us-east5-aiplatform.googleapis.com",
        location="us-east5",
        *args,
        **kwargs,
    )
    return model


def _Llama(*args, **kwargs):
    default_opts = {"content_formatter": CustomOpenAIChatContentFormatter()}
    default_opts.update(**kwargs)
    return AzureChatOpenAI(*args, **default_opts)


class LLMs(BaseModel):
    connections: dict
    models: dict = {}

    class Config:
        use_enum_values = True

    @property
    def all_model_names(self) -> Enum:
        return Enum("AllModelNames", list(self.connections.keys()))

    def __getattr__(self, __name: str) -> LLM:
        if __name in self.models:
            return self.models[__name]

        model_config = self.connections[__name]
        model_name = model_config["name"]
        self.models[model_name] = globals()[model_config["obj"]](
            **model_config["configs"],
        )
        return self.models[__name]

    def __getitem__(self, __name: str) -> LLM:
        return self.__getattr__(__name)


if __name__ == "__main__":
    from buttermilk import BM

    bm = BM()
    llm = LLMs()["haiku"]
    import pprint

    pprint.pprint(llm.invoke("hi what model are you?"))
