from enum import Enum
from typing import TYPE_CHECKING

import vertexai
from anthropic import AnthropicVertex, AsyncAnthropicVertex

# There are 'old' and 'new' harm categories. Use the new ones.
# see google/generativeai/types/safety_types.py
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.llms import LLM
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import BaseModel
from vertexai.generative_models import (
    HarmBlockThreshold as VertexHarmBlockThreshold,
    HarmCategory as VertexHarmCategory,
)

if TYPE_CHECKING:
    _ = [HarmBlockThreshold, HarmCategory]

# from ..libs.hf import Llama2ChatMod
# from ..libs.replicate import replicatellama3

MODEL_CLASSES = [
    ChatAnthropic,
    ChatOpenAI,
    AzureChatOpenAI,
    AnthropicVertex,
    ChatVertexAI,
]


class LLMTypes(Enum):
    openai = "openai"
    google_genai = "google-genai"
    google_vertexai = "google-vertexai"
    anthropic = "anthropic"
    llama = "llama"


VERTEX_SAFETY_SETTINGS = {
    VertexHarmCategory.HARM_CATEGORY_UNSPECIFIED: VertexHarmBlockThreshold.BLOCK_NONE,
    VertexHarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: VertexHarmBlockThreshold.BLOCK_NONE,
    VertexHarmCategory.HARM_CATEGORY_HATE_SPEECH: VertexHarmBlockThreshold.BLOCK_NONE,
    VertexHarmCategory.HARM_CATEGORY_HARASSMENT: VertexHarmBlockThreshold.BLOCK_NONE,
    VertexHarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: VertexHarmBlockThreshold.BLOCK_NONE,
}

GEMINI_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

GEMINI_SAFETY_SETTINGS_NONE = {
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
    "llama31_405b-azure",
    "llama31_405b",
    "llama31_8b",
    "llama32_11b",
    "llama32_90b",
    "llama32_90b_vision_instruct_azure",
    "llama31_70b",
    "o1-preview",
    "gpt4o",
    "sonnet",
    "sonnetanthropic",
    "haiku",
    "gemini15pro",
]
CHEAP_CHAT_MODELS = ["haiku", "llama31_8b"]
MULTIMODAL_MODELS = ["gemini15pro", "gpt4o", "sonnet", "llama32_90b"]


def VertexNoFilter(*args, **kwargs):
    return ChatVertexAI(
        *args,
        **kwargs,
        safety_settings=VERTEX_SAFETY_SETTINGS,
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
