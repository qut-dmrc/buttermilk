
import sys
import time
from cgitb import text
from logging import getLogger
from pathlib import Path
from typing import Optional, Self, TypedDict

import vertexai
import vertexai.generative_models as generative_models
from jinja2 import Environment, FileSystemLoader
from promptflow.core import Prompty, ToolProvider, tool
from promptflow.core._prompty_utils import convert_prompt_template
from promptflow.tracing import trace
from vertexai.generative_models import (
    FinishReason,
    GenerativeModel,
    Part,
    SafetySetting,
)

from buttermilk import BM
from buttermilk.tools.json_parser import ChatParser

BASE_DIR = Path(__file__).absolute().parent
TEMPLATE_PATH = BASE_DIR.parent / "templates"

logger = getLogger()


generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
}

class Prediction(TypedDict):
    metadata: dict
    record_id: str
    analysis: str


class Analyst():
    def __init__(self, *, model: str,  template='frames_system.jinja2', **kwargs) -> None:

        bm = BM()
        self.connections = bm._connections_azure
        self.model = model
        env = Environment(loader=FileSystemLoader(searchpath=[BASE_DIR, TEMPLATE_PATH]), trim_blocks=True, keep_trailing_newline=True)
        self.system_message = env.get_template(template).render()


    @tool
    def __call__(
        self, *, content: str='', media_attachment_uri: str, **kwargs) -> Prediction:
        vertexai.init(project="dmrc-platforms", location="us-central1")
        video_part = Part.from_uri(
            mime_type="video/mp4",
            uri=media_attachment_uri)

        llm_info = self.connections[self.model]

        model = GenerativeModel(
            "gemini-1.5-pro-001",
            system_instruction=[self.system_message]
        )
        responses = model.generate_content(
            [video_part, content],
            generation_config=generation_config,
            stream=False,
        )
        result = ChatParser().parse(responses.text)

        return result
