
from pathlib import Path
import time

from promptflow.core import (
    ToolProvider,
    tool,
)
from langchain_core.messages import HumanMessage
from pathlib import Path
from typing import Optional, Self, TypedDict
from jinja2 import Environment, FileSystemLoader
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import Runnable
from jinja2 import Environment, BaseLoader, Undefined
from buttermilk.llms import LLMs

from buttermilk import BM
from buttermilk.tools.json_parser import ChatParser
from langchain_core.prompts import MessagesPlaceholder
from promptflow.core import Prompty
from promptflow.core._prompty_utils import convert_prompt_template
BASE_DIR = Path(__file__).absolute().parent
TEMPLATE_PATHS = [BASE_DIR, BASE_DIR.parent / "common", BASE_DIR.parent / "templates"]

class LLMOutput(TypedDict):
    correct: bool
    alignment: float
    reasons: dict

class KeepUndefined(Undefined):
    def __str__(self):
        return '{{ ' + self._undefined_name + ' }}'

class Scorer(ToolProvider):
    def __init__(self, *, model: str, template_path: str = 'score.prompty') -> None:


        bm = BM()
        self.connections = bm._connections_azure

        # env = Environment(loader=FileSystemLoader(searchpath=TEMPLATE_PATHS), trim_blocks=True, keep_trailing_newline=True, undefined=KeepUndefined)

        # tpl = env.get_template(template_path).render(**kwargs)

        # load prompty as a flow
        prompty = Prompty.load(BASE_DIR / template_path)
        template = convert_prompt_template(prompty._template, api="chat", inputs={})

        # convert to a list of messages and roles expected by langchain
        langchain_messages = [(m['role'], m['content']) for m in template if m['content']]

        self.template = ChatPromptTemplate.from_messages(langchain_messages, template_format="jinja2")

        self.model = model

    @tool
    def __call__(
        self, *, groundtruth: dict,response: dict[str, str] ) -> LLMOutput:

        llm = LLMs(connections=self.connections)[self.model]

        chain = self.template.copy() | llm | ChatParser()
        input_vars = dict(expected_answer=groundtruth['answer'], expected_reasoning=groundtruth['reasoning'], prediction=response)
        output = chain.invoke(input=input_vars)

        for k in LLMOutput.__required_keys__:
            if k not in output:
                output[k] = None

        return  output
