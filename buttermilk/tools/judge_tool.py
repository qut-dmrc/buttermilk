
from logging import getLogger
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
from buttermilk.utils.json_parser import ChatParser
from langchain_core.prompts import MessagesPlaceholder

BASE_DIR = Path(__file__).absolute().parent
TEMPLATE_PATHS = [BASE_DIR, BASE_DIR / "includes" ]

logger = getLogger()
class KeepUndefined(Undefined):
    def __str__(self):
        return '{{ ' + self._undefined_name + ' }}'

class LLMOutput(TypedDict):
    result: dict
    reasons: list
    labels: list
    metadata: dict
    record_id: str
    scores: dict

class Judger(ToolProvider):
    def __init__(self, *, model: str, criteria: str = None, standards_path: str = None, template_path: str = 'apply.jinja2') -> None:

        bm = BM()
        self.connections = bm._connections_azure

        env = Environment(loader=FileSystemLoader(searchpath=TEMPLATE_PATHS), trim_blocks=True, keep_trailing_newline=True, undefined=KeepUndefined)

        partial_variables={}
        if standards_path and not criteria:
            standards = env.get_template(standards_path).render()
            partial_variables=dict(criteria=standards)
        elif criteria:
            partial_variables=dict(criteria=criteria)
        else:
            raise ValueError("You must provide criteria either as a string `criteria` or a filename `standards_path`, but not both.")

        tpl = env.get_template(template_path).render(**partial_variables)
        self.template = ChatPromptTemplate.from_messages([("system",tpl), MessagesPlaceholder("content", optional=True)], template_format="jinja2")

        self.model = model

    @tool
    def run(self, *, content: str) -> LLMOutput:

        llm = LLMs(connections=self.connections)[self.model]

        chain = self.template.copy() | llm | ChatParser()

        input_vars = {"content": [HumanMessage(content=content)]}
        input_vars.update({k: v for k, v in kwargs.items() if v})

        t0 = time.time()
        output = chain.invoke(input=input_vars)
        t1 = time.time()
        logger.debug(f"Judger invoked chain with {self.model} in {t1-t0:.2f} seconds")

        for k in LLMOutput.__required_keys__:
            if k not in output:
                output[k] = None

        return  output


    def __call__(        self, *, content: str) -> LLMOutput:
        return self.run(content=content)