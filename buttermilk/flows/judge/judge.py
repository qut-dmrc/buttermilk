
from logging import getLogger
from pathlib import Path
import time

from promptflow.core import (
    ToolProvider,
    tool,
)
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
BASE_DIR = Path(__file__).absolute().parent
TEMPLATE_PATHS = [BASE_DIR, BASE_DIR.parent / "common", BASE_DIR.parent / "templates"]

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

class Judger():
    def __init__(self, *, model: str, prompt: str = None, standards_path: str = None, template_path: str = None) -> None:

        bm = BM()
        self.connections = bm._connections_azure

        if template_path:
            env = Environment(loader=FileSystemLoader(searchpath=TEMPLATE_PATHS), trim_blocks=True, keep_trailing_newline=True, undefined=KeepUndefined)
            partial_variables={}
            if standards_path:
                criteria = env.get_template(standards_path).render()
                partial_variables=dict(criteria=criteria)

            tpl = env.get_template(template_path).render(**partial_variables)
            self.template = ChatPromptTemplate.from_messages([("system",tpl), MessagesPlaceholder("content", optional=True)], template_format="jinja2")

        else:
            self.template = ChatPromptTemplate.from_messages([("system", prompt), MessagesPlaceholder("content", optional=True)], template_format="jinja2")
        self.model = model

    @tool
    def __call__(
        self, *, content: str, record_id: str = 'not given', **kwargs) -> LLMOutput:

        llm = LLMs(connections=self.connections)[self.model]

        chain = self.template.copy() | llm | ChatParser()

        input_vars = dict(content=content)
        input_vars.update({k: v for k, v in kwargs.items() if v})

        logger.info(f"Judger invoking chain with {self.model} for record: {record_id}")
        t0 = time.time()
        output = chain.invoke(input=input_vars, **kwargs)
        t1 = time.time()
        logger.info(f"Judger invoked chain with {self.model} and record: {record_id} in {t1-t0:.2f} seconds")

        output['record_id'] = record_id
        for k in LLMOutput.__required_keys__:
            if k not in output:
                output[k] = None

        return  output



if __name__ == "__main__":
    judger = Judger(standards_path="criteria_ordinary.jinja2", template_path="system.jinja2",  model="gpt4o")
    output = judger(content="Hello, world!")
    print(output)