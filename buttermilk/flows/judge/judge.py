
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
from buttermilk.utils.utils import read_text, read_yaml, scrub_serializable
from buttermilk import BM
from buttermilk.tools.json_parser import ChatParser
from langchain_core.prompts import MessagesPlaceholder
import yaml

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

class Judger(ToolProvider):
    def __init__(self, *, model: str, criteria: str = None, standards_path: str = None, template_path: str = 'judge.jinja2', connection: dict ={}) -> None:

        self.connection = connection
        self.model = model
        loader=FileSystemLoader(searchpath=TEMPLATE_PATHS)
        env = Environment(loader=loader, trim_blocks=True, keep_trailing_newline=True, undefined=KeepUndefined)

        if standards_path and not criteria:
            criteria = env.get_template(standards_path).render()

        partial_variables=dict(criteria=criteria)

        self.template = env.get_template(template_path).render(**partial_variables)


    @tool
    def __call__(
        self, *, content: str, **kwargs) -> LLMOutput:

        llm = LLMs(connections=self.connection)[self.model]

        tpl = ChatPromptTemplate.from_messages([("system",self.template), MessagesPlaceholder("content", optional=True)], template_format="jinja2")

        chain = tpl | llm | ChatParser()

        input_vars = {"content": [HumanMessage(content=content)]}

        t0 = time.time()
        output = chain.invoke(input=input_vars)
        t1 = time.time()
        logger.info(f"Judger invoked chain with {self.model} in {t1-t0:.2f} seconds")

        for k in LLMOutput.__required_keys__:
            if k not in output:
                output[k] = None

        return  output



if __name__ == "__main__":
    bm = BM()
    conn = {'haiku': bm._connections_azure['haiku']}
    judger = Judger(standards_path="criteria_ordinary.jinja2", template_path="judge.jinja2",  model="haiku", connection=conn)
    output = judger(content="Hello, world!")
    print(output)