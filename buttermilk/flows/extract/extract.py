
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

from buttermilk import BM
from buttermilk.llms import LLMs
from buttermilk.tools.json_parser import ChatParser

BASE_DIR = Path(__file__).absolute().parent
TEMPLATE_PATH = BASE_DIR.parent / "templates"

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

class Analyst():
    def __init__(self, *, langchain_model_name: str, prompt_template_path: str) -> None:

        bm = BM()

        self.langchain_model_name = langchain_model_name
        self.llm = LLMs(connections=bm._connections_azure)[langchain_model_name]

        # Load the template from a prompty file
        from promptflow.core import Prompty
        from promptflow.core._prompty_utils import convert_prompt_template

        # load prompty as a flow
        prompty = Prompty.load(BASE_DIR / prompt_template_path)
        self.template = convert_prompt_template(prompty._template, api="chat", inputs={})

        # convert to a list of messages and roles expected by langchain
        self.langchain_template = [(m['role'], m['content']) for m in self.template]
        pass


    @tool
    def __call__(
        self, *, content: str, record_id: str = 'not given', **kwargs) -> LLMOutput:

        chain = ChatPromptTemplate.from_messages(self.langchain_template, template_format="jinja2") | self.llm | ChatParser()
        input_vars = dict(content=content)
        input_vars.update({k: v for k, v in kwargs.items() if v})

        logger.info(f"Invoking chain with {self.langchain_model_name} for record: {record_id}")
        t0 = time.time()
        output = chain.invoke(input=input_vars, **kwargs)
        t1 = time.time()
        logger.info(f"Invoked chain with {self.langchain_model_name} and record: {record_id} in {t1-t0:.2f} seconds")

        output['record_id'] = record_id
        for k in LLMOutput.__required_keys__:
            if k not in output:
                output[k] = None

        return  output

