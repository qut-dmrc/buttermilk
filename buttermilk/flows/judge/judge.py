
import datetime
import time
from logging import getLogger
from pathlib import Path
from typing import Optional, Self, TypedDict

import pandas as pd
from jinja2 import BaseLoader, Environment, FileSystemLoader, Undefined
from langchain_core.messages import HumanMessage
from langchain_core.prompts import (
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable
from promptflow.core import ToolProvider, tool
from promptflow.tracing import trace

from buttermilk import BM
from buttermilk.llms import LLMs
from buttermilk.tools.json_parser import ChatParser


logger = getLogger()
class KeepUndefined(Undefined):
    def __str__(self):
        return '{{ ' + self._undefined_name + ' }}'

class PredictionBatch(TypedDict):
    model_name: dict

class Prediction(TypedDict):
    timestamp: object
    result: dict
    reasons: list
    labels: list
    metadata: dict
    scores: dict

class Judger(ToolProvider):
    def __init__(self, *, model: str, criteria: str = None, standards_path: str = None, template_path: str = 'judge.jinja2', connections: Optional[dict] = None) -> None:
        if connections is not None:
            self.connections = connections
        else:
            bm = BM()
            self.connections = bm._connections_azure

        self.model = model
        loader=FileSystemLoader(searchpath=TEMPLATE_PATHS)
        env = Environment(loader=loader, trim_blocks=True, keep_trailing_newline=True, undefined=KeepUndefined)

        if standards_path and not criteria:
            criteria = env.get_template(standards_path).render()

        partial_variables=dict(criteria=criteria)

        self.template = env.get_template(template_path).render(**partial_variables)


    @tool
    def __call__(
        self, *, content: Optional[str] = None, record_id: Optional[str] = None) -> Prediction:

        llm = LLMs(connections=self.connections)[self.model]

        tpl = ChatPromptTemplate.from_messages([("system",self.template), MessagesPlaceholder("content", optional=True)], template_format="jinja2")

        chain = tpl | llm | ChatParser()

        input_vars = {}
        if content is not None:
            input_vars = {"content": [HumanMessage(content=content)]}

        output = self.invoke(chain=chain, input_vars=input_vars)

        if record_id is not None:
            output['record_id'] = record_id

        output["timestamp"] = pd.to_datetime(datetime.datetime.now(tz=datetime.UTC)).isoformat()

        for k in Prediction.__required_keys__:
            if k not in output:
                output[k] = None

        return  output

    @trace
    def invoke(self, chain, input_vars):
        t0 = time.time()
        output = chain.invoke(input=input_vars)
        t1 = time.time()
        logger.info(f"Invoked chain with {self.model} in {t1-t0:.2f} seconds")
        return output


if __name__ == "__main__":
    from buttermilk import BM
    bm = BM()
    conn = {'haiku': bm._connections_azure['haiku']}
    judger = Judger(standards_path="criteria_ordinary.jinja2", template_path="judge.jinja2",  model="haiku", connections=conn)
    output = judger(content="Hello, world!")
    print(output)