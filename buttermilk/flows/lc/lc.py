from dataclasses import dataclass
import datetime
import time
from typing import Any, List, Optional, TypedDict

import pandas as pd


from buttermilk.llms import LLMs
from promptflow.client import PFClient
from promptflow.connections import CustomConnection
from promptflow.tracing import trace
from buttermilk.tools.json_parser import ChatParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

from langchain_core.messages import HumanMessage

from buttermilk.utils.log import logger

from jinja2 import Environment, FileSystemLoader
from promptflow.core import (
    ToolProvider,
    tool
)

from buttermilk import BM
from buttermilk.flows.judge.judge import LLMOutput,TEMPLATE_PATHS,KeepUndefined




class LangChainMachine(ToolProvider):
    def __init__(self, *, model: str, connections: dict = None, custom_connection: Optional[CustomConnection] = None) -> None:
        if connections is not None:
            self.connections = connections
        else:
            bm = BM()
            self.connections = bm._connections_azure
        self.model = model

        self.llm = LLMs(connections=self.connections)[self.model]


    @tool
    def __call__(
        self,
        answers: dict[str, List[dict[str, str]]],
        *,
        content: Optional[str] = None,
        chain: Optional[Any] = None,
        **kwargs: Any,
    ) -> LLMOutput:
        """Evaluate with langchain evaluator."""

        input_vars = {}

        loader=FileSystemLoader(searchpath=TEMPLATE_PATHS)
        env = Environment(loader=loader, trim_blocks=True, keep_trailing_newline=True, undefined=KeepUndefined)

        original_instructions = env.get_template("criteria_ordinary.jinja2").render()

        self.template = env.get_template("synthesise.jinja2").render(original_instructions=original_instructions, content=content, answers=answers,feedback=None)

        if chain is None:
            chain = ChatPromptTemplate.from_messages([("human",self.template)]) | self.llm | ChatParser()
        else:
            chain = chain | self.llm | ChatParser()

        output = self.invoke(chain=chain, input_vars=input_vars)
        output["timestamp"] = pd.to_datetime(datetime.datetime.now(tz=datetime.UTC)).isoformat()

        return LLMOutput(**output)

    @trace
    def invoke(self, chain, input_vars):
        t0 = time.time()
        output = chain.invoke(input=input_vars)
        t1 = time.time()
        logger.info(f"Invoked chain with {self.model} in {t1-t0:.2f} seconds")
        return output

if __name__ == "__main__":
    from promptflow.tracing import start_trace
    from buttermilk import BM
    bm = BM()
    conn = {'haiku': bm._connections_azure['haiku']}

    start_trace()
    pf = PFClient()
    #connection = pf.connections.get(name="my_llm_connection")
    evaluator = LangChainMachine(connections=conn,model="haiku")
    result = evaluator(
        content="What's 2+2?",
    )
    print(result)