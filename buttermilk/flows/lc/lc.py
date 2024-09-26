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

from jinja2 import Environment, FileSystemLoader, Template
from promptflow.core import (
    ToolProvider,
    tool
)

from buttermilk import BM
from buttermilk.flows.judge.judge import LLMOutput,TEMPLATE_PATHS,KeepUndefined,LLMOutputBatch



class LangChainMulti(ToolProvider):
    def __init__(self, *, models: list, template_path: str, other_templates: dict = {}, other_vars: Optional[dict] = None) -> None:
        bm = BM()
        template_vars = {}
        if other_vars and isinstance(other_vars, dict):
            template_vars.update(other_vars)
        self.connections = bm._connections_azure
        self.models = models

        self.llm = LLMs(connections=self.connections)

        loader=FileSystemLoader(searchpath=TEMPLATE_PATHS)
        env = Environment(loader=loader, trim_blocks=True, keep_trailing_newline=True, undefined=KeepUndefined)

        for k, v in other_templates.items():
            template_vars[k] = env.get_template(v)

        # render sub-templates first
        # WARNING. Template vars have to be in REVERSE ORDER of dependency to render correctly.
        for k, v in template_vars.items():
            if isinstance(v, Template):
                template_vars[k] = v.render(**template_vars)

        self.template = env.get_template(template_path).render(**template_vars)

        breakpoint()
        pass


    @tool
    def __call__(
        self,
        *,
        inputs: dict = {},
        content: Optional[str] = None, **kwargs
    ) -> LLMOutputBatch:
        """Evaluate with langchain evaluator."""
        results = {}
        inputs.update(kwargs)
        breakpoint()
        for model in self.models:
            if content:
                chain = ChatPromptTemplate.from_messages([("system",self.template), MessagesPlaceholder("content", optional=True)], template_format="jinja2")
                inputs['content'] = [HumanMessage(content=content)]
            else:
                chain = ChatPromptTemplate.from_messages([("human",self.template)], template_format="jinja2")

            output = self.invoke(chain=chain, model=model, input_vars=inputs)
            output["timestamp"] = pd.to_datetime(datetime.datetime.now(tz=datetime.UTC)).isoformat()

            for k in LLMOutput.__required_keys__:
                if k not in output:
                    output[k] = None

            results[model] = LLMOutput(**output)

        return results

    @trace
    def invoke(self, chain, input_vars, model):
        t0 = time.time()
        chain = chain | self.llm[model] | ChatParser()
        output = chain.invoke(input=input_vars)
        t1 = time.time()
        logger.info(f"Invoked chain with {model} in {t1-t0:.2f} seconds")
        return output

if __name__ == "__main__":
    from promptflow.tracing import start_trace
    from buttermilk import BM
    bm = BM()
    conn = {'haiku': bm._connections_azure['haiku']}

    start_trace()
    pf = PFClient()
    #connection = pf.connections.get(name="my_llm_connection")
    evaluator = LangChainMulti(connections=conn,model="haiku")
    result = evaluator(
        content="What's 2+2?",
    )
    print(result)