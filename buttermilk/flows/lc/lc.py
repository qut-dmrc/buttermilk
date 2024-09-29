import datetime
import time
from dataclasses import dataclass
from random import shuffle
from typing import Any, List, Optional, TypedDict

import pandas as pd
import requests
import urllib3
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from anthropic import RateLimitError as AnthropicRateLimitError
from google.api_core.exceptions import ResourceExhausted
from google.generativeai.types.generation_types import (
    BlockedPromptException,
    StopCandidateException,
)
from jinja2 import Environment, FileSystemLoader, Template
from langchain_core.messages import HumanMessage
from langchain_core.prompts import (
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import RateLimitError as OpenAIRateLimitError
from promptflow.client import PFClient
from promptflow.connections import CustomConnection
from promptflow.core import ToolProvider, tool
from promptflow.tracing import trace
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
    wait_random,
    wait_random_exponential,
)

from buttermilk import BM
from buttermilk.exceptions import RateLimit
from buttermilk.flows.judge.judge import (
    TEMPLATE_PATHS,
    KeepUndefined,
    Prediction,
    PredictionBatch,
)
from buttermilk.llms import LLMs
from buttermilk.tools.json_parser import ChatParser
from buttermilk.utils.log import logger


class LangChainMulti(ToolProvider):
    def __init__(self, *, models: list, template_path: str, other_templates: dict = {}, other_vars: Optional[dict] = None) -> None:
        bm = BM()
        self.template_vars = {}
        if other_vars and isinstance(other_vars, dict):
            self.template_vars.update(other_vars)
        self.connections = bm._connections_azure
        self.models = models

        self.llm = LLMs(connections=self.connections)

        loader=FileSystemLoader(searchpath=TEMPLATE_PATHS)
        env = Environment(loader=loader, trim_blocks=True, keep_trailing_newline=True, undefined=KeepUndefined)

        for k, v in other_templates.items():
            self.template_vars[k] = env.get_template(v)

        self.template = env.get_template(template_path)

        pass


    @tool
    def __call__(
        self,
        *,
        content: Optional[str] = None, **kwargs
    ) -> PredictionBatch:
        """Evaluate with langchain evaluator."""

        results = {}
        local_inputs = {}
        local_inputs.update(kwargs)

        # render sub-templates first
        # WARNING. Template vars have to be in REVERSE ORDER of dependency to render correctly.
        for k, v in self.template_vars.items():
            if isinstance(v, Template):
                v = v.render(**local_inputs)
            local_inputs[k] = v

        local_template = self.template.render(**local_inputs)
        shuffle(self.models)
        for model in self.models:

            if content:
                chain = ChatPromptTemplate.from_messages([("system",local_template), MessagesPlaceholder("content", optional=True)], template_format="jinja2")
                local_inputs['content'] = [HumanMessage(content=content)]
            else:
                chain = ChatPromptTemplate.from_messages([("human",local_template)], template_format="jinja2")
            try:
                output = self.invoke(chain=chain, model=model, input_vars=local_inputs)
            except RetryError:
                output = dict(error="Retry timeout querying LLM")
            output["timestamp"] = pd.to_datetime(datetime.datetime.now(tz=datetime.UTC)).isoformat()

            for k in Prediction.__required_keys__:
                if k not in output:
                    output[k] = None

            results[model] = Prediction(**output)

        return results

    @retry(
        retry=retry_if_exception_type(
            exception_types=(
                RateLimit,TimeoutError,
                requests.exceptions.ConnectionError,
                urllib3.exceptions.ProtocolError,urllib3.exceptions.TimeoutError,
                OpenAIAPIConnectionError, OpenAIRateLimitError, AnthropicAPIConnectionError, AnthropicRateLimitError, ResourceExhausted
            ),
        ),
            wait=wait_random(min=10,max=60),
            stop=stop_after_attempt(4),
    )
    @trace
    def invoke(self, chain, input_vars, model) -> dict[str, str]:
        t0 = time.time()
        try:
            chain = chain | self.llm[model] | ChatParser()
            logger.info(f"Invoking chain with {model}...")
            output = chain.invoke(input=input_vars)
        except Exception as e:
            t1 = time.time()
            err = f"Error invoking chain with {model}: {e} after {t1-t0:.2f} seconds. {e.args=}"
            logger.error(err)
            raise e
            #return dict(error=err)
        t1 = time.time()
        logger.info(f"Invoked chain with {model} in {t1-t0:.2f} seconds")

        return output


if __name__ == "__main__":
    from promptflow.tracing import start_trace

    from buttermilk import BM
    bm = BM()

    start_trace()
    pf = PFClient()
    #connection = pf.connections.get(name="my_llm_connection")
    lc = LangChainMulti(models=["gpt4o"], template_path="judge.jinja2", other_templates={"criteria": "criteria_ordinary.jinja2"})
    result = lc(
        content="What's 2+2?",
    )
    print(result)