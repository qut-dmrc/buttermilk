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
from pydantic import BaseModel, Field, field_validator, model_validator
from pyparsing import cached_property
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
from buttermilk.runner import InputRecord

#########################################
###
### An interface to langchain chat models
###
#########################################
class LC(BaseModel):
    default_model: Optional[str] = None
    template_path: Optional[str] = None
    template_vars: Optional[dict] = {}
    other_template_paths: Optional[dict] = {}
    _connections: Optional[dict] = {}
    _template: Optional[Template] = None

 
    @cached_property
    def llm(self):
        return LLMs(connections=self._connections)

    @model_validator(mode="after")
    def load_connections_and_template(self) -> None:
        bm = BM()
        if not self._connections:
            self._connections = bm._connections_azure

        if self.template_path is None:
            return

        loader = FileSystemLoader(searchpath=TEMPLATE_PATHS)
        env = Environment(
            loader=loader,
            trim_blocks=True,
            keep_trailing_newline=True,
            undefined=KeepUndefined,
        )

        for k, v in self.other_template_paths.items():
            self.template_vars[k] = env.get_template(v).render()

        self._template = env.get_template(self.template_path)

    def __call__(
        self, content: Optional[str] = None, *, model: Optional[str] = None,  **kwargs
    ) -> PredictionBatch:
        if isinstance(content, InputRecord):
            content = content.text
        results = {}
        local_inputs = self.template_vars.copy()
        local_inputs.update(kwargs)

        if not (model := model or self.default_model):
            raise ValueError(
                "You must provide either model name or a default model when initialising."
            )

        local_template = self._template.render(**local_inputs)

        if content:
            chain = ChatPromptTemplate.from_messages(
                [
                    ("system", local_template),
                    MessagesPlaceholder("content", optional=True),
                ],
                template_format="jinja2",
            )
            local_inputs["content"] = [HumanMessage(content=content)]
        else:
            chain = ChatPromptTemplate.from_messages(
                [("human", local_template)], template_format="jinja2"
            )
        try:
            output = self.invoke(chain=chain, model=model, input_vars=local_inputs)
        except RetryError:
            output = dict(error="Retry timeout querying LLM")
        output["timestamp"] = pd.to_datetime(
            datetime.datetime.now(tz=datetime.UTC)
        ).isoformat()

        for k in Prediction.__required_keys__:
            if k not in output:
                output[k] = None

        results = Prediction(**output)

        return results

    @retry(
        retry=retry_if_exception_type(
            exception_types=(
                RateLimit,
                TimeoutError,
                requests.exceptions.ConnectionError,
                urllib3.exceptions.ProtocolError,
                urllib3.exceptions.TimeoutError,
                OpenAIAPIConnectionError,
                OpenAIRateLimitError,
                AnthropicAPIConnectionError,
                AnthropicRateLimitError,
                ResourceExhausted,
            ),
        ),
        wait=wait_random(min=10, max=60),
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
            # return dict(error=err)
        t1 = time.time()
        logger.info(f"Invoked chain with {model} in {t1-t0:.2f} seconds")

        return output


if __name__ == "__main__":
    lc = LC(
        default_model=["fake"],
        template_path="judge.jinja2",
        other_template_paths={"criteria": "criteria_ordinary.jinja2"},
    )
    result = lc(
        content="What's 2+2?",
    )
    print(result)
