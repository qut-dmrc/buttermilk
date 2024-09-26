from dataclasses import dataclass
import datetime
from random import shuffle
import time
from typing import Any, List, Optional, TypedDict

import pandas as pd
import requests
import urllib3

from google.api_core.exceptions import ResourceExhausted
from google.generativeai.types.generation_types import (
    BlockedPromptException,
    StopCandidateException,
)
from buttermilk.exceptions import RateLimit
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from anthropic import RateLimitError as AnthropicRateLimitError
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import RateLimitError as OpenAIRateLimitError
from buttermilk.llms import LLMs
from promptflow.client import PFClient
from promptflow.connections import CustomConnection
from promptflow.tracing import trace
from buttermilk.tools.json_parser import ChatParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
    wait_random_exponential,
)
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
    ) -> LLMOutputBatch:
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

            output = self.invoke(chain=chain, model=model, input_vars=local_inputs)
            output["timestamp"] = pd.to_datetime(datetime.datetime.now(tz=datetime.UTC)).isoformat()

            for k in LLMOutput.__required_keys__:
                if k not in output:
                    output[k] = None

            results[model] = LLMOutput(**output)
        breakpoint()
        results['metadata'] = self.try_sum_tokens(results)
        return results

    @retry(
        retry=retry_if_exception_type(
            exception_types=(
                RateLimit,
                requests.exceptions.ConnectionError,
                urllib3.exceptions.ProtocolError,urllib3.exceptions.TimeoutError,
                OpenAIAPIConnectionError, OpenAIRateLimitError, AnthropicAPIConnectionError, AnthropicRateLimitError, ResourceExhausted
            ),
        ),
            # Wait interval: increasing exponentially up to a max of 30s between retries
            wait=wait_exponential_jitter(initial=1, max=30, jitter=5),
            # Retry up to five times before giving up
            stop=stop_after_attempt(5),
    )
    @trace
    def invoke(self, chain, input_vars, model):
        t0 = time.time()
        try:
            chain = chain | self.llm[model] | ChatParser()
            logger.info(f"Invoking chain with {model}...")
            output = chain.invoke(input=input_vars)
        except Exception as e:
            t1 = time.time()
            err = f"Error invoking chain with {model}: {e} after {t1-t0:.2f} seconds. {e.args=}"
            logger.error(err)
            return dict(error=err)
        t1 = time.time()
        logger.info(f"Invoked chain with {model} in {t1-t0:.2f} seconds")
        breakpoint()
        return output

    def try_sum_tokens(self, results):
        breakpoint()
        try:
            cumulative_usage = results['metadata']['usage']
        except:
            try:
                cumulative_usage = results['usage']
            except:
                cumulative_usage = dict(total_tokens=0, prompt_tokens=0, completion_tokens=0)

        for model_output in results.keys():
            try:
                usage = results.get('metadata', results.get('usage').get('usage'))
                if usage:
                    cumulative_usage['total_tokens'] += usage.get('total_tokens', 0)
                    cumulative_usage['prompt_tokens'] += usage.get('prompt_tokens', 0)
                    cumulative_usage['completion_tokens'] += usage.get('completion_tokens', 0)
                else:
                    cumulative_usage['total_tokens'] += model_output.get('llm.usage.total_tokens', 0)
                    cumulative_usage['prompt_tokens'] += model_output.get('llm.usage.prompt_tokens', 0)
                    cumulative_usage['completion_tokens'] += model_output.get('llm.usage.completion_tokens', 0)
            except Exception as e:
                breakpoint()
                continue

        results['usage'] = cumulative_usage
        breakpoint()
        return results

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