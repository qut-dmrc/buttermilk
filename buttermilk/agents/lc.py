import datetime
from pathlib import Path
import time
from dataclasses import dataclass
from random import shuffle
from typing import Any, List, Literal, Optional, Self, Tuple, TypedDict

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
from jinja2 import FileSystemLoader, Template, TemplateNotFound
from jinja2.sandbox import SandboxedEnvironment
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
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator
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
import regex as re

from buttermilk import BM, BQ_SCHEMA_DIR, TEMPLATE_PATHS, BASE_DIR
from buttermilk._core.agent import Agent
from buttermilk._core.runner_types import Job, RecordInfo, Result
from buttermilk.exceptions import RateLimit
from buttermilk.utils.templating import KeepUndefined, _parse_prompty, make_messages
from buttermilk.llms import LLMs
from buttermilk.tools.json_parser import ChatParser
from buttermilk import logger
from buttermilk.utils.utils import read_file, read_text, scrub_keys

#########################################
###
### An interface to langchain chat models
###
#########################################


class LC(Agent):
    name: str = Field(default="lc", init=False)
    flow: Optional[str] = Field(default=None, init=False, description="The name of the flow or step in the process that this agent is responsible for.")
    model: Optional[str] = None
    template: Optional[str] = None
    template_vars: Optional[dict] = {}
    
    _env: Optional[SandboxedEnvironment] = None
    _connections: Optional[dict] = {}
    _template: Optional[Template] = None
    _template_messages: Optional[List[Tuple[str,str]]] = PrivateAttr(default_factory=list)
    _llm: Optional[LLMs] = None

    class Config:
        extra = "allow"

    def __init__(self, **kwargs):
        known_fields = {field: kwargs.pop(field) for field in self.model_fields.keys() if field in kwargs.keys()}

        # Add misc kwargs to template_vars
        if 'template_vars' not in known_fields.keys():
            known_fields['template_vars'] = {}

        known_fields['template_vars'].update(**kwargs)

        super().__init__(**known_fields)

    @cached_property
    def llm(self):
        return LLMs(connections=self._connections)

    @model_validator(mode="after")
    def load_connections_and_template(self) -> Self:
        if not self._connections:
            bm = BM()
            self._connections = bm._connections_azure

        recursive_paths = TEMPLATE_PATHS + [ x for p in TEMPLATE_PATHS for x in p.rglob('*') if x.is_dir()] 
        loader = FileSystemLoader(searchpath=recursive_paths)

        def placeholder(variable_name):
            """ Adds a placeholder for an array of messages to be inserted. """
            return MessagesPlaceholder(variable_name=variable_name, optional=True)

        self._env = SandboxedEnvironment(
            loader=loader,
            trim_blocks=True,
            keep_trailing_newline=True,
            undefined=KeepUndefined,
        )

        for k,v in self.template_vars.items():
            try:
                # Try to load a template if it's passed in by filename, otherwise use it
                # as a plain string replacement.
                filename = self._env.get_template(f'{v}.jinja2').filename
                var = read_text(filename)
                var = _parse_prompty(var)
                self._env.globals[k] = self._env.from_string(var).render()
            except TemplateNotFound as e:
                # Treat template as a string
                self._env.globals[k] = v
        
        # Load the template, reading it into a list of messages with smaller templates
        filename = self._env.get_template(f'{self.template}.jinja2').filename
        tpl = read_text(filename)
        messages = make_messages(local_template=tpl)

        for role, msg in messages:
            msg = self._env.from_string(msg).render()
            self._template_messages.append((role, msg))

        # Note that unfilled variables should still be left in to render again, but they don't always seem to work...
        # self._template = self._env.get_template() #, globals=self.template_vars)
        
        return self

    async def process_job(self, job: Job) -> Job:
        if not (model := job.parameters.pop('model', None) or self.model):
            raise ValueError(
                "You must provide either model name or provide a default model on initialisation."
            )

        params = {}
        for k, v in job.parameters.items():
            # Load params from content if they match, otherwise add them literally in the vars when we pass off.
            if v == 'record':
                # Add in the record as a placeholder message
                params[k] = [job.record.as_langchain_message(type='human')]
            elif v in job.record.model_fields or v in job.record.model_extra:
                params[k] = getattr(job.record, v)
            else:
                params[k] = v

        response = await self.invoke(input_vars=params, model=model)

        # # Add prompt to Job object
        # job.prompt = [ f"{role}:\n{message}" for role, message in messages]

        job.outputs = Result(**response)

        # Add this agent's details to the Job object
        job.agent_info = self._agent_info

        # Add model details to Job object
        job.parameters['connection'] = scrub_keys(self.llm.connections[model])
        job.parameters['model_params'] = scrub_keys(self.llm[model].dict())
        job.parameters['model'] = model

        return job



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
    async def invoke(self, input_vars, model) -> dict[str, str]:

        # Make the chain
        chain = ChatPromptTemplate(self._template_messages, template_format='jinja2')
        chain = chain | self.llm[model] | ChatParser()

        # Invoke the chain 
        try:
            t0 = time.time()
            try:
                logger.debug(f"Invoking chain with {model}...")
                output = await chain.ainvoke(input=input_vars)
            except Exception as e:
                t1 = time.time()
                elapsed = t1-t0
                err = f"Error invoking chain with {model}: {str(e)[:1000]} after {elapsed:.2f} seconds."
                logger.error(err)
                raise e
                # return dict(error=err)
            t1 = time.time()
            elapsed = t1-t0
            logger.debug(f"Invoked chain with {model} in {elapsed:.2f} seconds")


        except RetryError:
            output = dict(error="Retry timeout querying LLM")
        if 'metadata' not in output:
            output['metadata'] = {}
        output['metadata']['seconds_elapsed'] = elapsed

        return output


if __name__ == "__main__":
    lc = LC(
        model=["fake"],
        template="judge.jinja2", criteria="criteria_ordinary",
    )
    result = lc(
        content="What's 2+2?",
    )
    print(result)
