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
from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound
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

from buttermilk import BM, Agent, BQ_SCHEMA_DIR, TEMPLATE_PATHS, BASE_DIR
from buttermilk.exceptions import RateLimit
from buttermilk.flows.templating import KeepUndefined, make_messages
from buttermilk.llms import LLMs
from buttermilk import Job, Result,  RecordInfo
from buttermilk.tools.json_parser import ChatParser
from buttermilk import logger
from buttermilk.utils.utils import scrub_keys


#########################################
###
### An interface to langchain chat models
###
#########################################

class LC(Agent):
    name: str = Field(default="lc", init=False)
    model: Optional[str] = None
    template: Optional[str] = None
    template_vars: Optional[dict] = {}

    _connections: Optional[dict] = {}
    _template: Optional[Template] = None
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
        bm = BM()
        if not self._connections:
            self._connections = bm._connections_azure

        if self.template is None:
            return
        recursive_paths = TEMPLATE_PATHS + [ x for p in TEMPLATE_PATHS for x in p.rglob('*') if x.is_dir()] 
        loader = FileSystemLoader(searchpath=recursive_paths)

        def placeholder(variable_name):
            """ Adds a placeholder for an array of messages to be inserted. """
            return MessagesPlaceholder(variable_name="conversation", optional=True)

        env = Environment(
            loader=loader,
            trim_blocks=True,
            keep_trailing_newline=True,
            undefined=KeepUndefined,
        )

        for k in self.template_vars.keys():
            try:
                # Try to load a template if it's passed in by filename, otherwise use it
                # as a plain string replacement.
                self.template_vars[k] = env.get_template(self.template_vars[k] + '.jinja2').render()
            except TemplateNotFound as e:
                # Treat template as a string
                pass
        try:
            self._template = env.get_template(self.template + '.jinja2')
        except TemplateNotFound as e:
            # Treat template as a string
            self._template = env.from_string(self.template)

        return self

    async def process_job(self, job: Job) -> Job:
        if not (model := job.parameters.pop('model', None) or self.model):
            raise ValueError(
                "You must provide either model name or provide a default model on initialisation."
            )
        
        # Compile inputs and template variables
        local_inputs = self.template_vars.copy()

        for k, v in job.parameters.items():
            # Load params from content if they match, otherwise add them literally.
            if v in job.record.model_fields or v in job.record.model_extra:
                v = getattr(job.record, v)
            local_inputs[k] = v

        local_template = self._template.render(**local_inputs)
        messages = make_messages(local_template=local_template)

        # Add this agent's details to the Job object
        job.agent_info = self._agent_info

        # Add prompt to Job object
        job.prompt = [ f"{role}:\n{message}" for role, message in messages]

        # if there is a placeholder for {record}, add our record to list 
        # of messages (after saving the prompt)?
        # TODO: currently there is no code to add a placeholder, we haven't needed it yet.
        if any(isinstance(x, MessagesPlaceholder) for _, x in messages):
            record_msg = job.record.as_langchain_message(type='human')
            messages.append(dict(record=[record_msg]))

        # Get model
        llm = self.llm[model]

        # Add model details to Job object
        job.parameters['connection'] = scrub_keys(self.llm.connections[model])
        job.parameters['model_params'] = scrub_keys(llm.dict())
        job.parameters['model'] = model

        # Make the chain
        chain = ChatPromptTemplate.from_messages(
                messages, template_format="jinja2"
            )
        
        chain = chain | llm | ChatParser()

        # Invoke the chain  (input_vars have already been inserted into the template)
        response = await self.invoke(chain, input_vars={}, model=model)

        job.outputs = Result(**response)
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
    async def invoke(self, chain, input_vars, model) -> dict[str, str]:
        try:
            t0 = time.time()
            try:
                logger.debug(f"Invoking chain with {model}...")
                output = await chain.ainvoke(input=input_vars)
            except Exception as e:
                t1 = time.time()
                elapsed = t1-t0
                err = f"Error invoking chain with {model}: {e} after {elapsed:.2f} seconds. {e.args=}"
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
