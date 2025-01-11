import datetime
import time
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import regex as re
import requests
import urllib3
from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
    RateLimitError as AnthropicRateLimitError,
)
from google.api_core.exceptions import ResourceExhausted
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from omegaconf import DictConfig, ListConfig, OmegaConf
from openai import (
    APIConnectionError as OpenAIAPIConnectionError,
    RateLimitError as OpenAIRateLimitError,
)
from promptflow.tracing import trace
from pydantic import (
    Field,
    PrivateAttr,
    model_validator,
)
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random,
)

from buttermilk import BM, TEMPLATE_PATHS, logger
from buttermilk._core.agent import Agent
from buttermilk._core.runner_types import Job, RecordInfo, Result
from buttermilk.exceptions import RateLimit
from buttermilk.llms import LLMCapabilities, LLMs
from buttermilk.tools.json_parser import ChatParser
from buttermilk.utils.templating import KeepUndefined, _parse_prompty, make_messages, load_template_vars
from buttermilk.utils.utils import find_in_nested_dict, read_text, scrub_keys

#########################################
###
# An interface to langchain chat models
###
#########################################


class LC(Agent):

    @property
    def _llms(self) -> LLMs:
        bm = BM()
        return bm.llms


    async def process_job(
        self,
        *,
        job: Job,
        q: str | None = None,
        additional_data: dict[str, dict] = {},
    ) -> Job:

        model = job.parameters.pop('model')
        template = job.parameters.pop('template')
        if not model:
            raise ValueError(f"No model specified for agent LC for job {job.job_id}.")
        
        # Construct list of messages from the templates
        rendered_template = load_template_vars(template=template, **job.parameters, **job.inputs, **additional_data)

        # Add model details to Job object
        job.agent_info["connection"] = scrub_keys(self._llms.connections[model])
        job.agent_info["model_params"] = scrub_keys(self._llms[model].dict())
        job.parameters["model"] = model

        logger.debug(
            f"Invoking agent {self.name} for job {job.job_id} in flow {job.flow_id} with model {model}...",
        )
        response = await self.invoke(
            template=rendered_template,
            model=model,
            placeholders=job.inputs,
        )

        logger.debug(
            f"Finished agent {self.name} for job {job.job_id} in flow {job.flow_id} with model {model}, received response of {len(str(response.values()))} characters.",
        )
        error = response.pop("error", None)
        if error:
            job.error[self.name] = error
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
    async def invoke(
        self,
        *,
        template: str,
        placeholders: Mapping,
        model: str,
    ) -> dict[str, str]:
        input_vars = {}
        model_capabilities: LLMCapabilities = self._llms.connections[model].capabilities

        # Prepare placeholder variables
        for k, v in placeholders.items():
            if isinstance(v, RecordInfo):
                # Render record as a message part in OpenAI format
                if rendered := v.as_langchain_message(role="user", model_capabilities=model_capabilities):
                    input_vars[k] = [rendered]
            elif isinstance(v, str):
                input_vars[k] = v
            elif isinstance(v, Sequence):
                # Lists may need to be handled separately...?
                input_vars[k] = '\n\n'.join(v)

        # Substitute placeholder variables into the template
        # template = template.format(**input_vars)

        # Interpret the template as a Prompty; split it into separate messages with
        # role and content keys
        messages = make_messages(template)

        # Convert to langchain format
        # (Later we won't need this, because langchain ends up converting back to our json anyway)
        lc_messages = []
        for message in messages:
            role = message['role']
            content = message['content']
            lc_messages.append((role, content))

        # Make the chain
        chain = ChatPromptTemplate(lc_messages, template_format="jinja2") | self._llms[model] | ChatParser()

        elapsed = 0
        t0 = time.time()
        output = {}
        # Invoke the chain
        try:
            try:
                logger.debug(f"Invoking chain with {model}...")
                output = await chain.ainvoke(input=input_vars)
            except Exception as e:
                t1 = time.time()
                elapsed = t1 - t0
                err = f"Error invoking chain with {model} after {elapsed:.2f} seconds: {str(e)[:1000]} "
                logger.error(err)
                output = dict(error=err)
            t1 = time.time()
            elapsed = t1 - t0
            logger.debug(f"Invoked chain with {model} in {elapsed:.2f} seconds")

        except RetryError:
            output = dict(error="Retry timeout querying LLM")
        if "metadata" not in output:
            output["metadata"] = dict()
        output["metadata"]["seconds_elapsed"] = elapsed

        return output

