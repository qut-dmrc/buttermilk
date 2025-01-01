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
from jinja2 import FileSystemLoader, TemplateNotFound
from jinja2.sandbox import SandboxedEnvironment
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
from buttermilk.utils.templating import KeepUndefined, _parse_prompty, make_messages
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

    def load_template_vars(
        self, *, 
        template: str,
        **inputs,
    ):
        recursive_paths = TEMPLATE_PATHS + [
            x for p in TEMPLATE_PATHS for x in p.rglob("*") if x.is_dir()
        ]
        loader = FileSystemLoader(searchpath=recursive_paths)

        env = SandboxedEnvironment(
            loader=loader,
            trim_blocks=True,
            keep_trailing_newline=True,
            undefined=KeepUndefined,
        )

        # Convert template into a list of messages with smaller string templates
        filename = env.get_template(f"{template}.jinja2").filename
        logger.debug(f"Loading template {template} from {filename}.")
        tpl_text = read_text(filename)
        template_messages = make_messages(tpl_text)
        params = {}
        # Load template variables
        for k, v in inputs.items():
            if v and isinstance(v, str):
                # Try to load a template if it's passed in by filename, otherwise use it
                # as a plain string replacement.

                try:
                    filename = env.get_template(f"{v}.jinja2").filename
                    var = read_text(filename)
                    var = _parse_prompty(var)
                    env.globals[k] = env.from_string(var).render()
                    logger.debug(f"Loaded template variable {k} from {filename}.")
                except TemplateNotFound:
                    # Leave the value as is and pass it in separately
                    params[k] = v
            else:
                params[k] = v

        logger.debug(
            f"Loaded {len(inputs.keys())} template variables: {inputs.keys()}.",
        )

        logger.debug(f"Rendering template with {len(template_messages)} messages.")
        local_messages = []
        for role, msg in template_messages:
            content = env.from_string(msg).render(**params)

            # From this point, langchain expects single braces for replacement instead of double
            # we could do this with a custom template class, but it's easier to just do it here.
            content = re.sub(r"{{\s+([a-zA-Z0-9_]+)\s+}}", r"{\1}", content)

            local_messages.append((role, content))

        return local_messages

    async def process_job(
        self,
        *,
        job: Job,
        q: str | None = None
    ) -> Job:

        model = job.parameters.pop('model')
        if not model:
            raise ValueError(f"No model specified for agent LC for job {job.job_id}.")
        
        # Construct list of messages from the templates
        local_messages = self.load_template_vars(**job.parameters, **job.inputs)

        if q:
            job.prompt = q
            job.inputs["q"] = [q]

        # Add model details to Job object
        job.agent_info["connection"] = scrub_keys(self._llms.connections[model])
        job.agent_info["model_params"] = scrub_keys(self._llms[model].dict())
        job.parameters["model"] = model

        logger.debug(
            f"Invoking agent {self.name} for job {job.job_id} with model {model} and parameters: {job.parameters}...",
        )
        response = await self.invoke(
            prompt_messages=local_messages,
            model=model,
            placeholders=job.inputs,
        )

        logger.debug(
            f"Finished agent {self.name} for job {job.job_id} with model {model}, received response of {len(str(response.values()))} characters.",
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
        prompt_messages: list[tuple[str, str]],
        placeholders: Mapping,
        model: str,
    ) -> dict[str, str]:
        input_vars = {}
        model_capabilities: LLMCapabilities = self._llms.connections[model].capabilities
        # Fill placeholders
        for k, v in placeholders.items():
            if isinstance(v, RecordInfo):
                if rendered := v.as_langchain_message(role="user", model_capabilities=model_capabilities):
                    input_vars[k] = [rendered]
            elif v and v[0]:
                input_vars[k] = v

        # Make the chain
        logger.debug(f"Assembling the chain with model: {model}.")
        chain = ChatPromptTemplate(prompt_messages, template_format="jinja2")
        chain = chain | self._llms[model] | ChatParser()

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

