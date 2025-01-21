import time
from collections.abc import Mapping

import requests
import urllib3
from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
    RateLimitError as AnthropicRateLimitError,
)
from google.api_core.exceptions import ResourceExhausted
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from openai import (
    APIConnectionError as OpenAIAPIConnectionError,
    RateLimitError as OpenAIRateLimitError,
)
from promptflow.tracing import trace
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random,
)

from buttermilk import BM, logger
from buttermilk._core.agent import Agent
from buttermilk._core.runner_types import Job, Result
from buttermilk.exceptions import RateLimit
from buttermilk.llms import LLMCapabilities, LLMs
from buttermilk.tools.json_parser import ChatParser
from buttermilk.utils.templating import load_template_vars, make_messages
from buttermilk.utils.utils import scrub_keys

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
        additional_data: dict[str, dict] = {},
    ) -> Job:
        model = job.parameters.pop("model")
        template = job.parameters.pop("template")
        if not model:
            raise ValueError(f"No model specified for agent LC for job {job.job_id}.")

        # Construct list of messages from the templates
        rendered_template, remaining_inputs = load_template_vars(template=template, **job.parameters)

        # Prepare placeholder variables
        model_capabilities: LLMCapabilities = self._llms.connections[model].capabilities
        placeholders = {}
        placeholders.update(**job.inputs)
        if job.prompt:
            placeholders["q"] = job.prompt
        placeholders = {k: v for k, v in additional_data.items() if v}

        # Record will be passed in as a Langchain Placeholder, which means it has to be a list of messages
        if job.record:
            placeholders["record"] = [
                job.record.as_langchain_message(
                    role="user", model_capabilities=model_capabilities
                )
            ]

        # Remove unecessary variables before tracing the API call to the LLM
        placeholders = {k: v for k, v in placeholders.items() if k in remaining_inputs}

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
            placeholders=placeholders,
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
        # Interpret the template as a Prompty; split it into separate messages with
        # role and content keys
        messages = make_messages(template)

        # Convert to langchain format
        # (Later we won't need this, because langchain ends up converting back to our json anyway)
        lc_messages = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            lc_messages.append((role, content))

        # Make the chain
        chain = (
            ChatPromptTemplate(lc_messages, template_format="jinja2")
            | self._llms[model]
            | ChatParser()
        )

        elapsed = 0
        t0 = time.time()
        output = {}
        # Invoke the chain
        try:
            try:
                logger.debug(f"Invoking chain with {model}...")
                output = await chain.ainvoke(input=placeholders)
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
