import asyncio
import logging
import time
from collections.abc import Mapping
from typing import Any

import requests
import urllib3
import weave
from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
    RateLimitError as AnthropicRateLimitError,
)
from google.api_core.exceptions import ResourceExhausted, TooManyRequests
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from openai import (
    APIConnectionError as OpenAIAPIConnectionError,
    RateLimitError as OpenAIRateLimitError,
)
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from promptflow.tracing import trace
from pydantic import PrivateAttr
from tenacity import (
    RetryError,
    before_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random,
)

from buttermilk import bm, logger
from buttermilk._core.agent import Agent
from buttermilk.exceptions import RateLimit
from buttermilk.llms import LLMCapabilities, LLMs
from buttermilk.tools.json_parser import ChatParser
from buttermilk.utils.templating import (
    load_template,
    make_messages,
    prepare_placeholders,
)
from buttermilk.utils.utils import scrub_keys

#########################################
###
# An interface to langchain chat models
###
#########################################


class LC(Agent):
    _instrumentor: Any = PrivateAttr(default=None)
    template: str

    def model_post_init(self, __context) -> None:
        self._instrumentor = LangchainInstrumentor()
        if not self._instrumentor.is_instrumented_by_opentelemetry:
            self._instrumentor.instrument()

    @property
    def _llms(self) -> LLMs:
        return bm.llms

    async def process_job(
        self,
        *,
        job: Job,
        **kwargs,
    ) -> Job:
        model = job.parameters["model"]
        template = job.parameters["template"]
        if not model:
            raise ValueError(f"No model specified for agent LC for job {job.job_id}.")

        # Construct list of messages from the templates
        rendered_template, remaining_inputs = load_template(
            **job.parameters,
            **job.inputs,
        )

        # Interpret the template as a Prompty; split it into separate messages with
        # role and content keys
        messages = make_messages(rendered_template)

        # Prepare placeholder variables
        #
        # From this point, remaining input variables are placeholders -- ie. flexible lists of messages
        # that may include, for example, chat history or complex media objects.
        placeholders = {}

        placeholders.update({
            k: v for k, v in job.inputs.items() if k in remaining_inputs and v
        })
        # placeholders.update({
        #     k: v for k, v in additional_data.items() if k in remaining_inputs and v
        # })
        if job.q and "q" in remaining_inputs:
            placeholders["q"] = job.q
        if job.record and "record" in remaining_inputs:
            placeholders["record"] = job.record

        # Models should only be given input that they can handle. At this time we do not
        # fail if the model cannot handle, we just exclude the input. The user should provide,
        # for example, a textual caption input for models that do not support direct image inputs.
        model_capabilities: LLMCapabilities = self._llms[model].capabilities

        # In order to handle multimodal records, the job's record will be passed in as
        # a Langchain Placeholder, which means it has to be a list of messages
        placeholders = prepare_placeholders(
            model_capabilities=model_capabilities,
            **placeholders,
        )

        # Remove unecessary or empty variables before tracing the API call to the LLM
        placeholders = {
            k: v for k, v in placeholders.items() if k in remaining_inputs and v
        }

        # Add model details to Job object
        job.agent_info["connection"] = scrub_keys(self._llms[model].connection)
        job.agent_info["model_params"] = scrub_keys(self._llms[model].params)

        logger.debug(
            f"Invoking agent {self.agent_id} {template} for job {job.job_id} in flow {job.flow_id} with model {model}...",
        )
        response = await self.invoke(
            model=model,
            messages=messages,
            placeholders=placeholders,
        )

        elapsed = response["metadata"]["seconds_elapsed"]
        error = response.pop("error", None)

        if error and not str(error).lower() == "null":
            job.error[self.agent_id] = error
            logger.error(
                f"Unsuccessfully invoked chain with {model} in {elapsed:.2f} seconds: {error}",
            )
        else:
            logger.debug(
                f"Finished agent {self.agent_id} {template} for job {job.job_id} in flow {job.flow_id} with model {model} in {elapsed:.2f} seconds, received response of {len(str(response.values()))} characters.",
            )
        job.metadata.update(response.pop("metadata", {}))
        job.outputs = dict(**response)

        return job

    async def invoke(
        self,
        *,
        messages: list,
        placeholders: Mapping,
        model: str,
    ) -> dict[str, str]:
        # give our flows a little longer to set up
        loop = asyncio.get_event_loop()
        loop.slow_callback_duration = 5.0  # instead of default 0.1

        elapsed = 0
        t0 = time.time()
        output = {}

        # Invoke the chain
        try:
            logger.debug(f"Invoking chain with {model}...")
            output = await self.invoke_with_retry(
                messages=messages,
                placeholders=placeholders,
                model=model,
            )
        except RetryError:
            output = dict(error="Retry timeout querying LLM")
        except Exception as e:
            err = f"Error invoking chain with {model}: {str(e)[:1000]} "
            output = dict(error=err)
        finally:
            t1 = time.time()
            elapsed = t1 - t0
            if "metadata" not in output:
                output["metadata"] = dict()
            output["metadata"]["seconds_elapsed"] = elapsed

        return output

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
                TooManyRequests,
            ),
        ),
        wait=wait_random(min=2, max=30),
        stop=stop_after_attempt(6),
        before=before_log(logger, log_level=logging.DEBUG),
    )
    @weave.op
    @trace
    async def invoke_with_retry(
        self,
        *,
        messages: list,
        placeholders: Mapping,
        model: str,
    ) -> dict[str, str]:
        # Make the chain
        chain = (
            ChatPromptTemplate(messages, template_format="jinja2")
            | self._llms[model].client
            | ChatParser()
        )

        return await chain.ainvoke(input=placeholders)
