import datetime
import time
from collections.abc import Mapping, Sequence
from typing import Any, Self

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
from buttermilk._core.runner_types import Job, Result
from buttermilk.exceptions import RateLimit
from buttermilk.llms import LLMs
from buttermilk.tools.json_parser import ChatParser
from buttermilk.utils.templating import KeepUndefined, _parse_prompty, make_messages
from buttermilk.utils.utils import find_in_nested_dict, read_text, scrub_keys

#########################################
###
# An interface to langchain chat models
###
#########################################


class LC(Agent):
    name: str = Field(
        ...,
        init=False,
        description="The name of the flow or step in the process that this agent is responsible for.",
    )

    _connections: dict | None = PrivateAttr(default=dict)
    _llms: LLMs | None = PrivateAttr(default=dict)

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = False
        populate_by_name = True
        exclude_none = True
        exclude_unset = True

        json_encoders = {
            np.bool_: bool,
            datetime.datetime: lambda v: v.isoformat(),
            ListConfig: lambda v: OmegaConf.to_container(v, resolve=True),
            DictConfig: lambda v: OmegaConf.to_container(v, resolve=True),
        }

    @model_validator(mode="after")
    def load_connections(self) -> Self:
        bm = BM()
        self._connections = bm._connections_azure
        self._llms = LLMs(connections=self._connections)

        return self

    def load_template_vars(
        self,
        template: str,
        **inputs,
    ):
        def placeholder(variable_name):
            """Adds a placeholder for an array of messages to be inserted."""
            return MessagesPlaceholder(variable_name=variable_name, optional=True)

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
        model: str,
        template: str,
        additional_data: dict = None,
        **kwargs,
    ) -> Job:

        def resolve_value(value):
            """Recursively resolve values from different data sources."""
            if isinstance(value, str):
                # Handle special "record" case
                if value.lower() == "record":
                    return job.record

                # Handle dot notation
                if "." in value:
                    locator, field = value.split(".", maxsplit=1)
                    if locator in additional_data:
                        if isinstance(additional_data[locator], pd.DataFrame):
                            return additional_data[locator][field].values
                        return find_in_nested_dict(additional_data[locator], field)
                    if locator == "record":
                        return find_in_nested_dict(job.record.model_dump(), field)

                # Handle direct record field reference
                if value in job.record.model_fields or value in job.record.model_extra:
                    return getattr(job.record, value)

                # handle entire dataset
                if value in additional_data:
                    if isinstance(additional_data[value], pd.DataFrame):
                        return additional_data[value].to_dict(orient="records")
                    return additional_data[value]

                # No match
                return value

            if isinstance(value, Sequence) and not isinstance(value, str):
                return [resolve_value(item) for item in value]

            if isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}

            return value

        # Process all inputs into two categories.
        # Job objects have a .params mapping, which is usually the result of a combination of init variables that will be common to multiple runs over different records.
        # Job objects also have a .inputs mapping, which is the result of a combination of inputs that will be unique to a single record.
        # Then there are also extra **kwargs sent to this method.
        # In all cases, input values might be the name of a template, a literal value, or a reference to a field in the job.record object or in other supplied additional_data.
        # We need to resolve all inputs into a mapping that can be passed to the model.

        # First, log that we received extra **kwargs
        job.inputs.update(**kwargs)

        # Create a dictionary for complete prompt messages that we will not pass to the templating function
        placeholders = {"record": job.record}

        # And combine all sources of inputs into one dict
        all_params = {**job.parameters, **job.inputs}

        # but remove 'template', we deal with that explicitly, it's always required.
        _ = all_params.pop("template", None)

        input_vars = {}
        for key, value in all_params.items():
            resolved_value = resolve_value(value)
            if value == "record":  # Special case for full record placeholder
                placeholders[key] = resolved_value
            else:
                input_vars[key] = resolved_value

        # Construct list of messages from the templates
        local_messages = self.load_template_vars(
            template=template,
            **input_vars,
        )

        # Record final prompt in Job object (minus placeholders, which can contain large binary data)
        job.prompt = [f"{role}:\n{msg}" for role, msg in local_messages]

        # Add model details to Job object
        job.agent_info["connection"] = scrub_keys(self._connections[model])
        job.agent_info["model_params"] = scrub_keys(self._llms[model].dict())
        job.parameters["model"] = model

        logger.debug(
            f"Invoking agent {self.name} for job {job.job_id} with model {model} and parameters: {job.parameters}...",
        )
        response = await self.invoke(
            prompt_messages=local_messages,
            input_vars=input_vars,
            model=model,
            placeholders=placeholders,
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
        input_vars: dict[str, Any],
        placeholders: Mapping,
        model: str,
    ) -> dict[str, str]:
        # Filling placeholders
        for k, v in placeholders.items():
            input_vars[k] = [v.as_langchain_message(type="human")]

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


if __name__ == "__main__":
    lc = LC(
        model=["fake"],
        template="judge.jinja2",
        criteria="criteria_ordinary",
    )
    result = lc(
        content="What's 2+2?",
    )
    print(result)
