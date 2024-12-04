import datetime
from pathlib import Path
import time
from dataclasses import dataclass
from random import shuffle
from typing import Any, List, Literal, Optional, Self, Tuple, TypedDict

import shortuuid

from buttermilk.utils.gsheet import GSheet
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
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    convert_to_messages,
)
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



class GSheetExporter(Agent):
    name: str = Field(default="gsheetexporter", init=False)
    flow: Optional[str] = Field(default=None, init=False, description="The name of the flow or step in the process that this agent is responsible for.")

    convert_json_columns: list[str] = Field(default=[], description="List of columns to convert to JSON.")

    _gsheet: GSheet = PrivateAttr(default_factory=GSheet)

    class Config:
        extra = "allow"

    async def process_job(
        self,
        *,
        job: Job,
        additional_data: Any = None,
        **kwargs,
    ) -> Job:
         
        # save the input data from this step to a spreadsheet so that we can compare later.
        from buttermilk.utils.gsheet import GSheet, format_strings
        input_vars, placeholders = await self.prepare_inputs(
                    job=job,
                    additional_data=additional_data,
                    **kwargs,
                )
        # combine input vars and placeholders in this instance
        input_vars.update(placeholders)
        dataset = pd.DataFrame(input_vars, index=[shortuuid.uuid()])
        answers = format_strings(dataset, convert_json_columns=self.convert_json_columns)
        
        params = {}
        if job.parameters:
            params.update(job.parameters)
        if self.save:
            params.update(self.save.model_dump())

        sheet = self._gsheet.save_gsheet(df=answers, **params)
        params["sheet_id"]= sheet.id
        job.outputs = dict(sheet_url=sheet.url, **params)
        logger.info(f"Saved to sheet {job.outputs}")

        return job