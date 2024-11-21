import datetime
from pathlib import Path
import time
from dataclasses import dataclass
from random import shuffle
from typing import Any, List, Literal, Optional, Self, Tuple, TypedDict

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

#########################################
###
### An interface to langchain chat models
###
#########################################


class GSheetExporter(Agent):
    name: str = Field(default="gsheetexporter", init=False)
    flow: Optional[str] = Field(default=None, init=False, description="The name of the flow or step in the process that this agent is responsible for.")
    sheet_name: str
    sheet_id: Optional[str] = Field(default=None)
    uri: Optional[str] = Field(default=None, description="The URI of the Google Sheet to export to.")

    _gsheet: GSheet = PrivateAttr(default_factory=GSheet)

    class Config:
        extra = "allow"

    @model_validator(mode="after")
    def create_sheet(self) -> Self:
        self.sheet = self._gsheet.
        
        return self

    async def process_job(self, job: Job) -> Job:
        # save the input data from this step to a spreadsheet so that we can compare later.
        from buttermilk.utils.gsheet import GSheet, format_strings

        answers = format_strings(job.inpu, convert_json_columns=['outputs'])
        answers.info()

        sheet = gsheet.save_gsheet(df=answers, sheet_name="evals", title="results")
        sheet_id = sheet.id

        if not (model := job.parameters.pop('model', None) or self.model):
            raise ValueError(
                "You must provide either model name or provide a default model on initialisation."
            )

        params = {}
        placeholders = {}
        for k, v in job.parameters.items():