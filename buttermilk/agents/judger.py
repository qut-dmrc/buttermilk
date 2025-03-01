

from typing import Any, Awaitable, Callable, List, Sequence
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Handoff as HandoffBase
from autogen_core.memory import Memory
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import ChatCompletionClient
from autogen_core.tools import BaseTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

from buttermilk.bm import BM
from buttermilk.llms import LLMClient

from autogen_agentchat.agents import AssistantAgent, SocietyOfMindAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
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

from buttermilk import BM, logger
from buttermilk._core.agent import Agent
from buttermilk._core.runner_types import Job
from buttermilk.exceptions import RateLimit
from buttermilk.llms import LLMCapabilities, LLMs
from buttermilk.tools.json_parser import ChatParser
from buttermilk.utils.templating import (
    _parse_prompty,
    load_template_vars,
    make_messages,
    prepare_placeholders,
)
from buttermilk.utils.utils import scrub_keys


import asyncio
from autogen_agentchat.ui import Console
from autogen_agentchat.agents import AssistantAgent, SocietyOfMindAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination

import json
import string
import uuid
from typing import List

import openai
from autogen_core import (
    DefaultTopicId,
    FunctionCall,
    Image,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from IPython.display import display  # type: ignore
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown


import asyncio
from dataclasses import dataclass
from typing import List

from autogen_core import AgentId, MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient


@dataclass
class WorkerTask:
    task: str
    previous_results: List[str]


@dataclass
class WorkerTaskResult:
    result: str


@dataclass
class UserTask:
    task: str


@dataclass
class FinalResult:
    result: str


class ChatAgent(AssistantAgent):
    def __init__(self, name: str, model_client: ChatCompletionClient|str, *, tools: List[BaseTool[Any, Any] | Callable[..., Any] | Callable[..., Awaitable[Any]]] | None = None, handoffs: List[str] | None = None, model_context: ChatCompletionContext | None = None, description: str = "An agent that provides assistance with ability to use tools.", model_client_stream: bool = False, reflect_on_tool_use: bool = False, tool_call_summary_format: str = "{result}", memory: Sequence[Memory] | None = None, parameters: dict = {}):
        if isinstance(model_client, str):
            bm = BM()
            model_client = bm.llms.get_autogen_client(model_client)

        # Construct list of messages from the templates
        rendered_template, remaining_inputs = load_template_vars(
            **parameters
        )

        # Interpret the template as a Prompty; split it into separate messages with
        # role and content keys
        
        # Parse messages using Prompty format
        # First we strip the header information from the markdown
        prompty = _parse_prompty(rendered_template)

        # Next we use Prompty's format to set roles within the template
        from promptflow.core._prompty_utils import parse_chat

        messages = parse_chat(
            prompty,
            valid_roles=["system", "user", "developer", "human", "placeholder"],
        )
        system_message=None
        if not model_context:
            model_context = []
        if messages[0]['role'] in ('system', 'developer'):
            system_message = messages[0]['content']
            model_context.extend(messages[1:])
        
        super().__init__(name, model_client, tools=tools, handoffs=handoffs, model_context=model_context, description=description, system_message=system_message, model_client_stream=model_client_stream, reflect_on_tool_use=reflect_on_tool_use, tool_call_summary_format=tool_call_summary_format, memory=memory)
    
