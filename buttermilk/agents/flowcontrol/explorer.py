import asyncio
from collections.abc import Awaitable
from math import ceil
from huggingface_hub import User
from pydantic import BaseModel, Field, PrivateAttr
import weave
from buttermilk._core.contract import (
    COMMAND_SYMBOL,
    END,
    AssistantMessage,
    ConductorRequest,
    ConductorResponse,
    ManagerMessage,
    ManagerRequest,
    ManagerResponse,
    OOBMessages,
    StepRequest,
    TaskProcessingStarted,
    UserMessage,
)
from buttermilk.agents.flowcontrol.host import LLMHostAgent
from buttermilk.agents.llm import LLMAgent

from typing import Any, AsyncGenerator, Callable, Optional, Self, Union
from autogen_core import CancellationToken, MessageContext, message_handler

from buttermilk import logger
from buttermilk._core.config import DataSourceConfig
from buttermilk._core.contract import (
    AgentInput,
    AgentOutput,
    AllMessages,
    FlowMessage,
    GroupchatMessageTypes,
    OOBMessages,
    UserInstructions,
    TaskProcessingComplete,
    ProceedToNextTaskSignal,
)

TRUNCATE_LEN = 1000  # characters per history message


class ExplorerHost(LLMHostAgent):

    _output_model: Optional[type[BaseModel]] = StepRequest
    async def _choose(self, inputs: ConductorRequest) -> StepRequest:
        step = await self._process(inputs=inputs)
        return step
