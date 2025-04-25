"""
Defines the agent for interacting with the user over the api
"""

import asyncio
import json
from typing import Awaitable, Callable, Optional, Union  # Added List, Union, Optional

import regex as re
from autogen_core import CancellationToken  # Autogen types (used by base Agent)
from pydantic import PrivateAttr
from buttermilk import logger

# Import base agent and specific message types used
from buttermilk._core.agent import AgentInput, OOBMessages
from buttermilk._core.contract import (
    AgentOutput,
    ConductorResponse,
    FlowMessage,  # Base type for messages
    GroupchatMessageTypes,  # Union type for messages in group chat
    ManagerRequest,  # Requests sent *to* the manager (this agent)
    ManagerResponse,  # Responses sent *from* the manager (this agent)
    TaskProcessingComplete,  # Status updates
    ToolOutput,  # Potentially displayable tool output
    UserInstructions,  # Potentially displayable instructions
)
from buttermilk._core.types import Record  # For displaying record data
from buttermilk.agents.evaluators.scorer import QualResults, QualScore  # Specific format for scores
from buttermilk.agents.judge import AgentReasons  # Specific format for judge reasons
from buttermilk.agents.ui.generic import UIAgent  # Base class for UI agents


class ProxyUserAgent(UIAgent):
    """
    """

    # Callback function provided by the orchestrator/adapter to send ManagerResponse back.
    _input_callback: Callable[[ManagerResponse], Awaitable[None]] | None = PrivateAttr(default=None)
    # Background task for polling user input.
    _input_task: Optional[asyncio.Task] = PrivateAttr(default=None)
