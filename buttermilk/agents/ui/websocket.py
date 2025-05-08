"""Defines the agent for interacting with the user over the api
"""

import asyncio
from collections.abc import Awaitable, Callable  # Added List, Union, Optional

from pydantic import PrivateAttr

# Import base agent and specific message types used
from buttermilk._core.contract import (
    ManagerMessage,  # Responses sent *from* the manager (this agent)
)
from buttermilk.agents.ui.generic import UIAgent  # Base class for UI agents


class ProxyUserAgent(UIAgent):
    """
    """

    # Callback function provided by the orchestrator/adapter to send ManagerResponse back.
    _input_callback: Callable[[ManagerMessage], Awaitable[None]] | None = PrivateAttr(default=None)
    # Background task for polling user input.
    _input_task: asyncio.Task | None = PrivateAttr(default=None)
