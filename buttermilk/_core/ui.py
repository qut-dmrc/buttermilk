import asyncio
import os
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Type, Union

import hydra
from autogen_core import (
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from pydantic import BaseModel, Field, PrivateAttr, model_validator
import pydantic

from buttermilk._core.agent import Agent
from buttermilk._core.runner_types import Record
from buttermilk.bm import logger
from buttermilk.tools.json_parser import ChatParser
from buttermilk.utils.templating import (
    KeyValueCollector,
)


class IOInterface(BaseModel, ABC):

    @abstractmethod
    async def send_output(self, message: Any, source: str = "") -> None:
        """Send output to the user interface"""

    @abstractmethod
    async def get_input(self, message: Any, source: str = "") -> None:
        """Request input from the user interface"""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the interface"""

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources"""
