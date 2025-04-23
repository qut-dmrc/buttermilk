"""
Provides the adapter layer to integrate Buttermilk agents with the autogen-core runtime.

This module defines `AutogenAgentAdapter`, which wraps a standard Buttermilk `Agent`
and exposes it to the Autogen ecosystem as an `autogen_core.RoutedAgent`. It handles
message translation, routing via topics, and lifecycle management within the Autogen
runtime.
"""

import asyncio
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Sequence, Union
from uuid import uuid4  # Added Union for type hints
from bm import bm
from autogen_core import (
    DefaultTopicId,
    CancellationToken,
    MessageContext,
    RoutedAgent,
    TopicId,  # Identifier for message topics.
    message_handler,  # Decorator to register methods as message handlers.
)

from buttermilk._core.agent import ProcessingError
from buttermilk._core.config import  SaveInfo
from buttermilk._core.contract import (
    AgentOutput)
from buttermilk.utils.save import upload_rows_async
from buttermilk.utils.uploader import AsyncDataUploader 

BATCH_SIZE = 10

class SpyAgent(RoutedAgent):
    """Agent just lurks in the group chat, saving things to a database."""

    def __init__(
        self,
        save_dest: SaveInfo,
    ) -> None:
        
        self.save_dest = save_dest
        self.upload_fn = partial(upload_rows_async, schema=save_dest.db_schema, dataset=save_dest.dataset)  
        self.manager = AsyncDataUploader(upload_fn=self.upload_fn, buffer_size=BATCH_SIZE)

    @message_handler
    async def _agent_output(self, message: AgentOutput, ctx: MessageContext) -> None:
        """Captures outputs from other agents and saves them."""
        if isinstance(message, AgentOutput):
            await self.manager.add(message)
        else:
            raise ProcessingError(f"Spy database save agent received incompatible output type: {type(message)}")
