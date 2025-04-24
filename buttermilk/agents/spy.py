import asyncio
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Sequence, Union
from uuid import uuid4  # Added Union for type hints
from autogen_core import (
    DefaultTopicId,
    CancellationToken,
    MessageContext,
    RoutedAgent,
    TopicId,  # Identifier for message topics.
    message_handler,  # Decorator to register methods as message handlers.
)

from buttermilk._core.agent import Agent, AllMessages, ProcessingError
from buttermilk._core.config import  SaveInfo
from buttermilk._core.contract import (
    AgentOutput,
    GroupchatMessageTypes)
from buttermilk.utils.save import upload_rows, upload_rows_async
from buttermilk.utils.uploader import AsyncDataUploader 
from buttermilk.bm import bm, logger

BATCH_SIZE = 10

class SpyAgent(RoutedAgent):
    """Agent just lurks in the group chat, saving things to a database."""

    def __init__(
        self,
        save_dest: SaveInfo,
    ) -> None:
        super().__init__(description="Save results to BQ")
        self.manager = AsyncDataUploader(buffer_size=BATCH_SIZE, save_dest = save_dest)

    @message_handler
    async def agent_output(self, message: AgentOutput, ctx: MessageContext) -> None:
        logger.debug(f"SpyAgent received message of type: {type(message)} on topic {ctx.topic_id}") # Log received type and topic
    
        """Captures outputs from other agents and saves them."""
        if isinstance(message, AgentOutput):
            await self.manager.add(message)
        else:
            raise ProcessingError(f"Spy database save agent received incompatible output type: {type(message)}")
