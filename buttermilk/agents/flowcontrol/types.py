from huggingface_hub import User
from pydantic import PrivateAttr
from buttermilk._core.contract import ConductorRequest, ManagerMessage, ManagerRequest, ManagerResponse, OOBMessages
from buttermilk.agents.llm import LLMAgent

from typing import Any, AsyncGenerator, Self
from autogen_core import CancellationToken, MessageContext

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
)

class HostAgent(LLMAgent):
    """Coordinators for group chats that use an LLM."""

    _input_callback: Any = PrivateAttr(...)
    
    async def initialize(self, input_callback, **kwargs) -> None:
        """Initialize the interface"""
        self._input_callback = input_callback

    async def handle_control_message(
        self,
        message: OOBMessages,
    ) -> None:
        # Respond to certain control messages addressed to us
        # for now drop everything though.
        return

    async def listen(self, message: GroupchatMessageTypes, 
        ctx: MessageContext = None,
        **kwargs):
        """Listen as normal, except we want to cancel the current
        execution plan when a new user or agent message comes in."""
        if isinstance(message, (UserInstructions, AgentOutput)):
            # confirm negative
            await self._input_callback(
                ManagerResponse(
                    role=self.role,
                    source=self.id,
                    confirm=False,
                ),
            )
        await super().listen(message=message, ctx=ctx, **kwargs)

