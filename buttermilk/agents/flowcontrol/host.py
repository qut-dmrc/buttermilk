from huggingface_hub import User
from pydantic import PrivateAttr
from buttermilk._core.contract import ConductorRequest, ManagerMessage, ManagerRequest, ManagerResponse, OOBMessages
from buttermilk.agents.llm import LLMAgent

from typing import Any, AsyncGenerator, Self, Union
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
    TaskProcessingComplete,
    ProceedToNextTaskSignal,
)

class HostAgent(LLMAgent):
    """Coordinators for group chats that use an LLM. Can act as the 'beat' to regulate flow."""

    _input_callback: Any = PrivateAttr(...)
    _pending_agent_id: str | None = PrivateAttr(default=None) # Track agent waiting for signal

    _message_types_handled: type[Any] = PrivateAttr(default=Union[ConductorRequest])
    
    async def _ready_to_execute(self) -> bool:
        """Check if the agent is ready to execute."""
        return True
    
    async def initialize(self, input_callback, **kwargs) -> None:
        """Initialize the interface"""
        self._input_callback = input_callback
        await super().initialize(**kwargs) # Call parent initialize if needed

    async def handle_control_message( # type: ignore
        self,
        message: OOBMessages, ctx: MessageContext, **kwargs
    ) -> OOBMessages | ProceedToNextTaskSignal | None: 
        # --- Handle Conductor Request (existing logic) ---
        if isinstance(message, ConductorRequest):
            next_step_output = None
            async for next_step_output in self.__call__(message, ctx=ctx.cancellation_token):
                pass
            return next_step_output

        # --- Handle Task Completion from Worker Agents ---
        elif isinstance(message, TaskProcessingComplete):
            logger.info(f"Host received TaskComplete from {message.source} (Task {message.task_index}, More: {message.more_tasks_remain})")
            return None

        else:
            logger.debug(f"Host received unhandled OOB message type: {type(message)}")
            return None
