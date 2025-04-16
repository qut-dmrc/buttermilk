from collections.abc import Awaitable
from huggingface_hub import User
from pydantic import BaseModel, PrivateAttr
from buttermilk._core.contract import (
    COMMAND_SYMBOL,
    AssistantMessage,
    ConductorRequest,
    ConductorResponse,
    ManagerMessage,
    ManagerRequest,
    ManagerResponse,
    OOBMessages,
    StepRequest,
    UserMessage,
)
from buttermilk.agents.llm import LLMAgent

from typing import Any, AsyncGenerator, Callable, Optional, Self, Union
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

TRUNCATE_LEN = 1000  # characters per history message
class HostAgent(LLMAgent):
    """Coordinators for group chats that use an LLM. Can act as the 'beat' to regulate flow."""

    _input_callback: Any = PrivateAttr(...)
    _pending_agent_id: str | None = PrivateAttr(default=None) # Track agent waiting for signal

    _output_model: Optional[type[BaseModel]] = StepRequest
    _message_types_handled: type[Any] = PrivateAttr(default=Union[ConductorRequest])

    async def initialize(self, input_callback: Callable[..., Awaitable[None]] | None = None, **kwargs) -> None:
        """Initialize the agent"""
        self._input_callback = input_callback
        await super().initialize(**kwargs) # Call parent initialize if needed

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        cancellation_token: CancellationToken = None,
        public_callback: Callable = None,
        message_callback: Callable = None,
        source: str = "unknown",
        **kwargs,
    ) -> None:
        # Log messages to our local context cache, but truncate them

        if isinstance(message, (AgentOutput, ConductorResponse)):
            await self._model_context.add_message(AssistantMessage(content=str(message.content)[:TRUNCATE_LEN], source=source))
        else:
            if not message.content.startswith(COMMAND_SYMBOL):
                await self._model_context.add_message(UserMessage(content=str(message.content)[:TRUNCATE_LEN], source=source))

    async def _handle_control_message(
        self,
        message: OOBMessages,
        cancellation_token: CancellationToken = None,
        public_callback: Callable = None,
        message_callback: Callable = None,
        **kwargs,
    ) -> OOBMessages | None:

        # --- Handle Task Completion from Worker Agents ---
        if isinstance(message, TaskProcessingComplete):
            logger.info(f"Host received TaskComplete from {message.role} (Task {message.task_index}, More: {message.more_tasks_remain})")
        else:
            logger.debug(f"Host received unhandled OOB message type: {type(message)}")
        return None
