from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from typing import Any, Self

import pydantic
import weave
from autogen_agentchat.base import Response
from autogen_agentchat.messages import (  # noqa
    BaseAgentEvent,
    BaseChatMessage,
    ChatMessage,
    HandoffMessage,
    MultiModalMessage,
    StopMessage,
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
    UserInputRequestedEvent,
)
from autogen_core import CancellationToken
from autogen_core.model_context import (
    UnboundedChatCompletionContext,
)
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
)

from buttermilk import logger
from buttermilk._core.config import DataSource
from buttermilk._core.contract import (
    AgentInput,
    AgentOutput,
    UserInstructions,
)
from buttermilk._core.exceptions import FatalError, ProcessingError


class ToolConfig(BaseModel):
    id: str
    name: str
    description: str
    tool_obj: str | None = None

    data_cfg: list[DataSource] = Field(
        default=[],
        description="Specifications for data that the Agent should load",
    )

    def _run(self, *args, **kwargs):
        raise NotImplementedError


#########
# Agent
#
# A simple class with a function that process input.
#
#
##########


class AgentConfig(BaseModel):
    agent_obj: str = Field(
        default="",
        description="The object name to instantiate",
        exclude=True,
    )
    id: str = Field(
        ...,
        description="The unique name of this agent.",
    )
    name: str = Field(
        ...,
        description="A human-readable name for this agent.",
    )
    description: str = Field(
        ...,
        description="Short explanation of what this agent type does",
    )
    tools: list[ToolConfig] = Field(
        default=[],
        description="Tools the agent can invoke",
    )
    data: list[DataSource] = Field(
        default=[],
        description="Specifications for data that the Agent should load",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Initialisation parameters to pass to the agent",
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="A mapping of data to agent inputs",
    )
    outputs: dict[str, Any] = {}
    model_config = {
        "extra": "allow",
        "arbitrary_types_allowed": False,
        "populate_by_name": True,
    }


class Agent(AgentConfig, ABC):
    """Base Agent interface for all processing units"""

    _trace_this = True
    _run_fn: Callable | Awaitable = PrivateAttr()

    _model_context: UnboundedChatCompletionContext = PrivateAttr(
        default_factory=UnboundedChatCompletionContext,
    )

    @pydantic.model_validator(mode="after")
    def _get_process_func(self) -> Self:
        """Returns the appropriate processing function based on tracing setting."""

        def _process_fn():
            if self._trace_this:
                return weave.op(self._process, call_display_name=f"{self.id}")
            return self._process

        self._run_fn = _process_fn()
        return self

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (BaseChatMessage, MultiModalMessage)

    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> Response:
        final_response = None
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                final_response = message

        if final_response is not None:
            return final_response
        raise AssertionError("The stream should have returned the final result.")

    @abstractmethod
    async def on_messages_stream(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage, None]:
        # Add messages to the model context
        for msg in messages:
            await self._model_context.add_message(msg.to_model_message())

        # inputs = AgentIntput(...)
        response = await self._process(input_data=inputs)
        # # Create usage metadata
        # usage = RequestUsage(
        #     prompt_tokens=response.usage_metadata.prompt_token_count,
        #     completion_tokens=response.usage_metadata.candidates_token_count,
        # )

        # # Add response to model context
        # await self._model_context.add_message(
        #     AssistantMessage(content=response.text, source=self.name),
        # )

        # # Yield the final response
        # yield Response(
        #     chat_message=TextMessage(
        #         content=response.text,
        #         source=self.name,
        #         models_usage=usage,
        #     ),
        #     inner_messages=[],
        # )

    @abstractmethod
    async def _process(self, input_data: AgentInput, **kwargs) -> AgentOutput | None:
        """Process input data and return output

        Inputs:
            input_data: AgentInput with appropriate values required by the agent.

        Outputs:
            AgentOutput record with processed data or non-null Error field.

        """
        raise NotImplementedError

    async def handle_control_message(
        self,
        message: Any,
    ) -> None:
        """Most agents don't deal with control messages."""
        logger.debug(f"Agent {self.id} {self.name} dropping control message: {message}")

    async def __call__(
        self,
        input_data: AgentInput,
        **kwargs,
    ) -> AgentOutput | UserInstructions | None:
        """Allow agents to be called directly as functions"""
        try:
            return await self._run_fn(input_data, **kwargs)
        except ProcessingError as e:
            logger.error(
                f"Agent {self.id} {self.name} hit error: {e}. Task content: {input_data.content[:100]}",
            )
            return None
        except FatalError as e:
            raise e

    async def initialize(self, **kwargs) -> None:
        """Initialize the agent"""

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass
