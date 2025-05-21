
import asyncio
from collections.abc import AsyncGenerator, Callable
from enum import StrEnum
from typing import Any

from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
from pydantic import BaseModel, Field, PrivateAttr, field_validator

from buttermilk._core import StepRequest
from buttermilk._core.agent import ManagerMessage
from buttermilk._core.constants import END, MANAGER
from buttermilk._core.contract import AgentInput, GroupchatMessageTypes
from buttermilk.agents.flowcontrol.host import HostAgent
from buttermilk.agents.llm import LLMAgent

TRUNCATE_LEN = 1000


class CallOnAgent(BaseModel):
    """A request from a flow control agent for action from another agent.
    """

    role: str = Field(
        ...,
        description="The role of the agent to call on. This should be a valid role name.",
    )
    prompt: str = Field(
        ...,
        description="The prompt to send to the agent.",
    )

    @field_validator("role")
    @classmethod
    def _role_must_be_uppercase(cls, v: str) -> str:
        """Ensures the role field is always uppercase for consistency."""
        if v:
            return v.upper()
        return v


class LLMHostAgent(LLMAgent, HostAgent):
    """An agent that can call on other agents to perform actions.
    """

    _output_model: type[BaseModel] | None = CallOnAgent
    _user_feedback: list[str] = PrivateAttr(default_factory=list)

    max_user_confirmation_time: int = Field(
        default=7200,
        description="Maximum time to wait for agent responses in seconds",
    )

    async def _sequence(self) -> AsyncGenerator[StepRequest, None]:
        """Generate a sequence of steps to execute."""
        # First, say hello to the user
        yield StepRequest(
            role=MANAGER,
            content="Hi! What would you like to do?",
        )
        while True:
            # do nothing; we'll handle steps through the _listen method
            await asyncio.sleep(5)
            continue

        # This will never be reached, but is here for completeness
        yield StepRequest(role=END, content="End of sequence")

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        *,
        cancellation_token: CancellationToken,
        source: str = "",
        public_callback: Callable,
        message_callback: Callable,
        **kwargs: Any,
    ) -> None:
        """Listen to messages in the group chat respond to the manager."""
        # Save messages to context
        await super()._listen(
            message=message,
            cancellation_token=cancellation_token,
            source=source,
            public_callback=public_callback,
            message_callback=message_callback,
            **kwargs,
        )
        if isinstance(message, ManagerMessage):
            # If the message is from the manager, we need to process it

            # Assemble our list of participants as tools

            participant_names = list(self._participants.keys())
            RoleEnumType = StrEnum("RoleEnumType", {name: name for name in participant_names})

            async def _call_on_agent(role: RoleEnumType, prompt: str) -> None:
                """Call on another agent to perform an action."""
                # Create a new message for the agent
                choice = StepRequest(role=role, inputs={"prompt": prompt})
                # Send the message to the agent
                await self.callback_to_groupchat(choice)

            self._tools_list = [FunctionTool(_call_on_agent, description="Call on another agent to perform an action.")]

            # With user feedback, call the LLM to get the next step
            result = await self._process(message=AgentInput(inputs={"user_feedback": self._user_feedback, "participants": self._participants}))

            if result:
                # If the result is not None, we have a response from the LLM
                # Send the result to the group chat
                await self.callback_to_groupchat(result)
