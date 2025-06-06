import asyncio
from collections.abc import AsyncGenerator, Callable
from enum import StrEnum
from typing import Any

from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
from pydantic import BaseModel, Field, PrivateAttr, field_validator

from buttermilk._core import AgentTrace, StepRequest
from buttermilk._core.agent import ManagerMessage
from buttermilk._core.constants import COMMAND_SYMBOL, END, MANAGER
from buttermilk._core.contract import AgentInput, GroupchatMessageTypes
from buttermilk._core.log import logger
from buttermilk.agents.flowcontrol.host import HostAgent
from buttermilk.agents.llm import LLMAgent
from buttermilk.utils._tools import create_tool_functions

TRUNCATE_LEN = 1000


class CallOnAgentArgs(BaseModel):
    """Arguments for the _call_on_agent tool."""

    role: str = Field(
        ...,
        description="The role of the agent to call on. This should be one of the available participant roles.",
    )
    prompt: str = Field(
        ...,
        description="The prompt to send to the agent.",
    )


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
    _proposed_step: asyncio.Queue[StepRequest] = PrivateAttr(default_factory=asyncio.Queue)
    max_user_confirmation_time: int = Field(
        default=7200,
        description="Maximum time to wait for agent responses in seconds",
    )

    def _clear_pending_steps(self) -> None:
        """Clear all pending steps from the queue."""
        while not self._proposed_step.empty():
            try:
                self._proposed_step.get_nowait()
                logger.debug("Cleared pending step from queue due to new manager request")
            except asyncio.QueueEmpty:
                break

    async def _initialize(self, callback_to_groupchat: Any) -> None:
        await super()._initialize(callback_to_groupchat=callback_to_groupchat)

        # Assemble our list of participants as tools
        participant_names = list(self._participants.keys())
        # Define the enum type dynamically
        RoleEnumType = StrEnum("RoleEnumType", {name: name for name in participant_names})

        async def _call_on_agent(role: RoleEnumType, prompt: str) -> None:
            """Call on another agent to perform an action."""
            # Create a new message for the agent
            choice = StepRequest(role=role, inputs={"prompt": prompt})
            logger.info(f"Host {self.agent_name} calling on agent: {role} with prompt: {prompt}")
            # Send the message to the agent
            await self.callback_to_groupchat(choice)

        tool = FunctionTool(
            func=_call_on_agent,
            description="Call on another agent to perform an action.",
        )
        self._tools_list = []
        if self.tools:
            # reinitialize the tools list
            self._tools_list = create_tool_functions(self.tools)
        self._tools_list.append(tool)

    async def _sequence(self) -> AsyncGenerator[StepRequest, None]:
        """Generate a sequence of steps to execute."""
        # First, say hello to the user
        await asyncio.sleep(3)  # we need to let the group chat initialize
        yield StepRequest(
            role=MANAGER,
            content="Hi! What would you like to do?",
        )
        while True:
            # wait for the _listen method to add a proposed step to the queue
            # This is a blocking call, so it will wait until a message is received
            task = await self._proposed_step.get()
            yield task
            # Check if this is an END task to break the loop
            if task.role == END:
                break

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
            # Unless it's a command for another agent or something
            if message.content and str(message.content).startswith(COMMAND_SYMBOL):
                return

            # Skip processing if message content is None or empty
            if not message.content:
                logger.debug(f"Manager message received with empty content, skipping processing: {message}")
                return

            # Clear any pending steps since the manager has a new request
            self._clear_pending_steps()

            logger.info(f"Manager interrupted with new request: {message.content}")
            # Call the LLM to get the next step. Include the user feedback and participants in the input.
            result = await self.invoke(message=AgentInput(inputs={"user_feedback": self._user_feedback, "prompt": str(message.content), "participants": self._participants}), public_callback=public_callback, message_callback=message_callback, cancellation_token=cancellation_token, **kwargs)

            if result:
                # If the result is not None, we have a response from the LLM
                # Check if the result is a CallOnAgent object -- and convert it to StepRequest
                if isinstance(result, AgentTrace) and isinstance(result.outputs, CallOnAgent):
                    # Create a new message for the agent
                    explanation = f"Proposing to call on agent {result.outputs.role} with prompt: {result.outputs.prompt}"
                    choice = StepRequest(role=result.outputs.role, inputs={"prompt": result.outputs.prompt}, content=explanation)
                    logger.info(f"Host {self.agent_name} interpreted manager request. {explanation}")
                    # add to our queue
                    await self._proposed_step.put(choice)
                elif isinstance(result, StepRequest):
                    await self._proposed_step.put(result)
                else:
                    # If the result is not a step request of some kind, we can just send it to the group chat
                    await self.callback_to_groupchat(result)
