"""Defines the LLMHostAgent, an LLM-powered agent for dynamic flow control.

This module provides `LLMHostAgent`, which extends both `LLMAgent` and `HostAgent`.
It uses its own Language Model to decide which participant agent to call upon next,
based on the conversation history and user feedback. It dynamically creates tools
representing the available participant agents for its LLM to choose from.
"""
from collections.abc import AsyncGenerator, Awaitable, Callable  # For type hinting
from enum import StrEnum  # For creating string-based Enums for dynamic tool generation
from typing import Any  # For type hinting

from autogen_core import CancellationToken  # Autogen cancellation token
from autogen_core.tools import FunctionTool  # Autogen FunctionTool for LLM tool use
from pydantic import BaseModel, Field, PrivateAttr, field_validator  # Pydantic components

from buttermilk._core import StepRequest  # Core StepRequest message
from buttermilk._core.agent import ManagerMessage  # ManagerMessage for user interaction
from buttermilk._core.constants import MANAGER  # Buttermilk constants
from buttermilk._core.contract import AgentInput, GroupchatMessageTypes  # Buttermilk message contracts
from buttermilk._core.log import logger
from buttermilk.agents.flowcontrol.host import HostAgent  # Base HostAgent class
from buttermilk.agents.llm import LLMAgent  # Base LLMAgent class
from buttermilk.utils._tools import create_tool_functions

TRUNCATE_LEN = 1000  # Max characters for logging history messages
"""Maximum length for individual message content when logging to history."""


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


class CallOnAgentArgs(BaseModel):
    """A Pydantic model representing a structured request for the LLMHostAgent's LLM.

    When the `LLMHostAgent` uses its own LLM to decide the next step, it expects
    the LLM to generate output that can be parsed into this model. This model
    indicates which participant agent (`role`) should be called next and with
    what `prompt`.

    Attributes:
        role (str): The role name (uppercase) of the participant agent to be called.
            This must be one of the roles known to the `LLMHostAgent` via its
            `_participants` list.
        prompt (str): The prompt or instruction to be sent to the selected
            participant agent.

    """

    role: str = Field(
        ...,
        description="The role of the agent to call on. This should be one of the available participant roles.",
    )
    prompt: str = Field(
        ...,
        description="The specific prompt or instruction to send to the selected agent.",
    )


class CallOnAgent(BaseModel):
    """A Pydantic model representing a structured request for the LLMHostAgent's LLM.

    When the `LLMHostAgent` uses its own LLM to decide the next step, it expects
    the LLM to generate output that can be parsed into this model. This model
    indicates which participant agent (`role`) should be called next and with
    what `prompt`.

    Attributes:
        role (str): The role name (uppercase) of the participant agent to be called.
            This must be one of the roles known to the `LLMHostAgent` via its
            `_participants` list.
        prompt (str): The prompt or instruction to be sent to the selected
            participant agent.

    """

    role: str = Field(
        ...,
        description="The role of the participant agent to call. Must be a valid role name from the current participants list.",
    )
    prompt: str = Field(
        ...,
        description="The specific prompt or instruction to send to the selected agent.",
    )

    @field_validator("role")
    @classmethod
    def _role_must_be_uppercase(cls, v: str) -> str:
        """Ensures the `role` field is always in uppercase for consistency."""
        if v:
            return v.upper()
        return v  # Should ideally raise error if v is empty, but Pydantic handles required


class LLMHostAgent(LLMAgent, HostAgent):
    """An LLM-powered host agent that dynamically decides which agent to call next.

    This agent combines the capabilities of an `LLMAgent` (having its own LLM for
    decision making) and a `HostAgent` (managing a flow with multiple participants).
    When it receives input (typically from a user via a `ManagerMessage`), it uses
    its LLM to determine which of the available participant agents should be
    activated next and with what prompt.

    It achieves this by:
    1.  Maintaining a list of current participant agents (`self._participants`).
    2.  Dynamically creating an LLM tool (`_call_on_agent`) on-the-fly. The schema
        for this tool includes an Enum of the current participant roles, allowing
        the LLM to choose a valid role.
    3.  When user input is received via `_listen`, it invokes its own `_process`
        method (from `LLMAgent`). The prompt for this call includes the user
        feedback and the list of participants.
    4.  The LLM's response is expected to be a structured output conforming to
        `CallOnAgent`, indicating the chosen `role` and `prompt`.
    5.  The `_call_on_agent` tool (when "called" by the LLM's structured output)
        then constructs a `StepRequest` and sends it to the chosen participant
        via `self.callback_to_groupchat`.

    Configuration Parameters (from `AgentConfig.parameters` or direct attributes):
        - `model` (str): **Required (from LLMAgent)**. The LLM used by this HostAgent
          for its own decision-making.
        - `prompt_template` (str): **Required (from LLMAgent)**. The prompt template
          that guides this HostAgent's LLM to select the next agent and formulate
          a prompt for it, expecting output as `CallOnAgent`.
        - `max_user_confirmation_time` (int): Inherited from `HostAgent`, but can be
          overridden. Maximum time to wait for user responses if `human_in_loop`
          is active (though `LLMHostAgent`'s primary interaction is via `_listen`
          processing `ManagerMessage`). Default: 7200 seconds.

    Attributes:
        _output_model (Type[BaseModel] | None): Specifies `CallOnAgent` as the
            expected Pydantic model for this agent's own LLM's structured output.
        _user_feedback (list[str]): Stores recent user feedback messages, used
            to inform the LLM's decision for the next step.

    """

    _output_model: type[BaseModel] | None = CallOnAgent
    """The Pydantic model that this agent's LLM is expected to output,
    guiding its decision on which agent to call next and with what prompt.
    """
    _user_feedback: list[str] = PrivateAttr(default_factory=list)
    """Stores a list of recent feedback messages from the user, used as context
    for this agent's LLM."""

    max_user_confirmation_time: int = Field(
        default=7200,  # Increased from HostAgent's default
        description="Maximum time in seconds to wait for various user interactions or agent responses.",
    )

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
        """Generates the initial sequence of steps for the LLMHostAgent.

        This overridden method defines a simple startup sequence:
        1.  Sends a greeting message to the `MANAGER` (user interface) asking for
            the initial task or question.
        2.  Enters an indefinite loop, yielding `asyncio.sleep(5)` periodically.
            The actual flow progression is driven by user input received in the
            `_listen` method, which then triggers this agent's LLM to decide
            the next step.

        Yields:
            StepRequest: Initially, a `StepRequest` for the `MANAGER`. Subsequently,
            it effectively pauses, as further steps are determined by LLM responses
            to user input.

        """
        # 1. Greet the user and ask for initial input.
        yield StepRequest(
            role=MANAGER,  # Target the user interface/manager agent
            content="Hi! What would you like to do today?",
        )

        # 2. Enter a waiting loop.
        # The HostAgent's main loop in _run_flow will call this generator.
        # After the initial greeting, this LLMHostAgent waits for user input
        # (ManagerMessage) via its _listen method. The _listen method then
        # calls this agent's _process (LLM call) to decide the next actual step.
        # The _execute_step method in HostAgent will handle sending the StepRequest
        # that results from that LLM call.
        # This loop just keeps the _sequence generator alive if the HostAgent's
        # main loop tries to iterate it further after the initial greeting.
        # In practice, after the first yield, control flow is mostly event-driven by _listen.
        while True:
            # wait for the _listen method to add a proposed step to the queue
            # This is a blocking call, so it will wait until a message is received
            task = await self._proposed_step.get()
            yield task
            continue

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        *,
        cancellation_token: CancellationToken,  # Standard arg
        source: str = "",                   # Standard arg
        public_callback: Callable[[Any], Awaitable[None]],  # Standard arg for publishing
        message_callback: Callable[[Any], Awaitable[None]],  # Standard arg (less used here)
        **kwargs: Any,                      # Standard arg
    ) -> None:
        """Listens to group chat messages, processes user input (`ManagerMessage`), and triggers LLM decision.

        This method overrides the base `HostAgent._listen`. It first calls `super()._listen`
        to handle common message logging and user confirmation state updates.

        If the incoming `message` is a `ManagerMessage` (typically user input):
        1.  It dynamically creates a list of participant roles known to this host
            (from `self._participants`).
        2.  It defines an asynchronous function `_call_on_agent` which, when called,
            will construct a `StepRequest` for a specified participant role and prompt,
            and send it to the group chat using `self.callback_to_groupchat`.
        3.  This `_call_on_agent` function is wrapped into an Autogen `FunctionTool`.
            This tool is made available to this `LLMHostAgent`'s own LLM.
        4.  It then invokes its own `_process` method (inherited from `LLMAgent`).
            The input to `_process` includes the accumulated user feedback
            (`self._user_feedback`) and the list of current participants. The LLM is
            expected (via its prompt and the `_output_model = CallOnAgent`) to
            "call" the dynamically created `_call_on_agent` tool, effectively
            deciding which participant agent to engage next and with what prompt.

        Args:
            message: The incoming message from the group chat.
            cancellation_token: Token for cancellation.
            source: Identifier of the message sender.
            public_callback: Callback for publishing messages.
            message_callback: Callback for specific message topics.
            **kwargs: Additional keyword arguments.

        """
        # First, let the base HostAgent's _listen handle common tasks like
        # updating user feedback, confirmation events, and logging to history.
        await super()._listen(
            message=message,
            cancellation_token=cancellation_token,
            source=source,
            public_callback=public_callback,  # Pass through callbacks
            message_callback=message_callback,
            **kwargs,
        )

        if isinstance(message, ManagerMessage):
            # If the message is from the manager, we need to process it
            # Unless it's a command for another agent or something
            if message.content and str(message.content).startswith(COMMAND_SYMBOL):
                return

            # Call the LLM to get the next step. Include the user feedback and participants in the input.
            result = await self.invoke(message=AgentInput(inputs={"user_feedback": self._user_feedback, "prompt": str(message.content), "participants": self._participants}), public_callback=public_callback, message_callback=message_callback, cancellation_token=cancellation_token, **kwargs)

            if result:
                # If the result is not None, we have a response from the LLM
                # Check if the result is a CallOnAgent object -- and convert it to StepRequest
                if isinstance(result, AgentTrace) and isinstance(result.outputs, CallOnAgent):
                    # Create a new message for the agent
                    choice = StepRequest(role=result.outputs.role, inputs={"prompt": result.outputs.prompt})
                    logger.info(f"Host {self.agent_name} calling on agent: {result.outputs.role} with prompt: {result.outputs.prompt}")
                    # add to our queue
                    await self._proposed_step.put(choice)
                elif isinstance(result, StepRequest):
                    await self._proposed_step.put(result)
                else:
                    # If the result is not a step request of some kind, we can just send it to the group chat
                    await self.callback_to_groupchat(result)
