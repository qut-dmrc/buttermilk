"""Defines the LLMHostAgent, an LLM-powered agent for dynamic flow control.

This module provides `LLMHostAgent`, which extends both `LLMAgent` and `HostAgent`.
It uses its own Language Model to decide which participant agent to call upon next,
based on the conversation history and user feedback. It dynamically creates tools
representing the available participant agents for its LLM to choose from.
"""

import asyncio
from collections.abc import AsyncGenerator, Callable # For type hinting
from enum import StrEnum # For creating string-based Enums for dynamic tool generation
from typing import Any, Type # For type hinting

from autogen_core import CancellationToken # Autogen cancellation token
from autogen_core.tools import FunctionTool # Autogen FunctionTool for LLM tool use
from pydantic import BaseModel, Field, PrivateAttr, field_validator # Pydantic components

from buttermilk._core import StepRequest # Core StepRequest message
from buttermilk._core.agent import ManagerMessage # ManagerMessage for user interaction
from buttermilk._core.constants import END, MANAGER # Buttermilk constants
from buttermilk._core.contract import AgentInput, GroupchatMessageTypes # Buttermilk message contracts
from buttermilk.agents.flowcontrol.host import HostAgent # Base HostAgent class
from buttermilk.agents.llm import LLMAgent # Base LLMAgent class

TRUNCATE_LEN = 1000 # Max characters for logging history messages
"""Maximum length for individual message content when logging to history."""


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
        return v # Should ideally raise error if v is empty, but Pydantic handles required


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

    _output_model: Type[BaseModel] | None = CallOnAgent
    """The Pydantic model that this agent's LLM is expected to output,
    guiding its decision on which agent to call next and with what prompt.
    """
    _user_feedback: list[str] = PrivateAttr(default_factory=list)
    """Stores a list of recent feedback messages from the user, used as context
    for this agent's LLM."""

    max_user_confirmation_time: int = Field(
        default=7200, # Increased from HostAgent's default
        description="Maximum time in seconds to wait for various user interactions or agent responses.",
    )

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
            role=MANAGER, # Target the user interface/manager agent
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
            # This part means the default _run_flow loop in HostAgent might
            # just keep sleeping if not interrupted by user actions leading to an END.
            # Or, if _run_flow expects this generator to yield actual steps for other agents.
            # Given _listen triggers _process, this generator might not be the primary
            # source of StepRequests after the initial greeting.
            # Consider if a WAIT step should be yielded or if the generator should end.
            # For now, match original logic of sleeping.
            await asyncio.sleep(5) # Effectively pauses this sequence.
            # If we want HostAgent's loop to continue but wait for LLM-driven steps:
            # yield StepRequest(role=WAIT, content="LLMHost is waiting for user input to determine next action.")
            continue # Keep the generator alive if HostAgent's loop expects more.

        # This part is likely unreachable due to the `while True` loop.
        # yield StepRequest(role=END, content="LLMHostAgent sequence finished (should not happen).")


    async def _listen(
        self,
        message: GroupchatMessageTypes,
        *,
        cancellation_token: CancellationToken, # Standard arg
        source: str = "",                   # Standard arg
        public_callback: Callable[[Any], Awaitable[None]], # Standard arg for publishing
        message_callback: Callable[[Any], Awaitable[None]],# Standard arg (less used here)
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
            public_callback=public_callback, # Pass through callbacks
            message_callback=message_callback,
            **kwargs,
        )

        if isinstance(message, ManagerMessage):
            # This message is from the user/manager.
            # The LLMHostAgent now needs to decide what to do based on this input
            # and the current state of the conversation (participants, history).
            
            if not self.callback_to_groupchat:
                logger.error(f"LLMHostAgent '{self.agent_id}': callback_to_groupchat is not set. Cannot dispatch next step.")
                return

            participant_names = list(self._participants.keys())
            if not participant_names:
                logger.warning(f"LLMHostAgent '{self.agent_id}': No participants defined. Cannot create tools for LLM decision.")
                # Potentially send a message back to user or end.
                await self.callback_to_groupchat(StepRequest(role=MANAGER, content="There are no participant agents available to call."))
                return

            # Dynamically create an Enum for participant roles for type safety in the tool schema
            # Ensure names are valid Enum member names (e.g., no spaces, not starting with numbers if not quoted)
            valid_enum_participants = {name: name for name in participant_names if name.isidentifier()}
            if not valid_enum_participants:
                 logger.warning(f"LLMHostAgent '{self.agent_id}': No valid participant names for Enum creation. Participants: {participant_names}")
                 await self.callback_to_groupchat(StepRequest(role=MANAGER, content="No valid participant agents to call."))
                 return

            RoleEnumType = StrEnum("RoleEnumType", valid_enum_participants)

            async def _call_on_agent(role: RoleEnumType, prompt: str) -> str: # type: ignore[valid-type]
                """Dynamically created tool for the LLMHostAgent's LLM to call another agent.
                
                This function is intended to be "called" by the LLMHostAgent's own LLM
                as a structured output (tool call). When executed, it constructs a
                `StepRequest` and sends it to the specified participant agent.

                Args:
                    role: The role of the participant agent to call (selected by the LLM
                          from the dynamically generated `RoleEnumType`).
                    prompt: The prompt/instruction to send to the selected agent
                            (generated by the LLM).
                
                Returns:
                    str: A confirmation message indicating the action was dispatched.
                """
                if not self.callback_to_groupchat: # Should be set, but defensive check
                    err_msg = "LLMHostAgent: callback_to_groupchat not available to dispatch StepRequest."
                    logger.error(err_msg)
                    return f"Error: {err_msg}"

                # Create a StepRequest for the chosen agent and prompt
                step_to_dispatch = StepRequest(role=role.value, inputs={"prompt": prompt}, content=f"Responding to user: {prompt}")
                
                logger.info(f"LLMHostAgent '{self.agent_id}': LLM decided to call agent '{role.value}' with prompt: '{prompt[:100]}...'")
                await self.callback_to_groupchat(step_to_dispatch)
                return f"Successfully called agent '{role.value}' with the prompt."

            # Create a FunctionTool from the _call_on_agent function.
            # This tool will be available to the LLMHostAgent's own LLM.
            self._tools_list = [
                FunctionTool(
                    _call_on_agent,
                    description="Selects and calls upon another specialized agent to perform an action or answer a question based on the ongoing conversation and user feedback."
                )
            ]

            # Prepare AgentInput for this LLMHostAgent's own _process method.
            # The input should guide the LLM to use the _call_on_agent tool.
            # self._user_feedback is populated by the super()._listen method.
            llm_host_input_data = {
                "user_query_or_feedback": " ".join(self._user_feedback), # Concatenate recent user feedback
                "available_participant_roles": participant_names,
                # Could also include a summary of conversation history from self._model_context
            }
            # Clear user feedback after consuming it for this turn, or manage it as a window.
            self._user_feedback.clear() 
            
            input_for_llm_host = AgentInput(
                inputs=llm_host_input_data,
                # The prompt for LLMHostAgent's LLM is defined by its `prompt_template`
                # which should instruct it to use the `_call_on_agent` tool.
            )

            # Call this agent's own LLM (_process method from LLMAgent)
            # The LLM's response, if it's a tool call to `_call_on_agent`,
            # will trigger the execution of `_call_on_agent` above, which then
            # sends the StepRequest to the target participant.
            # The result of self._process here would be an AgentTrace from the LLMHostAgent's own LLM call.
            # This trace might contain the tool call request and result.
            # This result is not directly sent out by _listen; _listen's job is to trigger actions.
            await self._process(message=input_for_llm_host) # Result can be logged or handled if needed
            logger.debug(f"LLMHostAgent '{self.agent_id}': Finished processing ManagerMessage via its own LLM.")
