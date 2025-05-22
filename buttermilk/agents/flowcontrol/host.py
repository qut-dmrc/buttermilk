"""Provides the HostAgent, a coordinator for managing group chats and flow control.

The `HostAgent` acts as a conductor within a Buttermilk flow, typically assigned
the `CONDUCTOR` role. Its primary responsibility is to manage the sequence of
interactions between other participating agents. It can iterate through a predefined
sequence of agent roles, dispatch tasks (as `StepRequest` messages), and wait for
their completion before proceeding.

A key feature of the `HostAgent` is its ability to synchronize operations by
tracking pending tasks from other agents. It uses an `asyncio.Condition` to pause
its own execution until all agents involved in a particular step have signaled
completion (via `TaskProcessingComplete` messages).

The agent can also interact with a human user via a "MANAGER" interface if
`human_in_loop` is enabled, requesting confirmation before executing steps.
"""

import asyncio
from collections import defaultdict
from collections.abc import AsyncGenerator, Callable  # For type hinting
from typing import Any  # For type hinting

from autogen_core import CancellationToken  # Autogen cancellation token
from autogen_core.models import AssistantMessage, UserMessage  # Autogen message types
from pydantic import Field, PrivateAttr  # Pydantic components

from buttermilk import logger  # Centralized logger
from buttermilk._core.agent import Agent  # Buttermilk base Agent class
from buttermilk._core.constants import COMMAND_SYMBOL, END, WAIT  # Buttermilk constants
from buttermilk._core.contract import (  # Buttermilk message contracts
    AgentInput,
    AgentOutput,
    AgentTrace,  # Although not directly returned by _process, it's part of GroupchatMessageTypes
    ConductorRequest,
    ErrorEvent,
    FlowProgressUpdate,
    GroupchatMessageTypes,
    ManagerMessage,
    OOBMessages,  # Union type for Out-Of-Band messages
    StepRequest,
    TaskProcessingComplete,
    TaskProcessingStarted,
    UIMessage,
)
from buttermilk._core.exceptions import FatalError  # Custom Buttermilk exceptions

TRUNCATE_LEN = 1000  # Max characters for logging history messages
"""Maximum length for individual message content when logging to history."""


class HostAgent(Agent):
    """A coordinator agent for managing group chats and controlling workflow steps.

    The `HostAgent` typically acts as a "conductor" or "host" in a multi-agent
    system. Its main responsibilities include:
    -   Receiving a `ConductorRequest` which outlines the participants (other agents)
        and potentially the initial state or goal.
    -   Iterating through a sequence of steps, often corresponding to different
        participant roles.
    -   Dispatching `StepRequest` messages to the appropriate participant agent for each step.
    -   Synchronizing execution by waiting for all tasks initiated in a step to
        complete (using `TaskProcessingStarted` and `TaskProcessingComplete` events)
        before moving to the next step.
    -   Optionally, interacting with a human user (via a "MANAGER" agent) for
        step confirmation if `human_in_loop` is enabled.
    -   Maintaining a conversation history (`_model_context`) by listening to
        messages from other agents.

    Configuration Parameters (from `AgentConfig.parameters` or direct attributes):
        - `max_wait_time` (int): Maximum time in seconds to wait for all tasks in a
          step to complete before timing out. Default: 240.
        - `max_user_confirmation_time` (int): Maximum total time in seconds to wait
          for user confirmation if `human_in_loop` is True. Default: 240.
        - `human_in_loop` (bool): If True, the agent will request user confirmation
          (via a `UIMessage` to a MANAGER) before executing each step. Default: True.
        - `participants` (dict): This is typically provided via a `ConductorRequest`
          at the start of `_run_flow`, defining the roles and configurations of
          agents involved in the flow.

    Internal State Attributes:
        _message_types_handled (Type[Any]): The primary message type this agent's
            `_process` method is designed to handle (defaults to `ConductorRequest`).
        callback_to_groupchat (Callable | None): An asynchronous callback function
            provided during initialization, used to send messages (like `StepRequest`
            or `UIMessage`) to the broader group chat or communication channel.
        _step_generator (AsyncGenerator[StepRequest, None] | None): An asynchronous
            generator that yields the next `StepRequest` in the sequence.
        _tasks_condition (asyncio.Condition): Used to wait for pending tasks to complete.
        _step_starting (asyncio.Event): An event set when a new step is about to be
            dispatched, used in conjunction with `_tasks_condition`.
        _pending_tasks_by_agent (defaultdict[str, int]): Tracks the number of
            outstanding tasks for each agent ID involved in the current step.
        _participants (dict[str, Any]): Stores the participant configurations
            (role names to agent details) for the current flow.
        _conductor_task (asyncio.Task | None): The asyncio Task object for the main
            `_run_flow` execution loop.
        _current_step (str): Stores the role/name of the current step being processed.
        _user_confirmation (ManagerMessage | None): Stores the last received
            confirmation message from the user/MANAGER.
        _user_confirmation_received (asyncio.Event): An event that is set when a
            user confirmation (`ManagerMessage` with `confirm=True`) is received.
        _user_feedback (list[str]): A list to store textual feedback from the user.
    """

    _message_types_handled: type[Any] = PrivateAttr(default=type(ConductorRequest))
    """The primary message type this agent's `_process` method is designed for."""

    callback_to_groupchat: Callable[[Any], Awaitable[None]] | None = Field(
        default=None,
        description="Async callback to send messages to the group chat or UI.",
    )
    """An asynchronous callback function (e.g., provided by an adapter or orchestrator)
    used by the HostAgent to send messages (like `StepRequest` to other agents or
    `UIMessage` to a user interface) into the main communication channel.
    This needs to be set during or after initialization for the agent to communicate.
    """

    _step_generator: AsyncGenerator[StepRequest, None] | None = PrivateAttr(default=None)
    _tasks_condition: asyncio.Condition = PrivateAttr(default_factory=asyncio.Condition)
    _step_starting: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _pending_tasks_by_agent: defaultdict[str, int] = PrivateAttr(default_factory=lambda: defaultdict(int))
    _participants: dict[str, Any] = PrivateAttr(default_factory=dict)
    _conductor_task: asyncio.Task[None] | None = PrivateAttr(default=None)  # type: ignore

    max_wait_time: int = Field(
        default=240,
        description="Maximum time in seconds to wait for all agent responses in a step before timing out.",
    )
    _current_step: str = PrivateAttr(default="")
    """Stores the role/name of the current step being processed, used for progress reporting."""

    max_user_confirmation_time: int = Field(
        default=1220,  # Total time, not per request
        description="Maximum total time in seconds to wait for user confirmation across multiple retries.",
    )
    human_in_loop: bool = Field(
        default=True,
        description="If True, interact with a human/MANAGER for step confirmation.",
    )
    _user_confirmation: ManagerMessage | None = PrivateAttr(default=None)
    _user_confirmation_received: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _user_feedback: list[str] = PrivateAttr(default_factory=list)
    _progress_reporter_task: asyncio.Task | None = PrivateAttr(default=None)

    async def initialize(  # type: ignore[override]
        self,
        callback_to_groupchat: Callable[[Any], Awaitable[None]],
        **kwargs: Any,
    ) -> None:
        """Initializes the HostAgent, primarily by setting the callback for group communication.

        This method should be called before the agent starts its main processing loop
        (e.g., before `_run_flow` is invoked) to ensure it can send messages.

        Args:
            callback_to_groupchat: An asynchronous callable that the HostAgent will use
                to send messages (like `StepRequest` or `UIMessage`) to other
                participants or the UI.
            **kwargs: Additional keyword arguments for future extensions or compatibility.

        """
        await super().initialize(**kwargs)  # Call base Agent's initialize
        self.callback_to_groupchat = callback_to_groupchat
        logger.info(f"HostAgent '{self.agent_id}' initialized with group chat callback.")

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        *,
        cancellation_token: CancellationToken,  # Standard arg
        source: str = "",                   # Standard arg
        public_callback: Callable[[Any], Awaitable[None]],  # Standard arg
        message_callback: Callable[[Any], Awaitable[None]],  # Standard arg
        **kwargs: Any,                      # Standard arg
    ) -> None:
        """Listens to messages from the group chat and updates internal state.

        This method processes incoming messages to:
        - Log textual content from `AgentTrace` (as Assistant) or `ManagerMessage`
          (as User) to its internal conversation history (`_model_context`), truncating
          long messages.
        - Handle `ManagerMessage`:
            - Stores the message in `self._user_confirmation`.
            - Sets `self._user_confirmation_received` event if `message.confirm` is True.
            - Updates `self.human_in_loop` based on `message.human_in_loop`.
            - Appends textual content from `ManagerMessage` (if not a command) to `self._user_feedback`.
        - Ignores `TaskProgressUpdate` messages to avoid logging them to history.

        Args:
            message: The incoming message object from the group chat.
            cancellation_token: Token for cancellation.
            source: Identifier of the message sender.
            public_callback: Callback for publishing messages to a public/default topic.
            message_callback: Callback for publishing messages to the incoming message's topic.
            **kwargs: Additional keyword arguments.

        """
        content_to_log: str | None = None
        msg_type: type[UserMessage | AssistantMessage] = UserMessage  # Default to UserMessage

        if isinstance(message, AgentTrace):
            # Use message.content which is a property returning str(message.outputs)
            content_to_log = str(message.content)[:TRUNCATE_LEN]
            msg_type = AssistantMessage  # Traces are typically from other agents (assistants)
        elif isinstance(message, ManagerMessage):
            logger.info(f"Host '{self.agent_name}' received user input: {message.model_dump_json(indent=2, exclude_none=True)}")
            self._user_confirmation = message
            if message.confirm:
                self._user_confirmation_received.set()  # Signal that confirmation was received

            if message.human_in_loop is not None and self.human_in_loop != message.human_in_loop:
                logger.info(f"Host '{self.agent_name}': User request to set human_in_loop to {message.human_in_loop} (was {self.human_in_loop})")
                self.human_in_loop = message.human_in_loop

            # Get content for logging, preferring 'content' then 'params'
            loggable_content = getattr(message, "content", None)
            if loggable_content is None and message.params:
                loggable_content = str(message.params)  # Stringify params if content is None

            if loggable_content and isinstance(loggable_content, str) and not loggable_content.startswith(COMMAND_SYMBOL):
                content_to_log = loggable_content[:TRUNCATE_LEN]
                self._user_feedback.append(loggable_content)  # Store full feedback

        elif isinstance(message, FlowProgressUpdate):
             logger.debug(f"Host '{self.agent_name}': Received TaskProgressUpdate from '{source}'. Pending tasks: {dict(self._pending_tasks_by_agent)}. Not logging to history.")
             return  # Do not log TaskProgressUpdate to conversation history

        if content_to_log:
            # Add to internal model context for LLM decision making if HostAgent uses an LLM
            await self._model_context.add_message(msg_type(content=content_to_log, source=source))
            logger.debug(f"Host '{self.agent_name}': Logged message from '{source}' to internal context.")
        else:
            logger.debug(f"Host '{self.agent_name}': No loggable content from message type {type(message)} from '{source}'.")

    async def _handle_events(
        self,
        message: OOBMessages,  # Out-Of-Band messages
        cancellation_token: CancellationToken,  # Standard arg
        public_callback: Callable[[Any], Awaitable[None]],  # Standard arg
        message_callback: Callable[[Any], Awaitable[None]],  # Standard arg
        **kwargs: Any,  # Standard arg
    ) -> OOBMessages | None:
        """Handles Out-Of-Band (OOB) messages, primarily for task synchronization and flow control.

        This method processes specific OOB message types:
        -   `TaskProcessingComplete`: Decrements the pending task count for the
            source agent. If all tasks for all agents are complete, it notifies
            any waiters on `self._tasks_condition`.
        -   `TaskProcessingStarted`: Increments the pending task count for the
            source agent.
        -   `ConductorRequest`: If no flow is currently running (`_conductor_task`
            is None or done), it starts a new flow execution by creating an
            asyncio task for `self._run_flow(message)`.
        -   `TaskProgressUpdate`: Logs and ignores these if received by the host itself,
            as they are meant for UI/external observers.

        Args:
            message: The OOB message received.
            cancellation_token: Token for cancellation.
            public_callback: Callback for publishing general messages.
            message_callback: Callback for publishing messages to specific topics.
            **kwargs: Additional arguments.

        Returns:
            OOBMessages | None: Typically `None`, as this method handles events
            internally rather than producing a direct response message.

        """
        logger.debug(f"Host '{self.agent_name}' handling event: {type(message).__name__}")

        if isinstance(message, TaskProcessingComplete):
            # Logic for when a task completes
            # self._step_starting.clear() # This might be too early if multiple tasks start for one step
            agent_id_completed = message.agent_id
            async with self._tasks_condition:
                if agent_id_completed in self._pending_tasks_by_agent:
                    self._pending_tasks_by_agent[agent_id_completed] -= 1
                    log_msg_prefix = f"Host '{self.agent_name}': Received TaskComplete from agent '{agent_id_completed}' (role: '{message.role}')."

                    if self._pending_tasks_by_agent[agent_id_completed] <= 0:
                        if self._pending_tasks_by_agent[agent_id_completed] < 0:
                             logger.warning(f"{log_msg_prefix} Task count for agent went negative. This may indicate an issue.")
                        del self._pending_tasks_by_agent[agent_id_completed]
                        logger.debug(f"{log_msg_prefix} Agent has no more pending tasks.")
                    else:
                         logger.debug(
                            f"{log_msg_prefix} Agent has {self._pending_tasks_by_agent[agent_id_completed]} remaining tasks. "
                            f"(Task index: {message.task_index}, More tasks: {message.more_tasks_remain}, Error: {message.is_error})",
                         )

                    # If all tasks across all agents are now complete, notify waiters.
                    if not self._pending_tasks_by_agent:
                        logger.info(f"Host '{self.agent_name}': All pending tasks completed. Notifying waiters.")
                        self._tasks_condition.notify_all()
                else:
                    logger.warning(
                        f"Host '{self.agent_name}': Received TaskComplete from agent '{agent_id_completed}' (role: '{message.role}'), "
                        "but this agent was not tracked with pending tasks or was already cleared.",
                    )
                logger.debug(f"Host '{self.agent_name}': Current pending tasks: {dict(self._pending_tasks_by_agent)}")

        elif isinstance(message, TaskProcessingStarted):
            # Logic for when a task starts
            agent_id_started = message.agent_id
            async with self._tasks_condition:
                self._pending_tasks_by_agent[agent_id_started] += 1
                logger.debug(
                    f"Host '{self.agent_name}': Noted TaskStarted from agent '{agent_id_started}' (role: '{message.role}'). "
                    f"Pending tasks: {dict(self._pending_tasks_by_agent)}.",
                )

        elif isinstance(message, ConductorRequest):
            # Logic to start or manage the main flow execution task
            if self._conductor_task and not self._conductor_task.done():
                logger.warning(f"Host '{self.agent_name}': Received ConductorRequest but a flow task is already running. Ignoring new request.")
                return None  # Or handle as an error/queue if appropriate

            logger.info(f"Host '{self.agent_name}': Received ConductorRequest, starting new flow execution task.")
            self._conductor_task = asyncio.create_task(self._run_flow(message=message))
            # Consider adding error handling for the task itself, e.g., self._conductor_task.add_done_callback(...)

        elif isinstance(message, FlowProgressUpdate):
             logger.debug(f"Host '{self.agent_name}': Received its own TaskProgressUpdate. This is for external observers. Ignoring.")
             # The HostAgent itself doesn't act on these, they are for external monitoring.

        return None  # Default for OOB messages not producing a direct response

    async def request_user_confirmation(self, step: StepRequest) -> None:
        """Requests confirmation from the user/MANAGER for the proposed next step.

        This method is called when `self.human_in_loop` is True. It clears any
        previous user confirmation state and sends a `UIMessage` (with "confirm"
        and "reject" options) to the group chat, expecting a `ManagerMessage`
        in response.

        Args:
            step (StepRequest): The `StepRequest` object representing the proposed
                next step for which confirmation is sought. The `step.role` is
                used in the confirmation message.

        """
        if not self.callback_to_groupchat:
            logger.error(f"Host '{self.agent_name}': Cannot request user confirmation, callback_to_groupchat is not set.")
            return

        self._user_confirmation_received.clear()  # Reset the event for new confirmation
        self._user_confirmation = None  # Clear previous confirmation message

        confirmation_prompt = f"Confirm next step: Execute step for role '{step.role}'. Description: '{step.content or 'N/A'}'"
        ui_message_request = UIMessage(
            content=confirmation_prompt,
            options=["confirm", "reject"],  # Example options, could be True for simple Yes/No
        )
        logger.debug(f"Host '{self.agent_name}': Requesting user confirmation for step: {step.role}")
        await self.callback_to_groupchat(ui_message_request)

    async def _wait_for_user(self, step: StepRequest) -> bool:
        """Waits for user confirmation for a given step, retrying requests if needed.

        This method repeatedly calls `request_user_confirmation` and waits for
        the `_user_confirmation_received` event to be set (by `_listen` when a
        `ManagerMessage` with `confirm=True` arrives). It retries the request
        every 60 seconds up to `self.max_user_confirmation_time`.

        Args:
            step (StepRequest): The step for which confirmation is being awaited.

        Returns:
            bool: True if user confirmed, False if rejected or timed out

        """
        max_tries = self.max_user_confirmation_time // 60
        for i in range(max_tries):
            logger.debug(f"Host {self.agent_name} waiting for user confirmation for {step.role} step.")
            try:
                await self.request_user_confirmation(step)
                await asyncio.wait_for(self._user_confirmation_received.wait(), timeout=60)
            except TimeoutError:
                logger.warning(f"{self.agent_name} hit timeout waiting for manager response after 60 seconds.")
                continue

            if self._user_confirmation and getattr(self._user_confirmation, "confirm", False):
                logger.info(f"User confirmed step: {step.role}")
                return True
            logger.info(f"User rejected step: {step.role}")
            return False
        msg = "User did not respond to confirm step in time. Ending flow."
        logger.error(msg)
        await self.callback_to_groupchat(StepRequest(role=END, content=msg))
        return False

    async def _report_progress_periodically(self, interval: int = 10) -> None:
        """Periodically sends `TaskProgressUpdate` messages about pending tasks.

        This method runs as an asyncio task while `_wait_for_all_tasks_complete`
        is active. It sleeps for `interval` seconds, then checks `_pending_tasks_by_agent`.
        If there are pending tasks, it constructs and sends a `TaskProgressUpdate`
        message via `self.callback_to_groupchat`.

        The task is cancelled when the main waiting logic in
        `_wait_for_all_tasks_complete` finishes or times out.

        Args:
            interval (int): The time in seconds between progress reports. Defaults to 10.

        """
        try:
            while True:  # Loop indefinitely until cancelled
                await asyncio.sleep(interval)
                async with self._tasks_condition:  # Ensure thread-safe access to pending tasks
                    if self._pending_tasks_by_agent:
                        progress_message = FlowProgressUpdate(source=self.agent_id,
                                                              status="pending",
                                                              step_name=self._current_step,
                            waiting_on=dict(self._pending_tasks_by_agent),
                            message="Current pending tasks",
                        )
                    else:
                         progress_message = FlowProgressUpdate(source=self.agent_id,
                            status="idle",
                            step_name="IDLE",
                            waiting_on=dict(),
                            message="Flow currently idle",
                        )
                    logger.debug(f"Host {self.agent_name} sending progress update: {progress_message.status} {progress_message.waiting_on}")
                    await self.callback_to_groupchat(progress_message)
        except asyncio.CancelledError:
            logger.debug("Progress reporting task cancelled.")
        except Exception as e:
            logger.exception(f"Error in progress reporting task: {e}")
        finally:
            # Send one last update before exiting
            progress_message = FlowProgressUpdate(source=self.agent_id,
                                                    status="finished",
                                                    step_name=END,
                waiting_on=dict(),
                message="Flow finished",
            )
            logger.debug(f"Host {self.agent_name} sending final progress update.")
            await self.callback_to_groupchat(progress_message)

    async def _wait_for_all_tasks_complete(self) -> bool:
        """Wait until all tasks are completed, reporting progress periodically.
        
        This method uses an `asyncio.Condition` (`_tasks_condition`) to wait.
        The condition is met when `_step_starting` event is not set (signifying
        the current step's tasks have been dispatched and potentially started) AND
        the `_pending_tasks_by_agent` dictionary becomes empty.

        While waiting, it starts a periodic task (`_report_progress_periodically`)
        to send updates about the pending tasks.

        Returns:
            bool: `True` if all tasks completed within `self.max_wait_time`,
                  `False` if the wait timed out or an unexpected error occurred.
        
        Note:
            The predicate `lambda: not self._step_starting.is_set() and not self._pending_tasks_by_agent`
            implies that `_step_starting` must be cleared by some other part of the
            logic once tasks for a step are considered fully initiated or have started
            reporting completion.

        """
        try:
            async with self._tasks_condition:
                if self._pending_tasks_by_agent:
                    logger.info(f"Waiting for pending tasks to complete from: {list(self._pending_tasks_by_agent.keys())}...")
                    # Start the periodic progress reporter task
                    progress_reporter_task = asyncio.create_task(self._report_progress_periodically())
                else:
                    logger.info("No pending tasks yet, but we are expecting some this step. Waiting.")
                    # Still start the reporter, it will only send messages if tasks appear
                    progress_reporter_task = asyncio.create_task(self._report_progress_periodically())

                # wait_for releases the lock, waits for notification and predicate, then reacquires
                # The predicate checks if _step_starting is clear AND _pending_tasks_by_agent is empty.
                # This means we wait until the step is no longer considered "starting" AND all tasks are done. This provides insurance where
                # distributed tasks take a while to begin.
                await asyncio.wait_for(
                    self._tasks_condition.wait_for(
                        lambda: not self._step_starting.is_set() and not self._pending_tasks_by_agent,
                    ),
                    timeout=self.max_wait_time,
                )
                logger.info(f"Host '{self.agent_name}': All tasks completed or condition met.")
                return True
        except TimeoutError:
            logger.warning(
                f"Host '{self.agent_name}': Timeout after {self.max_wait_time}s waiting for task completion. "
                f"Pending tasks: {dict(self._pending_tasks_by_agent)}. Step starting event set: {self._step_starting.is_set()}.",
            )
            self._step_starting.clear()  # Ensure event is cleared on timeout to allow future steps
            return False
        except Exception as e:
            # Catch other potential errors during wait
            logger.exception(f"Unexpected error during task completion wait: {e}")
            self._step_starting.clear()  # Reset on error too
            return False  # Indicate failure due to error

    async def _sequence(self) -> AsyncGenerator[StepRequest, None]:
        """Generates a sequence of `StepRequest` objects, one for each participant role.

        This default implementation iterates through the roles defined in
        `self._participants` (which should be populated from a `ConductorRequest`)
        and yields a `StepRequest` for each. After all participant roles have
        been processed, it yields a final `StepRequest` with the `END` role to
        signal the completion of the sequence.

        Subclasses can override this method to implement different sequencing logic
        (e.g., conditional steps, loops, LLM-driven step generation).

        Yields:
            StepRequest: The next `StepRequest` in the sequence, targeting a
            participant role or signaling `END`.

        """
        if not self._participants:
            logger.warning(f"Host '{self.agent_name}': _sequence called with no participants defined.")
            yield StepRequest(role=END, content="No participants defined for the flow.")
            return

        for role_name in self._participants.keys():
            yield StepRequest(role=role_name, content=f"Requesting action from role: {role_name}")

        yield StepRequest(role=END, content="All participant roles in the default sequence have been processed.")

    async def _run_flow(self, message: ConductorRequest) -> None:
        """Executes the main flow logic based on a sequence of steps.

        This method is typically started as an asyncio task when the `HostAgent`
        receives a `ConductorRequest`. It initializes the participant list from
        the request, then iterates through a sequence of steps generated by
        `self._sequence()`.

        For each step:
        1.  It waits for all tasks from the *previous* step to complete using
            `wait_check_last_step_completions()`.
        2.  If `self.human_in_loop` is True, it waits for user confirmation for
            the current step via `_wait_for_user()`.
        3.  If confirmations are met (or not required), it executes the current
            step using `_execute_step()`.

        The loop breaks if task completions fail, user rejects a step, or an
        `END` step is encountered.

        Args:
            message (ConductorRequest): The initial request that starts the flow,
                containing the list of participants and their configurations.

        Raises:
            FatalError: If the `ConductorRequest` contains no participants.

        """
        logger.info(f"Host '{self.agent_name}': Starting flow execution based on ConductorRequest.")

        # Initialize participants from the request
        self._participants.update(message.participants)
        if not self._participants:
            msg = "Host received ConductorRequest with no participants."
            logger.error(f"{msg} Aborting.")
            # Send an END message with the error
            await self.callback_to_groupchat(StepRequest(role=END, content=msg))
            raise FatalError(msg)

        # Rerun initialization to set up the group chat
        await self.initialize(callback_to_groupchat=self.callback_to_groupchat)

        # Start the periodic progress reporter task
        self._progress_reporter_task = asyncio.create_task(self._report_progress_periodically())

        # Initialize generator now that participants are known
        self._step_generator = self._sequence()

        async for next_step_request in self._step_generator:
            # 1. Wait for tasks from the PREVIOUS step to complete.
            if not await self.wait_check_last_step_completions():
                # If the wait failed (timeout or error), stop the flow
                # break
                pass
            if self.human_in_loop and not await self._wait_for_user(next_step):
                # If user rejected or timed out, stop the flow
                continue
                # break

        logger.info(f"Host '{self.agent_name}': Flow execution loop finished.")
        # Final check for any outstanding tasks if loop finished due to reasons other than explicit END step from generator
        if not await self.wait_check_last_step_completions():
             logger.error(f"Host '{self.agent_name}': Outstanding tasks remaining after flow loop completion. This may indicate an issue.")
        logger.info(f"Host '{self.agent_name}': _run_flow completed.")

    async def _shutdown(self) -> None:
        """Shutdown the agent and clean up resources."""
        if self._conductor_task:
            self._conductor_task.cancel()
            try:
                await self._conductor_task
            except asyncio.CancelledError:
                logger.debug("Conductor task cancelled.")
        if self._step_generator:
            await self._step_generator.aclose()
        if self._tasks_condition:
            await self._tasks_condition.release()
            # Ensure the progress reporter task is cancelled when the wait finishes or times out
            if self._progress_reporter_task:
                self._progress_reporter_task.cancel()
                try:
                    await self._progress_reporter_task  # Await cancellation
                except asyncio.CancelledError:
                    pass  # Expected
        logger.info(f"Host {self.agent_name} shutdown complete.")

    async def _shutdown(self) -> None:
        """Shutdown the agent and clean up resources."""
        if self._conductor_task:
            self._conductor_task.cancel()
            try:
                await self._conductor_task
            except asyncio.CancelledError:
                logger.debug("Conductor task cancelled.")
        if self._step_generator:
            await self._step_generator.aclose()
        if self._tasks_condition:
            await self._tasks_condition.release()
            # Ensure the progress reporter task is cancelled when the wait finishes or times out
            if self._progress_reporter_task:
                self._progress_reporter_task.cancel()
                try:
                    await self._progress_reporter_task  # Await cancellation
                except asyncio.CancelledError:
                    pass  # Expected
        logger.info(f"Host {self.agent_name} shutdown complete.")

    async def wait_check_last_step_completions(self) -> bool:
        """Waits for all tasks initiated in the previous step to complete and checks success.

        Calls `_wait_for_all_tasks_complete` to pause until all pending agent tasks
        are done. If the wait is successful (returns True), it clears the
        `_pending_tasks_by_agent` dictionary to prepare for the next step.

        Returns:
            bool: `True` if all tasks from the previous step completed successfully
                  within the timeout, `False` otherwise.

        """
        all_tasks_completed = await self._wait_for_all_tasks_complete()
        if not all_tasks_completed:
            # Error already logged by _wait_for_all_tasks_complete if it's a timeout/error
            logger.error(
                f"Host '{self.agent_id}': Not all tasks for the previous step completed successfully or timed out. "
                f"Remaining pending tasks: {dict(self._pending_tasks_by_agent)}",
            )
            return False

        # If successful, clear the pending tasks dictionary for the next step.
        # This should be done under the condition lock to ensure atomicity with checks in _wait_for_all_tasks_complete.
        async with self._tasks_condition:
             self._pending_tasks_by_agent.clear()
        logger.info(f"Host '{self.agent_id}': All tasks for the previous step completed successfully. Pending tasks cleared.")
        return True

    async def _execute_step(self, step: StepRequest) -> None:
        """Executes a single step in the flow by sending the `StepRequest` to the group chat.

        This method updates `self._current_step` with the role from the `step`.
        - If the `step.role` is `WAIT`, it introduces an asyncio sleep.
        - If the `step.role` is `END`, it logs completion and sends the `END` step.
        - For other roles, it sets the `_step_starting` event (if the role is a known
          participant) to signal that tasks for this step are about to be dispatched.
          It then sends the `step` to the group chat via `self.callback_to_groupchat`.

        Args:
            step (StepRequest): The `StepRequest` object defining the step to execute.
        
        Note:
            This method relies on `self.callback_to_groupchat` being properly set
            during initialization to send messages.

        """
        logger.info(f"Host '{self.agent_id}': Executing step for role '{step.role}'. Content: '{step.content[:100]}...'")
        self._current_step = step.role  # Update current step being processed

        if not self.callback_to_groupchat:
            logger.error(f"Host '{self.agent_id}': callback_to_groupchat not set. Cannot execute step '{step.role}'.")
            # Depending on desired robustness, could raise an error or try to handle.
            return

        if step.role == WAIT:
            wait_duration = int(step.parameters.get("duration", 10))  # Allow duration from step params
            logger.info(f"Host '{self.agent_id}': Received WAIT step. Waiting for {wait_duration} seconds.")
            await asyncio.sleep(wait_duration)
        elif step.role == END:
            logger.info(f"Host '{self.agent_id}': Received END step. Signaling flow completion. Final content: '{step.content}'")
            await self.callback_to_groupchat(step)  # Send the END signal
        else:
            # For regular steps targeting a participant
            if step.role in self._participants:
                # Signal that tasks for this step are expected to start.
                # This event is used by _wait_for_all_tasks_complete.
                # It should be cleared once tasks actually start reporting completion (_handle_events).
                self._step_starting.set()
                logger.debug(f"Host '{self.agent_id}': Set _step_starting event for role '{step.role}'.")
            else:
                 logger.warning(f"Host '{self.agent_id}': Executing step for role '{step.role}' which is not in the known participants list: {list(self._participants.keys())}.")

            await self.callback_to_groupchat(step)  # Dispatch the step to the group/channel

    async def _process(
        self,
        *,
        message: AgentInput,
        cancellation_token: CancellationToken | None = None,  # Standard arg
        **kwargs: Any,  # Standard arg
    ) -> AgentOutput:
        """Handles direct invocations of the HostAgent.

        As a non-LLM agent focused on flow control, the `HostAgent` typically
        does not perform complex processing on direct `AgentInput` messages via
        its `_process` method in the same way an LLM-based agent would. Its main
        logic is within `_run_flow`, triggered by a `ConductorRequest` event.

        This implementation returns an `AgentOutput` wrapping an `ErrorEvent`
        to indicate that direct processing of generic `AgentInput` is not the
        primary mode of operation for this type of host.

        Args:
            message: The input `AgentInput` message.
            cancellation_token: Optional cancellation token.
            **kwargs: Additional keyword arguments.

        Returns:
            AgentOutput: An `AgentOutput` containing an `ErrorEvent` payload,
            signifying that this agent expects to be driven by `ConductorRequest`
            events or specific OOB messages rather than generic inputs.

        """
        error_payload = ErrorEvent(
            source=self.agent_id,
            content="HostAgent is designed for flow control and expects specific events (like ConductorRequest) rather than direct processing of generic AgentInput via _process.",
        )
        return AgentOutput(agent_id=self.agent_id, outputs=error_payload)
