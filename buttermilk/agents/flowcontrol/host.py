import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from autogen_core import CancellationToken
from autogen_core.models import AssistantMessage, UserMessage
from pydantic import BaseModel, Field, PrivateAttr

from buttermilk import logger
from buttermilk._core.agent import Agent, AgentResponse
from buttermilk._core.constants import COMMAND_SYMBOL, END, WAIT
from buttermilk._core.contract import (
    AgentInput,
    AgentTrace,
    ConductorRequest,
    ConductorResponse,
    ErrorEvent,
    GroupchatMessageTypes,
    ManagerMessage,
    ManagerResponse,
    OOBMessages,
    StepRequest,
    TaskProcessingComplete,
    TaskProcessingStarted,
    TaskProgressUpdate,
)
from buttermilk._core.exceptions import FatalError

TRUNCATE_LEN = 1000  # characters per history message


class HostAgent(Agent):
    """Base coordinator for group chats and flow control.

    This agent acts as a conductor (`CONDUCTOR` role). It iterates through a
    predefined sequence of participant roles. For each step, it sends a request
    to the corresponding role. Before proceeding to the next step, it waits
    until all agents that have started processing a task have reported completion.
    """

    _input_callback: Any = PrivateAttr(...)
    _pending_agent_id: str | None = PrivateAttr(default=None)  # Track agent waiting for signal

    _output_model: type[BaseModel] | None = StepRequest
    _message_types_handled: type[Any] = PrivateAttr(default=type(ConductorRequest))

    # Additional configuration
    max_wait_time: int = Field(
        default=240,
        description="Maximum time to wait for agent responses in seconds",
    )
    # human_in_loop removed for simplification

    _conductor_task: asyncio.Task | None = PrivateAttr(default=None)
    _current_step_name: str | None = PrivateAttr(default=None)
    # Simplified task tracking: single counter for all roles
    _total_outstanding_tasks: int = PrivateAttr(default=0)

    _step_generator: Any = PrivateAttr(default=None)
    _participants: dict = PrivateAttr(default={})  # Stores role descriptions
    # Event signals when _total_outstanding_tasks reaches 0
    _all_tasks_complete_event: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)

    # State tracking for exploration (kept for potential future use)
    _exploration_path: list[str] = PrivateAttr(default_factory=list)
    _exploration_results: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)
    _user_feedback: list[str] = PrivateAttr(default_factory=list)  # Kept in case ManagerResponse is still used elsewhere

    # Progress tracking
    _total_steps: int = PrivateAttr(default=0)
    _current_step: int = PrivateAttr(default=0)
    _step_sequence: list[str] = PrivateAttr(default_factory=list)

    async def initialize(self, input_callback: Callable[..., Awaitable[None]] | None = None, **kwargs) -> None:
        """Initialize the agent"""
        self._input_callback = input_callback
        self._all_tasks_complete_event.set()  # Initially set (no tasks active)
        self._step_generator = self._sequence()
        self._exploration_path = []
        self._exploration_results = {}
        self._user_feedback = []
        self._total_outstanding_tasks = 0
        self._total_steps = len(self._participants) if hasattr(self, "_participants") else 0
        self._current_step = 0
        self._step_sequence = list(self._participants.keys()) if hasattr(self, "_participants") else []
        await super().initialize(**kwargs)  # Call parent initialize if needed
        # self.human_in_loop = self.parameters.get("human_in_loop", True) # Removed

    async def _listen(
        self,
        message: GroupchatMessageTypes | StepRequest,
        *,
        cancellation_token: CancellationToken | None = None,
        source: str = "",
        public_callback: Callable | None = None,
        message_callback: Callable | None = None,
        **kwargs,
    ) -> None:
        """Listen to messages in the group chat and maintain conversation history."""
        # Log messages to our local context cache, but truncate them
        if isinstance(message, (AgentTrace, ConductorResponse)):
            await self._model_context.add_message(AssistantMessage(content=str(message.content)[:TRUNCATE_LEN], source=source))
        elif isinstance(message, StepRequest):
            if message.content and not message.content.startswith(COMMAND_SYMBOL):
                await self._model_context.add_message(UserMessage(content=str(message.content)[:TRUNCATE_LEN], source=source))
        elif isinstance(message, ManagerMessage):
            content = getattr(message, "content", getattr(message, "params", None))
            if content and not content.startswith(COMMAND_SYMBOL):
                await self._model_context.add_message(UserMessage(content=str(content)[:TRUNCATE_LEN], source=source))

        # Store user feedback if available (kept in case ManagerResponse is used elsewhere)
        if isinstance(message, ManagerResponse) and message.prompt:
            self._user_feedback.append(message.prompt)
            await self._model_context.add_message(UserMessage(content=f"User feedback: {message.prompt[:TRUNCATE_LEN]}", source="USER"))

    async def _handle_events(
        self,
        message: OOBMessages,
        cancellation_token: CancellationToken | None = None,
        public_callback: Callable | None = None,
        message_callback: Callable | None = None,
        **kwargs,
    ) -> OOBMessages | None:
        """Handle special events and messages."""
        logger.debug(f"Host {self.agent_id} handling event: {type(message).__name__}")

        # Handle ManagerResponse separately if still needed for other purposes
        if isinstance(message, ManagerResponse):
            # If human_in_loop was removed, this might need adjustment
            # depending on whether ManagerResponse is used for other things.
            # For now, just log it.
            logger.debug(f"Host received ManagerResponse: {message}")
            # If it was purely for confirmation, we might not need to do anything.
            # If it carries other data (like feedback), handle that:
            if message.prompt:
                self._user_feedback.append(message.prompt)
            return

        # Handle task completion signals from worker agents.
        if isinstance(message, TaskProcessingComplete):
            role = message.role
            if self._total_outstanding_tasks > 0:
                self._total_outstanding_tasks -= 1
                logger.info(
                    f"Host received TaskComplete from {message.agent_id} for role '{role}'. "
                    f"{self._total_outstanding_tasks} total tasks outstanding. "
                    f"Task {message.task_index}, More: {message.more_tasks_remain}, Error: {message.is_error}",
                )

                # If this was the last task, set the completion event
                if self._total_outstanding_tasks == 0:
                    logger.info("All outstanding tasks completed. Setting completion event.")
                    self._all_tasks_complete_event.set()
                    # Send progress update indicating current step (if any) is effectively done
                    if self._current_step_name and self._current_step_name != END:
                         await self._send_progress_update(
                             role=self._current_step_name,
                             status="completed",
                             message=f"All tasks completed after step {self._current_step_name}",
                             progress=(self._current_step / self._total_steps if self._total_steps > 0 else 1.0),
                         )
            else:
                logger.warning(f"Host received TaskComplete from agent {message.agent_id} for role '{role}', but no outstanding tasks were tracked.")

        elif isinstance(message, TaskProcessingStarted):
            role = message.role
            self._total_outstanding_tasks += 1
            logger.debug(f"Host noted TaskStarted from agent {message.agent_id} for role '{role}'. {self._total_outstanding_tasks} total outstanding tasks.")

            # If tasks become outstanding, clear the completion event
            if self._total_outstanding_tasks > 0:
                self._all_tasks_complete_event.clear()

            # Send a progress update that a task has started for the current step context
            await self._send_progress_update(
                role=self._current_step_name or role,  # Use current step name if available
                status="started",
                message=f"Task started by agent {message.agent_id} (Role: {role})",
                # Progress reflects the overall sequence progress
                progress=(self._current_step / self._total_steps if self._total_steps > 0 else 0.0),
            )

        # Handle conductor request to start running the flow
        elif isinstance(message, ConductorRequest):
            if not self._conductor_task or self._conductor_task.done():
                if self._conductor_task and self._conductor_task.done():
                    try:
                        self._conductor_task.result()
                    except Exception as e:
                        logger.error(f"Previous conductor task ended with exception: {e}")
                logger.info(f"Host {self.agent_id} starting new conductor task.")
                # Reset state before starting a new flow
                await self.on_reset()  # Ensure clean state
                self._conductor_task = asyncio.create_task(self._run_flow(message=message))
            else:
                 logger.warning(f"Host {self.agent_id} received ConductorRequest but task is already running.")

        return

    # Removed _wait_for_user as human_in_loop is removed

    async def _wait_for_all_tasks_complete(self) -> None:
        """Wait until the total count of outstanding tasks reaches zero."""
        # Removed initial check for self._total_outstanding_tasks == 0.
        # Rely primarily on the event state to handle rapid task completion.

        # Check if the event is already set (e.g., tasks completed very quickly
        # or no tasks were ever started for the preceding step(s)).
        if self._all_tasks_complete_event.is_set():
            # Re-verify the task count in case a new task started *just* after
            # the event was set but before this check.
            if self._total_outstanding_tasks == 0:
                logger.debug("All tasks complete (event is set and task count is 0).")
                return
            # This scenario implies the event was set (tasks hit 0), but a new task
            # started immediately after. Clear the event again and proceed to wait.
            logger.warning("Completion event was set, but new tasks were found. Re-clearing event and waiting.")
            self._all_tasks_complete_event.clear()

        # If we reach here, the event is not set (or was re-cleared), so we wait.
        try:
            logger.debug(f"Waiting for {self._total_outstanding_tasks} outstanding task(s) to complete...")
            await asyncio.wait_for(self._all_tasks_complete_event.wait(), timeout=self.max_wait_time)
            logger.debug("All outstanding tasks completed (wait unblocked).")
        except TimeoutError:
            msg = (f"Timeout waiting for all tasks completion. "
                   f"{self._total_outstanding_tasks} tasks still outstanding after {self.max_wait_time}s.")
            logger.error(f"{msg} Aborting.")
            # Reset count and set event to allow potential cleanup/reset
            self._total_outstanding_tasks = 0
            self._all_tasks_complete_event.set()
            raise FatalError(msg)

    async def _sequence(self) -> AsyncGenerator[StepRequest, None]:
        """Generate a sequence of steps to execute based on participants."""
        while not self._participants:
            # Wait for participants to be set by ConductorRequest in _run_flow
            await asyncio.sleep(0.1)

        self._step_sequence = list(self._participants.keys())
        self._total_steps = len(self._step_sequence)
        logger.info(f"Host {self.agent_id} initialized sequence with {self._total_steps} steps: {self._step_sequence}")

        for i, (step_name, cfg) in enumerate(self._participants.items()):
            self._current_step = i + 1  # 1-based step index for progress
            logger.info(f"Host {self.agent_id} generating step {self._current_step}/{self._total_steps}: {step_name}")
            yield StepRequest(role=step_name, content=f"Sequence host calling {step_name}.")

        logger.info(f"Host {self.agent_id} sequence completed, generating END step")
        yield StepRequest(role=END, content="Sequence wrapping up.")

    async def _run_flow(self, message: ConductorRequest) -> StepRequest:
        """Runs the predefined sequence of steps, waiting for task completion between steps."""
        logger.info(f"Host {self.agent_id} starting flow execution.")

        # Initialize participants from the request
        if not self._participants:
            self._participants.update(message.participants)
            if not self._participants:
                 logger.error("Host received ConductorRequest with no participants. Aborting.")
                 raise FatalError("Host received ConductorRequest with no participants.")
            # Re-initialize generator now that participants are known
            self._step_generator = self._sequence()
            # Wait briefly for generator to potentially initialize sequence/total steps
            await asyncio.sleep(0.1)
            logger.info(f"Host participants initialized to: {list(self._participants.keys())}")

        while True:
            # Get the next step from the sequence generator
            step = await self._choose(message=message)  # message might not be needed by _choose now

            if step.role == END:
                logger.info("Host sequence finished. Waiting for final tasks before ending.")
                break  # Exit the loop to perform final wait and send END

            # --- Execute the current step ---
            logger.info(f"Host proceeding with step for role: {step.role}")
            await self._execute_step(step)

            # --- Wait for ALL outstanding tasks to complete before next iteration ---
            # This ensures agents from the current step (or any previous step) finish
            # before we proceed to the *next* step in the sequence.
            await self._wait_for_all_tasks_complete()

            logger.info(f"Host completed waiting period after step: {step.role}")

        # --- Sequence finished, perform final wait and send END ---
        await self._wait_for_all_tasks_complete()
        logger.info("All tasks completed after final step. Sending END signal.")

        # Send a final progress update for the complete workflow
        await self._send_progress_update(
            role="WORKFLOW",
            status="completed",
            message="Workflow completed successfully",
            progress=1.0,
        )

        end_step = StepRequest(role=END, content="Flow completed.")
        if self._input_callback:
            await self._input_callback(end_step)

        logger.info(f"Host {self.agent_id} flow execution finished.")
        self._current_step_name = None  # Clear current step after finishing
        return end_step

    async def _execute_step(self, step: StepRequest) -> None:
        """Send the StepRequest for the current step."""
        if step.role == END or step.role == WAIT:
            logger.debug(f"Skipping execution for control step: {step.role}")
            self._current_step_name = step.role  # Still note the step name
            # Ensure completion event is set if we are just waiting or ending,
            # but only if no tasks are outstanding.
            if self._total_outstanding_tasks == 0 and not self._all_tasks_complete_event.is_set():
                 self._all_tasks_complete_event.set()
            return

        logger.info(f"Host executing step for role: {step.role}")
        self._current_step_name = step.role

        # Event management is now handled solely in _handle_events based on _total_outstanding_tasks.
        # No need to clear/set event here based on per-role counts.

        message_content = step.content or f"Executing step for role {step.role}"
        message = StepRequest(role=step.role, content=message_content, records=getattr(self, "_records", []))

        if not hasattr(self, "_input_callback") or not callable(self._input_callback):
            logger.error(f"Host {self.agent_id} cannot publish StepRequest for role {step.role}: _input_callback is not set or not callable.")
            # If publish fails, we might deadlock if agents were expected.
            # Consider raising an error or specific handling. For now, just log.
            # Setting the event might be dangerous if tasks *were* expected.
            return

        try:
            logger.debug(f"Host {self.agent_id} publishing StepRequest for role {step.role}...")
            await self._input_callback(message)
            logger.debug(f"Host {self.agent_id} successfully published StepRequest for role {step.role}")
        except Exception as e:
            logger.exception(f"Host {self.agent_id} encountered an error calling _input_callback for role {step.role}: {e}")
            # Consider raising FatalError or specific handling
            # Setting the event might be dangerous if tasks *were* expected.

    def _store_exploration_result(self, execution_id: str, output: AgentTrace) -> None:
        """Store the result of an agent execution in the exploration history."""
        if execution_id not in self._exploration_path:
            self._exploration_path.append(execution_id)
        self._exploration_results[execution_id] = {
            "id": output.call_id,
            "role": self._current_step_name or "unknown",  # Use current step name if available
            "inputs": getattr(output, "inputs", {}),
            "outputs": getattr(output, "outputs", getattr(output, "contents", "")),
            "is_error": getattr(output, "is_error", False),
            "error_details": getattr(output, "error", []) if getattr(output, "is_error", False) else None,
            "metadata": getattr(output, "metadata", {}),
        }
        logger.debug(f"Stored result for execution {execution_id}")

    # Removed request_user_confirmation

    async def _choose(self, message: ConductorRequest | None) -> StepRequest:
        """Choose the next step from the sequence generator."""
        # This implementation doesn't need the message context anymore
        try:
            step = await self._step_generator.__anext__()
            # Progress logging moved inside _sequence where step index is tracked
            # logger.info(f"Host {self.agent_id} chose next step: {step.role} (step {self._current_step}/{self._total_steps})")
            return step
        except StopAsyncIteration:
            logger.info(f"Host {self.agent_id} step generator exhausted, explicitly returning END step")
            return StepRequest(role=END, content="Sequence completed. All steps have been processed.")

    async def _send_progress_update(self, role: str, status: str, message: str, progress: float = 0.0) -> None:
        """Send a progress update to the UI agent."""
        if not self._input_callback:
            logger.warning("Cannot send progress update: input_callback is not set.")
            return

        try:
            # Ensure progress is between 0 and 1
            progress = max(0.0, min(1.0, progress))
            progress_percent = int(100 * progress)

            # Use tracked step counts for overall progress
            current_step_idx = self._current_step
            total_steps = self._total_steps

            # Create progress update message
            progress_update = TaskProgressUpdate(
                source=self.agent_id,
                role=role,  # Role associated with the event/step
                step_name=self._current_step_name or role,  # Current step context
                status=status,
                message=message,
                # Use overall sequence progress for the main bar
                total_steps=100,  # Represent as percentage
                current_step=progress_percent,
            )

            await self._input_callback(progress_update)
            logger.debug(f"Sent progress update for {role}: {status} - {message} (Overall: {progress_percent}%)")

        except Exception as e:
            logger.error(f"Error sending progress update: {e}")

    async def on_reset(self, cancellation_token=None) -> None:
        """Resets the HostAgent's internal state."""
        logger.info(f"Resetting Host agent {self.agent_id}...")
        await super().on_reset(cancellation_token)
        self._current_step_name = None
        self._total_outstanding_tasks = 0  # Reset single counter
        self._participants.clear()
        # Re-initialize the step generator (will wait for participants again).
        self._step_generator = self._sequence()
        # Set completion event to ready state.
        self._all_tasks_complete_event.set()
        # Clear user feedback list
        self._user_feedback = []
        # Reset exploration tracking
        self._exploration_path = []
        self._exploration_results = {}
        # Reset progress tracking
        self._total_steps = 0
        self._current_step = 0
        self._step_sequence = []
        # Cancel existing conductor task if running
        if self._conductor_task and not self._conductor_task.done():
            self._conductor_task.cancel()
            try:
                await self._conductor_task
            except asyncio.CancelledError:
                logger.debug("Conductor task cancelled during reset.")
            except Exception as e:
                 logger.error(f"Error retrieving result from cancelled conductor task during reset: {e}")
        self._conductor_task = None

        logger.info(f"Host agent {self.agent_id} reset complete.")

    async def _process(self, *, message: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs) -> AgentResponse:
        """Host _process implementation - not typically used directly."""
        placeholder = ErrorEvent(source=self.agent_id, content="Host agent does not process direct inputs via _process")
        return AgentResponse(
            metadata={"source": self.agent_id},
            outputs=placeholder,
        )
