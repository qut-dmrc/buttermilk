import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from autogen_core import CancellationToken
from autogen_core.models import AssistantMessage, UserMessage
from pydantic import BaseModel, Field, PrivateAttr

from buttermilk import logger
from buttermilk._core.agent import Agent, AgentResponse
from buttermilk._core.constants import COMMAND_SYMBOL, END, MANAGER, WAIT
from buttermilk._core.contract import (
    AgentInput,
    AgentTrace,
    ConductorRequest,
    ConductorResponse,
    ErrorEvent,
    GroupchatMessageTypes,
    ManagerMessage,
    ManagerRequest,
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
    
    This agent acts as a basic conductor (`CONDUCTOR` role). It is responsible for 
    determining the flow of conversations in a group chat, deciding which agents 
    to call next, and managing the overall interaction between agents. 
    It handles the substantive flow of conversation, allowing the orchestrator
    to focus on technical execution rather than conversation flow logic.
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
    human_in_loop: bool = Field(
        default=True,
        description="Whether to interact with the human/manager for step confirmation",
    )

    _conductor_task: asyncio.Task | None = PrivateAttr(default=None)
    _current_step_name: str | None = PrivateAttr(default=None)
    _outstanding_tasks_per_role: dict[str, int] = PrivateAttr(default_factory=dict)  # Track outstanding tasks per role

    _step_generator: Any = PrivateAttr(default=None)
    _participants: dict = PrivateAttr(default={})  # Stores role descriptions
    _step_completion_event: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)

    #  Queue for receiving confirmation responses from the MANAGER.
    _user_confirmation: asyncio.Queue[ManagerResponse] = PrivateAttr(default_factory=lambda: asyncio.Queue(maxsize=1))

    #  State tracking for exploration
    _exploration_path: list[str] = PrivateAttr(default_factory=list)
    _exploration_results: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)
    _user_feedback: list[str] = PrivateAttr(default_factory=list)

    # Progress tracking
    _total_steps: int = PrivateAttr(default=0)
    _current_step: int = PrivateAttr(default=0)
    _step_sequence: list[str] = PrivateAttr(default_factory=list)

    async def initialize(self, input_callback: Callable[..., Awaitable[None]] | None = None, **kwargs) -> None:
        """Initialize the agent"""
        self._input_callback = input_callback
        self._step_completion_event.set()  # Initially set (no step active)
        self._step_generator = self._sequence()
        self._exploration_path = []
        self._exploration_results = {}
        self._user_feedback = []
        self._outstanding_tasks_per_role = {}
        self._total_steps = len(self._participants) if hasattr(self, "_participants") else 0
        self._current_step = 0
        self._step_sequence = list(self._participants.keys()) if hasattr(self, "_participants") else []
        await super().initialize(**kwargs)  # Call parent initialize if needed

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
        """Listen to messages in the group chat and maintain conversation history.
        
        This method stores relevant messages in the agent's context to provide
        better context for future decision making.
        """
        # Log messages to our local context cache, but truncate them
        if isinstance(message, (AgentTrace, ConductorResponse)):
            await self._model_context.add_message(AssistantMessage(content=str(message.content)[:TRUNCATE_LEN], source=source))
        elif isinstance(message, StepRequest):
            # StepRequest has content field but it might be empty
            if message.content and not message.content.startswith(COMMAND_SYMBOL):
                await self._model_context.add_message(UserMessage(content=str(message.content)[:TRUNCATE_LEN], source=source))
        elif isinstance(message, ManagerMessage):
            # Handle ManagerMessage with content attribute or params attribute
            content = getattr(message, "content", getattr(message, "params", None))
            if content and not content.startswith(COMMAND_SYMBOL):
                await self._model_context.add_message(UserMessage(content=str(content)[:TRUNCATE_LEN], source=source))

        # Store user feedback if available
        if isinstance(message, ManagerResponse) and message.prompt:
            self._user_feedback.append(message.prompt)
            await self._model_context.add_message(UserMessage(content=f"User feedback: {message.prompt[:TRUNCATE_LEN]}", source="USER"))

    async def _handle_events(
        self,
        message: OOBMessages,
        cancellation_token: CancellationToken | None = None,
        public_callback: Callable | None = None,  # Callback provided by adapter
        message_callback: Callable | None = None,  # Callback provided by adapter
        **kwargs,
    ) -> OOBMessages | None:
        """Handle special events and messages in the group chat.
        
        This method processes various types of messages like task completion notifications,
        conductor requests, and user feedback. It's a central hub for managing the
        conversation flow state.
        """
        logger.debug(f"Host {self.agent_id} handling event: {type(message).__name__}")

        if isinstance(message, ManagerResponse):
            try:
                self._user_confirmation.put_nowait(message)
                return
            except asyncio.QueueFull:
                msg = f"Discarding user input because earlier input still hasn't been handled: {message}"
                logger.error(msg)
                if public_callback:
                    await public_callback(ErrorEvent(source=self.agent_id, content=msg))
                return

        # Handle task completion signals from worker agents.
        if isinstance(message, TaskProcessingComplete):
            # Decrement outstanding tasks for the role
            role = message.role
            if role in self._outstanding_tasks_per_role and self._outstanding_tasks_per_role[role] > 0:
                self._outstanding_tasks_per_role[role] -= 1
                logger.info(
                    f"Host received TaskComplete from {message.agent_id} for step '{role}'. "
                    f"{self._outstanding_tasks_per_role[role]} tasks still outstanding. "
                    f"Task {message.task_index}, More: {message.more_tasks_remain}, Error: {message.is_error}",
                )

                # Check if this was the current step and all tasks are complete
                if role == self._current_step_name and self._outstanding_tasks_per_role[role] == 0:
                    logger.info(f"All tasks completed for current step '{self._current_step_name}'. Setting completion event.")
                    self._step_completion_event.set()

                    # Send a progress update that this step is complete
                    await self._send_progress_update(
                        role=role,
                        status="completed",
                        message=f"All tasks for step {role} completed",
                        progress=1.0 if self._current_step == self._total_steps else
                            (self._current_step / self._total_steps if self._total_steps > 0 else 0.5),
                    )
            else:
                # Log if an unexpected completion is received
                logger.warning(f"Host received TaskComplete from agent {message.agent_id} for step '{role}', but no outstanding tasks were tracked.")
                # Initialize the counter if it doesn't exist
                if role not in self._outstanding_tasks_per_role:
                    self._outstanding_tasks_per_role[role] = 0

        elif isinstance(message, TaskProcessingStarted):
            # Increment outstanding tasks for the role
            role = message.role
            if role not in self._outstanding_tasks_per_role:
                self._outstanding_tasks_per_role[role] = 0

            self._outstanding_tasks_per_role[role] += 1
            logger.info(f"Host noted TaskStarted from agent {message.agent_id} for step '{role}'. {self._outstanding_tasks_per_role[role]} outstanding tasks.")

            # If this is the current step, make sure the completion event is cleared
            if role == self._current_step_name:
                self._step_completion_event.clear()

            # Send a progress update that a task has started
            await self._send_progress_update(
                role=role,
                status="started",
                message=f"Task started by agent {message.agent_id}",
                progress=0.0 if self._total_steps == 0 else (self._current_step / self._total_steps),
            )

        # Handle conductor request to start running the flow
        elif isinstance(message, ConductorRequest):
            if not self._conductor_task or self._conductor_task.done():
                 # Check if task exists and is done (e.g., due to previous error or completion)
                if self._conductor_task and self._conductor_task.done():
                    try:
                        # Retrieve potential exceptions from the completed task
                        self._conductor_task.result()
                    except Exception as e:
                        logger.error(f"Previous conductor task ended with exception: {e}")
                logger.info(f"Host {self.agent_id} starting new conductor task.")
                self._conductor_task = asyncio.create_task(self._run_flow(message=message))
            else:
                 logger.warning(f"Host {self.agent_id} received ConductorRequest but task is already running.")

        return

    async def _wait_for_user(self) -> ManagerResponse:
        try:
            response = await asyncio.wait_for(self._user_confirmation.get(), timeout=self.max_wait_time)
            logger.debug(f"Received manager response: confirm = {response.confirm}, halt = {response.halt}, interrupt = {response.interrupt}, selection = {response.selection}")
            return response
        except TimeoutError:
            logger.warning(f"{self.agent_id} hit timeout waiting for manager response after {self.max_wait_time} seconds.")
            return await self._wait_for_user()

    async def _wait_for_completions(self) -> None:
        """Wait for all outstanding tasks for the current step to complete.
        """
        if not self._current_step_name:
            # We aren't in a step, no use waiting.
            return

        current_role = self._current_step_name
        outstanding_tasks = self._outstanding_tasks_per_role.get(current_role, 0)

        # If no outstanding tasks or the event is already set
        if outstanding_tasks == 0 or self._step_completion_event.is_set():
             logger.debug(f"No outstanding tasks for '{current_role}' or completion event already set.")
             return

        try:
            logger.debug(f"Waiting for completion of '{current_role}'. Outstanding tasks: {outstanding_tasks}")
            await asyncio.wait_for(self._step_completion_event.wait(), timeout=self.max_wait_time)
            logger.debug(f"Step '{current_role}' completed.")
        except TimeoutError:
            # Timeout occurred, meaning the event was not set and we still have outstanding tasks.
            msg = (f"Timeout waiting for step '{current_role}' completion. "
                   f"There are still {outstanding_tasks} outstanding tasks that did not complete within {self.max_wait_time}s.")
            logger.error(f"{msg} Aborting.")
            # Reset the outstanding tasks count and set event to allow potential cleanup/reset
            self._outstanding_tasks_per_role[current_role] = 0
            self._step_completion_event.set()
            raise FatalError(msg)

    async def _sequence(self) -> AsyncGenerator[StepRequest, None]:
        """Generate a sequence of steps to execute in order.
        
        This default implementation simply processes each participant in order.
        Subclasses may override this to implement more complex sequencing logic.
        """
        while not self._participants:
            await asyncio.sleep(0.1)

        # Store the sequence and total steps for progress tracking
        self._step_sequence = list(self._participants.keys())
        self._total_steps = len(self._step_sequence)
        logger.info(f"Host {self.agent_id} initialized sequence with {self._total_steps} steps: {self._step_sequence}")

        # Generate steps in order
        for i, (step_name, cfg) in enumerate(self._participants.items()):
            self._current_step = i + 1  # 1-based step index
            logger.info(f"Host {self.agent_id} generating step {self._current_step}/{self._total_steps}: {step_name}")
            yield StepRequest(role=step_name, content=f"Sequence host calling {step_name}.")

        # Always yield the END step at the end
        logger.info(f"Host {self.agent_id} sequence completed, generating END step")
        yield StepRequest(role=END, content="Sequence wrapping up.")

    async def _run_flow(self, message: ConductorRequest) -> StepRequest:
        """Determine the next step in the conversation flow.
        
        This method is the core of the flow control logic. It:
        1. Waits for current step completion if needed
        2. Initializes participants if not done
        3. Chooses the next step using _choose
        4. Validates the step against available participants
        5. Prepares for the next step execution
        6. Gets user confirmation if needed
        7. Executes the step directly if appropriate
        
        Args:
            message: The ConductorRequest containing context information
            
        """
        while True:
            # Initialize participants if not done yet
            if not self._participants:
                # Store role descriptions
                self._participants.update(message.participants)
                if not self._participants:
                    raise FatalError("Host received ConductorRequest with no participants.")

            # Wait for enough completions, with a timeout
            await self._wait_for_completions()

            # Get the next step using the strategy provided by the specific host implementation
            step = await self._choose(message=message)

            if step.role == self.role:
                # Don't call ourselves please
                logger.warning(f"Host chose itself ({self.role}) for the next step. Switching to WAIT.")
                step.role = WAIT

            # Validate against known roles (keys of the participants dict)
            if step.role not in self._participants and step.role not in [WAIT, END]:
                logger.warning(f"Host chose unknown role '{step.role}'. Known roles: {list(self._participants.keys())}. Switching to WAIT.")
                step.role = WAIT

            if step.role == END:
                logger.info("Host received END step. Stopping flow.")
                break

            # Get user confirmation if needed
            if step.role != WAIT and self.human_in_loop:
                confirmation_request = await self.request_user_confirmation(step)
                if self._input_callback:
                     await self._input_callback(confirmation_request)
                else:
                     logger.error("Cannot request user confirmation: input_callback is not set.")
                     step.role = WAIT  # Fallback to WAIT

                if step.role != WAIT:
                    manager_response: ManagerResponse = await self._wait_for_user()
                    if not manager_response.confirm:
                        logger.info(f"User rejected step for role: {step.role}. Switching to WAIT.")
                        step.role = WAIT
                        if manager_response.prompt:
                             logger.info(f"User provided feedback: {manager_response.prompt}")
                             self._user_feedback.append(manager_response.prompt)

            logger.info(f"Host proceeding with role: {step.role}")
            # Pass only the step, participants dict is internal state now
            await self._execute_step(step)

        # Send a final progress update for the complete workflow
        await self._send_progress_update(
            role="WORKFLOW",
            status="completed",
            message="Workflow completed successfully",
            progress=1.0,
        )

        # Create the END step message
        end_step = StepRequest(role=END, content="Flow completed.")

        # Also send the end step to the agents
        if self._input_callback:
            await self._input_callback(end_step)

        logger.info(f"Host {self.agent_id} flow execution finished.")
        return end_step

    async def _execute_step(self, step: StepRequest) -> None:
        """Prepare for a step execution by setting state and sending the StepRequest.

        Sets the current step name, resets completion tracking state, and publishes
        the StepRequest message via the input callback.

        Args:
            step: The step to execute

        """
        if step.role == END or step.role == WAIT:
            logger.debug(f"Skipping execution setup for control step: {step.role}")
            # Ensure completion event is set if we are just waiting or ending
            if not self._step_completion_event.is_set():
                 self._step_completion_event.set()
            # Clear current step name if waiting/ending
            self._current_step_name = None
            return

        logger.info(f"Host executing step for role: {step.role}")

        # --- Set up state for the new step ---
        self._current_step_name = step.role

        # Make sure counter exists for this role
        if step.role not in self._outstanding_tasks_per_role:
            self._outstanding_tasks_per_role[step.role] = 0

        # Clear the event if there are outstanding tasks for this role
        if self._outstanding_tasks_per_role[step.role] > 0:
            self._step_completion_event.clear()
        else:
            # If no tasks are running for this step, ensure the completion event is set
            self._step_completion_event.set()

        logger.debug(f"Starting step '{step.role}' with {self._outstanding_tasks_per_role[step.role]} outstanding tasks. Event state: {'SET' if self._step_completion_event.is_set() else 'CLEARED'}")

        # Create the message for the target role
        message_content = step.content or f"Executing step for role {step.role}"
        # Pass records if needed (assuming self._records exists or is handled)
        message = StepRequest(role=step.role, content=message_content, records=getattr(self, "_records", []))

        # Publish the message using the _input_callback
        if not hasattr(self, "_input_callback") or not callable(self._input_callback):
            logger.error(f"Host {self.agent_id} cannot publish StepRequest for role {step.role}: _input_callback is not set or not callable.")
            # If the step can't be sent, reset state and ensure event is set to unblock
            self._current_step_name = None
            if step.role in self._outstanding_tasks_per_role:
                self._outstanding_tasks_per_role[step.role] = 0
            self._step_completion_event.set()  # Set here to prevent deadlock if publish fails
            return

        try:
            logger.debug(f"Host {self.agent_id} attempting to publish StepRequest for role {step.role}...")
            await self._input_callback(message)
            logger.debug(f"Host {self.agent_id} successfully published StepRequest for role {step.role}")
        except Exception as e:
            logger.exception(f"Host {self.agent_id} encountered an error calling _input_callback for role {step.role}: {e}")
            # Reset state and ensure event is set to prevent deadlock
            self._current_step_name = None
            if step.role in self._outstanding_tasks_per_role:
                self._outstanding_tasks_per_role[step.role] = 0
            self._step_completion_event.set()  # Set here to prevent deadlock on exception

    def _store_exploration_result(self, execution_id: str, output: AgentTrace) -> None:
        """Store the result of an agent execution in the exploration history.
        
        Args:
            execution_id: A unique identifier for this execution
            output: The AgentTrace from the agent execution

        """
        # Add to exploration path if not already there
        if execution_id not in self._exploration_path:
            self._exploration_path.append(execution_id)

        # Store the result
        self._exploration_results[execution_id] = {
            "id": output.call_id,
            "role": self._current_step_name or "unknown",
            "inputs": getattr(output, "inputs", {}),
            "outputs": getattr(output, "outputs", getattr(output, "contents", "")),
            "is_error": getattr(output, "is_error", False),
            "error_details": getattr(output, "error", []) if getattr(output, "is_error", False) else None,
            "metadata": getattr(output, "metadata", {}),
        }
        logger.debug(f"Stored result for execution {execution_id}")

    async def request_user_confirmation(self, step: StepRequest) -> StepRequest:
        """Request confirmation from the user for the next step.
        
        This method is used when human_in_loop is True to get user approval
        before executing a step.
        
        Args:
            step: The proposed next step
            
        Returns:
            StepRequest: message asking the UI for input

        """
        request_content = (
            f"**Next Proposed Step:**\n"
            f"- **Agent Role:** {step.role}\n"
            f"- **Description:** {step.content or '(No description)'}\n"
            f"- **Prompt Snippet:** {step.prompt[:100] + '...' if step.prompt else '(No prompt)'}\n\n"
            f"Confirm (Enter), provide feedback, or reject ('n'/'q')."
            )
        logger.debug(f"Requesting info from user about proposed step {step.role}.")
        return ManagerRequest(role=MANAGER, prompt=request_content, inputs=dict(confirm=True, selection=[True, False]))

    async def _choose(self, message: ConductorRequest | None) -> StepRequest:
        """Choose the next step in the conversation.
        
        This default implementation simply takes the next step from the sequence generator.
        Subclasses should override this to implement more sophisticated decision making.
        
        Args:
            message: The ConductorRequest containing context for decision making
            
        Returns:
            A StepRequest representing the next step to execute

        """
        try:
            # Manual implementation to avoid using anext directly
            step = await self._step_generator.__anext__()
            logger.info(f"Host {self.agent_id} chose next step: {step.role} (step {self._current_step}/{self._total_steps})")
            return step
        except StopAsyncIteration:
            # If the generator is exhausted, ensure we return the END step
            logger.info(f"Host {self.agent_id} step generator exhausted, explicitly returning END step")
            return StepRequest(role=END, content="Sequence completed. All steps have been processed.")

    async def _send_progress_update(self, role: str, status: str, message: str, progress: float = 0.0) -> None:
        """Send a progress update to the UI agent.
        
        Args:
            role: The role associated with this task
            status: Current status (e.g., 'started', 'in_progress', 'completed', 'error')
            message: Human-readable message explaining the current progress
            progress: Numeric progress indicator (0.0 to 1.0)

        """
        if not self._input_callback:
            logger.warning("Cannot send progress update: input_callback is not set.")
            return

        try:
            progress = int(100 * progress)
            # Get the current step if it exists in the sequence
            if self._current_step_name and self._current_step_name in self._step_sequence:
                current_step_idx = self._step_sequence.index(self._current_step_name) + 1
            else:
                # If not found or not set, use the tracked current step or default to 0
                current_step_idx = self._current_step

            # Update total steps count if needed
            if not self._total_steps and self._participants:
                self._total_steps = len(self._participants)
                self._step_sequence = list(self._participants.keys())

            # Create progress update message
            progress_update = TaskProgressUpdate(
                source=self.agent_id,
                role=role,
                step_name=self._current_step_name or role,
                status=status,
                message=message,
                total_steps=100,
                current_step=progress,
            )

            # Send to the UI agent via the input callback
            await self._input_callback(progress_update)
            logger.debug(f"Sent progress update for {role}: {status} - {message}")

        except Exception as e:
            logger.error(f"Error sending progress update: {e}")

    async def on_reset(self, cancellation_token=None) -> None:
        """Resets the HostAgent's internal state."""
        logger.info(f"Resetting Host agent {self.agent_id}...")
        await super().on_reset(cancellation_token)
        self._current_step_name = None
        self._outstanding_tasks_per_role.clear()  # Reset task counters
        self._participants.clear()
        # Re-initialize the step generator (will wait for participants again).
        self._step_generator = self._sequence()
        # Set completion event to ready state.
        self._step_completion_event.set()
        # Clear user confirmation queue
        while not self._user_confirmation.empty():
            try:
                self._user_confirmation.get_nowait()
            except asyncio.QueueEmpty:
                break
        # Reset exploration tracking
        self._exploration_path = []
        self._exploration_results = {}
        self._user_feedback = []
        # Cancel existing conductor task if running
        if self._conductor_task and not self._conductor_task.done():
            self._conductor_task.cancel()
            try:
                await self._conductor_task  # Allow cancellation to propagate
            except asyncio.CancelledError:
                logger.debug("Conductor task cancelled during reset.")
            except Exception as e:
                 logger.error(f"Error retrieving result from cancelled conductor task during reset: {e}")
        self._conductor_task = None

        logger.info(f"Host agent {self.agent_id} reset complete.")

    async def _process(self, *, message: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs) -> AgentResponse:
        """Host _process implementation - returns an empty response as a placeholder.
        
        This is not generally used directly as the host operates via _handle_events
        """
        # Create placeholder response with an error event
        placeholder = ErrorEvent(source=self.agent_id, content="Host agent has no direct processing behavior")
        return AgentResponse(
            metadata={"source": self.agent_id},
            outputs=placeholder,
        )
