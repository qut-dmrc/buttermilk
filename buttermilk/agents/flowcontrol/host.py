import asyncio
from collections.abc import Awaitable
from typing import Any, AsyncGenerator, Callable, Optional, Union, cast, Dict, List

from autogen_core import CancellationToken, DefaultTopicId
from pydantic import BaseModel, Field, PrivateAttr

from autogen_core.models import AssistantMessage, UserMessage
from buttermilk import logger
from buttermilk._core.agent import Agent, FatalError, ToolOutput
from buttermilk._core.contract import (
    COMMAND_SYMBOL,
    END,
    MANAGER,
    WAIT,
    AgentInput,
    AgentOutput,
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
)
from buttermilk.agents.llm import LLMAgent

TRUNCATE_LEN = 1000  # characters per history message


class HostAgent(Agent):
    """
    Base coordinator for group chats and flow control.
    
    This agent acts as a basic conductor (`CONDUCTOR` role). It is responsible for 
    determining the flow of conversations in a group chat, deciding which agents 
    to call next, and managing the overall interaction between agents. 
    It handles the substantive flow of conversation, allowing the orchestrator
    to focus on technical execution rather than conversation flow logic.
    """

    _input_callback: Any = PrivateAttr(...)
    _pending_agent_id: str | None = PrivateAttr(default=None)  # Track agent waiting for signal

    _output_model: Optional[type[BaseModel]] = StepRequest
    _message_types_handled: type[Any] = PrivateAttr(default=type(ConductorRequest))

    # Additional configuration
    max_wait_time: int = Field(
        default=240,
        description="Maximum time to wait for agent responses in seconds",
    )
    human_in_loop: bool = Field(
        default=True,
        description="Whether to interact with the human/manager for step confirmation"
    )

    _conductor_task: asyncio.Task|None = PrivateAttr(default=None)
    _current_step_name: str | None = PrivateAttr(default=None)
    _active_agents_current_step: set[str] = PrivateAttr(default_factory=set) # New: Track agents that started

    _step_generator: Any = PrivateAttr(default=None)
    _participants: dict = PrivateAttr(default={}) # Stores role descriptions
    _step_completion_event: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
     
    #  Queue for receiving confirmation responses from the MANAGER.
    _user_confirmation: asyncio.Queue[ManagerResponse] = PrivateAttr(default_factory=lambda: asyncio.Queue(maxsize=1)) 
    
    #  State tracking for exploration
    _exploration_path: List[str] = PrivateAttr(default_factory=list)
    _exploration_results: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)
    _user_feedback: List[str] = PrivateAttr(default_factory=list)

    async def initialize(self, input_callback: Callable[..., Awaitable[None]] | None = None, **kwargs) -> None:
        """Initialize the agent"""
        self._input_callback = input_callback
        self._step_completion_event.set()  # Initially set (no step active)
        self._step_generator = self._sequence()
        self._exploration_path = []
        self._exploration_results = {}
        self._user_feedback = []
        self._active_agents_current_step = set()
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
        """
        Listen to messages in the group chat and maintain conversation history.
        
        This method stores relevant messages in the agent's context to provide
        better context for future decision making.
        """
        # Log messages to our local context cache, but truncate them
        if isinstance(message, (AgentOutput, ConductorResponse)):
            await self._model_context.add_message(AssistantMessage(content=str(message.contents)[:TRUNCATE_LEN], source=source))
        elif isinstance(message, StepRequest):
            # StepRequest has content field but it might be empty
            if message.content and not message.content.startswith(COMMAND_SYMBOL):
                await self._model_context.add_message(UserMessage(content=str(message.content)[:TRUNCATE_LEN], source=source))
        elif isinstance(message, ManagerMessage) and message.params:
            # ManagerMessage and subclasses have content field
            if not message.params.startswith(COMMAND_SYMBOL):
                await self._model_context.add_message(UserMessage(content=str(message.params)[:TRUNCATE_LEN], source=source))
        
        # Store user feedback if available
        if isinstance(message, ManagerResponse) and message.prompt:
            self._user_feedback.append(message.prompt)
            await self._model_context.add_message(UserMessage(content=f"User feedback: {message.prompt[:TRUNCATE_LEN]}", source="USER"))


    async def _handle_events(
        self,
        message: OOBMessages,
        cancellation_token: CancellationToken | None = None,
        **kwargs,
    ) -> OOBMessages | None:
        """
        Handle special events and messages in the group chat.
        
        This method processes various types of messages like task completion notifications,
        conductor requests, and user feedback. It's a central hub for managing the
        conversation flow state.
        """
        logger.debug(f"Host {self.id} handling event: {type(message).__name__}")

        if isinstance(message, ManagerResponse):
            try:
                self._user_confirmation.put_nowait(message)
                return None
            except asyncio.QueueFull:
                logger.error(f"Discarding user input because earlier input still hasn't been handled.")
                return message
            
        # Handle task completion signals from worker agents.
        if isinstance(message, TaskProcessingComplete):
            # Only track completions for the currently active step.
            if message.role == self._current_step_name:
                # Store task completion info (optional, can be kept)
                execution_id = f"{message.role}-{message.agent_id}"

                # Check if this agent was active for the current step
                if message.agent_id in self._active_agents_current_step:
                    self._active_agents_current_step.remove(message.agent_id)
                    logger.info(
                        f"Host received TaskComplete from {message.agent_id} for step '{self._current_step_name}'. "
                        f"{len(self._active_agents_current_step)} agents still active. "
                        f"Task {message.task_index}, More: {message.more_tasks_remain}, Error: {message.is_error}"
                    )
                    # Check if all active agents have now completed
                    if not self._active_agents_current_step:
                        logger.info(f"All active agents completed step '{self._current_step_name}'. Setting completion event.")
                        self._step_completion_event.set()
                else:
                    # Log if an unexpected agent sends completion (wasn't tracked as started).
                    logger.warning(f"Host received TaskComplete from agent {message.agent_id} for step '{self._current_step_name}', but it wasn't tracked as active.")
            else:
                # Ignore completions for steps other than the current one.
                logger.debug(
                    f"Host ignored TaskComplete for inactive step '{message.role}' (current: '{self._current_step_name}') from {message.agent_id}."
                )

        elif isinstance(message, TaskProcessingStarted):
            # Track which agents have started the current step.
            if message.role == self._current_step_name:
                if message.agent_id not in self._active_agents_current_step:
                    self._active_agents_current_step.add(message.agent_id)
                    logger.info(f"Host noted TaskStarted from agent {message.agent_id} for step '{self._current_step_name}'. {len(self._active_agents_current_step)} active agents.")
                    # Clear the event now that we know at least one agent is running
                    self._step_completion_event.clear()
                else:
                     logger.warning(f"Host received duplicate TaskStarted from agent {message.agent_id} for step '{self._current_step_name}'.")
            else:
                # Ignore starts for steps other than the current one.
                logger.debug(
                    f"Host ignored TaskStarted for inactive step '{message.role}' (current: '{self._current_step_name}') from {message.agent_id}."
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
                logger.info(f"Host {self.id} starting new conductor task.")
                self._conductor_task = asyncio.create_task(self._run_flow(message=message))
            else:
                 logger.warning(f"Host {self.id} received ConductorRequest but task is already running.")

            
        return None

    async def _wait_for_user(self) -> ManagerResponse:
        try: 
            response = await asyncio.wait_for(self._user_confirmation.get(), timeout=self.max_wait_time)
            logger.debug(f"Received manager response: confirm = {response.confirm}, halt = {response.halt}, interrupt = {response.interrupt}, selection = {response.selection}")
            return response
        except asyncio.TimeoutError:
            logger.warning(f"{self.id} hit timeout waiting for manager response after {self.max_wait_time} seconds.")
            return await self._wait_for_user()


    async def _wait_for_completions(self) -> None:
        """
        Wait for all agents that started the current step to complete.
        """
        if not self._current_step_name:
            # We aren't in a step, no use waiting.
            return 
        
        # If the event is already set (e.g., no agents started, or they finished instantly)
        if self._step_completion_event.is_set():
             logger.debug(f"Completion event for '{self._current_step_name}' already set or no agents active.")
             return

        try: 
            logger.debug(f"Waiting for completion of '{self._current_step_name}'. Active agents: {self._active_agents_current_step}")
            await asyncio.wait_for(self._step_completion_event.wait(), timeout=self.max_wait_time)
            logger.debug(f"Step '{self._current_step_name}' completed.")
        except asyncio.TimeoutError:
            # Timeout occurred, meaning the event was not set.
            # This implies _active_agents_current_step is not empty.
            msg = (f"Timeout waiting for step '{self._current_step_name}' completion. "
                   f"The following agents started but did not complete within {self.max_wait_time}s: "
                   f"{self._active_agents_current_step}.")
            logger.error(f"{msg} Aborting.")
            # Clear active agents and set event to allow potential cleanup/reset
            self._active_agents_current_step.clear()
            self._step_completion_event.set()
            raise FatalError(msg)


    async def _sequence(self) -> AsyncGenerator[StepRequest, None]:
        """
        Generate a sequence of steps to execute in order.
        
        This default implementation simply processes each participant in order.
        Subclasses may override this to implement more complex sequencing logic.
        """
        while not self._participants:
            await asyncio.sleep(0.1)
        for step_name, cfg in self._participants.items():
            yield StepRequest(role=step_name, content=f"Sequence host calling {step_name}.")
        yield StepRequest(role=END, content="Sequence wrapping up.")


    async def _run_flow(self, message: ConductorRequest) -> StepRequest:
        """
        Determine the next step in the conversation flow.
        
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
                self._participants = message.participants # Store role descriptions
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
                     step.role = WAIT # Fallback to WAIT

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

        logger.info(f"Host {self.id} flow execution finished.")
        return StepRequest(role=END, content="Flow completed.")


    async def _execute_step(self, step: StepRequest) -> None:
        """
        Prepare for a step execution by setting state and sending the StepRequest.

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
            self._active_agents_current_step.clear()
            return

        logger.info(f"Host executing step for role: {step.role}")

        # --- Set up state for the new step ---
        self._current_step_name = step.role
        self._active_agents_current_step.clear()
        # Clear the event. It will be cleared again by TaskProcessingStarted
        # and only set by TaskProcessingComplete when the last agent finishes.
        self._step_completion_event.clear()
        logger.debug(f"Cleared active agents for step '{step.role}'. Completion event initially CLEARED.")

        # Create the message for the target role
        message_content = step.content or f"Executing step for role {step.role}"
        # Pass records if needed (assuming self._records exists or is handled)
        message = StepRequest(role=step.role, content=message_content, records=getattr(self, '_records', []))

        # Publish the message using the _input_callback
        if not hasattr(self, '_input_callback') or not callable(self._input_callback):
            logger.error(f"Host {self.id} cannot publish StepRequest for role {step.role}: _input_callback is not set or not callable.")
            # If the step can't be sent, reset state and ensure event is set to unblock
            self._current_step_name = None
            self._active_agents_current_step.clear()
            self._step_completion_event.set() # Set here to prevent deadlock if publish fails
            return

        try:
            logger.debug(f"Host {self.id} attempting to publish StepRequest for role {step.role}...")
            await self._input_callback(message)
            logger.debug(f"Host {self.id} successfully published StepRequest for role {step.role}")
        except Exception as e:
            logger.exception(f"Host {self.id} encountered an error calling _input_callback for role {step.role}: {e}")
            # Reset state and ensure event is set to prevent deadlock
            self._current_step_name = None
            self._active_agents_current_step.clear()
            self._step_completion_event.set() # Set here to prevent deadlock on exception

    def _store_exploration_result(self, execution_id: str, output: AgentOutput) -> None:
        """
        Store the result of an agent execution in the exploration history.
        
        Args:
            execution_id: A unique identifier for this execution
            output: The AgentOutput from the agent execution
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
        """
        Request confirmation from the user for the next step.
        
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
        return StepRequest(role=MANAGER, prompt=request_content, inputs=dict(confirm=True, selection=[True, False]))

    async def _choose(self, message: ConductorRequest|None) -> StepRequest:
        """
        Choose the next step in the conversation.
        
        This default implementation simply takes the next step from the sequence generator.
        Subclasses should override this to implement more sophisticated decision making.
        
        Args:
            message: The ConductorRequest containing context for decision making
            
        Returns:
            A StepRequest representing the next step to execute
        """
        step = await anext(self._step_generator)
        logger.debug(f"Host {self.id} suggests next step: {step.role}.")
        return step


    async def on_reset(self, cancellation_token=None) -> None:
        """Resets the HostAgent's internal state."""
        logger.info(f"Resetting Host agent {self.id}...")
        await super().on_reset(cancellation_token)
        self._current_step_name = None
        self._active_agents_current_step.clear() # Reset active agents
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
                await self._conductor_task # Allow cancellation to propagate
            except asyncio.CancelledError:
                logger.debug("Conductor task cancelled during reset.")
            except Exception as e:
                 logger.error(f"Error retrieving result from cancelled conductor task during reset: {e}")
        self._conductor_task = None

        logger.info(f"Host agent {self.id} reset complete.")

    async def _process(self, *, message: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs) -> AgentOutput | StepRequest | ManagerRequest | ManagerMessage | ToolOutput | ErrorEvent | None:
        pass  # this host does nothing when called
        return None
