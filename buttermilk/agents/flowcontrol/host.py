import asyncio
from collections.abc import Awaitable
from math import ceil
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
        default=300,
        description="Maximum time to wait for agent responses in seconds",
    )
    completion_threshold_ratio: float = Field(
        default=0.8,
        description="Ratio of agents that must complete a step before proceeding (0.0 to 1.0)",
    )
    human_in_loop: bool = Field(
        default=True,
        description="Whether to interact with the human/manager for step confirmation"
    )

    _conductor_task: asyncio.Task|None = PrivateAttr(default=None)
    _current_step_name: str | None = PrivateAttr(default=None)
    _completed_agents_current_step: set[str] = PrivateAttr(default_factory=set)
    _expected_agents_current_step: set[str] = PrivateAttr(default_factory=set)

    _step_generator: Any = PrivateAttr(default=None)
    _participants: dict = PrivateAttr(default={})
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
        self._step_completion_event.set()  # Ready to process
        self._step_generator = self._sequence()
        self._exploration_path = []
        self._exploration_results = {}
        self._user_feedback = []
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
                self._completed_agents_current_step.add(message.agent_id)
                logger.debug(f"Host received TaskComplete from {message.agent_id} (Task {message.task_index}, More: {message.more_tasks_remain})")
                
                # Store task completion info in exploration path
                execution_id = f"{message.role}-{message.agent_id}"
                self._exploration_path.append(execution_id)
                self._exploration_results[execution_id] = {
                    "agent_id": message.agent_id,
                    "role": message.role,
                    "task_index": message.task_index,
                    "is_error": message.is_error,
                }
                if message.agent_id not in self._completed_agents_current_step:
                    self._completed_agents_current_step.add(message.agent_id)
                    logger.info(
                        f"Host received TaskComplete from {message.agent_id} for step '{self._current_step_name}' "
                        f"(Task {message.task_index}, More: {message.more_tasks_remain}, Error: {message.is_error})"
                    )
                    if  self._completed_agents_current_step == self._expected_agents_current_step:
                        # Finished all expected agents.
                        self._step_completion_event.set()
                else:
                    # Log if we receive a duplicate completion signal.
                    logger.warning(f"Host received duplicate TaskComplete from {message.agent_id} for step '{self._current_step_name}'.")
            else:
                # Ignore completions for steps other than the current one.
                logger.debug(
                    f"Host ignored TaskComplete for inactive step '{message.role}' (current: '{self._current_step_name}') from {message.agent_id}."
                )

        elif isinstance(message, TaskProcessingStarted):
            # Track which agents have started the current step.
            if message.role == self._current_step_name:
                if message.agent_id not in self._expected_agents_current_step:
                    self._expected_agents_current_step.add(message.agent_id)
                    logger.info(f"Host noted TaskStarted from {message.agent_id} for step '{self._current_step_name}'.")
                # else: Agent already known to have started.
            else:
                # Ignore starts for steps other than the current one.
                logger.debug(
                    f"Host ignored TaskStarted for inactive step '{message.role}' (current: '{self._current_step_name}') from {message.agent_id}."
                )
        
        # Handle conductor request to start running the flow
        elif isinstance(message, ConductorRequest):
            if not self._conductor_task:
                self._conductor_task = asyncio.create_task(self._run_flow(message=message))
            
        return None

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
    
    async def _wait_for_user(self) -> ManagerResponse:
        try: 
            response = await asyncio.wait_for(self._user_confirmation.get(), timeout=self.max_wait_time)
            logger.debug(f"Received manager response: confirm = {response.confirm}, halt = {response.halt}, interrupt = {response.interrupt}, selection = {response.selection}")
            return response
        except asyncio.TimeoutError:
            logger.warning(f"{self.id} hit timeout waiting for manager response after {self.max_wait_time} seconds.")
            return ManagerResponse(confirm=False)

    async def _wait_for_completions(self) -> None:
        """
        Check if enough agents have completed the current step to proceed.
        
        This method uses the completion_threshold_ratio to determine if enough
        agents have completed their tasks to move on to the next step.
        """
        if not self._current_step_name:
            # We aren't in a step, no use waiting.
            return 
        
        try: 
            logger.debug(f"Waiting for completion of '{self._current_step_name}'.")
            await asyncio.wait_for(self._step_completion_event.wait(), timeout=self.max_wait_time)
            logger.debug(f"Previous step '{self._current_step_name}' cleared, moving on.")
        except asyncio.TimeoutError:
            required_completions = ceil(len(self._expected_agents_current_step) * self.completion_threshold_ratio)
            msg= f"Timeout waiting for step completion. {self.id} heard back from {len(self._completed_agents_current_step)}/{len(self._expected_agents_current_step)} completed agents (required: {required_completions})."
            logger.warning(msg)
            if len(self._completed_agents_current_step) >= required_completions:
                logger.info(f"{msg}. Completion threshold reached for step '{self._current_step_name}'.")
                return
            logger.error(f"{msg}. Aborting.")
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
            
        Returns:
            An AgentOutput containing the next StepRequest
        """
        while True:
            # Wait for enough completions, with a timeout
            await self._wait_for_completions()
        
            if not self._participants:
                # Initialize from the message if not done yet
                self._participants = message.participants
                
            # Build enhanced context for decision making
            conductor_context = self._build_conductor_context(message)
            
            # Get the next step using the strategy provided by the specific host implementation
            step = await self._choose(message=message)

            if step.role == self.role:
                # Don't call ourselves please
                step.role = WAIT

            if step.role not in self._participants and step.role not in [WAIT, END]:
                logger.warning(f"Host could not find next step. Suggested {step.role}, which doesn't exist.")
                step.role = WAIT
            
            # If this is an END step, wrap it up
            if step.role == END:
                raise StopAsyncIteration
            
            # Get user confirmation if needed
            if step.role != WAIT and self.human_in_loop:
                confirmation = await self.request_user_confirmation(step)
                await self._input_callback(confirmation)
                manager_response: ManagerResponse = await self._wait_for_user()
                if not manager_response.confirm:
                    # User rejected the step
                    logger.info(f"User rejected step for role: {step.role}")
                    # Use a WAIT step instead to signal orchestrator to try again
                    step.role = WAIT

                    # TODO: Add some way to incorporate user feedback here.

            logger.info(f"Host calling for execution of role: {step.role}")
            await self._execute_step(step)

    async def _execute_step(self, step: StepRequest) -> None:
        """
        Execute a step by sending a message directly to the target agent.
        
        This method allows the host to bypass the orchestrator and directly
        handle the execution of steps, giving it more control over the
        conversation flow. It uses topic-based publishing to send messages
        to the agents.
        
        Args:
            step: The step to execute
        """
        if step.role == END or step.role == WAIT:
            logger.debug(f"Skipping direct execution for control step: {step.role}")
            return
        
        logger.info(f"Host executing step for role: {step.role}")
        
        # Create the message for the target agent
        message = StepRequest(role=step.role, records=getattr(self, '_records', []))
        
        # Set step as current for completion tracking
        self._current_step_name = step.role
        self._expected_agents_current_step.clear()
        self._completed_agents_current_step.clear()
        self._step_completion_event.clear()
        
        # Publish the message using the _input_callback if available
        # This callback is provided by the AutogenAgentAdapter during initialization
        # and allows publishing messages to topics
        if hasattr(self, '_input_callback') and self._input_callback:
            await self._input_callback(message)
        else:
            logger.warning(f"No publish callback available to execute step {step.role}")
            
        # Note: Completion will be tracked when TaskProcessingComplete messages are received
    
    def _build_conductor_context(self, message: ConductorRequest) -> Dict[str, Any]:
        """
        Build an enhanced context dictionary for decision making.
        
        This collects relevant information from the host's state and the incoming
        message to provide better context for the next step decision.
        
        Args:
            message: The ConductorRequest containing base context
            
        Returns:
            An enhanced context dictionary
        """
        # Start with the inputs from the message
        context = dict(message.inputs)
        
        # Add exploration history
        context.update({
            "exploration_path": self._exploration_path,
            "latest_results": (
                self._exploration_results.get(self._exploration_path[-1]) 
                if self._exploration_path else None
            ),
            "user_feedback": self._user_feedback,
        })
        
        # Clear feedback after using it
        self._user_feedback = []
        
        return context
    
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
        # if message:
        #     # We've been asked a specific question. This time we won't go with the flow, 
        #     # we should make sure we respond to it.
        #     return StepRequest(role=WAIT)

        step = await anext(self._step_generator)
        logger.debug(f"Host {self.id} suggests next step: {step.role}.")
        return step


    async def on_reset(self, cancellation_token=None) -> None:
        """Resets the Sequencer's internal state."""
        logger.info(f"Resetting Sequencer agent {self.id}...")
        await super().on_reset(cancellation_token)
        self._current_step_name = None
        self._completed_agents_current_step.clear()
        self._expected_agents_current_step.clear()
        self._participants.clear()
        # Re-initialize the step generator (will wait for participants again).
        self._step_generator = self._sequence()
        # Set completion event to ready state.
        self._step_completion_event.set()
        logger.info(f"Sequencer agent {self.id} reset complete.")

    async def _process(self, *, message: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs) -> AgentOutput | StepRequest | ManagerRequest | ManagerMessage | ToolOutput | ErrorEvent:
        pass  # this host does nothing when called
     