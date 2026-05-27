import asyncio
from collections import defaultdict
from collections.abc import (
    AsyncGenerator,
    Mapping,  # Import Dict
)
from typing import Any

from autogen_core import DefaultTopicId, MessageContext, message_handler
from autogen_core.models import AssistantMessage, UserMessage
from buttermilk._core.tool_types import Tool

from buttermilk import logger
from buttermilk._core.agent import Agent
from buttermilk._core.constants import COMMAND_SYMBOL, END, MANAGER, WAIT
from buttermilk._core.contract import (
    AgentAnnouncement,
    AgentInput,
    AgentOutput,
    ConductorRequest,
    ExecutionTrace,
    FlowEvent,
    FlowProgressUpdate,
    StepRequest,
    SystemPromptMessage,
    TaskProcessingComplete,
    TaskProcessingStarted,
    UserResponseMessage,
)
from buttermilk._core.exceptions import FatalError, ProcessingError
from buttermilk._core.types import BaseRecord

TRUNCATE_LEN = 1000  # characters per history message


class HostAgent(Agent):
    """Base coordinator for group chats and flow control.

    This agent acts as a conductor (`CONDUCTOR` role). It iterates through a
    predefined sequence of participant roles. For each step, it sends a request
    to the corresponding role. Before proceeding to the next step, it waits
    until all agents that have started processing a task have reported completion.
    Uses a dictionary to count pending tasks per agent ID and a Condition variable
    for synchronization.
    """

    def __init__(self, **kwargs):
        """Initialize HostAgent with all required attributes."""
        super().__init__(**kwargs)

        # Initialize private attributes that were previously using PrivateAttr
        self._message_types_handled: type[Any] = type(ConductorRequest)
        self._step_generator: AsyncGenerator[StepRequest, None] | None = None
        self._tasks_condition: asyncio.Condition = asyncio.Condition()
        self._step_starting: asyncio.Event = asyncio.Event()
        self._pending_tasks_by_agent: defaultdict[str, int] = defaultdict(int)
        self._participants: dict[str, Any] = {}
        self._host_initial_inputs: dict[str, Any] = {}
        # Error tracking for current step
        self._failed_tasks_by_agent: defaultdict[str, int] = defaultdict(int)
        self._total_tasks_in_step: int = 0
        self._conductor_task: asyncio.Task | None = None

        # Record propagation - stores latest record from agent outputs
        self._current_record: BaseRecord | None = None

        # Agent registry attributes
        self._agent_registry: dict[str, AgentAnnouncement] = {}
        self._registry_lock: asyncio.Lock = asyncio.Lock()
        self._tool_to_agent_map: dict[str, str] = {}  # Maps tool names to agent IDs
        self._registry_summary_cache: dict[str, Any] | None = None  # Cache for registry summaries

        # Tool schemas for LLM-based hosts
        self._tools: list[Tool] = []
        self._proposed_step: asyncio.Queue[StepRequest] = asyncio.Queue()
        self._current_step: str = ""

        # User confirmation attributes
        self._user_confirmation: UserResponseMessage | None = None
        self._user_confirmation_received: asyncio.Event = asyncio.Event()
        self._user_feedback: list[str] = []
        self._progress_reporter_task: asyncio.Task | None = None

        # Configuration parameters - stored internally, accessed via properties
        self._max_wait_time_value: int | None = None
        self._max_user_confirmation_time_value: int | None = None
        self._error_threshold_value: float | None = None

    # human_in_loop is now read from self.parameters instead of being a direct field
    @property
    def human_in_loop(self) -> bool:
        """Whether to interact with the human/manager for step confirmation.

        Must be explicitly configured in parameters - no defaults allowed.
        """
        if "human_in_loop" not in self.parameters:
            raise ValueError(f"Host agent '{self.agent_name}': 'human_in_loop' must be explicitly set in parameters")
        return self.parameters["human_in_loop"]

    @human_in_loop.setter
    def human_in_loop(self, value: bool) -> None:
        """Set the human_in_loop value in parameters."""
        self.parameters["human_in_loop"] = value

    @property
    def _max_wait_time(self) -> int:
        """Maximum time to wait for agent responses in seconds.

        Must be explicitly configured in parameters - no defaults allowed.
        """
        if self._max_wait_time_value is None:
            if "max_wait_time" not in self.parameters:
                raise ValueError(f"Host agent '{self.agent_name}': 'max_wait_time' must be explicitly set in parameters")
            self._max_wait_time_value = self.parameters["max_wait_time"]
        return self._max_wait_time_value

    @property
    def _max_user_confirmation_time(self) -> int:
        """Maximum time to wait for user confirmation in seconds.

        Must be explicitly configured in parameters - no defaults allowed.
        """
        if self._max_user_confirmation_time_value is None:
            if "max_user_confirmation_time" not in self.parameters:
                raise ValueError(f"Host agent '{self.agent_name}': 'max_user_confirmation_time' must be explicitly set in parameters")
            self._max_user_confirmation_time_value = self.parameters["max_user_confirmation_time"]
        return self._max_user_confirmation_time_value

    @property
    def _error_threshold(self) -> float:
        """Error tolerance: fraction of tasks that can fail before stopping flow.

        Must be explicitly configured in parameters - no defaults allowed.
        """
        if self._error_threshold_value is None:
            if "error_threshold" not in self.parameters:
                raise ValueError(f"Host agent '{self.agent_name}': 'error_threshold' must be explicitly set in parameters")
            self._error_threshold_value = self.parameters["error_threshold"]
        return self._error_threshold_value

    @message_handler
    async def handle_conductor_request(  # type: ignore
        self,
        message: ConductorRequest,
        ctx: MessageContext,
    ) -> None:
        """Handle ConductorRequest to start the flow."""
        logger.debug(
            "Host received ConductorRequest",
            agent_name=self.agent_name,
            num_participants=len(message.participants),
            participants=list(message.participants.keys()),
        )
        await super().handle_conductor_request(message=message, ctx=ctx)

        if hasattr(self, "_conductor_task") and self._conductor_task:
            logger.warning(
                "Host received ConductorRequest but task is already running.",
                agent_name=self.agent_name,
            )
            return

        # Store the participants from the message
        self._participants = dict(message.participants)

        self._conductor_task = "starting"  # Mark as starting to avoid re-entrance
        logger.debug("Host starting new conductor task", agent_name=self.agent_name)
        self._conductor_task = asyncio.create_task(self._run_flow(message=message))

    @message_handler
    async def handle_task_complete(
        self,
        message: TaskProcessingComplete,
        ctx: MessageContext,
    ) -> None:
        """Handle task completion signals from worker agents."""
        self._step_starting.clear()  # Clear this event as soon as any task completes
        agent_id_to_update = message.agent_id

        async with self._tasks_condition:
            if agent_id_to_update in self._pending_tasks_by_agent:
                self._pending_tasks_by_agent[agent_id_to_update] -= 1

                # Track errors if this task failed
                if message.is_error:
                    self._failed_tasks_by_agent[agent_id_to_update] += 1
                    logger.warning(
                        "Host noted TaskComplete with ERROR from agent",
                        agent_id=agent_id_to_update,
                        role=message.role,
                        failed_tasks=dict(self._failed_tasks_by_agent),
                    )

                if self._pending_tasks_by_agent[agent_id_to_update] <= 0:
                    del self._pending_tasks_by_agent[agent_id_to_update]

                logger.debug(
                    "Host noted TaskComplete from agent",
                    agent_id=agent_id_to_update,
                    role=message.role,
                    error=message.is_error,
                    pending_tasks=dict(self._pending_tasks_by_agent),
                )
                self._tasks_condition.notify_all()
            else:
                logger.debug(
                    "Host received TaskComplete from agent but it was not in pending tasks.",
                    agent_id=agent_id_to_update,
                    role=message.role,
                )

    @message_handler
    async def handle_task_started(
        self,
        message: TaskProcessingStarted,
        ctx: MessageContext,
    ) -> None:
        """Handle task start signals from worker agents."""
        agent_id_to_update = message.agent_id

        async with self._tasks_condition:
            self._total_tasks_in_step += 1
            self._pending_tasks_by_agent[agent_id_to_update] += 1
            logger.debug(
                "Host noted TaskStarted from agent",
                agent_id=agent_id_to_update,
                role=message.role,
                pending_tasks=dict(self._pending_tasks_by_agent),
                total_tasks_in_step=self._total_tasks_in_step,
            )

    @message_handler
    async def handle_flow_progress_update(
        self,
        message: FlowProgressUpdate,
        ctx: MessageContext,
    ) -> None:
        """Handle FlowProgressUpdate messages."""
        logger.debug(
            "Host received FlowProgressUpdate message. Ignoring.",
            agent_name=self.agent_name,
        )
        # Do nothing with progress updates received by the host

    @message_handler
    async def handle_agent_trace(
        self,
        message: ExecutionTrace,
        ctx: MessageContext,
    ) -> None:
        """Handle ExecutionTrace messages and add to conversation history.

        Also captures any BaseRecord in message.outputs for propagation to subsequent steps.
        This ensures records flow through the pipeline (e.g., FETCH -> JUDGE -> SYNTHESISER).
        """
        content_to_log = str(message.content)[:TRUNCATE_LEN]
        await self._model_context.add_message(
            AssistantMessage(content=content_to_log, source=ctx.sender.key if ctx.sender else ""),
        )

        # Capture record output for propagation to next steps
        if message.outputs is not None and isinstance(message.outputs, BaseRecord):
            self._current_record = message.outputs
            logger.debug(
                "Host captured record from agent trace for propagation",
                agent_name=self.agent_name,
                record_id=self._current_record.record_id,
                source_agent=ctx.sender.key if ctx.sender else "unknown",
            )

    @message_handler
    async def handle_manager_message(
        self,
        message: UserResponseMessage,
        ctx: MessageContext,
    ) -> None:
        """Handle UserResponseMessage for user confirmations and feedback."""
        logger.debug("Host received user input", agent_name=self.agent_name, message=message)
        self._user_confirmation = message
        self._user_confirmation_received.set()

        # Handle halt request - user wants to stop the entire flow
        if message.halt:
            logger.debug(
                "Host received halt request from user - terminating flow",
                agent_name=self.agent_name,
            )
            # Send END message to signal flow termination
            end_step = StepRequest(role=END, content="Flow halted by user request")
            await self._publish(end_step)
            return

        if message.human_in_loop is not None and self.human_in_loop != message.human_in_loop:
            logger.debug(
                "Host received user request to set human in the loop",
                agent_name=self.agent_name,
                new_value=message.human_in_loop,
                old_value=self.human_in_loop,
            )
            self.human_in_loop = message.human_in_loop

        content = getattr(message, "content", getattr(message, "params", None))
        if content and not str(content).startswith(COMMAND_SYMBOL):
            content_to_log = str(content)[:TRUNCATE_LEN]
            # store in user feedback separately as well
            self._user_feedback.append(content)
            # Add to conversation history
            await self._model_context.add_message(
                UserMessage(content=content_to_log, source=ctx.sender.key if ctx.sender else ""),
            )

    # --- Agent Registry Methods ---

    @message_handler
    async def update_agent_registry(
        self,
        message: AgentAnnouncement,
        ctx: MessageContext = None,
    ) -> None:
        """Update registry with agent announcement.

        Thread-safe update of agent registry.

        Args:
            announcement: The agent announcement to process.

        """
        async with self._registry_lock:
            agent_id = message.agent_config.agent_id
            message.agent_config.role.upper()  # Normalize to uppercase

            if message.status == "leaving":
                logger.warning(
                    "Host received notification to remove agent, but functionality is not implemented.",
                    agent_name=self.agent_name,
                    agent_id=agent_id,
                )
            else:
                # Add or update agent in registry
                self._agent_registry[agent_id] = message
                # Update tool registry
                self._tools.extend(message.tool_definitions)

                # Build tool-to-agent mapping from the tools this agent provides
                tool_names = []
                for tool in message.tool_definitions:
                    # Extract tool name from either name attribute or schema.name
                    tool_name = None
                    if isinstance(tool, Tool):
                        tool_name = getattr(tool, "name", None)
                    elif hasattr(tool, "schema"):
                        tool_name = getattr(tool.schema, "name", None)
                    elif isinstance(tool, Mapping):
                        tool_name = tool.get("name") or tool.get("schema", {}).get("name")
                    if tool_name:
                        self._tool_to_agent_map[tool_name] = agent_id
                        tool_names.append(tool_name)
                    else:
                        tool_names.append("unknown")

                logger.debug(
                    "Host registered agent",
                    agent_name=self.agent_name,
                    agent_id=agent_id,
                    tools=tool_names,
                )

            # Invalidate cache
            self._registry_summary_cache = None

    def create_registry_summary(self) -> dict[str, Any]:
        """Create a summary of the agent registry for UI display.

        Uses caching to avoid redundant computation.

        Returns:
            dict: Summary containing active agents, available tools, and counts.

        """
        # Return cached summary if available
        if self._registry_summary_cache is not None:
            return self._registry_summary_cache

        active_agents = []
        for agent_id, announcement in self._agent_registry.items():
            agent_info = {
                "agent_id": agent_id,
                "role": announcement.agent_config.role,
                "description": announcement.agent_config.description,
                "tools": announcement.available_tools,
                "model": announcement.agent_config.parameters.get("model") if announcement.agent_config.parameters else None,
            }
            active_agents.append(agent_info)

        summary = {
            "active_agents": active_agents,
            "total_agents": len(self._agent_registry),
        }

        # Cache the summary
        self._registry_summary_cache = summary
        return summary

    # TODO: convert this to a handler for a command message
    def create_ui_message_with_registry(
        self,
        content: str,
        options: bool | list[str] | None = None,
        **kwargs: Any,
    ) -> SystemPromptMessage:
        """Create a UI message that includes the agent registry summary.

        Args:
            content: The message content.
            options: Optional interaction options.
            **kwargs: Additional SystemPromptMessage fields.

        Returns:
            SystemPromptMessage: UI message with registry summary.

        """
        registry_summary = self.create_registry_summary()
        return SystemPromptMessage(
            content=content,
            options=options,
            agent_registry_summary=registry_summary,
            **kwargs,
        )

    async def request_user_confirmation(self, step: StepRequest) -> None:
        """Request confirmation from the user for the next step.

        This method is used when human_in_loop is True to get user approval
        before executing a step.

        Args:
            step: The proposed next step

        Returns:
            None

        """
        self._user_confirmation_received.clear()
        # Send the request to the user
        confirmation_request = SystemPromptMessage(
            content=step.content or f"Confirm next step: {step.role}",
            options=["confirm", "reject"],
        )
        # This is callback to groupchat, because we're the host agent right now.
        # Our message goes to the groupchat and then gets picked up by the UI.
        await self._publish(confirmation_request)

    async def _wait_for_user(self, step) -> bool:
        """Wait for user confirmation before proceeding with the next step.

        Returns:
            bool: True if user confirmed, False if rejected or timed out

        """
        logger.info(
            "Host waiting for user confirmation",
            agent_name=self.agent_name,
            step_role=step.role,
        )
        max_tries = self._max_user_confirmation_time // 60
        for _ in range(max_tries):
            logger.debug(
                "Host waiting for user confirmation",
                agent_name=self.agent_name,
                step_role=step.role,
            )
            try:
                await self.request_user_confirmation(step)
                await asyncio.wait_for(self._user_confirmation_received.wait(), timeout=60)
            except TimeoutError:
                logger.info(
                    "Host hit timeout waiting for manager response after 60 seconds.",
                    agent_name=self.agent_name,
                )
                continue

            if self._user_confirmation and getattr(self._user_confirmation, "confirm", False):
                logger.info("User confirmed step", step_role=step.role)
                return True
            logger.info("User rejected step", step_role=step.role)
            return False
        msg = "User did not respond to confirm step in time. Ending flow."
        logger.error(msg)
        await self._publish(StepRequest(role=END, content=msg))
        return False

    async def _report_progress_periodically(self, interval: int = 10):
        """Periodically report the status of pending tasks."""
        try:
            while True:
                await asyncio.sleep(interval)
                async with self._tasks_condition:
                    if self._pending_tasks_by_agent:
                        progress_message = FlowProgressUpdate(
                            source=self.agent_id,
                            status="pending",
                            step_name=self._current_step,
                            waiting_on=dict(self._pending_tasks_by_agent),
                            message="Current pending tasks",
                        )
                    else:
                        progress_message = FlowProgressUpdate(
                            source=self.agent_id,
                            status="idle",
                            step_name="IDLE",
                            waiting_on=dict(),
                            message="Flow currently idle",
                        )
                    await self._publish(progress_message)
        except asyncio.CancelledError:
            logger.debug("Progress reporting task cancelled.")
        except Exception as e:
            logger.exception("Error in progress reporting task", error=e)
        finally:
            logger.debug("Host progress reporting task terminated.", agent_name=self.agent_name)

    async def _wait_for_all_tasks_complete(self) -> bool:
        """Wait until all tasks are completed, reporting progress periodically."""
        try:
            async with self._tasks_condition:
                if self._pending_tasks_by_agent:
                    logger.debug(
                        "Waiting for pending tasks to complete",
                        pending_from=list(self._pending_tasks_by_agent.keys()),
                    )

                # Calculate dynamic timeout based on number of tasks
                # Base timeout + (120 seconds per task / 6 parallel capacity)
                # With 6 parallel tasks potentially taking 2 minutes total
                # TODO: Need to add some allowance for rate limits
                # But capped between at no more than 5 minutes per step
                total_pending_tasks = sum(self._pending_tasks_by_agent.values())
                additional_time = (total_pending_tasks * 60) / 6  # Assuming 6 parallel workers
                calculated_timeout = self._max_wait_time + additional_time

                dynamic_timeout = max(120, min(calculated_timeout, 300))

                logger.debug(
                    "Using dynamic timeout",
                    timeout=f"{dynamic_timeout:.0f}s",
                    pending_tasks=total_pending_tasks,
                )

                # wait_for releases the lock, waits for notification and predicate, then reacquires
                # The predicate checks if _step_starting is clear AND _pending_tasks_by_agent is empty.
                # This means we wait until the step is no longer considered "starting" AND all tasks are done. This provides insurance where
                # distributed tasks take a while to begin.
                await asyncio.wait_for(
                    self._tasks_condition.wait_for(lambda: not self._step_starting.is_set() and not self._pending_tasks_by_agent),
                    timeout=dynamic_timeout,
                )
                return True
        except TimeoutError:
            # Lock is released automatically on timeout exception from wait_for
            # Note: We can't access total_pending_tasks here as it was in the async with block
            # but we can still show the pending tasks
            msg = (
                f"Timeout waiting for task completion. "
                f"Pending tasks will be treated as failed: {dict(self._pending_tasks_by_agent)}. "
                f"Flow will continue if error threshold not exceeded."
            )
            logger.warning(msg)
            self._step_starting.clear()  # Reset the event to allow for new steps
            return False  # Indicate timeout occurred
        except Exception as e:
            # Catch other potential errors during wait
            logger.exception("Unexpected error during task completion wait", error=e)
            self._step_starting.clear()  # Reset on error too
            return False  # Indicate failure due to error

    async def _sequence(self) -> AsyncGenerator[StepRequest, None]:
        """Generate a sequence of steps to execute.

        Uses captured record from previous agent outputs to propagate data through the flow.
        The first step uses the record from initial inputs, subsequent steps use the
        most recent record captured from agent ExecutionTrace outputs.

        Yields:
            StepRequest: The next step request in the sequence.

        """
        for role, description in self._participants.items():
            # Create more descriptive step content using the participant description
            step_description = f"Executing {role.lower()} step: {description}"

            # Separate record from inputs dict - record should be in the record field, not inputs
            step_inputs = self._host_initial_inputs.copy()
            initial_record = step_inputs.pop("record", None)

            # Use the most recent captured record from agent outputs, or fall back to initial record
            record = self._current_record or initial_record

            # Reconstruct record if needed
            if record:
                # If it's a list, take the last item (most recent)
                if isinstance(record, list):
                    record = record[-1] if record else None

                # If it's a dict, reconstruct as BaseRecord
                if isinstance(record, dict):
                    record = BaseRecord.from_dict(record)

            yield StepRequest(
                role=role,
                content=step_description,
                inputs=step_inputs,
                record=record,
            )
        yield StepRequest(role=END, content="Sequence completed.")

    async def _run_flow(self, message: ConductorRequest) -> None:
        """Run the predefined sequence of steps, waiting for task completion between steps.

        Args:
            message: The initial request containing participants.

        Returns:
            StepRequest: The final END step request.

        Raises:
            FatalError: If no participants are provided in the request.

        """
        try:
            logger.debug(
                "Host starting flow execution",
                agent_name=self.agent_name,
                num_participants=len(self._participants),
                participants=list(self._participants.keys()),
            )

            if not self._participants:
                msg = "Host received ConductorRequest with no participants."
                logger.error(msg, aborting=True)
                # Send an END message with the error
                await self._publish(StepRequest(role=END, content=msg))
                raise FatalError(msg)

            # Store additional tools if provided
            self._tools.extend(message.additional_tools)

            # Store any parameters passed in
            self._host_initial_inputs = message.inputs

            # Parameters will be passed to agents via StepRequest

            # Announce, and trigger agents to announce themselves
            hello_message = f"Starting a new flow with parameters: {message.parameters} and participants: {', '.join(self._participants.keys())}"

            msg = AgentAnnouncement(
                content=hello_message,
                agent_config=self._config,
                announcement_type="initial",
            )
            await self._publish(msg)

            # Start the periodic progress reporter task
            self._progress_reporter_task = asyncio.create_task(self._report_progress_periodically())

            # Initialize generator now that participants are known
            self._step_generator = self._sequence()
            logger.debug(
                "Host participants initialized",
                participants=list(self._participants.keys()),
            )

            flow_stopped_early = False
            early_stop_reason = ""

            async for next_step in self._step_generator:
                logger.debug(
                    f"Host processing step {next_step.role}",
                    agent_name=self.agent_name,
                    step_role=next_step.role,
                )

                if self.human_in_loop and next_step.role != MANAGER and not await self._wait_for_user(next_step):
                    # If user rejected or timed out, stop the flow
                    flow_stopped_early = True
                    early_stop_reason = "Flow stopped: user rejected or timed out"
                    break

                # Execute the current step
                await self._execute_step(next_step)

                # Wait for the current step to complete before moving to the next one
                # Skip this check for END steps since they don't generate tasks
                if next_step.role != END:
                    if not await self.wait_check_current_step_completions():
                        logger.debug(
                            "Step completion check failed - stopping flow",
                            agent_name=self.agent_name,
                        )
                        flow_stopped_early = True
                        early_stop_reason = "Flow stopped: error threshold exceeded or step failed"
                        break

            # --- Sequence finished ---
            logger.debug("Host flow execution finished.", agent_name=self.agent_name)

            # Send END message if we stopped early
            if flow_stopped_early:
                logger.warning(
                    "Sending END message due to early termination",
                    agent_name=self.agent_name,
                    reason=early_stop_reason,
                )
                await self._publish(StepRequest(role=END, content=early_stop_reason))
                # Signal flow failure to pipeline via TaskProcessingComplete
                await self._publish(
                    TaskProcessingComplete(
                        agent_id=self.agent_id,
                        role=self.role,
                        is_error=True,
                        error=early_stop_reason,
                    ),
                    topic_id=self._topic_id,
                )

            # Send final progress update before any cleanup begins
            final_progress_message = FlowProgressUpdate(
                source=self.agent_id,
                status="finished",
                step_name=END,
                waiting_on={},
                message="Flow completed",
            )
            logger.debug(
                "Host sending final progress update before cleanup.",
                agent_name=self.agent_name,
            )
            await self._publish(final_progress_message)

        except KeyboardInterrupt:
            logger.info("Flow terminated by user.")
            # Send END message to terminate orchestrator
            await self._publish(StepRequest(role=END, content="Flow terminated by user interrupt"))
        except (FatalError, Exception) as e:
            logger.exception("Unexpected and unhandled fatal error", error=e)
            # Send END message to terminate orchestrator even when exceptions occur
            error_message = f"Flow terminated due to error: {type(e).__name__}: {str(e)[:200]}"
            await self._publish(StepRequest(role=END, content=error_message))
        finally:
            # Cancel the progress reporter task
            if self._progress_reporter_task:
                self._progress_reporter_task.cancel()
                try:
                    await self._progress_reporter_task
                except asyncio.CancelledError:
                    pass

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
            self._tasks_condition.release()
            # Ensure the progress reporter task is cancelled when the wait finishes or times out
            if self._progress_reporter_task:
                self._progress_reporter_task.cancel()
                try:
                    await self._progress_reporter_task  # Await cancellation
                except asyncio.CancelledError:
                    pass  # Expected
        logger.debug("Host shutdown complete.", agent_name=self.agent_name)

    async def wait_check_current_step_completions(self) -> bool:
        """Wait for tasks from the current step to complete and check for errors."""
        # Wait for pending tasks to complete
        last_step_successful = await self._wait_for_all_tasks_complete()

        # If timeout occurred, treat timed-out tasks as errors
        if not last_step_successful:
            # Record timed-out tasks as failures
            async with self._tasks_condition:
                timed_out_agents = dict(self._pending_tasks_by_agent)  # Copy pending tasks
                for agent_id, pending_count in timed_out_agents.items():
                    self._failed_tasks_by_agent[agent_id] += pending_count
                    logger.warning(
                        "Host marking timed-out tasks as failed",
                        agent_id=self.agent_id,
                        timed_out_agent_id=agent_id,
                        pending_count=pending_count,
                    )
                # Clear the pending tasks since we're treating them as completed (with errors)
                self._pending_tasks_by_agent.clear()

        # Check if too many tasks failed (including timeouts)
        total_failed = sum(self._failed_tasks_by_agent.values())
        if self._total_tasks_in_step == 0:
            logger.warning("Host encountered no tasks in the current step.", agent_id=self.agent_id)
            error_ratio = 0  # No tasks started, error ratio is undefined or treated as 0
        else:
            error_ratio = total_failed / self._total_tasks_in_step

        if error_ratio > self._error_threshold:
            logger.error(
                "Host stopping flow: error threshold exceeded",
                agent_id=self.agent_id,
                total_failed=total_failed,
                total_tasks=self._total_tasks_in_step,
                error_ratio=error_ratio,
                error_threshold=self._error_threshold,
                failed_tasks_by_agent=dict(self._failed_tasks_by_agent),
            )
            return False

        if total_failed > 0:
            logger.warning(
                "Host proceeding despite failed tasks",
                agent_id=self.agent_id,
                failed_tasks=f"{total_failed}/{self._total_tasks_in_step}",
                error_ratio=f"{error_ratio:.1%}",
                threshold=f"{self._error_threshold:.1%}",
            )

        # Clear error tracking for the next step
        async with self._tasks_condition:
            self._failed_tasks_by_agent.clear()
            self._pending_tasks_by_agent.clear()  # Defensive: should already be empty
            self._total_tasks_in_step = 0

        return True

    async def _execute_step(self, step: StepRequest) -> None:
        """Process a single step."""
        self._current_step = step.role
        if step.role == WAIT:
            logger.debug("Host waiting for 10 seconds as requested by WAIT step.")
            await asyncio.sleep(10)
        elif step.role == END:
            logger.debug("Flow completed and all tasks finished. Sending END signal", step=step)
            await self._publish(step)
        else:
            if step.role in self._participants:
                # Signal that we expect at least one response/task start for this step
                # This event is used in the wait_for predicate.
                self._step_starting.set()
                logger.debug("Host set _step_starting event", role=step.role)

                # Send a FlowEvent to the main topic to notify UI about the step starting
                flow_event = FlowEvent(
                    source=self.agent_id,
                    content=f"Starting {step.role} step: {self._participants.get(step.role, step.role)}",
                )
                await self._publish(flow_event)  # This goes to the main topic

            elif step.role == MANAGER:
                # MANAGER steps don't spawn trackable worker tasks, so don't set _step_starting
                # Convert StepRequest to SystemPromptMessage for frontend display
                ui_message = SystemPromptMessage(
                    content=step.content or "What would you like to do?",
                    options=None,  # No specific options, just free text response
                )
                await self._publish(ui_message)
                return  # Don't send the StepRequest itself
            else:
                logger.warning("Host executing step for unknown participant role", role=step.role)

            # Route StepRequest to role-specific topic
            role_topic = DefaultTopicId(type=step.role)
            await self._publish(step, topic_id=role_topic)

    async def _process(
        self,
        *,
        message: AgentInput,
        **kwargs: Any,
    ) -> AgentOutput | None:
        """Process messages.

        Base implementation returns an error since non-LLM hosts don't process direct inputs.
        Subclasses that support LLM-based processing should override this method.
        """
        raise ProcessingError("Host agent does not process direct inputs via _process")

    async def _route_tool_calls_to_agents(
        self,
        tool_calls: list[Any],  # FunctionCall objects
    ) -> None:
        """Route tool calls to the appropriate agents as StepRequests.

        This is a helper method that can be used by LLM-based host subclasses
        to convert tool calls into StepRequests for the appropriate agents.
        """
        import json

        for call in tool_calls:
            # Look up the agent that owns this tool
            agent_id = self._tool_to_agent_map.get(call.name)
            if not agent_id:
                logger.error(
                    "No agent found for tool",
                    tool_name=call.name,
                    available_tools=list(self._tool_to_agent_map.keys()),
                )
                continue

            # Get the agent's role from the registry
            agent_announcement = self._agent_registry.get(agent_id)
            if not agent_announcement:
                logger.error(
                    "Agent not found in registry for tool",
                    agent_id=agent_id,
                    tool_name=call.name,
                )
                continue

            role = agent_announcement.agent_config.role.upper()

            # Parse the arguments
            try:
                arguments = json.loads(call.arguments)
                if "inputs" in arguments:
                    # If inputs are present, use them directly
                    arguments = arguments["inputs"]
            except json.JSONDecodeError:
                logger.error("Failed to parse tool arguments", arguments=call.arguments)
                continue

            step_request = StepRequest(
                role=role,
                inputs=arguments,
                metadata={"tool_name": call.name, "tool_call_id": call.id},
            )

            # Create a more descriptive log message
            tool_desc = self._describe_tool_call(call.name, arguments)
            logger.debug(
                "Host routing tool to agent",
                tool_name=call.name,
                agent_id=agent_id,
                role=role,
                tool_description=tool_desc,
            )

            if self.human_in_loop:
                await self._proposed_step.put(step_request)
            else:
                # If human_in_loop is False, we send the step request directly
                logger.debug(
                    "Host routing tool call to agent",
                    agent_name=self.agent_name,
                    agent_id=agent_id,
                    step_request=step_request,
                )
                # Route to role-specific topic
                role_topic = DefaultTopicId(type=role)
                await self._publish(step_request, topic_id=role_topic)

    def _describe_tool_call(self, tool_name: str, arguments: dict) -> str:
        """Generate a concise description of a tool call.

        Args:
            tool_name: Name of the tool being called
            arguments: Parsed arguments for the tool

        Returns:
            str: A human-readable description

        """
        # Common patterns for better descriptions
        if "query" in arguments:
            query = str(arguments["query"])
            return f"{tool_name}('{query[:40]}{'...' if len(query) > 40 else ''}')"
        if "message" in arguments:
            msg = str(arguments["message"])
            return f"{tool_name}('{msg[:40]}{'...' if len(msg) > 40 else ''}')"
        if "content" in arguments:
            content = str(arguments["content"])
            return f"{tool_name}('{content[:40]}{'...' if len(content) > 40 else ''}')"
        if "target" in arguments:
            return f"{tool_name}(target='{arguments['target']}')"
        if "inputs" in arguments and isinstance(arguments["inputs"], dict):
            # Handle nested inputs
            if "query" in arguments["inputs"]:
                query = str(arguments["inputs"]["query"])
                return f"{tool_name}(query='{query[:30]}{'...' if len(query) > 30 else ''}')"
            return f"{tool_name}({len(arguments['inputs'])} inputs)"
        # Show first meaningful argument
        for key, value in arguments.items():
            if key not in ["metadata", "context", "options"]:
                val_str = str(value)
                return f"{tool_name}({key}='{val_str[:30]}{'...' if len(val_str) > 30 else ''}')"

        # Fallback
        return f"{tool_name}({len(arguments)} args)"
