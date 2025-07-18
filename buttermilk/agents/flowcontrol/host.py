import asyncio
from collections import defaultdict
from collections.abc import AsyncGenerator
from typing import Any  # Import Dict

from autogen_core import CancellationToken, MessageContext, message_handler
from autogen_core.models import AssistantMessage, UserMessage
from autogen_core.tools import (
    Tool,
)

from buttermilk import logger
from buttermilk._core.agent import Agent
from buttermilk._core.constants import COMMAND_SYMBOL, END, MANAGER, WAIT
from buttermilk._core.contract import (
    AgentAnnouncement,
    AgentInput,
    AgentOutput,
    AgentTrace,
    ConductorRequest,
    ErrorEvent,
    FlowProgressUpdate,
    ManagerMessage,
    StepRequest,
    TaskProcessingComplete,
    TaskProcessingStarted,
    UIMessage,
)
from buttermilk._core.exceptions import FatalError

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
        self._conductor_task: asyncio.Task | None = None

        # Agent registry attributes
        self._agent_registry: dict[str, AgentAnnouncement] = {}
        self._registry_lock: asyncio.Lock = asyncio.Lock()

        # Tool schemas for LLM-based hosts
        self._tools: list[Tool] = []
        self._proposed_step: asyncio.Queue[StepRequest] = asyncio.Queue()
        self._current_step: str = ""

        # User confirmation attributes
        self._user_confirmation: ManagerMessage | None = None
        self._user_confirmation_received: asyncio.Event = asyncio.Event()
        self._user_feedback: list[str] = []
        self._progress_reporter_task: asyncio.Task | None = None

        # Maximum time to wait for agent responses in seconds
        self._max_wait_time: int = kwargs.get("max_wait_time", 240)

        # Maximum time to wait for agent responses in seconds
        self._max_user_confirmation_time: int = kwargs.get("max_user_confirmation_time", 1220)

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

    @message_handler
    async def handle_conductor_request(  # type: ignore
        self,
        message: ConductorRequest,
        ctx: MessageContext,
    ) -> None:
        """Handle ConductorRequest to start the flow."""
        await super().handle_conductor_request(message, ctx)

        logger.info(
            f"[HostAgent.handle_conductor_request] Host {self.agent_name} received ConductorRequest "
            f"with {len(message.participants)} participants: {list(message.participants.keys())}",
        )

        if hasattr(self, "_conductor_task") and self._conductor_task:
            logger.warning(f"[HostAgent.handle_conductor_request] Host {self.agent_name} received ConductorRequest but task is already running.")
            return

        # Store the participants from the message
        self._participants = dict(message.participants)

        self._conductor_task = "starting"  # Mark as starting to avoid re-entrance
        logger.debug(f"[HostAgent.handle_conductor_request] Host {self.agent_name} starting new conductor task")
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
                if self._pending_tasks_by_agent[agent_id_to_update] <= 0:
                    del self._pending_tasks_by_agent[agent_id_to_update]

                logger.info(
                    f"Host noted TaskComplete from agent {agent_id_to_update} for role '{message.role}'. "
                    f"Pending tasks: {dict(self._pending_tasks_by_agent)}.",
                )
                self._tasks_condition.notify_all()
            else:
                logger.warning(
                    f"Host received TaskComplete from agent {agent_id_to_update} but it was not in pending tasks.",
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
            self._pending_tasks_by_agent[agent_id_to_update] += 1
            logger.info(
                f"Host noted TaskStarted from agent {agent_id_to_update} for role '{message.role}'. "
                f"Pending tasks: {dict(self._pending_tasks_by_agent)}.",
            )

    @message_handler
    async def handle_flow_progress_update(
        self,
        message: FlowProgressUpdate,
        ctx: MessageContext,
    ) -> None:
        """Handle FlowProgressUpdate messages."""
        logger.debug(f"Host {self.agent_name} received FlowProgressUpdate message. Ignoring.")
        # Do nothing with progress updates received by the host

    @message_handler
    async def handle_agent_trace(
        self,
        message: AgentTrace,
        ctx: MessageContext,
    ) -> None:
        """Handle AgentTrace messages and add to conversation history."""
        content_to_log = str(message.content)[:TRUNCATE_LEN]
        await self._model_context.add_message(
            AssistantMessage(content=content_to_log, source=ctx.sender.key if ctx.sender else ""),
        )

    @message_handler
    async def handle_manager_message(
        self,
        message: ManagerMessage,
        ctx: MessageContext,
    ) -> None:
        """Handle ManagerMessage for user confirmations and feedback."""
        logger.info(f"Host {self.agent_name} received user input: {message}")
        self._user_confirmation = message
        self._user_confirmation_received.set()

        if message.human_in_loop is not None and self.human_in_loop != message.human_in_loop:
            logger.info(
                f"Host {self.agent_name} received user request to set human in the loop to {message.human_in_loop} (was {self.human_in_loop})",
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
        ctx: MessageContext,
    ) -> None:
        """Update registry with agent announcement.
        
        Thread-safe update of agent registry.
        
        Args:
            announcement: The agent announcement to process.

        """
        async with self._registry_lock:
            agent_id = message.agent_config.agent_id
            role = message.agent_config.role.upper()  # Normalize to uppercase

            if message.status == "leaving":
                logger.warning(f"Host {self.agent_name} received notification to remove agent {agent_id}, but functionality is not implemented.")
            else:
                # Add or update agent in registry
                self._agent_registry[agent_id] = message
                # Update tool registry
                self._tools.extend(message.tool_definitions)
                logger.info(f"Host {self.agent_name} registered agent {agent_id} with tools: {[tool.name for tool in message.tool_definitions]}")

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
    ) -> UIMessage:
        """Create a UI message that includes the agent registry summary.
        
        Args:
            content: The message content.
            options: Optional interaction options.
            **kwargs: Additional UIMessage fields.
            
        Returns:
            UIMessage: UI message with registry summary.

        """
        registry_summary = self.create_registry_summary()
        return UIMessage(
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
        confirmation_request = UIMessage(
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
        logger.info(f"Host {self.agent_name}: _wait_for_user called for step {step.role}")
        max_tries = self._max_user_confirmation_time // 60
        for _ in range(max_tries):
            logger.info(f"Host {self.agent_name} waiting for user confirmation for {step.role} step.")
            try:
                await self.request_user_confirmation(step)
                await asyncio.wait_for(self._user_confirmation_received.wait(), timeout=60)
            except TimeoutError:
                logger.info(f"{self.agent_name} hit timeout waiting for manager response after 60 seconds.")
                continue

            if self._user_confirmation and getattr(self._user_confirmation, "confirm", False):
                logger.info(f"User confirmed step: {step.role}")
                return True
            logger.info(f"User rejected step: {step.role}")
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
            logger.exception(f"Error in progress reporting task: {e}")
        finally:
            logger.debug(f"Host {self.agent_name} progress reporting task terminated.")

    async def _wait_for_all_tasks_complete(self) -> bool:
        """Wait until all tasks are completed, reporting progress periodically."""
        try:
            async with self._tasks_condition:
                if self._pending_tasks_by_agent:
                    logger.info(f"Waiting for pending tasks to complete from: {list(self._pending_tasks_by_agent.keys())}...")

                # wait_for releases the lock, waits for notification and predicate, then reacquires
                # The predicate checks if _step_starting is clear AND _pending_tasks_by_agent is empty.
                # This means we wait until the step is no longer considered "starting" AND all tasks are done. This provides insurance where
                # distributed tasks take a while to begin.
                await asyncio.wait_for(
                    self._tasks_condition.wait_for(lambda: not self._step_starting.is_set() and not self._pending_tasks_by_agent),
                    timeout=self._max_wait_time,
                )
                return True
        except TimeoutError:
            # Lock is released automatically on timeout exception from wait_for
            msg = (
                f"Timeout waiting for task completion condition. "
                f"Pending tasks: {dict(self._pending_tasks_by_agent)} after {self._max_wait_time}s."
            )
            logger.warning(msg)
            self._step_starting.clear()  # Reset the event to allow for new steps
            return False  # Indicate failure due to timeout
        except Exception as e:
            # Catch other potential errors during wait
            logger.exception(f"Unexpected error during task completion wait: {e}")
            self._step_starting.clear()  # Reset on error too
            return False  # Indicate failure due to error

    async def _sequence(self) -> AsyncGenerator[StepRequest, None]:
        """Generate a sequence of steps to execute.

        Yields:
            StepRequest: The next step request in the sequence.

        """
        for role, description in self._participants.items():
            # Create more descriptive step content using the participant description
            step_description = f"Executing {role.lower()} step: {description}"

            # Create StepRequest with initial parameters if available
            step_inputs = {}
            step_parameters = {}

            # Add initial query/prompt if available
            if hasattr(self, "_initial_query") and self._initial_query:
                step_inputs["query"] = self._initial_query
                step_inputs["prompt"] = self._initial_query

            # Add any parameters from the initial inputs
            if hasattr(self, "_initial_inputs") and isinstance(self._initial_inputs, dict):
                # Extract parameters from the initial inputs
                if "parameters" in self._initial_inputs:
                    step_parameters.update(self._initial_inputs["parameters"])

                # Also check for direct prompt in initial inputs
                if "prompt" in self._initial_inputs and "prompt" not in step_inputs:
                    step_inputs["prompt"] = self._initial_inputs["prompt"]
                if "query" in self._initial_inputs and "query" not in step_inputs:
                    step_inputs["query"] = self._initial_inputs["query"]

            yield StepRequest(
                role=role,
                content=step_description,
                inputs=step_inputs,
                parameters=step_parameters,
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
            logger.info(f"Host {self.agent_name} starting flow execution.")

            logger.info(f"Host {self.agent_name} has {len(self._participants)} participants after update: {list(self._participants.keys())}")
            if not self._participants:
                msg = "Host received ConductorRequest with no participants."
                logger.error(f"{msg} Aborting.")
                # Send an END message with the error
                await self._publish(StepRequest(role=END, content=msg))
                raise FatalError(msg)

            # Store additional tools if provided
            self._tools.extend(message.additional_tools)

            # Announce, and trigger agents to announce themselves
            msg = AgentAnnouncement(
                content="Host joining",
                agent_config=self._config,
                announcement_type="initial",
            )
            await self._publish(msg)

            # Extract initial query/prompt from ConductorRequest if available
            # Parse inputs using the typed model for cleaner extraction
            from buttermilk._core.contract import HostInputModel

            # Store the raw inputs for backward compatibility
            self._initial_inputs = message.inputs if hasattr(message, "inputs") else {}

            try:
                # Parse inputs into our typed model
                if isinstance(self._initial_inputs, dict):
                    host_inputs = HostInputModel(**self._initial_inputs)
                else:
                    host_inputs = HostInputModel()

                # Extract fields cleanly
                self._initial_query = host_inputs.initial_query or ""
                self._initial_parameters = host_inputs.parameters or {}

                logger.info(f"Host {self.agent_name} extracted initial query: {self._initial_query[:100]}, parameters: {list(self._initial_parameters.keys())}")

            except Exception as e:
                # If parsing fails, fall back to empty values
                logger.warning(f"Host {self.agent_name} failed to parse inputs: {e}")
                self._initial_query = None
                self._initial_parameters = {}

            # Start the periodic progress reporter task
            self._progress_reporter_task = asyncio.create_task(self._report_progress_periodically())

            # Initialize generator now that participants are known
            self._step_generator = self._sequence()
            logger.info(f"Host participants initialized to: {list(self._participants.keys())}")

            async for next_step in self._step_generator:
                logger.info(f"Host {self.agent_name}: Processing step {next_step.role}")
                # Wait for tasks from the *previous* step to complete before starting the *next* step
                if not await self.wait_check_last_step_completions():
                    # If the wait failed (timeout or error), stop the flow
                    # break
                    pass
                # Don't seek confirmation from the manager to send a request to the manager
                logger.info(f"Host {self.agent_name}: human_in_loop={self.human_in_loop}, next_step.role={next_step.role}, MANAGER={MANAGER}")
                if self.human_in_loop and next_step.role != MANAGER and not await self._wait_for_user(next_step):
                    # If user rejected or timed out, stop the flow
                    continue
                    # break

                # Execute the current step
                await self._execute_step(next_step)

            # --- Sequence finished ---
            logger.info(f"Host {self.agent_name} flow execution finished.")

            # Send final progress update before any cleanup begins
            final_progress_message = FlowProgressUpdate(
                source=self.agent_id,
                status="finished",
                step_name=END,
                waiting_on={},
                message="Flow completed",
            )
            logger.info(f"Host {self.agent_name} sending final progress update before cleanup.")
            await self._publish(final_progress_message)

        except (KeyboardInterrupt):
            logger.info("Flow terminated by user.")
        except (FatalError, Exception) as e:
            logger.exception(f"Unexpected and unhandled fatal error: {e}", exc_info=True)
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
        logger.info(f"Host {self.agent_name} shutdown complete.")

    async def wait_check_last_step_completions(self) -> bool:
        """Wait for tasks from the previous step to complete."""
        # Wait for pending tasks to complete
        last_step_successful = await self._wait_for_all_tasks_complete()
        if not last_step_successful:
            msg = f"Host {self.agent_id} failed to complete all tasks for the previous step: {dict(self._pending_tasks_by_agent)}"
            logger.error(msg)
            return False
        # If successful, clear the pending tasks dictionary for the next step
        async with self._tasks_condition:
            self._pending_tasks_by_agent.clear()
        logger.info("No pending tasks left over from previous steps, clear to proceed.")
        return True

    async def _execute_step(self, step: StepRequest) -> None:
        """Process a single step."""
        logger.info(f"Host executing step: {step}")
        self._current_step = step.role
        if step.role == WAIT:
            logger.info("Host waiting for 10 seconds as requested by WAIT step.")
            await asyncio.sleep(10)
        elif step.role == END:
            logger.info(f"Flow completed and all tasks finished. Sending END signal: {step}")
            await self._publish(step)
        else:
            if step.role in self._participants:
                # Signal that we expect at least one response/task start for this step
                # This event is used in the wait_for predicate.
                self._step_starting.set()
                logger.debug(f"Host set _step_starting event for role: {step.role}")
            elif step.role == MANAGER:
                # MANAGER steps don't spawn trackable worker tasks, so don't set _step_starting
                # Convert StepRequest to UIMessage for frontend display
                ui_message = UIMessage(
                    content=step.content or "What would you like to do?",
                    options=None,  # No specific options, just free text response
                )
                await self._publish(ui_message)
                return  # Don't send the StepRequest itself
            else:
                logger.warning(f"Host executing step for unknown participant role: {step.role}")

            await self._publish(step)

    async def _process(
        self,
        *,
        message: AgentInput,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> AgentOutput:
        """Process messages.
        
        Base implementation returns an error since non-LLM hosts don't process direct inputs.
        Subclasses that support LLM-based processing should override this method.
        """
        placeholder = ErrorEvent(source=self.agent_id, content="Host agent does not process direct inputs via _process")
        return AgentOutput(agent_id=self.agent_id, outputs=placeholder)

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

            # First check if it's a participant "ask_" tool
            if call.name.endswith("_call"):
                # Extract role from tool name (e.g., "zotero_researcher_call" -> "ZOTERO_RESEARCHER")
                role_part = call.name[:-5].upper()  # Remove "_call" suffix and uppercase
                # Parse the arguments
                try:
                    arguments = json.loads(call.arguments)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse tool arguments: {call.arguments}")
                    continue

                step_request = StepRequest(
                    role=role_part, inputs=arguments,
                    metadata={"tool_name": call.name, "tool_call_id": call.id})

                # Create a more descriptive log message
                tool_desc = self._describe_tool_call(call.name, arguments)
                logger.info(f"Host routing to {role_part}: {tool_desc}")

                if self.human_in_loop:
                    await self._proposed_step.put(step_request)
                else:
                    # If human_in_loop is False, we send the step request directly
                    logger.info(f"Host {self.agent_name} routing tool call to agent {role_part}: {step_request}")
                    await self._publish(step_request)

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
