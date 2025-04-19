"""
SelectorOrchestrator: an interactive orchestrator with host agent and user guidance.
"""
import asyncio
import time
from typing import Any, Self, Sequence

from autogen_core import ClosureAgent, ClosureContext, MessageContext, TypeSubscription
from pydantic import Field, PrivateAttr, model_validator
import shortuuid

from buttermilk._core.agent import ProcessingError
from buttermilk._core.contract import (
    CLOSURE,
    CONDUCTOR,
    CONFIRM,
    END,
    MANAGER,
    AgentInput,
    AgentOutput,
    ConductorRequest,
    ManagerMessage,
    ManagerRequest,
    ManagerResponse,
    StepRequest,
    UserInstructions,
)
from buttermilk._core.types import RunRequest
from buttermilk.bm import logger
from buttermilk.runner.groupchat import AutogenOrchestrator


class SelectorOrchestrator(AutogenOrchestrator):
    """
    An orchestrator that enables interactive exploration of agent variants
    with direct user involvement and an active host LLM agent.
    
    This orchestrator extends the AutogenOrchestrator with:
    1. A dedicated host agent to manage conversations
    2. Rich interactive options with the user
    3. Step-by-step exploration of agent variants
    4. Tracking of exploration paths and results
    """
    
    # Core attributes for managing state
    _active_variants: dict[str, list[tuple]] = PrivateAttr(default_factory=dict)
    _exploration_path: list[str] = PrivateAttr(default_factory=list)
    _host_agent: Any = PrivateAttr(default=None)
    _user_confirmation: asyncio.Queue = PrivateAttr()
    _exploration_results: dict[str, dict] = PrivateAttr(default_factory=dict)
    
    @model_validator(mode="after")
    def open_queue(self) -> Self:
        """Initialize user confirmation queue."""
        self._user_confirmation = asyncio.Queue(maxsize=1)
        return self
    
    async def _setup(self) -> None:
        """Initialize runtime, register agents and set up communication channels."""
        # Initialize the base components from AutogenOrchestrator
        await super()._setup()
        
        # Initialize exploration tracking
        for agent_name, agent_variants in self._agent_types.items():
            self._active_variants[agent_name] = agent_variants
            
        # Set up interactive components
        await self._register_host_agent()
        
        # Send welcome message
        intro_msg = ManagerMessage(
            role="orchestrator",
            content=f"Started {self.name}: {self.description}. The host agent will guide our exploration step by step. You can provide feedback and guidance at each step.",
        )
        await self._runtime.publish_message(intro_msg, topic_id=self._topic)
    
    async def _register_host_agent(self) -> None:
        """Register a host agent responsible for guiding the conversation."""
        # Find the host agent in our agent types
        if "host" in self._agent_types:
            # Use the first host agent variant 
            self._host_agent = self._agent_types["host"][0]
            logger.info(f"Registered host agent: {self._host_agent[1].id}")
        else:
            logger.warning("No host agent found in agent types. Using conductor for step determination.")
    
    async def _wait_for_human(self, timeout=240) -> bool:
        """Wait for human confirmation with timeout."""
        t0 = time.time()
        while True:
            try:
                msg = self._user_confirmation.get_nowait()
                if msg.halt:
                    raise StopAsyncIteration("User requested halt.")
                return msg.confirm
            except asyncio.QueueEmpty:
                if time.time() - t0 > timeout:
                    return False
                await asyncio.sleep(1)
    
    async def _in_the_loop(self, step: StepRequest) -> bool:
        """Enhanced interaction with user for guidance, not just confirmation."""
        # Prepare a richer message with more context about the step
        variant_info = ""
        if step.role in self._active_variants:
            variants = self._active_variants[step.role]
            if len(variants) > 1:
                variant_info = (
                    f"\n\nThis step has {len(variants)} variants available:\n" + 
                    "\n".join([f"- {v[1].id}: {v[1].role}" for v in variants])
                )
        
        # Create rich message with exploration context
        confirm_step = ManagerRequest(
            role="orchestrator",
            content=(
                f"Here's the next proposed step:\n\n"
                f"**Step**: {step.role}\n"
                f"**Description**: {step.description}\n"
                f"**Prompt**: {step.prompt}"
                f"{variant_info}\n\n"
                f"Do you want to proceed with this exploration step? "
                f"You can also suggest a different approach or request to try a specific variant."
            ),
            prompt=step.prompt,
            description=step.description,
        )

        await self._send_ui_message(confirm_step)
        response = await self._wait_for_human()
        return response
    
    async def _get_next_step(self) -> StepRequest:
        """Determine next step based on host agent recommendation and user input."""
        # If we have a host agent, use it to determine the next step
        if self._host_agent:
            host_type, host_config = self._host_agent
            
            # Create a request for the host to determine the next step
            host_context = {
                "exploration_path": self._exploration_path,
                "available_agents": {name: [v[1].id for v in variants] 
                                    for name, variants in self._active_variants.items()},
                "task": self.params.get("task", "Analyze the content"),
                "results": self._exploration_results,
            }
            
            host_request = ConductorRequest(
                role="host",
                inputs=host_context,
            )
            
            # Get recommendation from host
            host_id = await self._runtime.get(host_type)
            response = await self._runtime.send_message(host_request, recipient=host_id)
            
            if response and isinstance(response, AgentOutput) and response.outputs:
                if isinstance(response.outputs, StepRequest):
                    return response.outputs
                else:
                    logger.warning(f"Host agent returned unexpected output: {response.outputs}")
        
        # Fall back to the conductor-based approach from AutogenOrchestrator
        # Each step, we proceed by asking the CONDUCTOR agent what to do.
        request = ConductorRequest(
            role=self.name,
            inputs={"participants": dict(self._agent_types.items()), "task": self.params.get("task")},
        )
        responses = await self._ask_agents(
            CONDUCTOR,
            message=request,
        )

        # Determine the next step based on the response
        if len(responses) != 1 or not (instructions := responses[0].outputs) or not (isinstance(instructions, StepRequest)):
            logger.warning("Conductor could not get next step.")
            return None

        next_step = instructions
        if next_step.role == END:
            raise StopAsyncIteration("Host signaled that flow has been completed.")

        if next_step.role.lower() not in self._agent_types:
            raise ProcessingError(
                f"Step {next_step.role} not found in registered agents.",
            )

        # We're going to wait a bit between steps.
        await asyncio.sleep(2)
        return next_step
    
    async def _execute_step(
        self,
        step: str,
        input: AgentInput,
        variant_index: int = 0,
    ) -> AgentOutput | None:
        """
        Execute step with selected variant and capture results for exploration.
        
        Args:
            step: The step name to execute (corresponds to agent role)
            input: The input data for the agent
            variant_index: Which variant of the agent to use (default: 0, first variant)
            
        Returns:
            AgentOutput: The output from the executed agent
        """
        step_lower = step.lower()
        if step_lower not in self._agent_types:
            logger.warning(f"Step {step} not found in registered agents.")
            return None
            
        # Select the specified variant
        variants = self._agent_types[step_lower]
        if variant_index >= len(variants):
            logger.warning(f"Variant index {variant_index} out of range for step {step}. Using first variant.")
            variant_index = 0
            
        agent_type, agent_config = variants[variant_index]
        
        # Track this step in our exploration path
        step_id = f"{step_lower}_{variant_index}_{shortuuid.uuid()[:4]}"
        self._exploration_path.append(step_id)
        
        # Execute the agent
        agent_id = await self._runtime.get(agent_type)
        response = await self._runtime.send_message(input, recipient=agent_id)
        
        # Store the results for later comparison
        if response and isinstance(response, AgentOutput):
            self._exploration_results[step_id] = {
                "agent": agent_config.id,
                "role": agent_config.role,
                "variant": variant_index,
                "outputs": response.outputs,
            }
            
        return response
    
    async def _run(self, request: RunRequest | None = None) -> None:
        """
        Main execution method with interactive exploration capabilities.
        
        This extends the base _run method to provide more interactive capabilities
        and step-by-step exploration guided by the host agent and user input.
        """
        try:
            # Initialize components
            await self._setup()
            
            # Handle initial data loading if request provided
            if request:
                # Initialize with records from request if available
                if request.records:
                    self._records = request.records
                # Otherwise, try to fetch by ID or URI if provided
                elif request.record_id or request.uri:
                    await self._fetch_record(request)
            
            # Main interactive loop
            while True:
                try:
                    # Small delay to prevent busy-waiting
                    await asyncio.sleep(1)
                    
                    # Get next recommended step from host agent
                    next_step = await self._get_next_step()
                    if not next_step:
                        # If no step recommended, wait before checking again
                        await asyncio.sleep(5)
                        continue
                        
                    # Get user confirmation with enhanced context
                    if not await self._in_the_loop(next_step):
                        logger.info("User did not confirm step. Waiting for new instructions.")
                        continue
                        
                    # Prepare and execute the confirmed step
                    step_input = await self._prepare_step(next_step)
                    await self._execute_step(step=next_step.role, input=step_input)
                    
                except ProcessingError as e:
                    # Non-fatal error, log and continue
                    logger.error(f"Error in Selector orchestrator run: {e}")
                    continue
                except (StopAsyncIteration, KeyboardInterrupt):
                    # Flow completion or user termination
                    raise
                except Exception as e:
                    logger.exception(f"Unexpected error in Selector orchestrator: {e}")
                    raise
                    
        except (StopAsyncIteration, KeyboardInterrupt):
            logger.info("Selector orchestrator run completed or terminated by user.")
        finally:
            # Clean up resources
            await self._cleanup()
            
    async def _fetch_record(self, request: RunRequest) -> None:
        """Fetch a record based on record_id or URI."""
        try:
            from buttermilk.agents.fetch import FetchRecord
            fetch = FetchRecord(data=self.data)
            output = await fetch._run(record_id=request.record_id, uri=request.uri, prompt=request.prompt)
            if output:
                self._records = output.results
        except Exception as e:
            logger.error(f"Error fetching record: {e}")
