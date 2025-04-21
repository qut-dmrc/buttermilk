"""
SelectorOrchestrator: an interactive orchestrator with host agent and user guidance.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Self, Sequence, Union, cast

from pydantic import BaseModel, PrivateAttr, model_validator
import shortuuid

from buttermilk._core.agent import ProcessingError
from buttermilk._core.contract import (
    CONDUCTOR,
    CONFIRM,
    END,
    AgentInput,
    AgentOutput,
    ConductorRequest,
    ConductorResponse,
    ManagerMessage,
    ManagerRequest,
    ManagerResponse,
    StepRequest,
    UserInstructions,
)
from buttermilk._core.types import RunRequest
from buttermilk.bm import logger
from buttermilk.runner.groupchat import AutogenOrchestrator


class Selector(AutogenOrchestrator):
    """
    An orchestrator that enables interactive exploration of agent variants
    with direct user involvement and an active host LLM agent.

    This orchestrator extends the AutogenOrchestrator with:
    1. Rich interactive options with the user
    2. Step-by-step exploration of agent variants
    3. Tracking of exploration paths and results
    4. Comparison capabilities between different agent variants
    """

    # Core attributes for managing state
    _active_variants: Dict[str, List[tuple]] = PrivateAttr(default_factory=dict)
    _exploration_path: List[str] = PrivateAttr(default_factory=list)
    _exploration_results: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)
    _user_feedback: List[str] = PrivateAttr(default_factory=list)
    _last_user_selection: Optional[str] = PrivateAttr(default=None)
    _variant_mapping: Dict[str, int] = PrivateAttr(default_factory=dict)

    async def _setup(self) -> None:
        """Initialize runtime, register agents and set up communication channels."""
        # Initialize the base components from AutogenOrchestrator
        await super()._setup()

        # Initialize exploration tracking
        for agent_name, agent_variants in self._agent_types.items():
            self._active_variants[agent_name] = agent_variants

            # Build mapping of variant IDs to indices for easy lookup
            for i, (_, config) in enumerate(agent_variants):
                self._variant_mapping[config.id] = i

        # Send welcome message
        await self._in_the_loop(
            prompt=(
                f"Started {self.name}: {self.description}. "
                f"The conductor will guide our exploration step by step. "
                f"You can provide feedback and guidance at each step, "
                f"including selecting specific variants to try. "
                f"Let me know when you're ready to begin, or just enter your prompt."
            ),
        )

    async def _wait_for_human(self, timeout: int = 20) -> ManagerResponse:
        """
        Wait for human confirmation with timeout.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            bool: True if user confirmed, False if rejected or timed out
        """
            try:
            msg = await asyncio.wait_for(self._user_confirmation.get(), timeout=timeout)
                if msg.halt:
                    raise StopAsyncIteration("User requested halt.")

                # Store feedback if provided
                if msg.prompt:
                    self._user_feedback.append(msg.prompt)

                # Store selected option if provided
                if msg.selection:
                    self._last_user_selection = msg.selection

            return True
        except asyncio.TimeoutError:
            return ManagerResponse(error=[f"No response from Manager within timeout period ({timeout} seconds)."], confirm=False, halt=False)

    async def _in_the_loop(self, step: StepRequest = None, prompt: str = "") -> ManagerResponse:
        """
        Enhanced interaction with user for guidance, not just confirmation.

        Args:
            step: The step request to confirm with the user

        Returns:
            ManagerResponse
        """
        if step:
            # Prepare a richer message with more context about the step
            variant_info = ""
            if step.role in self._active_variants:
                variants = self._active_variants[step.role]
                if len(variants) > 1:
                    variant_info = f"\n\nThis step has {len(variants)} variants available:\n" + "\n".join(
                        [f"- {v[1].id}: {v[1].role} ({str(v[1].parameters)})" for v in variants]
                    )

            # Create rich message with exploration context
            confirm_step = ManagerRequest(
                role=step.role,
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
        else:
            confirm_step = ManagerRequest(prompt=prompt, role="user")

        await self._send_ui_message(confirm_step)
        response = await self._wait_for_human()
        return response

    async def _get_host_suggestion(self) -> StepRequest:
        """
        Determine next step based on conductor recommendation and user input.

        Returns:
            StepRequest: The next step to execute or a wait step if no step can be determined
        """
        # Create enhanced context with exploration history and user feedback
        conductor_context = {
            "exploration_path": self._exploration_path,
            "task": self.params.get("task", "Assist the user"),
            "results": self._exploration_results,
            "user_feedback": self._user_feedback if self._user_feedback else [],
            "participants": {name: variants[0][1].description for name, variants in self._agent_types.items()},
            "available_variants": {name: [v[1].id for v in variants] for name, variants in self._active_variants.items()},
        }

        # Create the request for the conductor
        request = ConductorRequest(
            inputs=conductor_context,
        )

        # Ask the conductor for the next step
        responses = await self._ask_agents(
            CONDUCTOR,
            message=request,
        )

        # Default response if we can't determine a step
        wait_step = StepRequest(
            role="wait",
            description="Waiting for conductor to provide instructions",
            prompt="Please wait...",
        )

        # Check if we have a valid response
        if not responses or len(responses) != 1:
            logger.warning("Conductor could not get next step.")
            return wait_step

        # Get the response - this is an AgentOutput or a ConductorResponse
        agent_output = responses[0]
        outputs = agent_output.outputs

        # Case 1: Response is a ConductorResponse with special instructions
        if isinstance(agent_output, ConductorResponse):
            await self._handle_host_message(agent_output)
            return await self._get_host_suggestion()

        # Case 2: response.Output is a StepRequest object
        try:
            outputs = StepRequest.model_validate(outputs)
        except Exception as e:
            raise ProcessingError(f"Received uninterpretable result from Selector host LLM: {e}")

        if isinstance(outputs, StepRequest):
            next_step = outputs
            if next_step.role == END:
                raise StopAsyncIteration("Host signaled that flow has been completed.")

            if next_step.role.upper() not in self._agent_types and next_step.role != END and next_step.role != "wait":
                raise ProcessingError(f"Step {next_step.role} not found in registered agents.")

            await asyncio.sleep(2)
            return next_step

        logger.warning(f"Unexpected conductor response type: {type(agent_output)}, outputs: {type(outputs)}")
        return wait_step

    async def _handle_host_message(self, message: ConductorResponse) -> None:
        """
        Process special messages from the host agent (conductor).

        Args:
            message: The ConductorResponse from the host agent
        """
        outputs = message.outputs
        if not isinstance(outputs, dict):
            logger.warning(f"Expected dict outputs, got {type(outputs)}")
            return

        msg_type = outputs.get("type", "")

        if msg_type == "question":
            # Host is asking user a question
            options = outputs.get("options", [])
            question = ManagerRequest(
                role="user",
                prompt=(f"{message.content or ''}\n\n" + (f"Options:\n" + "\n".join([f"- {opt}" for opt in options]) if options else "")),
            )
            await self._runtime.publish_message(question, topic_id=self._topic)

            # Wait for user's response
            response = await self._wait_for_human()

        elif msg_type == "comparison":
            # Host is providing a comparison between variants
            await self._handle_comparison(message)

        else:
            # Just forward the message to the UI
            await self._runtime.publish_message(
                ManagerMessage(
                    content=message.content or "",
                ),
                topic_id=self._topic,
            )

    async def _handle_comparison(self, message: ConductorResponse) -> None:
        """
        Format and present variant comparisons to the user.

        Args:
            message: The ConductorResponse containing comparison data
        """
        outputs = message.outputs
        variants = outputs.get("variants", [])
        results = outputs.get("results", {})

        # Create a nicely formatted comparison
        content = message.content or ""
        comparison_text = content + "\n\n"
        comparison_text += "## Comparison of Results\n\n"

        for variant in variants:
            variant_results = results.get(variant, {})
            comparison_text += f"### {variant}\n\n"

            # Format the results as a bullet list
            for key, value in variant_results.items():
                comparison_text += f"- **{key}**: {value}\n"
            comparison_text += "\n"

        # Send the formatted comparison to the UI
        await self._send_ui_message(
            ManagerMessage(
                content=comparison_text,
            )
        )

    async def _execute_step(
        self,
        step: Union[str, StepRequest],
        input: AgentInput,
        variant_index: int = 0,
    ) -> Optional[AgentOutput]:
        """
        Execute step with selected variant and capture results for exploration.

        Args:
            step: The step name to execute (corresponds to agent role)
            input: The input data for the agent
            variant_index: Which variant of the agent to use (default: 0, first variant)

        Returns:
            AgentOutput: The output from the executed agent, or None if there was an error
        """
        # Handle both string and StepRequest inputs
        step_name = step.role.upper() if isinstance(step, StepRequest) else step.upper()

        if step_name not in self._agent_types:
            logger.warning(f"Step {step_name} not found in registered agents.")
            return None

        # Select the specified variant
        variants = self._agent_types[step_name]
        if variant_index >= len(variants):
            logger.warning(f"Variant index {variant_index} out of range for step {step_name}. Using first variant.")
            variant_index = 0

        agent_type, agent_config = variants[variant_index]

        # Track this step in our exploration path
        step_id = f"{step_name}_{variant_index}_{shortuuid.uuid()[:4]}"
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

        return cast(Optional[AgentOutput], response)

    async def _run(self, request: Optional[RunRequest] = None) -> None:
        """
        Main execution method with interactive exploration capabilities.

        This extends the base _run method to provide more interactive capabilities
        and step-by-step exploration guided by the host agent and user input.

        Args:
            request: Optional RunRequest containing record information
        """
        try:
            # Initialize components
            await self._setup()

            # Handle initial data loading if request provided
            if request:
                # Initialize with records from request if available
                if request.records:
                    message = UserInstructions(records=request.records, prompt=request.prompt)
                    self._records = request.records
                # Otherwise, try to fetch by ID or URI if provided
                elif request.record_id or request.uri:
                    message = AgentInput(prompt=request.prompt, inputs={"record_id": request.record_id, "uri": request.uri})
                else:
                    message = request
                response = await self._ask_agents("fetch", message)
                await self._runtime.publish_message(response, topic_id=self._topic, sender="orchestrator")

            # Main interactive loop
            while True:
                try:
                    # Small delay to prevent busy-waiting
                    await asyncio.sleep(1)

                    # Get next recommended step from host agent
                    next_step = await self._get_host_suggestion()
                    # Handle the "wait" step specially
                    if next_step.role == "wait":
                        # If waiting for next step, pause before checking again
                        await asyncio.sleep(5)
                        continue

                    # Get user confirmation with enhanced context
                    response = await self._in_the_loop(next_step)
                    if response.halt:
                        raise StopAsyncIteration("User requested halt.")
                    if not response.confirm:
                        logger.info("User did not confirm step. Waiting for new instructions.")
                        continue

                    # Prepare the step input
                    step_input = await self._prepare_step(next_step)

                    # Check if a specific variant was selected by the user
                    variant_index = 0  # Default to first variant
                    if response.selection:
                        # Look up the variant index from the name
                        if response.selection in self._variant_mapping:
                            variant_index = self._variant_mapping[response.selection]
                            logger.info(f"Using selected variant: {response.selection} (index {variant_index})")
                        else:
                            logger.warning(f"Selected variant {response.selection} not found. Using default.")

                    # Execute the step with the selected variant
                    await self._execute_step(step=next_step, input=step_input, variant_index=variant_index)

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
        """
        Fetch a record based on record_id or URI.

        Args:
            request: The RunRequest containing record_id or URI
        """
        try:
            from buttermilk.agents.fetch import FetchRecord

            fetch = FetchRecord(role="fetch", description="fetch records and urls", data=list(self.data))
            output = await fetch._run(record_id=request.record_id, uri=request.uri, prompt=request.prompt)
            if output and output.results:
                self._records = output.results
        except Exception as e:
            logger.error(f"Error fetching record: {e}")
