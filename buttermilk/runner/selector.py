"""
Selector Orchestrator: Enables interactive, user-guided exploration of agent workflows.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Self, Sequence, Union, cast
from pydantic import BaseModel, PrivateAttr, model_validator
import shortuuid

# Buttermilk core imports
from buttermilk._core.agent import ProcessingError, ToolOutput
from buttermilk._core.contract import (
    CONDUCTOR,  # Role constant for the conductor/host agent
    CONFIRM,  # Role constant (unused directly here, but related to confirmation flow)
    END,  # Special role indicating flow completion
    WAIT,  # Special role indicating the orchestrator should wait
    AgentInput,
    AgentOutput,
    ConductorRequest,  # Request sent *to* the conductor
    ConductorResponse,  # Response *from* the conductor (can contain special types)
    ManagerMessage,  # Message *to* the user/manager
    ManagerRequest,  # Request *to* the user/manager (for confirmation/input)
    ManagerResponse,  # Response *from* the user/manager
    StepRequest,  # Request defining the next step to execute
    UserInstructions,
)
from buttermilk._core.types import Record, RunRequest  # Type for initial run request data
from buttermilk.bm import logger  # Buttermilk logger

# Base orchestrator class
from buttermilk.runner.groupchat import AutogenOrchestrator


class Selector(AutogenOrchestrator):
    """
    An orchestrator facilitating interactive exploration of multi-agent workflows.

    Extends `AutogenOrchestrator` to incorporate direct user guidance, feedback,
    and the ability to select between different agent variants at each step.
    It maintains state about the exploration process and interacts closely with
    both the user (via `ManagerRequest`/`ManagerResponse`) and a `CONDUCTOR` agent.

    Key Features:
    - Step-by-step execution driven by `CONDUCTOR` agent suggestions.
    - User confirmation and feedback collection at each step via `_in_the_loop`.
    - Ability for the user to select specific agent variants for a step.
    - Tracking of the exploration path and results (`_exploration_path`, `_exploration_results`).
    - Handling of special messages from the `CONDUCTOR` (e.g., questions for the user, comparisons).
    """

    # --- Internal State Tracking ---
    # Stores available agent variants for each role. Maps RoleName -> List[(AgentType, AgentConfig)]
    _active_variants: Dict[str, List[tuple]] = PrivateAttr(default_factory=dict)
    # Records the sequence of executed steps (including variant choice). List[step_id]
    _exploration_path: List[str] = PrivateAttr(default_factory=list)
    # Stores results of executed steps. Maps step_id -> result_data_dict
    _exploration_results: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)
    # Collects free-text feedback provided by the user during interactions. List[str]
    _user_feedback: List[str] = PrivateAttr(default_factory=list)
    # Stores the last variant ID selected by the user. Optional[str]
    _last_user_selection: Optional[str] = PrivateAttr(default=None)
    # Maps variant ID (AgentConfig.id) to its index within the role's variant list. Dict[str, int]
    _variant_mapping: Dict[str, int] = PrivateAttr(default_factory=dict)

    async def _setup(self) -> None:
        """
        Initializes the Selector orchestrator.

        Calls the base class setup, initializes exploration tracking structures,
        builds the variant mapping, and sends an initial welcome message to the user.
        """
        logger.info("Setting up Selector orchestrator...")
        # Initialize Autogen runtime, register agents via base class method.
        await super()._setup()

        # Initialize Selector-specific state.
        self._exploration_path = []
        self._exploration_results = {}
        self._user_feedback = []
        self._last_user_selection = None
        self._active_variants = {}
        self._variant_mapping = {}

        # Populate active variants and create the ID-to-index mapping.
        # _agent_types is populated by the base class _setup -> _register_agents.
        for agent_name_upper, agent_variants in self._agent_types.items():
            self._active_variants[agent_name_upper] = agent_variants
            # Map the unique ID of each variant config to its index for quick lookup.
            for i, (_, config) in enumerate(agent_variants):
                # Assumes config object has an 'id' attribute.
                variant_id = getattr(config, "id", None)
                if variant_id:
                    self._variant_mapping[variant_id] = i
                else:
                    logger.warning(f"Agent variant config for role {agent_name_upper} at index {i} lacks an 'id'. Cannot map for selection.")

        logger.debug(f"Selector setup complete. Variant mapping created: {self._variant_mapping}")

        # Send initial welcome message to the user via the MANAGER interface.
        # Uses _in_the_loop without a step to just send a prompt.
        await self._in_the_loop(
            prompt=(
                f"✨ Welcome to the Buttermilk Selector ✨\n\n"
                f"Flow: '{self.name}' - {self.description}\n\n"
                f"I will propose steps suggested by the conductor agent. "
                f"At each step, you can:\n"
                f"  ✅ **Confirm** (press Enter with no text) to proceed.\n"
                f"  💬 **Provide Feedback/Instructions** (type text and press Enter) before confirming.\n"
                f"  ❌ **Reject** (type 'n', 'q', 'stop', etc.) to ask the conductor for a different step.\n"
                f"  🚪 **Exit** (type 'exit') to stop the flow.\n\n"
                f"Ready to begin?"
            ),
        )

    async def _wait_for_human(self, timeout: int = 300) -> ManagerResponse:  # Increased default timeout
        """
        Waits for a response from the user via the confirmation queue.

        Overrides the base class potentially to handle timeouts or specific feedback structures.

        Args:
            timeout: Maximum time in seconds to wait for user input.

        Returns:
            The ManagerResponse received from the user, or a timeout error response.

        Raises:
            StopAsyncIteration: If the user response indicates a desire to halt.
        """
        logger.debug(f"Waiting for user confirmation/input (timeout: {timeout}s)...")
        try:
            # Wait for a message on the queue filled by the manager interface agent.
            response: ManagerResponse = await asyncio.wait_for(self._user_confirmation.get(), timeout=timeout)
            logger.info(
                f"Received user response: Confirm={response.confirm}, Halt={response.halt}, Selection='{response.selection}', Prompt='{str(response.prompt)[:50]}...'"
            )

            # Check if user wants to stop the entire flow.
            if response.halt:
                logger.warning("User requested halt.")
                raise StopAsyncIteration("User requested halt.")

            # Store any feedback provided by the user.
            if response.prompt:
                self._user_feedback.append(response.prompt)
                logger.debug(f"Stored user feedback: '{str(response.prompt)[:100]}...'")

            # Store the specific variant selected by the user, if any.
            if response.selection:
                # TODO: Validate if selection corresponds to an actual available variant ID?
                # Currently just stores whatever string the user provided.
                self._last_user_selection = response.selection
                logger.debug(f"Stored user selection: '{response.selection}'")
            else:
                self._last_user_selection = None  # Clear last selection if none provided

            # Mark the item as done in the queue.
            self._user_confirmation.task_done()  # Important for queue management
            return response

        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for user response after {timeout} seconds.")
            # Return a specific response indicating timeout.
            return ManagerResponse(error=[f"Timeout: No response within {timeout} seconds."], confirm=False, halt=False)
        except StopAsyncIteration:
            raise  # Re-raise halt signal
        except Exception as e:
            logger.error(f"Error waiting for user confirmation: {e}")
            # Return an error response
            return ManagerResponse(error=[f"Error processing user input: {e}"], confirm=False, halt=False)

    async def _in_the_loop(self, step: StepRequest | None = None, prompt: str = "") -> ManagerResponse:
        """
        Sends a request to the user (MANAGER) for confirmation, feedback, or selection,
        and waits for their response.

        Overrides the base method to provide richer context, including available agent variants.

        Args:
            step: The proposed `StepRequest` object to present to the user (optional).
            prompt: A simple text prompt to send if `step` is not provided (optional).

        Returns:
            The `ManagerResponse` from the user.
        """
        if step:
            logger.info(f"Requesting user confirmation for step: {step.role}")
            # Prepare context about available variants for the proposed step.
            variant_info = ""
            options_list = []  # For ManagerRequest options
            # role_upper = step.role.upper()
            # if role_upper in self._active_variants:
            #     variants = self._active_variants[role_upper]
            # if len(variants) > 1:
            #     variant_info_lines = [f"    {i+1}. {v[1].id} (Model: {v[1].parameters.get('model', 'N/A')})" for i, v in enumerate(variants)]
            #     # TODO: Include more descriptive variant info if available?
            #     variant_info = f"\n\nThis step has {len(variants)} variants available:\n" + "\n".join(variant_info_lines)
            #     options_list = [v[1].id for v in variants]  # Provide variant IDs as selectable options

            # Construct the message for the user.
            request_content = (
                f"**Next Proposed Step:**\n"
                f"- **Agent Role:** {step.role}\n"
                f"- **Description:** {step.description or '(No description)'}\n"
                f"- **Prompt Snippet:** {step.prompt[:100] + '...' if step.prompt else '(No prompt)'}"
                f"{variant_info}\n\n"
                f"Confirm (Enter), provide feedback, select a variant ID, or reject ('n'/'q')."
            )
            manager_request = ManagerRequest(
                role=step.role,  # Indicate which role this confirmation relates to
                content=request_content,
                prompt=step.prompt,  # Full prompt can be useful context
                description=step.description,
                options=options_list,  # Pass variant IDs as options
            )
        elif prompt:
            logger.info("Sending general prompt to user.")
            # Send a simple prompt if no step is provided.
            manager_request = ManagerRequest(prompt=prompt, role="user", content=prompt)  # Content=prompt for simple display
        else:
            logger.error("Selector._in_the_loop called without step or prompt.")
            # Cannot proceed without something to ask the user.
            return ManagerResponse(error=["Internal error: _in_the_loop called inappropriately."], confirm=False)

        # Send the request to the MANAGER interface agent.
        await self._send_ui_message(manager_request)
        await asyncio.sleep(0.1)  # Short delay (optional, aids debugging/readability)

        # Wait for and return the user's response.
        response = await self._wait_for_human()
        return response

    async def _get_host_suggestion(self) -> StepRequest:
        """
        Asks the CONDUCTOR agent for the next step suggestion.

        Overrides the base method to provide richer context to the CONDUCTOR,
        including exploration history, user feedback, and available variants.

        Returns:
            The `StepRequest` suggested by the CONDUCTOR, or a WAIT step if none is provided.
        """
        logger.debug("Asking CONDUCTOR for next step suggestion with selector context...")
        # Prepare enhanced context for the CONDUCTOR.
        conductor_context = {
            "task": self.params.get("task", "Assist the user"),  # Overall task goal
            "exploration_path": self._exploration_path,  # History of executed steps/variants
            "latest_results": (
                self._exploration_results.get(self._exploration_path[-1]) if self._exploration_path else None
            ),  # Result of the last step
            # TODO: Consider sending more results history? Might bloat context.
            "user_feedback": self._user_feedback,  # Accumulated user feedback
            # Provide descriptions of participant roles.
            "participants": {name: variants[0][1].description for name, variants in self._agent_types.items()},
            # List available variant IDs for each role.
            "available_variants": {name: [v[1].id for v in variants] for name, variants in self._active_variants.items()},
        }
        self._user_feedback = []  # Clear feedback after sending it to conductor

        # Create the request for the CONDUCTOR agent.
        request = ConductorRequest(
            inputs=conductor_context,
            prompt="Based on the current state, task, history, and user feedback, what is the next logical step or question?",  # Explicit prompt
            records=self._records,  # Include current data records
        )

        # Ask the CONDUCTOR agent(s).
        responses = await self._ask_agents(CONDUCTOR, message=request)

        # Define a default WAIT step if conductor fails.
        wait_step = StepRequest(role=WAIT, description="Waiting for conductor instructions.", prompt="")

        # Process the response(s). Expecting a single AgentOutput.
        if not responses:
            logger.warning("Conductor did not provide a response.")
            return wait_step
        if len(responses) > 1:
            logger.warning(f"Received multiple responses ({len(responses)}) from CONDUCTOR role, using the first.")

        agent_output = responses[0]

        # Handle if the conductor returned an error
        if not isinstance(agent_output, AgentOutput) or agent_output.is_error:
            err_details = agent_output.outputs if agent_output else "No response object"
            logger.error(f"Conductor returned an error or invalid output: {err_details}")
            # Maybe ask user what to do? For now, just wait.
            return wait_step

        conductor_outputs = agent_output.outputs

        # Check if the output is already a structured StepRequest (ideal case)
        if isinstance(conductor_outputs, StepRequest):
            next_step = conductor_outputs
            logger.debug(f"Conductor returned StepRequest for role: {next_step.role}")
        else:
            # If not StepRequest, try to validate it as one (e.g., if LLM returned dict)
            try:
                next_step = StepRequest.model_validate(conductor_outputs)
                logger.debug(f"Conductor output validated as StepRequest for role: {next_step.role}")
            except Exception as e:
                # If validation fails, assume it might be a special message (like question/comparison) or invalid.
                # TODO: Handle special ConductorResponse types more explicitly here if the conductor
                #       is expected to return things other than StepRequest via this path.
                #       Currently, _handle_host_message seems intended for responses *not* returned directly.
                logger.warning(f"Conductor output is not a valid StepRequest: {conductor_outputs}. Error: {e}. Assuming WAIT.")
                # Consider sending the raw output to the user for inspection?
                # await self._send_ui_message(ManagerMessage(content=f"Conductor proposed an invalid step:\n```\n{conductor_outputs}\n```"))
                return wait_step

        # Final checks and return
        if next_step.role == END:
            logger.info("Conductor signaled flow completion.")
            # Don't raise StopAsyncIteration here, let the main loop handle it after user confirmation.
            return next_step  # Return the END step

        role_upper = next_step.role.upper()
        if role_upper not in self._agent_types and role_upper != WAIT:
            logger.error(f"Conductor suggested step for unknown role: {next_step.role}. Available roles: {list(self._agent_types.keys())}")
            # Ask user or default to WAIT?
            return wait_step  # Defaulting to WAIT

        # TODO: Remove arbitrary sleep if not necessary for timing.
        await asyncio.sleep(2)
        return next_step

    async def _handle_host_message(self, message: ConductorResponse) -> None:
        """
        Processes special message types potentially returned by the CONDUCTOR agent.
        (Currently seems less used if _get_host_suggestion expects StepRequest).

        Args:
            message: The `ConductorResponse` message from the host/conductor.
        """
        # This method might be called if _ask_agents returned ConductorResponse directly,
        # or potentially if the conductor publishes these messages instead of returning StepRequest.
        logger.debug(f"Handling special host message: {message.outputs}")
        outputs = message.outputs
        if not isinstance(outputs, dict):
            logger.warning(f"Host message outputs are not a dict: {type(outputs)}. Forwarding content.")
            await self._send_ui_message(ManagerMessage(content=message.contents or "(Host sent non-dict message)"))
            return

        msg_type = outputs.get("type", "").lower()

        if msg_type == "question":
            # Host asks user a question, potentially with options.
            logger.info("Host is asking the user a question.")
            question_text = outputs.get("question", message.contents or "Host has a question:")
            options = outputs.get("options", [])
            options_text = "\nOptions:\n" + "\n".join([f"- {opt}" for opt in options]) if options else ""

            question_request = ManagerRequest(
                role="user",  # Request directed at the user
                content=f"{question_text}{options_text}",
                options=options,  # Pass options for potential UI elements
            )
            # Send question to user and wait for response.
            await self._send_ui_message(question_request)
            user_response = await self._wait_for_human()
            # TODO: What happens with the user_response here? It needs to be sent back
            #       to the conductor or used to influence the next _get_host_suggestion call.
            #       Currently, it's just received and stored in _user_feedback/_last_user_selection.
            logger.info(
                f"Received user answer to host question: Confirm={user_response.confirm}, Selection='{user_response.selection}', Prompt='{user_response.prompt}'"
            )

        elif msg_type == "comparison":
            # Host provides a comparison of results (e.g., between variants).
            logger.info("Host provided a comparison.")
            await self._handle_comparison(message)  # Format and display comparison

        else:
            # Unknown or generic message type, just display content to user.
            logger.debug(f"Forwarding generic host message to user.")
            await self._send_ui_message(ManagerMessage(content=message.contents or "(Host sent message with unknown type)"))

    async def _handle_comparison(self, message: ConductorResponse) -> None:
        """
        Formats and displays a comparison message from the host to the user.

        Args:
            message: The `ConductorResponse` containing comparison data in its `outputs` dict.
                     Expected keys: 'variants' (list of IDs), 'results' (dict mapping ID to results).
        """
        outputs = message.outputs
        if not isinstance(outputs, dict):
            return  # Should be checked before calling

        variants_compared = outputs.get("variants", [])
        results_data = outputs.get("results", {})
        comparison_intro = message.contents or "The conductor provided the following comparison:"

        logger.debug(f"Formatting comparison for variants: {variants_compared}")

        # Build the markdown text for the comparison.
        comparison_text = f"{comparison_intro}\n\n## Comparison of Results\n"
        for variant_id in variants_compared:
            variant_result = results_data.get(variant_id, {})
            comparison_text += f"\n### Variant: {variant_id}\n"
            if variant_result:
                # Format the results dictionary nicely.
                for key, value in variant_result.items():
                    # Simple key-value list for now. Could use tables or JSON.
                    comparison_text += f"- **{key}**: {pretty_repr(value, max_string=200)}\n"
            else:
                comparison_text += "- (No results provided for this variant)\n"

        # Send the formatted comparison to the user.
        await self._send_ui_message(ManagerMessage(content=comparison_text))

    async def _execute_step(  # type: ignore
        self,
        step: StepRequest,
        variant_index: int = 0,  # Allow specifying which variant to use
    ) -> AgentOutput | None:
        """
        Executes a specific variant of an agent for the given step.

        Overrides the base method to handle variant selection based on `variant_index`
        and stores the execution result in `_exploration_results`.

        Args:
            step: The `StepRequest` defining the role and prompt.
            variant_index: The index of the agent variant to execute for the specified role.

        Returns:
            The `AgentOutput` from the executed agent, or None if an error occurred.
        """
        role_upper = step.role.upper()
        if role_upper not in self._agent_types:
            logger.error(f"Cannot execute step: Role '{step.role}' not found in registered agents.")
            return None

        # Prepare the input message for the agent.
        message = AgentInput(prompt=step.prompt, records=self._records, parameters={})

        # Select the requested variant based on the index.
        variants = self._agent_types[role_upper]
        if not (0 <= variant_index < len(variants)):
            logger.warning(f"Variant index {variant_index} out of range for role {step.role} (0-{len(variants)-1}). Using index 0.")
            variant_index = 0

        agent_type, agent_config = variants[variant_index]
        variant_id = getattr(agent_config, "id", f"variant_{variant_index}")  # Get unique ID
        logger.info(f"Executing step '{step.role}' using variant '{variant_id}' (Index: {variant_index}).")

        # Create a unique ID for this specific execution instance in the exploration path.
        execution_id = f"{step.role}-{variant_id}-{shortuuid.uuid()[:4]}"
        self._exploration_path.append(execution_id)
        logger.debug(f"Added step execution to path: {execution_id}")

        # --- Execute the agent using the runtime ---
        response: AgentOutput | None = None
        try:
            # Get the specific agent instance ID from the runtime.
            agent_instance_id = await self._runtime.get(agent_type)
            # Send the message to the specific agent instance.
            response = await self._runtime.send_message(message, recipient=agent_instance_id)
            logger.debug(f"Received response from agent instance {agent_instance_id} for execution {execution_id}.")

            # Validate the response type.
            if response and not isinstance(response, AgentOutput):
                logger.warning(f"Agent {variant_id} returned unexpected type {type(response)}. Expected AgentOutput.")
                # Attempt to wrap it? Or discard? For now, store raw if possible.
                raw_output = response
                response = AgentOutput(agent_id=variant_id, role=step.role, inputs=message.inputs)
                response.set_error(f"Agent returned unexpected type: {type(raw_output)}")
                response.outputs = {"raw_output": raw_output}

        except Exception as e:
            logger.error(f"Error sending message to agent {variant_id} for execution {execution_id}: {e}")
            # Create an error response
            response = AgentOutput(agent_id=variant_id, role=step.role, inputs=message.inputs)
            response.set_error(f"Failed to execute agent: {e}")

        # --- Store results ---
        if response:
            # Store relevant information about the execution and its result.
            self._exploration_results[execution_id] = {
                "agent_id": variant_id,
                "role": step.role,
                "variant_index": variant_index,
                "parameters": agent_config.parameters if agent_config else {},
                "inputs": message.model_dump(exclude={"records"}),  # Store input data (excluding potentially large records)
                "outputs": (
                    response.outputs.model_dump() if isinstance(response.outputs, BaseModel) else response.outputs
                ),  # Store parsed outputs if possible
                "is_error": response.is_error,
                "error_details": response.error if response.is_error else None,
                "metadata": response.metadata,
            }
            logger.debug(f"Stored result for execution {execution_id}.")
        else:
            # Handle case where send_message returned None (e.g., timeout, internal error)
            logger.warning(f"No response received from agent {variant_id} for execution {execution_id}.")
            self._exploration_results[execution_id] = {
                "agent_id": variant_id,
                "role": step.role,
                "variant_index": variant_index,
                "is_error": True,
                "error_details": "No response received from agent.",
            }

        # The Autogen runtime implicitly handles publishing TaskProcessingComplete.
        # The adapter publishes the AgentOutput `response` itself if needed based on its logic.
        return cast(Optional[AgentOutput], response)  # Return the result (or None if execution failed)

    async def _run(self, request: Optional[RunRequest] = None) -> None:
        """
        Main execution loop for the Selector orchestrator.

        Handles setup, initial data loading, and the interactive loop involving
        conductor suggestions, user feedback/selection, and step execution.
        """
        try:
            logger.info(f"Starting Selector orchestrator run for flow '{self.name}'.")
            await self._setup()

            # Handle initial data loading if RunRequest is provided.
            if request:
                # TODO: Refactor fetch logic. _fetch_record seems like a utility, not core run logic.
                #       Perhaps call Fetch agent via _ask_agents or rely on initial setup message?
                result = await self._fetch_record(request)
                await self._runtime.publish_message(result, self._topic)

            # --- Main Interactive Loop ---
            while True:
                try:
                    await asyncio.sleep(0.5)  # Small delay

                    # 1. Get suggested step from Conductor
                    suggested_step = await self._get_host_suggestion()

                    # Handle WAIT response from conductor
                    if suggested_step.role == WAIT:
                        logger.info("Conductor suggested WAIT. Pausing before asking again.")
                        await asyncio.sleep(5)  # Wait longer if conductor says to wait
                        continue

                    # Handle END response from conductor (allow user final say)
                    if suggested_step.role == END:
                        logger.info("Conductor suggested END.")
                        # Optionally confirm end with user? For now, break loop.
                        # response = await self._in_the_loop(prompt="Conductor indicates the flow is complete. Press Enter to finish or provide final feedback.")
                        # if response.halt: raise StopAsyncIteration... etc.
                        break  # Exit the main loop on END suggestion

                    # 2. Interact with User (Confirm/Feedback/Select Variant)
                    user_response = await self._in_the_loop(suggested_step)

                    # Handle user halting the process
                    if user_response.halt:  # halt is checked within _wait_for_human now
                        # StopAsyncIteration is raised by _wait_for_human if halt=True
                        pass  # Should not be reached if exception is raised

                    # Handle user rejecting the step
                    if not user_response.confirm:
                        logger.info("User rejected the proposed step. Asking conductor for alternatives.")
                        # Loop back to _get_host_suggestion, providing the rejection context via _user_feedback.
                        continue

                    # Handle user providing feedback with interrupt flag
                    if user_response.interrupt:
                        logger.info("User provided feedback with interrupt flag. Requesting conductor to review feedback before proceeding.")
                        # Loop back to _get_host_suggestion, where the feedback is already stored in _user_feedback
                        continue

                    # 3. Determine Variant and Execute Step
                    selected_variant_index = 0  # Default to the first variant
                    if self._last_user_selection:  # Check if user made a selection in _in_the_loop
                        # Attempt to find the index for the selected variant ID.
                        if self._last_user_selection in self._variant_mapping:
                            selected_variant_index = self._variant_mapping[self._last_user_selection]
                            logger.info(f"Executing step with user-selected variant: '{self._last_user_selection}' (Index: {selected_variant_index})")
                        else:
                            logger.warning(
                                f"User selected variant '{self._last_user_selection}' not found for role '{suggested_step.role}'. Using default variant 0."
                            )
                        self._last_user_selection = None  # Clear selection after use

                    # Execute the confirmed step with the chosen variant.
                    await self._execute_step(step=suggested_step, variant_index=selected_variant_index)
                    # Result is stored in _exploration_results, loop continues.

                # --- Loop Exception Handling ---
                except ProcessingError as e:
                    # Log non-fatal processing errors (e.g., template errors) and continue loop.
                    logger.error(f"Processing error during Selector loop: {e}")
                    # Optionally inform the user about the error?
                    await self._send_ui_message(ManagerMessage(content=f"⚠️ Encountered an error: {e}. Trying to continue..."))
                    continue  # Attempt to recover by getting next step suggestion
                except (StopAsyncIteration, KeyboardInterrupt):
                    # Catch termination signals (END step processed or Ctrl+C).
                    raise  # Re-raise to be caught by the outer handler
                except Exception as e:
                    # Catch unexpected errors within the loop.
                    logger.exception(f"Unexpected error in Selector loop: {e}")
                    # Depending on severity, might try to inform user and continue, or re-raise.
                    try:
                        await self._send_ui_message(ManagerMessage(content=f"🚨 Unexpected Error: {e}. Attempting to recover..."))
                    except Exception:
                        pass  # Avoid errors during error reporting
                    continue  # Try to continue the loop

        # --- Outer Exception Handling & Cleanup ---
        except (StopAsyncIteration, KeyboardInterrupt):
            logger.info(f"Selector orchestrator run for '{self.name}' finished or terminated.")
        except Exception as e:
            logger.exception(f"Fatal error during Selector orchestrator run for '{self.name}': {e}")
        finally:
            logger.info(f"Cleaning up Selector orchestrator for '{self.name}'.")
            await self._cleanup()  # Ensure base class cleanup runs

    async def _fetch_record(self, request: RunRequest) -> ToolOutput | None:
        """
        Utility to fetch initial record(s) based on RunRequest.

        Args:
            request: The RunRequest containing record_id or uri.
        """
        # TODO: This seems like application-specific logic that might belong elsewhere
        #       or be handled by a dedicated 'fetch' agent called at the start of the run.
        #       Keeping it for now but consider refactoring.
        if not (request.record_id or request.uri):
            logger.debug("No record_id or uri provided in RunRequest, skipping fetch.")
            return None

        logger.info(f"Fetching initial record (ID: {request.record_id}, URI: {request.uri})...")
        try:
            # Use the FetchRecord agent directly (consider if this should be part of the flow instead)
            # Assumes 'data' sources are configured correctly for the orchestrator.
            from buttermilk.agents.fetch import FetchRecord

            fetch_agent = FetchRecord(role="fetch_init", data=list(self.data))  # Create temporary instance
            fetch_output = await fetch_agent._run(record_id=request.record_id, uri=request.uri, prompt=request.prompt)
            if fetch_output and fetch_output.results:
                self._records = fetch_output.results
                logger.info(f"Successfully fetched {len(self._records)} initial record(s).")
                return fetch_output
            else:
                logger.warning("Fetch agent did not return any results.")
        except ImportError:
            logger.error("Could not import FetchRecord agent for initial fetch.")
        except Exception as e:
            logger.error(f"Error fetching initial record: {e}")
            # Should this prevent the flow from starting?
        return None
