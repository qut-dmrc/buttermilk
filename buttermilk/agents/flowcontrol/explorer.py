"""Defines the ExplorerHost agent for managing interactive exploration workflows.

The ExplorerHost agent guides conversations where the path of inquiry is not
predefined but is discovered through interaction, user feedback, and LLM-driven
suggestions. It's a specialized form of `LLMHostAgent` designed to facilitate
step-by-step exploration of topics or data.
"""

from typing import Any  # For type hinting class types

from autogen_core import CancellationToken  # Autogen cancellation token
from pydantic import BaseModel, Field, PrivateAttr  # Pydantic components

from buttermilk import logger  # Centralized logger
from buttermilk._core.agent import AgentTrace  # For storing exploration results
from buttermilk._core.constants import END, WAIT  # Buttermilk constants
from buttermilk._core.contract import (  # Buttermilk message contracts
    AgentInput,
    AgentOutput,
    AgentTrace,
    ConductorRequest,  # Expected input type for choosing next step
    StepRequest,  # Expected output type for next step
)

from .llmhost import LLMHostAgent  # Base class for LLM-powered host agents

TRUNCATE_LEN = 1000  # Max characters per history message for summarization
"""Maximum length for individual message summaries in exploration history."""


class ExplorerHost(LLMHostAgent):
    """An advanced host agent for guiding interactive and dynamic exploration workflows.

    The `ExplorerHost` specializes in managing conversations where the sequence of
    actions or topics is not fixed in advance. Instead, it relies on suggestions
    (often from an LLM-based "conductor" agent, received as `ConductorRequest`),
    user feedback, and its own internal logic (potentially LLM-driven via `_choose`)
    to determine the next step in the exploration.

    Key Features:
        - Manages step-by-step execution, where each step is typically a `StepRequest`
          directed at another agent.
        - Can be configured to operate in interactive or autonomous modes.
        - Tracks the history of the exploration path and the results obtained at each step.
        - Uses an LLM (via `LLMHostAgent`'s capabilities) in its `_choose` method
          to decide on subsequent actions, potentially considering past steps,
          user feedback, and unexplored avenues.
        - Can be configured with limits (e.g., `max_exploration_steps`) and strategies
          (e.g., `prioritize_unexplored`) to guide the exploration.

    Configuration Parameters (from `AgentConfig.parameters` or direct attributes):
        - `exploration_mode` (str): "interactive" (allows user feedback, though
          `_in_the_loop` from `LLMHostAgent` is the primary interaction point)
          or "autonomous". Default: "interactive".
        - `max_exploration_steps` (int): Maximum number of steps before the
          agent suggests concluding the exploration. Default: 20.
        - `consider_previous_steps` (bool): If True, the LLM considers past steps
          when choosing the next one. Default: True. (Note: Actual usage in `_choose`
          depends on the prompt template for the LLM.)
        - `prioritize_unexplored` (bool): If True, the agent may try to guide
          the LLM to suggest steps involving agents/roles not yet used in the
          exploration. Default: True.

    Internal State:
        _exploration_path (list[str]): A list of execution IDs representing the
            chronological path of the exploration.
        _exploration_results (dict[str, dict[str, Any]]): A dictionary storing
            detailed results from each step of the exploration, keyed by execution ID.
    
    Expected Input to `_process`:
        Primarily `ConductorRequest` messages, which provide context and suggestions
        for the next step. Other `AgentInput` types might result in a "WAIT" state.

    Output from `_process` (and subsequently `_choose`):
        An `AgentOutput` wrapping a `StepRequest` that defines the next action to be
        taken in the flow (e.g., which agent to call next, with what prompt).
        Can also output `StepRequest(role=END)` or `StepRequest(role=WAIT)`.
    """

    _output_model: type[BaseModel] | None = StepRequest
    """Specifies that the LLM called by this host (e.g., in `_choose` via `_process`)
    is expected to produce output parsable into a `StepRequest` model.
    """

    # Configuration fields for exploration behavior
    exploration_mode: str = Field(
        default="interactive",
        description="Mode of exploration: 'interactive' (allows user feedback) or 'autonomous'.",
    )
    max_exploration_steps: int = Field(
        default=20,
        description="Maximum number of exploration steps before suggesting completion.",
    )
    consider_previous_steps: bool = Field(  # Note: Prompt engineering needed for LLM to use this
        default=True,
        description="Whether the LLM should consider previous steps when choosing the next one (requires prompt support).",
    )
    prioritize_unexplored: bool = Field(  # Note: Prompt engineering needed for LLM to use this
        default=True,
        description="Whether to guide the LLM to prioritize unexplored agents/roles (requires prompt support).",
    )

    # Private attributes for state tracking
    _exploration_path: list[str] = PrivateAttr(default_factory=list)
    _exploration_results: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)

    async def _process(
        self,
        *,
        message: AgentInput,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> AgentOutput:
        """Processes an incoming message, primarily to determine the next exploration step.

        If the message is a `ConductorRequest`, this method calls `self._choose`
        to decide the next `StepRequest` for the exploration. For other input
        types, it defaults to outputting a `StepRequest` with a "WAIT" role,
        indicating it's awaiting further instructions or a valid `ConductorRequest`.

        Args:
            message: The input `AgentInput` message. Expected to be a
                `ConductorRequest` for active step choosing.
            cancellation_token: An optional token for cancelling the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            AgentOutput: An `AgentOutput` object where `outputs` is a `StepRequest`
            indicating the next action (or WAIT/END).

        """
        # Import locally if it causes issues, but AgentOutput is usually fine at module level
        # from buttermilk._core.agent import AgentOutput

        next_step: StepRequest
        if isinstance(message, ConductorRequest):
            next_step = await self._choose(message=message)
        else:
            logger.debug(f"ExplorerHost '{self.agent_id}' received non-ConductorRequest input type: {type(message)}. Waiting.")
            next_step = StepRequest(role=WAIT, content="Waiting for conductor request or further instructions.")

        return AgentOutput(
            agent_id=self.agent_id,  # ID of this ExplorerHost agent
            metadata={"role": self.role, "source_message_id": getattr(message, "message_id", None)},
            outputs=next_step,  # The chosen StepRequest
        )

    async def _choose(self, message: ConductorRequest | None) -> StepRequest:
        """Chooses the next step in the exploration based on the current context and strategy.

        This method implements the core decision-making logic for the `ExplorerHost`.
        It considers the maximum exploration steps, enhances the incoming `ConductorRequest`
        with exploration-specific history and context, and then uses its LLM
        (via the inherited `_process` method, which is assumed to be from `LLMHostAgent`
        and ultimately calls an LLM) to suggest the next `StepRequest`.

        Args:
            message: The `ConductorRequest` providing context and suggestions from
                a conductor agent, or `None` if called without explicit input.

        Returns:
            StepRequest: A `StepRequest` object representing the chosen next step.
            This could be a step for another agent, a "WAIT" signal, or an "END" signal
            if the exploration limit is reached or the LLM suggests termination.

        """
        if len(self._exploration_path) >= self.max_exploration_steps:
            logger.info(f"ExplorerHost '{self.agent_id}': Reached maximum exploration steps ({self.max_exploration_steps}). Suggesting END.")
            return StepRequest(role=END, content="Maximum exploration steps reached.")

        if message is None:
            logger.warning(f"ExplorerHost '{self.agent_id}': _choose called with None message. Defaulting to WAIT.")
            return StepRequest(role=WAIT, content="Waiting for a valid conductor request.")

        # Enhance the incoming message with exploration history and context
        enhanced_conductor_request = await self._enhance_message_for_exploration(message)

        # Use the LLM (via superclass's _process or similar mechanism) to determine the next step.
        # This assumes LLMHostAgent._process takes AgentInput and returns AgentOutput wrapping StepRequest.
        # The current class's _process is what's called here, which then calls self._choose.
        # This creates a recursive call if not careful.
        # The intent is that LLMHostAgent has a _process method that calls an LLM.
        # We need to call that LLMHostAgent._process here, not self._process.
        # For now, assuming super()._process will invoke the LLMHostAgent's LLM call.

        # TODO: Verify this call sequence. If LLMHostAgent._process also calls _choose,
        # this will recurse. LLMHostAgent._process should be the one making the LLM call
        # based on a prompt constructed from enhanced_conductor_request.
        # The current structure of ExplorerHost._process calling self._choose, and
        # _choose calling self._process (even if it's intended to be super()._process)
        # needs careful review.
        # Assuming LLMHostAgent's _process is the LLM call:
        llm_decision_output: AgentOutput = await super()._process(message=enhanced_conductor_request)
                                                            # Pass empty kwargs if super()._process expects it

        chosen_step: StepRequest
        if isinstance(llm_decision_output.outputs, StepRequest):
            chosen_step = llm_decision_output.outputs
        else:
            logger.warning(
                f"ExplorerHost '{self.agent_id}': LLM call in _choose returned unexpected output type: "
                f"{type(llm_decision_output.outputs)}. Defaulting to WAIT.",
            )
            chosen_step = StepRequest(role=WAIT, content="Waiting after unexpected LLM output for step choice.")

        if not chosen_step.role:  # Validate that the LLM provided a role
            logger.warning(f"ExplorerHost '{self.agent_id}': LLM suggested step without a role. Defaulting to WAIT.")
            chosen_step = StepRequest(role=WAIT, content="Waiting after LLM suggested step without a role.")

        # TODO: Add logic to record the chosen_step in _exploration_path if it's not END/WAIT?
        # Or is that handled when the step is actually executed and result stored?
        # Current _store_exploration_result happens after execution.

        return chosen_step

    async def _enhance_message_for_exploration(self, message: ConductorRequest) -> ConductorRequest:
        """Enhances a `ConductorRequest` with context about the ongoing exploration.

        This method augments the input `message` (typically from a conductor agent)
        by adding information such as:
        - Statistics about the exploration (steps taken, roles explored/unexplored).
        - Recent user feedback (if tracked by `LLMHostAgent`).
        - A summary of the exploration history.
        - If `prioritize_unexplored` is True, it may also prepend a suggestion to
          the prompt to consider unexplored roles.

        Args:
            message: The original `ConductorRequest` to be enhanced.

        Returns:
            ConductorRequest: An enhanced copy of the input `ConductorRequest` with
            added exploration-specific context in its `inputs` field and potentially
            an updated `prompt`.

        """
        enhanced_request = message.model_copy(deep=True)

        # Gather exploration statistics
        explored_roles = {self._exploration_results[step_id].get("role") for step_id in self._exploration_path if step_id in self._exploration_results and self._exploration_results[step_id].get("role")}
        # _participants is from LLMHostAgent, assumed to be a dict of role_name: AgentConfig/AgentVariants
        available_roles = set(getattr(self, "_participants", {}).keys())
        unexplored_roles = available_roles - explored_roles

        exploration_context_stats = {
            "steps_taken": len(self._exploration_path),
            "max_steps": self.max_exploration_steps,
            "explored_roles": sorted(list(explored_roles)),  # Sorted for consistent prompting
            "unexplored_roles": sorted(list(unexplored_roles)),
            "available_roles": sorted(list(available_roles)),
        }

        # Prepare exploration context to be added to inputs
        # _user_feedback is assumed to be an attribute from LLMHostAgent or similar
        exploration_inputs_update = {
            "exploration_statistics": exploration_context_stats,
            "recent_user_feedback": getattr(self, "_user_feedback", "No recent user feedback available."),  # Provide default
            "exploration_history_summary": self._summarize_exploration_history(),
        }

        # Update inputs of the enhanced request
        if enhanced_request.inputs:  # If inputs dict already exists
            enhanced_request.inputs.update(exploration_inputs_update)
        else:  # If inputs dict doesn't exist, create it
            enhanced_request.inputs = exploration_inputs_update

        # Optionally update the prompt to encourage exploring new roles
        if self.prioritize_unexplored and unexplored_roles:
            unexplored_roles_str = ", ".join(sorted(list(unexplored_roles)))
            prioritization_hint = f"Consider exploring roles not yet used: {unexplored_roles_str}. "
            original_prompt = enhanced_request.prompt or ""
            enhanced_request.prompt = prioritization_hint + original_prompt

        return enhanced_request

    def _store_exploration_result(self, execution_id: str, output: AgentTrace) -> None:
        """Stores the result of an executed exploration step.

        This method updates the internal state of the `ExplorerHost` by appending
        the `execution_id` to `_exploration_path` (if not already present) and
        storing relevant details from the `output` (`AgentTrace`) into the
        `_exploration_results` dictionary.

        Args:
            execution_id (str): A unique identifier for this specific execution
                step (e.g., a hash of the step details or a UUID). This ID is
                used as the key in `_exploration_results` and added to `_exploration_path`.
            output (AgentTrace): The `AgentTrace` object resulting from the
                execution of the exploration step. It contains the inputs, outputs,
                metadata, and error status of the step.

        """
        if execution_id not in self._exploration_path:
            self._exploration_path.append(execution_id)

        # _current_step_name is likely an attribute from LLMHostAgent tracking the role of the agent that just ran
        current_role_executed = getattr(self, "_current_step_name", "unknown_role")

        self._exploration_results[execution_id] = {
            "call_id": output.call_id,  # Trace's own call_id
            "role": current_role_executed,
            "inputs": getattr(output, "inputs", {}),  # Original AgentInput to the executed agent
            "outputs": getattr(output, "outputs", getattr(output, "content", "")),  # Actual output data
            "is_error": output.is_error,  # From FlowMessage base
            "error_details": output.error if output.is_error else [],  # error is list[str]
            "metadata": getattr(output, "metadata", {}),
        }
        logger.debug(f"ExplorerHost '{self.agent_id}': Stored result for execution step ID '{execution_id}' by role '{current_role_executed}'.")

    def _summarize_exploration_history(self) -> list[dict[str, Any]]:
        """Creates a summarized, chronological list of the exploration steps taken.

        This summary is intended to be passed to an LLM (as part of the context
        in `_enhance_message_for_exploration`) to inform its decision-making
        for the next step. It focuses on the sequence of roles executed,
        whether each step was successful, and a truncated summary of the output.

        Returns:
            list[dict[str, Any]]: A list of dictionaries, where each dictionary
            summarizes one step in the exploration history. Each summary includes
            `id` (execution ID), `role` (role of the agent executed), `success`
            (boolean), and an optional `output_summary`.

        """
        summary_list: list[dict[str, Any]] = []
        for step_execution_id in self._exploration_path:
            if step_execution_id in self._exploration_results:
                result_data = self._exploration_results[step_execution_id]
                step_summary_entry = {
                    "id": step_execution_id,  # The unique ID for this step in the exploration path
                    "role": result_data.get("role", "unknown_role"),
                    "success": not result_data.get("is_error", False),
                }

                # Add a condensed output summary
                outputs_data = result_data.get("outputs")
                if isinstance(outputs_data, str):
                    summary_text = outputs_data[:TRUNCATE_LEN] + "..." if len(outputs_data) > TRUNCATE_LEN else outputs_data
                    step_summary_entry["output_summary"] = summary_text
                elif isinstance(outputs_data, (dict, list)):
                    # For structured data, provide a type hint or a brief representation
                    try:
                        # Attempt a compact JSON representation for structured data
                        json_summary = json.dumps(outputs_data)
                        step_summary_entry["output_summary"] = json_summary[:TRUNCATE_LEN] + "..." if len(json_summary) > TRUNCATE_LEN else json_summary
                    except TypeError:  # Handle non-serializable data
                        step_summary_entry["output_summary"] = f"Structured data of type {type(outputs_data).__name__}"
                elif outputs_data is not None:
                     step_summary_entry["output_summary"] = str(outputs_data)[:TRUNCATE_LEN] + "..." if len(str(outputs_data)) > TRUNCATE_LEN else str(outputs_data)

                summary_list.append(step_summary_entry)
        return summary_list
