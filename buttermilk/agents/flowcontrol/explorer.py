from typing import Any

from autogen_core import CancellationToken
from pydantic import BaseModel, Field

from buttermilk import logger
from buttermilk._core.contract import (
    END,
    WAIT,
    AgentInput,
    ConductorRequest,
    StepRequest,
)

from .llmhost import LLMHostAgent

TRUNCATE_LEN = 1000  # characters per history message

###########
####
# TODO:
#####
# Key Features:
        # - Step-by-step execution driven by `CONDUCTOR` agent suggestions.
        # - User confirmation and feedback collection at each step via `_in_the_loop`.
        # - Ability for the user to select specific agent variants for a step.
        # - Tracking of the exploration path and results (`_exploration_path`, `_exploration_results`).
        # - Handling of special messages from the `CONDUCTOR` (e.g., questions for the user, comparisons).
####
####
#


class ExplorerHost(LLMHostAgent):
    """An advanced host agent for interactive exploration workflows.
    
    This agent specializes in guiding conversations where the path isn't predetermined,
    but rather discovered through interaction and exploration. It maintains state
    about the exploration history, understands user feedback, and adapts its
    recommendations based on the evolving context.
    """

    _output_model: type[BaseModel] | None = StepRequest

    async def _process(self, *, message: AgentInput, cancellation_token: CancellationToken | None = None, **kwargs):
        """Process a message for the Explorer agent - calls _choose to determine the next step"""
        from buttermilk._core.agent import AgentResponse  # Import locally to avoid being removed

        if isinstance(message, ConductorRequest):
            step = await self._choose(message=message)
            return AgentResponse(
                metadata={"source": self.agent_id, "role": self.role},
                outputs=step,
            )

        step = StepRequest(role=WAIT, content="Waiting for conductor request")
        return AgentResponse(
            metadata={"source": self.agent_id, "role": self.role},
            outputs=step,
        )

    # Explorer-specific configuration
    exploration_mode: str = Field(
        default="interactive",
        description="Mode of exploration: 'interactive' (with user feedback) or 'autonomous'",
    )
    max_exploration_steps: int = Field(
        default=20,
        description="Maximum number of exploration steps before suggesting completion",
    )
    consider_previous_steps: bool = Field(
        default=True,
        description="Whether to consider previous steps when choosing the next one",
    )
    prioritize_unexplored: bool = Field(
        default=True,
        description="Whether to prioritize unexplored agents over previously used ones",
    )

    async def _choose(self, message: ConductorRequest | None) -> StepRequest:
        """Choose the next step based on exploration context.
        
        This method analyzes the conversation history, user feedback,
        and exploration state to determine the most appropriate next step.
        It uses LLM reasoning through the _process method, which employs
        the agent's language model to make decisions.
        
        Args:
            message: The ConductorRequest containing context for decision
            
        Returns:
            A StepRequest representing the chosen next step

        """
        # Check if we've reached the maximum exploration steps
        if len(self._exploration_path) >= self.max_exploration_steps:
            logger.info(f"Reached maximum exploration steps ({self.max_exploration_steps}), suggesting END")
            return StepRequest(role=END, content="Reached maximum exploration steps")

        # Handle the None case
        if message is None:
            logger.warning("Explorer received None message in _choose, using fallback")
            return StepRequest(role=WAIT, content="Waiting for valid conductor request")

        # Enhance message context with exploration-specific information
        enhanced_message = await self._enhance_message_for_exploration(message)

        # Use LLM to determine the next step
        result = await self._process(message=enhanced_message)

        # Process the result
        if isinstance(result, StepRequest):
            step = result
        # Check more specific types that we know have outputs
        elif (hasattr(result, "outputs") and
              hasattr(result, "metadata") and
              isinstance(result.outputs, StepRequest)):
            # Handle AgentResponse or similar structures that contain a StepRequest
            step = result.outputs
        else:
            # Fallback for invalid or unexpected return types
            logger.warning(f"Explorer received unexpected result type from LLM: {type(result)}")
            step = StepRequest(role=WAIT, content="Waiting after receiving invalid result type")

        # Validate the step has a role
        if not step.role:
            logger.warning("Explorer received step without role from LLM, using fallback")
            step = StepRequest(role=WAIT, content="Waiting after receiving invalid step suggestion")

        return step

    async def _enhance_message_for_exploration(self, message: ConductorRequest) -> ConductorRequest:
        """Enhance the conductor request with exploration-specific context.
        
        This method adds information about exploration history, which roles have been
        explored, user feedback patterns, and suggested areas for further exploration.
        
        Args:
            message: The original ConductorRequest
            
        Returns:
            An enhanced ConductorRequest with exploration context

        """
        # Create a copy to avoid modifying the original
        enhanced = message.model_copy(deep=True)

        # Get all roles that have been explored so far
        explored_roles = set()
        for path_id in self._exploration_path:
            if path_id in self._exploration_results:
                role = self._exploration_results[path_id].get("role")
                if role:
                    explored_roles.add(role)

        # Get all available roles from participants
        available_roles = set(self._participants.keys())

        # Find unexplored roles
        unexplored_roles = available_roles - explored_roles

        # Add exploration analytics to the inputs
        exploration_context = {
            "exploration_statistics": {
                "steps_taken": len(self._exploration_path),
                "max_steps": self.max_exploration_steps,
                "explored_roles": list(explored_roles),
                "unexplored_roles": list(unexplored_roles),
                "available_roles": list(available_roles),
            },
            "recent_user_feedback": self._user_feedback,
            "exploration_history": self._summarize_exploration_history(),
        }

            # # add extra information about next step
            # request_content = (
            #     f"**Next Proposed Step:**\n"
            #     f"- **Agent Role:** {step.role}\n"
            #     f"- **Description:** {step.content or '(No description)'}\n"
            #     f"- **Prompt Snippet:** {step.prompt[:100] + '...' if step.prompt else '(No prompt)'}"
            #     f"{variant_info}\n\n"
            #     f"Confirm (Enter), provide feedback, select a variant ID, or reject ('n'/'q')."
            # )
        # Update inputs with exploration context
        if "inputs" in enhanced.__dict__:
            if isinstance(enhanced.inputs, dict):
                enhanced.inputs.update(exploration_context)
            else:
                # If inputs is not a dict, create a new one
                enhanced.inputs = {**exploration_context}

        # Update the prompt to encourage exploration if needed
        if self.prioritize_unexplored and unexplored_roles:
            prioritize_message = f"Consider exploring these roles that haven't been used yet: {', '.join(unexplored_roles)}. "
            enhanced.prompt = prioritize_message + (enhanced.prompt or "")

        return enhanced

    def _summarize_exploration_history(self) -> list[dict[str, Any]]:
        """Create a summarized version of the exploration history.
        
        This creates a chronological summary of exploration steps taken,
        focusing on role sequence, outcomes, and patterns.
        
        Returns:
            A list of summary dictionaries for each exploration step

        """
        summary = []
        for step_id in self._exploration_path:
            if step_id in self._exploration_results:
                result = self._exploration_results[step_id]
                step_summary = {
                    "id": step_id,
                    "role": result.get("role", "unknown"),
                    "success": not result.get("is_error", False),
                }

                # Add a condensed output summary if available
                if "outputs" in result:
                    outputs = result["outputs"]
                    if isinstance(outputs, str):
                        # Truncate long string outputs
                        step_summary["output_summary"] = outputs[:100] + "..." if len(outputs) > 100 else outputs
                    elif isinstance(outputs, (dict, list)):
                        # For structured data, just note the type
                        step_summary["output_summary"] = f"{type(outputs).__name__} data"

                summary.append(step_summary)

        return summary
