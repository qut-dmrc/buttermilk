"""Defines the Judge agent, an LLM-based agent specialized for evaluating content
against predefined criteria.
"""

from typing import Literal

# Import Autogen core components needed for type hints and potential interaction (though handled by adapter)
from pydantic import BaseModel, Field

# Buttermilk core imports
from buttermilk._core.agent import AgentInput, AgentTrace, buttermilk_handler  # Base types and decorator
from buttermilk.agents.llm import LLMAgent  # Base class for LLM-powered agents
from buttermilk.bm import logger  # Global Buttermilk instance and logger


# --- Pydantic Models ---
class Reasons(BaseModel):
    conclusion: str = Field(..., description="Your conclusion or final answer summarizing current tentative conclusions and outlining any uncertainty.")
    reasons: list[str] = Field(..., description="Each element should represent a single step in logical reasoning.")

    def __str__(self):
        """Returns a nicely formatted MarkDown representation of the evaluation."""
        reasons_str = "\n\n\t".join(f"- {reason}" for reason in self.reasons)
        return (
            f"**Reasoning:**\n\n{reasons_str}"
        )


class JudgeReasons(Reasons):
    """Structured output model for the Judge agent's evaluation.

    Defines the expected JSON structure returned by the LLM after evaluating
    content against the provided criteria.
    """

    prediction: bool = Field(
        ...,
        description="Boolean flag indicating if the content violates the policy/guidelines. This should be derived logically from the reasoning and criteria application.",
    )
    reasons: list[str] = Field(
        ...,
        description="A list of strings, where each string represents a distinct step in the reasoning process leading to the conclusion and prediction.",
    )
    uncertainty: Literal["high", "medium", "low"] = Field(
        ...,
        description="The scope for reasonable minds to differ on this conclusion.",
    )

    def __str__(self):
        """Returns a nicely formatted MarkDown representation of the evaluation."""
        reasons_str = "\n\n".join(f"\t- {reason}" for reason in self.reasons)
        return (
            f"**Conclusion:** {self.conclusion}\n"
            f"**Violates Policy:** {'Yes' if self.prediction else 'No'}\n"
            f"**Confidence:** {self.uncertainty.capitalize()}\n"
            f"**Reasoning:**\n{reasons_str}"
        )


# --- Judge Agent ---
class Judge(LLMAgent):
    """An LLM agent specialized in evaluating content based on provided criteria.

    Inherits from `LLMAgent`, leveraging its capabilities for LLM interaction,
    prompt templating, and structured output parsing. The `Judge` agent is
    configured with a specific prompt template (likely focused on evaluation)
    and expects the LLM to return results conforming to the `AgentReasons` model.

    The `buttermilk_handler` decorator registers methods to handle specific message
    types within the Buttermilk/Autogen ecosystem.
    """

    # Specifies that the expected structured output from the LLM should conform to AgentReasons.
    # LLMAgent's _process method will attempt to parse the LLM response into this model.
    _output_model: type[BaseModel] | None = JudgeReasons

    # Initialization (`__init__`) is handled by the parent LLMAgent, which takes
    # configuration (like model, template, parameters) via its AgentConfig.

    # --- Buttermilk/Autogen Message Handler ---

    # This decorator registers the method with the Buttermilk framework.
    # When this agent (wrapped by AutogenAgentAdapter) receives an AgentInput message
    # via Autogen, the adapter will likely route it to this handler.
    @buttermilk_handler(AgentInput)
    async def evaluate_content(
        self,
        message: AgentInput,
    ) -> AgentTrace:
        """Handles an AgentInput request to evaluate content based on the agent's configured criteria.

        Args:
            message: The AgentInput message containing the content/prompt to evaluate.

        Returns:
            An AgentTrace message containing the structured evaluation (AgentReasons)
            or an error if processing fails.

        """
        raise NotImplementedError("@buttermilk_handler is only an idea at this stage.")
        logger.debug(f"Judge agent '{self.agent_id}' received evaluation request.")
        # Note that we don't do error handling here. If the call fails, the Autogen Adapter
        # or whatever else called us has to deal with it.

        # Delegate the core LLM call and output parsing to the parent LLMAgent's _process method.
        # This method handles template rendering, API calls, retries, and parsing into _output_model.
        response = await self._process(message=message)

        # Create a new AgentTrace from the response
        trace = AgentTrace(agent_id=self.agent_id, session_id=self.session_id,
            agent_info=self._cfg,
            inputs=message,
            outputs=response.outputs,
            metadata=response.metadata,
        )

        return trace

    # Note: Other handlers (like _listen, _handle_events) can be added here if the Judge
    # needs to react to other message types or perform background tasks, inheriting or
    # overriding behavior from Agent/LLMAgent as needed.
