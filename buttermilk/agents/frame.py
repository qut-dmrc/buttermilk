"""Defines the Frame agent and associated Pydantic models for structured frame analysis.

This module provides the `Frame` agent, an LLM-based agent specialized for
analyzing news articles using Entman's (1993) framing theory. It identifies
frame elements at the statement level, including speakers, problem definitions,
causal attributions, moral evaluations, and treatment recommendations.
"""

from pydantic import BaseModel, Field  # Pydantic components

# Buttermilk core imports
from buttermilk._core.agent import AgentInput, ExecutionTrace  # Base types
from buttermilk._core.log import logger  # Centralized logger
from buttermilk.agents.llm import LLMAgent  # Base class for LLM-powered agents

# --- Pydantic Models for Frame Analysis Output ---


class FramedStatement(BaseModel):
    """A model representing a single framed statement from a news article.

    This model captures all elements of Entman's (1993) framing theory for a
    single value-laden statement, including speaker identification, problem
    definition, causal attribution, moral evaluation, and treatment recommendations.

    Attributes:
        statement (str): The exact quote of the value-laden statement from the article.
        speaker_name (str): Name of the person making the statement, or 'journalist' if narration.
        speaker_affiliation (str): Organization, role, or institutional affiliation of the speaker.
        solution_addressee (Optional[str]): Person or entity to whom the solution/action is directed.
        problem_definition (str): What is presented as the core issue or problem.
        blame_attribution (Optional[str]): Who or what is held responsible for causing the problem.
        moral_evaluation (Optional[str]): Moral judgment or evaluative stance toward the issue.
        recommendation (Optional[str]): Proposed solution or course of action.
        confidence_score (float): Confidence level in the frame identification (0-1).

    """

    statement: str = Field(..., description="Exact quote of the value-laden statement from the article")
    speaker_name: str = Field(
        ...,
        description="Name of the person making the statement, or 'journalist' if narration",
    )
    speaker_affiliation: str = Field(
        ...,
        description="Organization, role, or institutional affiliation of the speaker",
    )
    solution_addressee: str | None = Field(None, description="Person or entity to whom the solution/action is directed")
    problem_definition: str = Field(..., description="What is presented as the core issue or problem")
    blame_attribution: str | None = Field(None, description="Who or what is held responsible for causing the problem")
    moral_evaluation: str | None = Field(None, description="Moral judgment or evaluative stance toward the issue")
    recommendation: str | None = Field(None, description="Proposed solution or course of action")
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence level in the frame identification (0-1)",
    )

    def as_markdown(self, agent_id: str = None, call_id: str = None) -> str:
        """Returns a Markdown formatted string for insertion into templates.

        Note: This is for a single statement. Usually used within FrameAnalysisResults.

        Args:
            agent_id: The agent identifier (e.g., "FRAME-gpt4")
            call_id: The call identifier for this execution

        Returns:
            str: Formatted markdown string suitable for template insertion
        """
        # For individual statements, we don't include header (handled by FrameAnalysisResults)
        return (
            f'"{self.statement}" - {self.speaker_name} ({self.speaker_affiliation})\n'
            f"Problem: {self.problem_definition}\n"
            f"Blame: {self.blame_attribution or 'Not specified'}\n"
            f"Evaluation: {self.moral_evaluation or 'Not specified'}\n"
            f"Recommendation: {self.recommendation or 'Not specified'}"
        )

    def __str__(self) -> str:
        """Returns a formatted string representation of the framed statement."""
        return (
            f'**Statement:** "{self.statement}"\n'
            f"**Speaker:** {self.speaker_name} ({self.speaker_affiliation})\n"
            f"**Problem:** {self.problem_definition}\n"
            f"**Blame:** {self.blame_attribution or 'Not specified'}\n"
            f"**Moral Evaluation:** {self.moral_evaluation or 'Not specified'}\n"
            f"**Recommendation:** {self.recommendation or 'Not specified'}\n"
            f"**Solution Addressee:** {self.solution_addressee or 'Not specified'}\n"
            f"**Confidence:** {self.confidence_score:.2f}"
        )


class FrameAnalysisResults(BaseModel):
    """Container for multiple framed statements from an article analysis.

    This model aggregates all the framed statements identified in a news article,
    providing a complete frame analysis following Entman's (1993) theory.

    Attributes:
        statements (list[FramedStatement]): List of all framed statements identified.
        article_summary (str): Brief summary of the article being analyzed.
        dominant_frame (Optional[str]): The predominant framing pattern in the article.
    """

    statements: list[FramedStatement] = Field(..., description="List of all framed statements identified in the article")
    article_summary: str = Field(..., description="Brief summary of the article being analyzed")
    dominant_frame: str | None = Field(None, description="The predominant framing pattern identified in the article")

    def as_markdown(self, agent_id: str = None, call_id: str = None) -> str:
        """Returns a Markdown formatted string for insertion into templates.

        Format follows the standard: agent identifier on first line, followed by
        content-specific fields without empty lines between components.

        Args:
            agent_id: The agent identifier (e.g., "FRAME-gpt4")
            call_id: The call identifier for this execution

        Returns:
            str: Formatted markdown string suitable for template insertion
        """
        header = ""
        if agent_id and call_id:
            # Use only the last 8 characters of call_id for brevity
            short_call_id = call_id[-8:] if len(call_id) > 8 else call_id
            header = f"**{agent_id} #{short_call_id}**\n"

        # Format statements
        statements_str = ""
        if self.statements:
            statement_parts = []
            for stmt in self.statements[:3]:  # Show first 3 statements
                statement_parts.append(f'- {stmt.speaker_name}: "{stmt.statement[:100]}..."')
            statements_str = "\n".join(statement_parts)
            if len(self.statements) > 3:
                statements_str += f"\n- ... and {len(self.statements) - 3} more statements"

        frame_str = f"Frame: {self.dominant_frame}\n" if self.dominant_frame else ""

        return f"{header}{self.article_summary}\n{frame_str}Statements analyzed: {len(self.statements)}\n{statements_str}"

    def __str__(self) -> str:
        """Returns a Markdown formatted string representation.

        When agent context is available (via _agent_id and _call_id attributes),
        includes the full header. Otherwise returns the summary.
        """
        # Check if agent context is available (set by ExecutionTrace)
        agent_id = getattr(self, "_agent_id", None)
        call_id = getattr(self, "_call_id", None)

        if agent_id and call_id:
            return self.as_markdown(agent_id, call_id)

        # Fallback to summary
        return self.article_summary


# --- Frame Agent ---
class Frame(LLMAgent):
    """An LLM-based agent specialized in frame analysis of news articles using Entman's (1993) theory.

    The `Frame` agent inherits from `LLMAgent`, utilizing its capabilities for
    interacting with Language Models, managing prompt templates, and handling
    structured output. It is specifically configured to perform frame analysis tasks
    on news articles, particularly focused on climate activism content.

    The core of its operation involves:
    1. Receiving article content to be analyzed (via an `AgentInput` message).
    2. Using a configured LLM and a specialized prompt template that instructs
       the LLM on Entman's framing theory and the desired output format.
    3. Expecting the LLM to return a structured response that conforms to the
       `FrameAnalysisResults` Pydantic model, which captures all frame elements
       for each statement in the article.

    Key Configuration Parameters (from `AgentConfig.parameters`):
        - `model` (str): **Required**. The name of the LLM to use for frame analysis.
        - `template` (str): **Required**. The name of the prompt template
          that guides the LLM to perform frame analysis and output `FrameAnalysisResults`.

    Attributes:
        output_model (Type[BaseModel] | None): Specifies `FrameAnalysisResults` as the
            expected Pydantic model for the LLM's structured output.

    """

    def __init__(self, **kwargs):
        """Initializes the Frame agent with its specific configuration and output model."""
        super().__init__(**kwargs)
        # Set the expected output model for the LLM's response
        self.output_model = FrameAnalysisResults

    async def analyze_article(
        self,
        message: AgentInput,
    ) -> ExecutionTrace:
        """Handles an `AgentInput` request to analyze article content using frame analysis.

        This method is intended to be the primary entry point when the `Frame`
        agent is invoked to perform frame analysis, particularly within systems
        that use the `@buttermilk_handler` for message routing.

        It delegates the core LLM interaction and structured output parsing to
        the `_process` method inherited from `LLMAgent`.

        Args:
            message (AgentInput): The `AgentInput` message containing the article
                content to be analyzed. The `message.inputs` should align with what the
                Frame's prompt template expects (e.g., article text, analysis focus).

        Returns:
            ExecutionTrace: An `ExecutionTrace` object. If successful, `outputs` will
            contain an instance of `FrameAnalysisResults` (the structured analysis).
            If processing or the LLM call fails, the `error` field within the
            `ExecutionTrace` will be populated.

        Raises:
            NotImplementedError: Currently raised as a placeholder, indicating this
                handler's direct usage path might be conceptual or for specific integrations.

        """
        raise NotImplementedError("@buttermilk_handler is only an idea at this stage.")
        logger.debug(f"Frame agent '{self.agent_name}' received analysis request.")

        # Delegate the core LLM call and output parsing to the parent LLMAgent's _process method.
        # This method handles template rendering, API calls, retries, and parsing into output_model.
        trace = await self._process(message=message)

        if trace.outputs:
            logger.info(f"Frame analysis completed: {trace.outputs.preview}")

        return trace

    async def analyze_climate_activism(
        self,
        message: AgentInput,
    ) -> ExecutionTrace:
        """Specialized method for climate activism frame analysis.

        This method provides a domain-specific entry point for analyzing climate
        activism articles, potentially with specialized prompting or processing.

        Args:
            message (AgentInput): The `AgentInput` message containing the article content.

        Returns:
            ExecutionTrace: An `ExecutionTrace` object with the frame analysis results.

        """
        # Ensure the analysis focus is set appropriately
        if hasattr(message, "inputs") and isinstance(message.inputs, dict):
            message.inputs["analysis_focus"] = "climate_activism"

        logger.debug(f"Frame agent '{self.agent_name}' performing climate activism analysis.")

        return await self.analyze_article(message)

    # Note: The primary logic for the Frame agent is handled by the LLMAgent._process method,
    # which will use the `output_model = FrameAnalysisResults` to parse the LLM's response.
    # The prompt template should instruct the LLM to follow Entman's (1993) framing theory
    # and output the analysis in the expected FrameAnalysisResults structure.
