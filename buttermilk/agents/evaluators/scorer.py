"""Defines agents and Pydantic models for evaluating and scoring LLM outputs.

This module provides the `LLMScorer` agent, which uses a Language Model to
qualitatively assess the output of another agent based on predefined criteria
and, potentially, ground truth information. It also defines several Pydantic
models (`QualScoreCRA`, `QualScore`, `QualResults`) to structure the scoring
criteria, individual assessments, and the overall scoring output.
"""

from typing import Any

from autogen_core import (
    MessageContext,  # Autogen cancellation token
    message_handler,
)
from pydantic import BaseModel, Field  # Pydantic components

# Buttermilk core imports
from buttermilk import logger  # Centralized logger
from buttermilk._core.contract import (  # Buttermilk message contracts
    AgentInput,
    AgentOutput,  # Used as return type hint for _process
    ExecutionTrace,
)
from buttermilk._core.message_data import (
    extract_message_data,
)  # Utility for data extraction
from buttermilk.agents.judge import (
    JudgeReasons,
)  # Expected input model from Judge agent
from buttermilk.agents.llm import LLMAgent  # Base class for LLM-powered agents

# --- Pydantic Models for Scoring ---

"""Pydantic models for evaluating and scoring LLM outputs.

This module defines Pydantic models for structured scoring output including
error detection and ground truth alignment assessment.
"""

from enum import StrEnum

# --- Pydantic Models for Scoring ---


class QualityRating(StrEnum):
    """Quality rating for ground truth alignment."""

    NO_ANSWER = "No answer"
    INCORRECT = "Incorrect"
    KEY_ERRORS = "Key errors or inaccuracies"
    MINOR_ERRORS = "Minor errors or inaccuracies"
    CORRECT = "Correct"


class KeyPointType(StrEnum):
    """Type of key point in ground truth alignment."""

    PROHIBITED = "prohibited"
    REQUIRED = "required"
    OPTIONAL = "optional"


class Confidence(StrEnum):
    """Scorer's confidence level in the assessment."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ErrorType(StrEnum):
    """Types of critical errors in analyst reasoning."""

    HALLUCINATED_RULE = "hallucinated_rule"
    HALLUCINATED_FACT = "hallucinated_fact"
    MISINTERPRETED_FACT = "misinterpreted_fact"
    MISAPPLIED_RULE = "misapplied_rule"
    LOGICAL_ERROR = "logical_error"
    ABUSE_OF_DISCRETION = "abuse_of_discretion"


class CriticalErrorEvidence(BaseModel):
    """Evidence for a specific critical error.

    Attributes:
        error_type: The type of error found
        analyst_reasoning: Specific quoted string from the analyst's reasons list
        explanation: Why this is an error
        source_excerpt: Relevant article excerpt showing the error
        rule_excerpt: Which rule was misapplied
    """

    model_config = {"extra": "forbid"}  # Required for Azure OpenAI structured output

    error_type: ErrorType = Field(..., description="The type of error found")
    analyst_reasoning: str = Field(..., description="Specific quoted string from reasons list")
    explanation: str = Field(..., description="Why this is an error")
    source_excerpt: str = Field(..., description="Relevant article excerpt showing the error")
    rule_excerpt: str = Field(..., description="Which rule was misapplied")


class CriticalErrors(BaseModel):
    """Binary flags for critical errors and supporting evidence.

    Any true flag means the judgment is incorrect.

    Attributes:
        hallucinated_rule: Analyst invented a rule that doesn't exist
        hallucinated_fact: Analyst invented facts not in the article
        misinterpreted_fact: Analyst misunderstood facts in the article
        misapplied_rule: Analyst applied a rule incorrectly
        logical_error: Analyst made a logical error in reasoning
        abuse_of_discretion: Analyst's discretion was unreasonable
        reasons: Evidence array, only populated if errors found
    """

    model_config = {"extra": "forbid"}  # Required for Azure OpenAI structured output

    hallucinated_rule: bool = Field(default=False, description="Analyst invented a rule that doesn't exist")
    hallucinated_fact: bool = Field(default=False, description="Analyst invented facts not in the article")
    misinterpreted_fact: bool = Field(default=False, description="Analyst misunderstood facts in the article")
    misapplied_rule: bool = Field(default=False, description="Analyst applied a rule incorrectly")
    logical_error: bool = Field(default=False, description="Analyst made a logical error in reasoning")
    abuse_of_discretion: bool = Field(default=False, description="Analyst's discretion was unreasonable")
    reasons: list[CriticalErrorEvidence] = Field(
        default_factory=list,
        description="Evidence array, only populated if errors found",
    )


class GroundTruthAlignment(BaseModel):
    """Assessment of alignment with a single ground truth key point.

    Attributes:
        key_point: The ground truth key point being assessed
        key_point_type: Whether this point is prohibited, required, or optional
        key_point_mentioned: Whether the analyst mentioned this key point
        alignment: Quality rating for how well the analyst aligned with this point
    """

    model_config = {"extra": "forbid"}  # Required for Azure OpenAI structured output

    key_point: str = Field(..., description="The ground truth key point being assessed")
    key_point_type: KeyPointType = Field(..., description="Whether this point is prohibited, required, or optional")
    key_point_mentioned: bool = Field(..., description="Whether the analyst mentioned this key point")
    alignment: QualityRating = Field(..., description="Quality rating for alignment with this point")


class QualScore(BaseModel):
    """Represents a qualitative score of a analyst's output.

    This model assesses whether the analyst made critical errors and how well
    the analyst's reasoning aligns with ground truth key points.

    Attributes:
        critical_errors: Binary flags and evidence for critical reasoning errors
        ground_truth_alignment: Assessment of alignment with each ground truth key point
        confidence: Scorer's self-assessment of confidence in this evaluation
        summary: 1-2 sentence explanation of the overall assessment
    """

    model_config = {"extra": "forbid"}  # Required for Azure OpenAI structured output

    critical_errors: CriticalErrors = Field(..., description="Binary flags and evidence for critical reasoning errors")
    ground_truth_alignment: list[GroundTruthAlignment] = Field(..., description="Assessment of alignment with each ground truth key point")
    confidence: Confidence = Field(..., description="Scorer's self-assessment of confidence")
    summary: str = Field(..., description="1-2 sentence explanation of the overall assessment")


class QualResults(QualScore):
    """Extends `QualScore` to include metadata about the answer/output being assessed.

    This model is designed for presenting or logging scoring results externally,
    linking the qualitative assessments back to the specific agent output that
    was evaluated.

    Attributes:
        assessed_agent_id (str): The unique identifier of the agent whose output
            was assessed.
        assessed_call_id (str): A unique identifier for the specific answer, call,
            or output instance that was assessed. This helps in pinpointing the
            exact piece of work evaluated.

    """

    assessed_agent_id: str = Field(..., description="The ID of the agent whose output was assessed.")
    assessed_call_id: str = Field(
        ...,
        description="A unique identifier for the specific answer/output being assessed.",
    )


# --- LLM Scorer Agent ---
class LLMScorer(LLMAgent):
    """An LLM-based agent that qualitatively scores another agent's output.

    This agent listens for `ExecutionTrace` messages, particularly those containing
    `JudgeReasons` in their outputs (typically from a `Judge` agent). When such a
    message is detected and relevant ground truth information is available (either
    attached to the original record or inferred from the context), the `LLMScorer`
    triggers its own Language Model to perform an evaluation.

    The LLM is guided by a scoring-specific prompt template (configured via
    `AgentConfig.parameters.template`) to produce a structured score
    conforming to the `QualScore` Pydantic model. This structured score is then
    wrapped in a `QualResults` model, adding metadata about the assessed item,
    and included in the `ExecutionTrace` produced by this scorer.

    The agent can integrate with Weave for logging scores against the trace of
    the original agent's output that was scored.

    Key Configuration Parameters (from `AgentConfig.parameters`):
        - `model` (str): **Required**. The LLM to use for scoring.
        - `template` (str): **Required**. Template guiding the LLM to score.
          The output of this template should be parsable into `QualScore`.

    Attributes:
        output_model (Type[BaseModel] | None): Specifies `QualScore` as the
            expected Pydantic model for the LLM's structured output.

    """

    def __init__(self, **kwargs):
        """Initializes the Scorer agent with its specific configuration and output model."""
        # Fail explicitly if config tries to override output_model - Scorer requires QualScore
        if "output_model" in kwargs and kwargs["output_model"] is not None:
            raise ValueError(
                f"Scorer agent requires output_model=QualScore. Cannot override with {kwargs['output_model']}. Remove 'output_model' from config."
            )
        kwargs.pop("output_model", None)  # Remove None values to avoid duplicate kwarg
        super().__init__(output_model=QualScore, **kwargs)

    @message_handler(match=lambda msg, ctx: isinstance(msg.outputs, JudgeReasons))
    async def _score_judge(
        self,
        message: ExecutionTrace,
        ctx: MessageContext,
    ) -> None:
        """Listens for relevant `ExecutionTrace` messages and triggers the scoring process.

        This method is invoked when the `LLMScorer` passively receives a message
        in a group chat or similar context. It performs the following checks:
        1.  Verifies if the incoming `message` is an `ExecutionTrace`.
        2.  Checks if `message.outputs` is an instance of `JudgeReasons` (indicating
            it's likely a structured reasoning output from another evaluation agent like `Judge`).
        3.  Ensures `message.inputs` (the original input to the judged agent) is present,
            as this often contains the records and ground truth needed for scoring.

        If these conditions are met, it extracts necessary data (records, answers
        from `JudgeReasons`) using `extract_message_data` based on `self.inputs`
        mappings. It then constructs an `AgentInput` tailored for its own `_process`
        method (which will call the LLM for scoring).

        Finally, it invokes its own processing logic (via `self.__call__`, which
        wraps `self._process`) to perform the scoring. The resulting score
        (as an `ExecutionTrace` containing `QualResults`) is published using the
        `public_callback`.

        Args:
            message: The incoming message object. Expected to be an `ExecutionTrace`
                from another agent (e.g., a `Judge` agent).
            cancellation_token: An optional token for cancelling the operation.
            source: The identifier of the agent that sent the `message`.
            public_callback: An asynchronous callback function used to publish
                the scoring results (as an `ExecutionTrace`) back to the flow or UI.
            **kwargs: Additional keyword arguments.

        """
        # Validate the incoming message type and content
        if not isinstance(message, ExecutionTrace) or not isinstance(message.outputs, JudgeReasons) or not message.inputs:  # Ensure inputs exist
            logger.debug(
                "Scorer received message that is not a suitable ExecutionTrace with JudgeReasons and inputs. Skipping.",
                agent_id=self.agent_id,
                message_agent_id=message.agent_info.get("agent_id"),
            )
            return

        logger.debug(
            "Scorer received potential scoring target",
            scorer_agent_id=self.agent_id,
            from_agent_id=message.agent_info.get("agent_id"),
            call_id=message.call_id,
        )

        # Extract data based on `self.inputs` mappings.
        # These mappings should define how to get 'records', 'answers' (from JudgeReasons),
        # and 'criteria' (if the criteria template is dynamic).
        extracted_data = extract_message_data(
            message=message,  # The ExecutionTrace from the Judge
            source=message.agent_info.get("agent_id"),  # The Judge agent's ID/name
            input_mappings=self.inputs,  # Configured mappings for the Scorer
        )

        # `records` for scoring should come from the original input to the agent being judged.
        # `answers` for scoring are the `JudgeReasons` from the `message.outputs`.
        # `criteria` might be predefined in the scorer's prompt or passed dynamically.
        # The `scorer_agent_input` needs to be structured according to what the
        # scorer's prompt template expects.

        # Construct the AgentInput for this Scorer's _process method.
        # parent_call_id links this scoring trace back to the Judge's trace.
        scorer_agent_input = AgentInput(
            parent_call_id=message.call_id,  # Link to the Judge's trace
            inputs=extracted_data,  # Remaining extracted data (should include 'answers', 'criteria')
            # Context might not be needed if the scorer's prompt is self-contained with inputs.
        )

        # Invoke the scoring process using the LLM
        logger.debug(
            "Scorer scoring request",
            scorer_agent_name=self.agent_name,
            assessed_agent_id=message.agent_info.get("agent_id"),
            assessed_call_id=message.call_id,
        )
        await self.invoke(message=scorer_agent_input)

        # We don't publish here, because the invoke() method have already published the result.

    async def _process(
        self,
        *,
        message: AgentInput,  # Input for the Scorer LLM
        **kwargs: Any,
    ) -> AgentOutput | None:
        """Performs the LLM-based scoring and formats the output.

        This method overrides the base `LLMAgent._process`. It first calls
        `super()._process()` to get the raw scoring output from the LLM (which
        should conform to `QualScore` due to `output_model` setting).
        It then transforms this `QualScore` into a richer `QualResults` object
        by adding metadata about the assessed agent and call ID, extracted from
        the input `message` (which should contain details from the `Judge` agent's
        output, specifically the `answers` field).

        Args:
            message: The `AgentInput` for the scoring task. `message.inputs` is expected
                to contain information about the answer being assessed (e.g., under a
                key like "answers", often from `JudgeReasons.answers`) including
                `agent_id` and `answer_id` (which corresponds to a `call_id`).
            **kwargs: Additional keyword arguments for the LLM call.

        Returns:
            AgentOutput | None: An `AgentOutput` where the `outputs` field is populated
            with a `QualResults` object. If the LLM call fails or parsing is unsuccessful,
            the `outputs` might be an error structure or the raw LLM response.

        """
        # Call the base LLMAgent's _process to get the LLM's structured score (QualScore)
        llm_output_base = await super()._process(
            message=message,  # This message is the input for the Scorer's LLM
            **kwargs,
        )

        # Process the score for richer logging and output (QualResults)
        if llm_output_base and isinstance(llm_output_base.outputs, QualScore):
            qual_score_from_llm = llm_output_base.outputs

            # Extract details of the assessed answer from the input message.
            # This assumes message.inputs["answers"] is a list and we take the first.
            # The structure of "answers" depends on how it was mapped in _listen via input_mappings.
            assessed_answer_info = None
            if message.inputs and "answers" in message.inputs:
                answers_input = message.inputs["answers"]
                if isinstance(answers_input, list) and answers_input:
                    # Assuming the relevant answer info is the first item if it's a list
                    # This might need adjustment if 'answers' can have multiple items or different structure
                    assessed_answer_info = answers_input[0]
                elif isinstance(answers_input, dict):  # If 'answers' is a single dict
                    assessed_answer_info = answers_input

            if assessed_answer_info and isinstance(assessed_answer_info, dict):
                assessed_agent_id = assessed_answer_info.get("agent_id", "UnknownAgent")
                # 'answer_id' from JudgeReasons usually corresponds to the call_id of the trace being judged
                assessed_call_id = assessed_answer_info.get("answer_id", "UnknownCall")

                qual_results = QualResults(
                    critical_errors=qual_score_from_llm.critical_errors,
                    ground_truth_alignment=qual_score_from_llm.ground_truth_alignment,
                    confidence=qual_score_from_llm.confidence,
                    summary=qual_score_from_llm.summary,
                    assessed_agent_id=assessed_agent_id,
                    assessed_call_id=assessed_call_id,
                )
                # Replace the simpler QualScore in outputs with the richer QualResults
                llm_output_base.outputs = qual_results
                logger.debug(
                    "Scorer successfully processed score into QualResults",
                    scorer_agent_id=self.agent_id,
                    assessed_call_id=assessed_call_id,
                )
            else:
                logger.warning(
                    "Scorer could not extract assessed agent/call ID from message.inputs to create QualResults.",
                    scorer_agent_id=self.agent_id,
                    answers_data=message.inputs.get("answers"),
                )
                # llm_output_base.outputs remains QualScore in this case
        elif llm_output_base:
            logger.warning(
                "Scorer LLM output was not of type QualScore. Raw output will be returned.",
                scorer_agent_id=self.agent_id,
                actual_type=type(llm_output_base.outputs),
            )
        # If llm_output_base is None or an error, it will be returned as is.
        return llm_output_base
