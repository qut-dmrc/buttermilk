"""Defines agents and Pydantic models for evaluating and scoring LLM outputs.

This module provides the `LLMScorer` agent, which uses a Language Model to
qualitatively assess the output of another agent based on predefined criteria
and, potentially, ground truth information. It also defines several Pydantic
models (`QualScoreCRA`, `QualScore`, `QualResults`) to structure the scoring
criteria, individual assessments, and the overall scoring output.
"""

from collections.abc import (
    Awaitable,  # For type hinting and class types
    Callable,  # For typing callables
)
from typing import Any

from autogen_core import CancellationToken  # Autogen cancellation token
from pydantic import BaseModel, Field, computed_field  # Pydantic components

# Buttermilk core imports
from buttermilk import logger  # Centralized logger
from buttermilk._core.contract import (  # Buttermilk message contracts
    AgentInput,
    AgentOutput,  # Used as return type hint for _process
    AgentTrace,
    GroupchatMessageTypes,
)
from buttermilk._core.message_data import extract_message_data  # Utility for data extraction
from buttermilk.agents.judge import JudgeReasons  # Expected input model from Judge agent
from buttermilk.agents.llm import LLMAgent  # Base class for LLM-powered agents

# --- Pydantic Models for Scoring ---


class QualScoreCRA(BaseModel):
    """Represents a single Criterion-Referenced Assessment (CRA).

    This model captures the evaluation against one specific criterion, including
    whether the content met the criterion and textual feedback explaining the assessment.

    Attributes:
        correct (bool): Indicates whether the content meets this specific criterion.
            `True` if met, `False` otherwise.
        feedback (str): A concise explanation (e.g., one sentence) for the
            assessment against this criterion, detailing why it was judged as
            correct or incorrect.

    """

    correct: bool = Field(..., description="Does the content meet this specific criterion? (True/False)")
    feedback: str = Field(..., description="Concise explanation for the assessment against this criterion.")


class QualScore(BaseModel):
    """Represents a qualitative score of an LLM's output or any text.

    This model aggregates multiple Criterion-Referenced Assessments (`QualScoreCRA`)
    to form an overall qualitative evaluation. It's typically the direct output
    structure expected from an LLM tasked with scoring.

    Attributes:
        assessments (list[QualScoreCRA]): A list of `QualScoreCRA` objects,
            each representing an assessment against a specific criterion.

    """

    assessments: list[QualScoreCRA] = Field(
        ...,
        description="A list of individual assessments, one for each criterion evaluated.",
    )


class QualResults(QualScore):
    """Extends `QualScore` to include metadata about the answer/output being assessed.

    This model is designed for presenting or logging scoring results externally,
    linking the qualitative assessments back to the specific agent output that
    was evaluated. It inherits the `assessments` field from `QualScore`.

    Attributes:
        assessed_agent_id (str): The unique identifier of the agent whose output
            was assessed.
        assessed_call_id (str): A unique identifier for the specific answer, call,
            or output instance that was assessed. This helps in pinpointing the
            exact piece of work evaluated.
        correctness (float | None): A computed property that calculates the overall
            correctness score as a simple average of the `correct` fields from
            all assessments (fraction of criteria marked as correct). Returns
            `None` if there are no assessments. This is an unweighted average.
        score_text (str): A property that provides a human-readable string
            representation of the `correctness` score (e.g., "75%").

    """

    assessed_agent_id: str = Field(..., description="The ID of the agent whose output was assessed.")
    assessed_call_id: str = Field(..., description="A unique identifier for the specific answer/output being assessed.")
    # 'assessments' is inherited from QualScore. The description is repeated for clarity if needed:
    # assessments: list[QualScoreCRA] = Field(..., description="A list of assessments, one for each criterion evaluated.")

    @computed_field
    @property
    def correctness(self) -> float | None:
        """Calculates the overall correctness score as a simple average.

        The score is the fraction of criteria marked as 'correct'.
        This is an unweighted average.

        Returns:
            float | None: The correctness score (0.0 to 1.0), or `None` if
            there are no assessments or if an error occurs during calculation.

        """
        if not self.assessments:  # Avoid division by zero if list is empty
            return None
        try:
            return sum(cra.correct for cra in self.assessments) / len(self.assessments)
        except Exception as e:  # Catch any potential errors during calculation
            logger.error(f"Error calculating correctness for QualResults (call_id: {self.assessed_call_id}): {e!s}")
            return None

    @property
    def score_text(self) -> str:
        """Provides a human-readable string representation of the numerical correctness score.

        Formats the `correctness` score as a percentage (e.g., "75%").

        Returns:
            str: The score as a percentage string, or "N/A" if `correctness` is None.

        """
        corr = self.correctness  # Calculate once
        if corr is not None:
            return f"{int(corr * 100)}%"
        return "N/A"

    def __str__(self) -> str:
        """Provides a human-readable Markdown representation of the qualitative scoring results.

        Includes the assessed answer's call ID, the overall numerical score,
        and a list of individual criterion assessments with feedback.

        Returns:
            str: A Markdown formatted string summarizing the scoring results.

        """
        score_val = self.correctness
        score_str = f"{score_val:.2f}" if score_val is not None else "N/A"

        assessment_lines = [
            f"**{'✔️ Correct' if cra.correct else '✘ Incorrect'}**: {cra.feedback}"
            for cra in self.assessments
        ]
        return (
            f"**Assessed Answer Call ID**: {self.assessed_call_id}\n"
            f"**Overall Score**: {score_str} ({self.score_text})\n\n"
            f"**Criterion Assessments**:\n\t- " + "\n\t- ".join(assessment_lines)
        )


# --- LLM Scorer Agent ---
class LLMScorer(LLMAgent):
    """An LLM-based agent that qualitatively scores another agent's output.

    This agent listens for `AgentTrace` messages, particularly those containing
    `JudgeReasons` in their outputs (typically from a `Judge` agent). When such a
    message is detected and relevant ground truth information is available (either
    attached to the original record or inferred from the context), the `LLMScorer`
    triggers its own Language Model to perform an evaluation.

    The LLM is guided by a scoring-specific prompt template (configured via
    `AgentConfig.parameters.prompt_template`) to produce a structured score
    conforming to the `QualScore` Pydantic model. This structured score is then
    wrapped in a `QualResults` model, adding metadata about the assessed item,
    and included in the `AgentTrace` produced by this scorer.

    The agent can integrate with Weave for logging scores against the trace of
    the original agent's output that was scored.

    Key Configuration Parameters (from `AgentConfig.parameters`):
        - `model` (str): **Required**. The LLM to use for scoring.
        - `prompt_template` (str): **Required**. Template guiding the LLM to score.
          The output of this template should be parsable into `QualScore`.

    Attributes:
        _output_model (Type[BaseModel] | None): Specifies `QualScore` as the
            expected Pydantic model for the LLM's structured output.

    """

    _output_model: type[BaseModel] | None = QualScore  # Expected Pydantic model for LLM output

    async def _listen(
        self,
        message: GroupchatMessageTypes,
        *,
        cancellation_token: CancellationToken | None = None,
        source: str = "",
        public_callback: Callable[[Any], Awaitable[None]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Listens for relevant `AgentTrace` messages and triggers the scoring process.

        This method is invoked when the `LLMScorer` passively receives a message
        in a group chat or similar context. It performs the following checks:
        1.  Verifies if the incoming `message` is an `AgentTrace`.
        2.  Checks if `message.outputs` is an instance of `JudgeReasons` (indicating
            it's likely a structured reasoning output from another evaluation agent like `Judge`).
        3.  Ensures `message.inputs` (the original input to the judged agent) is present,
            as this often contains the records and ground truth needed for scoring.

        If these conditions are met, it extracts necessary data (records, answers
        from `JudgeReasons`) using `extract_message_data` based on `self.inputs`
        mappings. It then constructs an `AgentInput` tailored for its own `_process`
        method (which will call the LLM for scoring).

        Finally, it invokes its own processing logic (via `self.invoke`, which
        wraps `self._process`) to perform the scoring. The resulting score
        (as an `AgentTrace` containing `QualResults`) is published using the
        `public_callback`.

        Args:
            message: The incoming message object. Expected to be an `AgentTrace`
                from another agent (e.g., a `Judge` agent).
            cancellation_token: An optional token for cancelling the operation.
            source: The identifier of the agent that sent the `message`.
            public_callback: An asynchronous callback function used to publish
                the scoring results (as an `AgentTrace`) back to the flow or UI.
            **kwargs: Additional keyword arguments.

        """
        # Validate the incoming message type and content
        if not isinstance(message, AgentTrace) or \
           not hasattr(message, "outputs") or \
           not isinstance(message.outputs, JudgeReasons) or \
           not hasattr(message, "inputs") or not message.inputs:  # Ensure inputs exist
            logger.debug(
                f"Scorer '{self.agent_name}' received message from agent '{source}' that "
                "is not a suitable AgentTrace with JudgeReasons and inputs. Skipping.",
            )
            return

        logger.debug(f"Scorer '{self.agent_name}' received potential scoring target from agent '{source}' (Call ID: {message.call_id}).")

        # Extract data based on `self.inputs` mappings.
        # These mappings should define how to get 'records', 'answers' (from JudgeReasons),
        # and 'criteria' (if criteria are dynamic or passed through).
        # Example mapping for 'answers': "SourceAgentName.outputs" (if source is Judge's name)
        # Example mapping for 'records': "SourceAgentName.inputs.records"
        extracted_data = extract_message_data(
            message=message,  # The AgentTrace from the Judge
            source=source,   # The Judge agent's ID/name
            input_mappings=self.inputs,  # Configured mappings for the Scorer
        )

        # Ignore messages that don't have ground truth in the input record
        record = extracted_data.pop("records", [])
        if not record or not isinstance(record, list) or not record[0] or "ground_truth" not in record[0]:
            logger.debug(f"Scorer {self.agent_name} received message from agent {source} without ground truth.")
            return

        # `records` for scoring should come from the original input to the agent being judged.
        # `answers` for scoring are the `JudgeReasons` from the `message.outputs`.
        # `criteria` might be predefined in the scorer's prompt or passed dynamically.
        # The `scorer_agent_input` needs to be structured according to what the
        # scorer's prompt template expects.

        # Ensure 'records' and 'answers' are present, as they are crucial for scoring.
        # 'answers' would typically be mapped from message.outputs (the JudgeReasons).
        # 'records' would typically be mapped from message.inputs.records (original records judged).
        # Extract the first record
        record = record[0]

        # Create an AgentInput with minimal state
        scorer_agent_input = AgentInput(parent_call_id=message.call_id, records=[record], inputs=extracted_data)

        # Construct the AgentInput for this Scorer's _process method.
        # parent_call_id links this scoring trace back to the Judge's trace.
        scorer_agent_input = AgentInput(
            parent_call_id=message.call_id,  # Link to the Judge's trace
            records=extracted_data.pop("records", []),  # Original records that were judged
            inputs=extracted_data,  # Remaining extracted data (should include 'answers', 'criteria')
            # Context might not be needed if the scorer's prompt is self-contained with inputs.
        )

        logger.debug(f"Scorer '{self.agent_name}' prepared AgentInput for scoring (Parent Call ID: {message.call_id}).")

        # Invoke this Scorer's own processing logic.
        # The public_callback will be used by `invoke` to publish the resulting AgentTrace.
        if public_callback:
            await self.invoke(  # self.invoke calls self._process internally
                message=scorer_agent_input,
                public_callback=public_callback,
                cancellation_token=cancellation_token,
                **kwargs,
            )
        else:
            logger.warning(f"Scorer '{self.agent_name}': No public_callback provided; scoring result will not be published.")

    async def _process(
        self,
        message: AgentInput,  # Input for the Scorer LLM
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> AgentOutput:  # Returns AgentOutput, which AgentTrace inherits from
        """Performs the LLM-based scoring and formats the output.

        This method overrides the base `LLMAgent._process`. It first calls
        `super()._process()` to get the raw scoring output from the LLM (which
        should conform to `QualScore` due to `_output_model` setting).
        It then transforms this `QualScore` into a richer `QualResults` object
        by adding metadata about the assessed agent and call ID, extracted from
        the input `message` (which should contain details from the `Judge` agent's
        output, specifically the `answers` field).

        Args:
            message: The `AgentInput` for the scoring task. `message.inputs` is expected
                to contain information about the answer being assessed (e.g., under a
                key like "answers", often from `JudgeReasons.answers`) including
                `agent_id` and `answer_id` (which corresponds to a `call_id`).
            cancellation_token: An optional token for cancelling the LLM call.
            **kwargs: Additional keyword arguments for the LLM call.

        Returns:
            AgentOutput: An `AgentOutput` (typically an `AgentTrace` instance via
            `self.invoke`) where the `outputs` field is populated with a
            `QualResults` object. If the LLM call fails or parsing is unsuccessful,
            the `outputs` might be an error structure or the raw LLM response.

        """
        # Call the base LLMAgent's _process to get the LLM's structured score (QualScore)
        llm_output_base = await super()._process(
            message=message,  # This message is the input for the Scorer's LLM
            cancellation_token=cancellation_token,
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
                    assessments=qual_score_from_llm.assessments,
                    assessed_agent_id=assessed_agent_id,
                    assessed_call_id=assessed_call_id,
                )
                # Replace the simpler QualScore in outputs with the richer QualResults
                llm_output_base.outputs = qual_results
                logger.debug(f"Scorer '{self.agent_id}' successfully processed score into QualResults for assessed call ID '{assessed_call_id}'.")
            else:
                logger.warning(
                    f"Scorer '{self.agent_id}': Could not extract assessed agent/call ID from "
                    f"message.inputs['answers'] to create QualResults. Input answers data: {message.inputs.get('answers')}. "
                    "LLM output (QualScore) will be returned directly in AgentOutput.",
                )
                # llm_output_base.outputs remains QualScore in this case
        elif llm_output_base:
             logger.warning(
                f"Scorer '{self.agent_id}': LLM output was not of type QualScore. Actual type: {type(llm_output_base.outputs)}. "
                "Raw output will be returned.",
            )
        # If llm_output_base is None or an error, it will be returned as is.
        return llm_output_base
