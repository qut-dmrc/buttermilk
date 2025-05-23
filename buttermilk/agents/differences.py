"""Defines Pydantic models for structuring the analysis of differences between texts.

This module provides a set of Pydantic models (`Expert`, `Position`, `Divergence`,
`Differences`, `DifferencesOutput`) used to represent and format the output of
agents that analyze and identify divergences or disagreements between multiple
textual inputs (e.g., answers from different experts or LLMs).

The `Differentiator` agent, also defined here, leverages these models to structure
its output when performing such difference analysis.
"""

from typing import Type  # For type hinting a class type

from pydantic import BaseModel, Field  # Pydantic components for data validation

# Buttermilk core imports
from buttermilk.agents.llm import LLMAgent  # Base class for LLM-powered agents


class Expert(BaseModel):
    """Represents an expert or source providing a specific answer or viewpoint.

    Attributes:
        name (str): The name or identifier of the expert or source.
        answer_id (str): A unique identifier for the specific answer or statement
            provided by this expert. This can be used to link back to the
            original input text.
    """

    name: str = Field(..., description="Name or identifier of the expert or source.")
    answer_id: str = Field(..., description="Unique identifier for the expert's specific answer or statement.")


class Position(BaseModel):
    """Represents a distinct position or viewpoint on a topic, potentially held by multiple experts.

    Attributes:
        experts (list[Expert]): A list of `Expert` objects who hold or support
            this particular position.
        position (str): A concise summary or statement of the position itself.
    """

    experts: list[Expert] = Field(..., description="A list of experts who hold or support this position.")
    position: str = Field(..., description="A concise summary or statement of the position.")


class Divergence(BaseModel):
    """Describes a specific point of divergence or disagreement on a particular topic.

    It outlines a topic and the different positions taken by experts regarding that topic.

    Attributes:
        topic (str): The key topic, point, fact, or aspect where differences
            in opinion or statement have been identified.
        positions (list[Position]): A list of `Position` objects, where each
            represents a distinct stance or viewpoint on the `topic`. This list
            should ideally only include materially different positions.
    """

    topic: str = Field(..., description="The key topic, point, or fact where differences are noted.")
    positions: list[Position] = Field(..., description="A list of distinct positions held by experts on this topic.")


class Differences(BaseModel):
    """Represents a structured summary of differences and divergences found in a set of texts.

    This model is typically used as the structured output format for an LLM tasked
    with analyzing multiple inputs (e.g., expert opinions, different model responses)
    to identify and summarize their key differences.

    Attributes:
        conclusion (str): A summary conclusion that synthesizes the overall findings,
            highlighting current tentative agreements and outlining any remaining
            uncertainty or significant divergences.
        divergences (list[Divergence]): A list of `Divergence` objects, where
            each object details a specific topic of disagreement and the
            various positions taken on it.
    """

    conclusion: str = Field(
        ...,
        description="A summary conclusion outlining overall findings, agreements, and unresolved divergences."
    )
    divergences: list[Divergence] = Field(
        ...,
        description="A list of specific topics where divergences or disagreements were identified."
    )


class DifferencesOutput(Differences):
    """Extends `Differences` to provide a formatted string representation of the analysis.

    This class inherits all fields from `Differences` and adds a `__str__` method
    to generate a human-readable (Markdown formatted) summary of the conclusion
    and the identified divergences.
    """

    def __str__(self) -> str:
        """Returns a Markdown formatted string representation of the differences analysis.

        The output includes the main conclusion followed by sections for each
        identified divergence, listing the topic and the different positions
        held by experts on that topic.

        Returns:
            str: A Markdown formatted string summarizing the analysis.
        """
        divergences_str_parts: list[str] = []
        for divergence_item in self.divergences:
            positions_str = "\n".join(
                # Ensure position.position and expert.name are used for clarity
                f"\t- Position: \"{pos.position}\" (Held by: {', '.join([exp.name for exp in pos.experts])})"
                for pos in divergence_item.positions
            )
            divergences_str_parts.append(
                f"### Divergence on Topic: {divergence_item.topic}\n{positions_str}"
            )

        final_divergences_str = "\n\n".join(divergences_str_parts)

        return (
            f"## Overall Conclusion:\n{self.conclusion}\n\n"
            f"## Detailed Divergences:\n{final_divergences_str if final_divergences_str else 'No specific divergences listed.'}"
        )


class Differentiator(LLMAgent):
    """An LLM-based agent that analyzes multiple text inputs to identify and summarize their differences and similarities.

    The `Differentiator` agent is designed to take a collection of texts (e.g.,
    answers from different experts, responses from various LLMs to the same prompt)
    and produce a structured analysis highlighting the key points of agreement,
    disagreement, and overall conclusions.

    It uses a Language Model, configured via its `LLMAgent` base class, and expects
    the LLM to generate output conforming to the `Differences` Pydantic model.
    This structured output is then available in the `AgentTrace.outputs`.

    Key Configuration Parameters (inherited from `LLMAgent` and used here):
        - `model` (str): **Required**. The name of the LLM to use for the analysis.
        - `prompt_template` (str): **Required**. The name of the prompt template that
          guides the LLM to perform the difference analysis and structure its
          output according to the `Differences` model.

    Input:
        Expects an `AgentInput` where `message.inputs` (or `message.records`)
        contains the texts to be compared. The specific format of these inputs
        (e.g., a list of strings, a dictionary mapping expert names to texts)
        should align with what the configured prompt template expects.

    Output:
        Produces an `AgentTrace` where `agent_trace.outputs` is an instance of
        `Differences` (or `DifferencesOutput`), providing a structured breakdown
        of the analysis.

    Attributes:
        _output_model (Type[BaseModel] | None): Specifies that this agent expects
            its LLM output to be parsable into the `Differences` model.
            This is used by the `LLMAgent` base class to automatically attempt
            parsing the LLM's JSON output into this Pydantic model.
    """

    _output_model: Type[BaseModel] | None = Differences
    # This tells the LLMAgent base class to attempt to parse the LLM's output
    # (if JSON) into the `Differences` model.
