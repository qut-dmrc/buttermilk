"""Defines the Judge agent and associated Pydantic models for structured evaluation.

This module provides the `Judge` agent, an LLM-based agent specialized for
evaluating content (e.g., text, other agent outputs) against a predefined set of
criteria or a policy. It uses Pydantic models like `Reasons` and `JudgeReasons`
to structure the LLM's output, ensuring a consistent and parsable evaluation format.
"""

import random  # For random emoji selection in preview
from typing import Literal  # For type hinting

from pydantic import BaseModel, Field, computed_field  # Pydantic components

# Buttermilk core imports
from buttermilk._core.agent import AgentInput, AgentTrace  # Base types
from buttermilk._core.log import logger  # Centralized logger
from buttermilk.agents.llm import LLMAgent  # Base class for LLM-powered agents

# --- Pydantic Models for Evaluation Output ---


class Reasons(BaseModel):
    """A base model for structuring conclusions and supporting reasons.

    This model provides a common structure for outputs that include a main
    conclusion and a list of reasons that support or lead to that conclusion.

    Attributes:
        conclusion (str): The main conclusion, final answer, or summary statement.
            Should outline any uncertainty if applicable.
        reasons (list[str]): A list of strings, where each string represents a
            single step, justification, or piece of evidence in the logical
            reasoning process that supports the `conclusion`.

    """

    conclusion: str = Field(
        ...,
        description="The overall conclusion, final answer, or summary, outlining any uncertainty.",
    )
    reasons: list[str] = Field(
        ...,
        description="A list of strings, each representing a distinct step or justification in the reasoning process.",
    )

    def __str__(self) -> str:
        """Returns a Markdown formatted string representation of the reasons.

        Returns:
            str: A string with the conclusion followed by a list of reasons,
                 formatted for Markdown display.

        """
        reasons_str = "\n\n\t".join(f"- {reason}" for reason in self.reasons)
        return (
            f"**Conclusion:** {self.conclusion}\n\n"
            f"**Reasoning Steps:**\n\t{reasons_str or 'No specific reasons provided.'}"
        )


class JudgeReasons(Reasons):
    """Structured output model for the `Judge` agent's evaluation.

    This model defines the expected JSON structure that the `Judge` agent's LLM
    should return after evaluating content against provided criteria or a policy.
    It extends `Reasons` to include a specific prediction (e.g., policy violation)
    and a level of uncertainty.

    Attributes:
        prediction (bool): A boolean flag indicating the outcome of the judgment.
            For example, `True` if the content violates a policy or meets a
            negative criterion, `False` otherwise. This should be logically
            derived from the reasoning and criteria application.
        reasons (list[str]): Overrides the description from `Reasons`. A list of strings,
            where each string represents a distinct step in the reasoning process
            leading to the `conclusion` and `prediction` regarding the
            policy/guidelines or criteria.
        uncertainty (Literal["high", "medium", "low"]): An assessment of the
            uncertainty or confidence in the prediction and conclusion.
            "high" uncertainty means there's significant room for reasonable
            minds to differ.
        preview (str): A computed property that returns a short, emoji-enhanced
            preview string of the evaluation, summarizing the conclusion,
            prediction (with an icon), and uncertainty level.

    """

    # `conclusion` is inherited from the `Reasons` base model.
    reasons: list[str] = Field(
        ...,
        description="A list of strings, where each string represents a distinct step in the reasoning process leading to the conclusion and prediction regarding policy/guidelines.",
    )
    prediction: bool = Field(
        ...,
        description="Boolean flag indicating the judgment (e.g., True if content violates policy/guidelines, False otherwise). This should be logically derived from the reasoning.",
    )
    uncertainty: Literal["high", "medium", "low"] = Field(
        ...,
        description="Assesses the scope for reasonable minds to differ on this conclusion (high, medium, or low uncertainty).",
    )

    def __str__(self) -> str:
        """Returns a Markdown formatted string representation of the full judgment.

        Includes the conclusion, policy violation prediction, uncertainty level,
        and detailed reasoning steps.

        Returns:
            str: A comprehensive Markdown formatted summary of the judgment.

        """
        reasons_str = "\n".join(f"\t- {reason}" for reason in self.reasons)
        return (
            f"**Conclusion:** {self.conclusion}\n"
            f"**Prediction (e.g., Violates Policy):** {'Yes' if self.prediction else 'No'}\n"
            f"**Uncertainty Level:** {self.uncertainty.capitalize()}\n\n"
            f"**Detailed Reasoning:**\n{reasons_str or 'No specific reasons provided.'}"
        )


# --- Judge Agent ---
class Judge(LLMAgent):
    """An LLM-based agent specialized in evaluating content against predefined criteria or policies.

    The `Judge` agent inherits from `LLMAgent`, utilizing its capabilities for
    interacting with Language Models, managing prompt templates, and handling
    structured output. It is specifically configured to perform evaluative tasks.
    The core of its operation involves:
    1.  Receiving content to be judged (via an `AgentInput` message).
    2.  Using a configured LLM and a specialized prompt template (which should
        instruct the LLM on the criteria and desired output format).
    3.  Expecting the LLM to return a structured response that conforms to the
        `JudgeReasons` Pydantic model. This model captures the judgment (a boolean
        `prediction`), the supporting `reasons`, the overall `conclusion`, and
        the level of `uncertainty`.

    The `buttermilk_handler` decorator (if used and implemented in the framework)
    would typically register methods like `evaluate_content` to handle specific
    message types within a Buttermilk or Autogen-based multi-agent system.

    Key Configuration Parameters (from `AgentConfig.parameters`):
        - `model` (str): **Required**. The name of the LLM to use for judgment.
        - `prompt_template` (str): **Required**. The name of the prompt template
          that guides the LLM to perform the evaluation and output `JudgeReasons`.

    Attributes:
        output_model (Type[BaseModel] | None): Specifies `JudgeReasons` as the
            expected Pydantic model for the LLM's structured output. This is
            used by the base `LLMAgent`'s `_process` method to automatically
            attempt parsing of the LLM's JSON output.

    """

    def __init__(self, **kwargs):
        """Initializes the Judge agent with its specific configuration and output model."""
        super().__init__(**kwargs)
        # Set the expected output model for the LLM's response
        self.output_model = JudgeReasons
