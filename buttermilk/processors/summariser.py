from pydantic import BaseModel, Field

from buttermilk import LLMCore


class Summary(BaseModel):
    """A base model for structuring case summaries.

    This model provides a common structure for outputs that include various
    components of a case summary, such as content summary, procedural history,
    relevant policies, reasoning analysis, and final summary.

    Attributes:
        content (str): A summary of the content and factual context of the case.
        procedural_history (str): A summary of the procedural history of the case.
        content_policies (list[str]): A list of relevant content policies quoted verbatim.
        human_rights_standards (list[str]): A list of relevant human rights standards quoted verbatim.
        recommendations (list[str]): A list of the Board's policy recommendations, if applicable.
        reasoning (list[str]): A list of reasoning steps analyzing contentious issues in the case.
        ratio (list[str]): A list of major principles established by the case.
        summary (str): A summary of the key findings, rationale, and outcome of the case.

    """

    facts: str = Field(
        ...,
        description="A summary of the content and factual context of the case.",
    )
    procedural_history: str = Field(
        ...,
        description="A summary of the procedural history of the case.",
    )
    content_policies: list[str] = Field(
        ...,
        description="A list of relevant content policies quoted verbatim.",
    )
    human_rights_standards: list[str] = Field(
        ...,
        description="A list of relevant human rights standards quoted verbatim.",
    )
    recommendations: list[str] = Field(
        ...,
        description="A list of the Board's policy recommendations, if applicable.",
    )
    reasoning: list[str] = Field(
        ...,
        description="A list of reasoning steps analyzing contentious issues in the case.",
    )
    ratio: list[str] = Field(
        ...,
        description="A list of major principles established by the case.",
    )
    summary: str = Field(
        ...,
        description="A summary of the key findings, rationale, and outcome of the case.",
    )

    def as_markdown(self, agent_id: str | None = None, call_id: str | None = None) -> str:
        """Returns a Markdown formatted string for insertion into templates.

        Format follows the standard: agent identifier on first line, followed by
        content-specific fields without empty lines between components.

        Args:
            agent_id: The agent identifier (e.g., "JUDGE-gpt4")
            call_id: The call identifier for this execution

        Returns:
            str: Formatted markdown string suitable for template insertion
        """
        header = ""
        if agent_id and call_id:
            # Use only the last 8 characters of call_id for brevity
            short_call_id = call_id[-8:] if len(call_id) > 8 else call_id
            header = f"**{agent_id} #{short_call_id}**\n"

        content_policies_str = "\n".join(f"- {policy}" for policy in self.content_policies)
        human_rights_standards_str = "\n".join(f"- {standard}" for standard in self.human_rights_standards)
        recommendations_str = "\n".join(f"- {rec}" for rec in self.recommendations)
        reasoning_str = "\n".join(f"- {reason}" for reason in self.reasoning)
        ratio_str = "\n".join(f"- {principle}" for principle in self.ratio)

        formatted_string = (
            f"{header}"
            f"**Facts:**\n{self.facts}\n\n"
            f"**Procedural History:**\n{self.procedural_history}\n\n"
            f"**Content Policies:**\n{content_policies_str or 'None provided.'}\n\n"
            f"**Human Rights Standards:**\n{human_rights_standards_str or 'None provided.'}\n\n"
            f"**Recommendations:**\n{recommendations_str or 'None provided.'}\n\n"
            f"**Reasoning Analysis:**\n{reasoning_str or 'None provided.'}\n\n"
            f"**Ratio (Major Principles):**\n{ratio_str or 'None provided.'}\n\n"
            f"**Summary of Findings and Outcome:**\n{self.summary}"
        )

        return formatted_string

    def __str__(self) -> str:
        """Returns a Markdown formatted string representation.

        When agent context is available (via _agent_id and _call_id attributes),
        includes the full header. Otherwise returns a simpler format.
        """
        # Check if agent context is available (set by ExecutionTrace)
        agent_id = getattr(self, "_agent_id", None)
        call_id = getattr(self, "_call_id", None)

        return self.as_markdown(agent_id, call_id)


class Summariser(LLMCore):
    """An LLMCore agent specialized in summarizing content using structured reasoning."""

    def __init__(self, **kwargs):
        """Initializes the agent with its specific configuration and output model."""
        output_model = kwargs.pop("output_model", Summary)

        parameters = kwargs.pop("parameters", {})
        parameters["template"] = parameters.pop("template", "summarise_case")

        super().__init__(output_model=output_model, parameters=parameters, **kwargs)
