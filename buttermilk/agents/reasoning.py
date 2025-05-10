"""Defines output formats for individual components of reasoning agents.
"""


# Import Autogen core components needed for type hints and potential interaction (though handled by adapter)
from pydantic import BaseModel, Field

# Buttermilk core imports
from buttermilk.agents.llm import LLMAgent  # Base class for LLM-powered agents


class Expert(BaseModel):
    """Expert class representing an expert's name and answer."""

    name: str
    answer_id: str


class Divergence(BaseModel):
    topic: str = Field(..., description="The key topic, point, or fact at hand.")
    positions: list[dict[list[Expert], str]] = Field(..., description="Summary of positions held by each group of experts (only include material differences).")


class Differences(BaseModel):
    conclusion: str = Field(..., description="Your conclusion or final answer summarizing current tentative conclusions and outlining any uncertainty or divergence.")
    divergences: list[Divergence] = Field(..., description="A list of divergences, where each divergence represents a disagreement or difference in opinion among experts.")

    def __str__(self):
        """Returns a nicely formatted MarkDown representation of the evaluation."""
        divergences_str = "\n\n".join(
            f"### Divergence: {divergence.topic}\n"
            + "\n".join(
                f"\t- {position} ({', '.join([f'{expert.name} ({expert.answer_id})' for expert in group])})"
                for group, position in divergence.positions
            )
            for divergence in self.divergences
        )
        return (
            f"## Conclusion:\n {self.conclusion}\n{divergences_str}"
        )


class Differentiator(LLMAgent):
    """Analysis of differences and similarities between answers."""

    _output_model: type[BaseModel] | None = Differences
