"""Defines output formats for individual components of reasoning agents.
"""


# Import Autogen core components needed for type hints and potential interaction (though handled by adapter)
from pydantic import BaseModel, Field

# Buttermilk core imports
from buttermilk.agents.llm import LLMAgent  # Base class for LLM-powered agents

class Agreement(BaseModel):
    topic: str = Field(..., description="The key topic, point, or fact at hand.")
    divergence: str = Field(..., description="The specific divergence or difference in opinion.")
    agreement: str = Field(..., description="The specific agreement or common ground.")

    

class Reasons(BaseModel):
    conclusion: str = Field(..., description="Your conclusion or final answer summarizing current tentative conclusions and outlining any uncertainty or divergence.")
    reasons: list[str] = Field(..., description="Each element should represent a single step in logical reasoning.")

    def __str__(self):
        """Returns a nicely formatted MarkDown representation of the evaluation."""
        reasons_str = "\n\n\t".join(f"- {reason}" for reason in self.reasons)
        return (
            f"**Reasoning:**\n\n{reasons_str}"
        )


class Differentiator(LLMAgent):
    """Analysis of differences and similarities between answers."""

    _output_model: type[BaseModel] | None = Reasons
