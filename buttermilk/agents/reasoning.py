"""
Defines output formats for individual components of reasoning agents.
"""

from typing import Literal, Optional, Type

# Import Autogen core components needed for type hints and potential interaction (though handled by adapter)
from pydantic import BaseModel, Field

# Buttermilk core imports
from buttermilk.agents.llm import LLMAgent  # Base class for LLM-powered agents

class Reasons(BaseModel):
    reasons: list[str] = Field(..., description="Each element should represent a single step in logical reasoning.")

    def __str__(self):
        """Returns a nicely formatted string representation of the evaluation."""
        reasons_str = "\n\n".join(f"- {reason}" for reason in self.reasons)
        return (
            f"**Reasoning:**\n\n{reasons_str}"
        )


class Differentiator(LLMAgent):
    """Analysis of differences and similarities between answers."""
    _output_model: Optional[Type[BaseModel]] = Reasons
