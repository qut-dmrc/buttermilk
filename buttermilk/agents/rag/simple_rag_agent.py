"""Simplified RAG agent that uses composition over inheritance.

This module provides a clean RAG agent that:
- Uses external search tools instead of embedded logic
- Guarantees structured outputs with citations
- Relies on templates for orchestration
"""


from pydantic import BaseModel, Field

from buttermilk.agents.llm import LLMAgent


class Reference(BaseModel):
    """Represents a single cited reference within a research result."""

    summary: str = Field(..., description="Summary of the key information from this reference.")
    source: str = Field(..., description="Source identifier or title for the reference.")


class ResearchResult(BaseModel):
    """Structured output of a RAG process with citations."""

    literature: list[Reference] = Field(
        ...,
        description="List of literature references used in generating the response.",
    )
    response: str = Field(
        ...,
        description="The synthesized textual response.",
    )


class RagAgent(LLMAgent):
    """Base RAG agent ensuring structured outputs with citations.
    
    This simplified agent:
    - Extends LLMAgent to inherit all tool usage capabilities
    - Forces ResearchResult as the output format
    - Uses external search tools configured in YAML
    - Relies on Jinja2 templates for search orchestration
    
    The template is responsible for:
    1. Calling search tools
    2. Extracting citations from results
    3. Synthesizing the response
    4. Returning in ResearchResult format
    """

    def __init__(self, **kwargs):
        """Initialize RagAgent with template configuration."""
        super().__init__(**kwargs)
        
        # Force structured output - can be overridden by subclasses
        self._output_model: type[BaseModel] | None = ResearchResult
        
        # Template configuration - moved from Field declaration
        self.template: str = kwargs.get("template", "rag")
