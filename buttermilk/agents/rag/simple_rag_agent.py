"""Simplified RAG agent that uses composition over inheritance.

This module provides a clean RAG agent that:
- Uses external search tools instead of embedded logic
- Guarantees structured outputs with citations
- Relies on templates for orchestration
"""


from typing import Any

from pydantic import BaseModel, Field

from buttermilk.agents.llm import LLMAgent


class Reference(BaseModel):
    """Represents a single cited reference within a research result."""

    summary: str = Field(..., description="Summary of the key information from this reference.")
    citation: str = Field(..., description="Source identifier for the reference.")


class ResearchResult(BaseModel):
    """Structured output of a RAG process with citations."""

    literature: list[Reference] = Field(
        ...,
        description="List of literature references used in generating the response.",
    )
    response: str = Field(
        ...,
        description="The synthesized textual response (markdown permitted).",
    )
    summary: str = Field(
        ...,
        description="A brief summary (plain text)",
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
    3. Iterating up to max_tool_iterations
    4. Synthesizing the response
    5. Returning in ResearchResult format
    """

    def __init__(self, *, output_model: type[BaseModel] = None, **kwargs: Any) -> None:
        """Initialize RagAgent with template configuration."""
        if output_model is None:
            output_model = ResearchResult
        super().__init__(output_model=output_model, **kwargs)

    def get_tool_definitions(self) -> list[Tool]:
        """Generate structured tool definitions for this agent."""
        internal_tools = [
            FunctionTool(
                name="fetch_uri",
                description=("Get a record from a given URI."),
                func=self.fetch_uri,
                strict=True,
            ),
        ]

        # Create dataset-specific fetch_record tools using partial
        dataset_tools = [
            FunctionTool(
                name=f"fetch_record_from_{dataset_name}",
                description=f"Get a record from the {dataset_name} dataset by record ID.",
                func=partial(self.fetch_record, dataset_name=dataset_name),
                strict=True,
            )
            for dataset_name in self._data_sources.keys()
        ]

        return internal_tools + dataset_tools
