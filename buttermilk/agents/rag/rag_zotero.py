"""Zotero-specific RAG agent with academic citation formatting.

This module provides a Zotero-focused RAG agent that inherits from the
simplified RagAgent base class and adds only Zotero-specific output formatting.
"""

from typing import Any

import pydantic
from autogen_core.tools import Tool
from pydantic import Field

from buttermilk._core.tool_definition import AgentToolDefinition
from buttermilk.agents.rag.simple_rag_agent import RagAgent, Reference, ResearchResult


class ZoteroReference(Reference):
    """Zotero literature reference with full academic citation.

    Extends the base Reference class to include proper academic citations.
    """

    citation: str = Field(..., description="Brief, complete academic citation for the reference.")
    doi: str | None = Field(default=None, description="DOI of the reference if available.")
    uri: str | None = Field(default=None, description="URI of the reference if available.")


class ZoteroResearchResult(ResearchResult):
    """Research result with Zotero academic literature references.

    Uses ZoteroReference objects for proper academic citation formatting.
    """

    literature: list[ZoteroReference] = Field(
        ...,
        description="List of Zotero literature references with full citations.",
    )


class RagZotero(RagAgent):
    """RAG agent specialized for Zotero academic literature.

    This agent inherits all functionality from RagAgent but uses
    Zotero-specific output formats for academic citation compliance.

    The only differences from base RagAgent:
    - Uses ZoteroResearchResult for output formatting
    - Expects search results to include citation metadata
    - May use a specialized template for academic formatting

    All search functionality is handled by configured tools.
    """

    def __init__(self, *, output_model: type[pydantic.BaseModel] = None, **kwargs: Any) -> None:
        """Initialize RagZotero with Zotero-specific output model."""
        if output_model is None:
            output_model = ZoteroResearchResult
        super().__init__(output_model=ZoteroResearchResult, **kwargs)

    def get_tool_definitions(self) -> list[Tool]:
        """Generate structured tool definitions for this agent."""
        # Create a tool definition for this agent's main capability
        tool_def = AgentToolDefinition(
            name="review_literature",
            description="Independently search, review, and synthesise the scholarly literature. Provide an extended description in natural language of a research question or task that can be answered by reviewing existing scholarly materials. This agent will independently develop an efficient and effective search strategy from this description. It will benefit from any additional context or guidance that you can provide.",
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "An extended description in natural language of a research question or task that can be answered by reviewing existing scholarly materials."
                    }
                },
                "required": ["prompt"]
            },
            output_schema={
                "type": "object",
                "description": "Research result with Zotero academic literature references",
            }
        )

        return [tool_def]
