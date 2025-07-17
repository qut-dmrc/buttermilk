"""Zotero-specific RAG agent with academic citation formatting.

This module provides a Zotero-focused RAG agent that inherits from the
simplified RagAgent base class and adds only Zotero-specific output formatting.
"""

from pydantic import BaseModel, Field

from buttermilk.agents.rag.iterative_rag_agent import IterativeRagAgent
from buttermilk.agents.rag.simple_rag_agent import Reference, ResearchResult


class ZoteroReference(Reference):
    """Zotero literature reference with full academic citation.

    Extends the base Reference class to include proper academic citations.
    """

    citation: str = Field(..., description="Full academic citation for the reference.")
    doi: str | None = Field(default=None, description="DOI of the reference if available.")


class ZoteroResearchResult(ResearchResult):
    """Research result with Zotero academic literature references.

    Uses ZoteroReference objects for proper academic citation formatting.
    """

    literature: list[ZoteroReference] = Field(
        ...,
        description="List of Zotero literature references with full citations.",
    )


class RagZotero(IterativeRagAgent):
    """RAG agent specialized for Zotero academic literature.

    This agent inherits all functionality from RagAgent but uses
    Zotero-specific output formats for academic citation compliance.

    The only differences from base RagAgent:
    - Uses ZoteroResearchResult for output formatting
    - Expects search results to include citation metadata
    - May use a specialized template for academic formatting

    All search functionality is handled by configured tools.
    """

    def __init__(self, **kwargs):
        """Initialize RagZotero with Zotero-specific output model."""
        super().__init__(**kwargs)

        # Override output model for Zotero-specific formatting - moved from class attribute
        self._output_model: type[BaseModel] = ZoteroResearchResult
