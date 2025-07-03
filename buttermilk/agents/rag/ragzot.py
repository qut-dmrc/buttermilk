"""Provides a Retrieval Augmented Generation (RAG) agent using Zotero and ChromaDB.

This module defines the `RagZot` agent, which leverages a Zotero library as its
knowledge base. It assumes Zotero data has been processed and embedded into a
ChromaDB vector store. The agent inherits from the generic RagAgent base class
and adds Zotero-specific citation and metadata handling.
"""

from typing import Any, Self

from pydantic import Field, PrivateAttr

from buttermilk import logger
from buttermilk._core.contract import ToolOutput
from buttermilk._core.types import UserMessage
from buttermilk.agents.rag.rag_agent import RagAgent, RefResult, Reference, ResearchResult

import pydantic


class ZoteroRefResult(RefResult):
    """Zotero-specific extension of RefResult with citation and DOI fields.

    Adds Zotero-specific metadata fields while inheriting core functionality
    from the base RefResult class.

    Additional Attributes:
        citation (str | None): Optional citation string for the document.
        doi_or_url (str | None): DOI or URL associated with the parent document.
        _extra_zotero_data (str): Additional raw Zotero metadata.
    """

    citation: str | None = Field(default="", description="Optional citation string for the document.")
    doi_or_url: str | None = Field(default=None, description="DOI or URL of the parent document.")
    _extra_zotero_data: str = PrivateAttr()

    def __str__(self) -> str:
        """Return a formatted string representation with Zotero-specific formatting.

        Includes citation and DOI information specific to Zotero documents.

        Returns:
            str: A human-readable string summary of the Zotero reference.
        """
        return (
            f"**Title:** {self.document_title or 'N/A'}\n"
            f"**Document ID:** {self.document_id} (Source: {self.doi_or_url or 'N/A'}, Chunk: {self.chunk_index})\n"
            f"**Citation:** {self.citation or 'N/A'}\n"
            f"**Full Text Snippet:** {self.full_text[:2000] + '...' if len(self.full_text) > 2000 else self.full_text}\n"
        )

    @classmethod
    def from_chroma(cls, results: dict[str, list[list[Any]]], index: int) -> "ZoteroRefResult":
        """Create a Zotero-specific RefResult from ChromaDB query results.

        Args:
            results (dict): The raw query results dictionary from ChromaDB.
            index (int): The index within the inner lists to pick the specific item's data.

        Returns:
            ZoteroRefResult: A Zotero-specific RefResult instance.

        Raises:
            ValueError: If the results cannot be parsed properly.
        """
        try:
            item_id = results["ids"][0][index]
            item_metadata = results["metadatas"][0][index] if results.get("metadatas") and results["metadatas"][0] else {}
            item_document = results["documents"][0][index] if results.get("documents") and results["documents"][0] else ""
        except (IndexError, KeyError) as e:
            logger.error(f"Error parsing ChromaDB result at index {index}: {e!s}. Results structure: {results}")
            raise ValueError(f"Could not parse ChromaDB result at index {index} due to missing data or structure mismatch.") from e

        # Extract Zotero-specific fields from metadata
        return cls(
            id=item_id,
            full_text=item_document,
            uri=item_metadata.get("uri"),
            chunk_index=item_metadata.get("chunk_index", 0),
            document_id=item_metadata.get("document_id", item_id),
            document_title=item_metadata.get("document_title"),
            metadata=item_metadata,
            citation=item_metadata.get("citation", ""),
            doi_or_url=item_metadata.get("doi_or_url"),
        )


class ZoteroReference(Reference):
    """Represents a single Zotero literature reference in a research result.

    Extends the base Reference class with Zotero-specific citation handling.

    Attributes:
        summary (str): Summary of the key information from this reference.
        citation (str): Full academic citation for the reference.
    """

    citation: str = pydantic.Field(..., description="Full academic citation for the reference.")


class ZoteroResearchResult(ResearchResult):
    """Zotero-specific research result with academic literature references.

    Uses ZoteroReference objects for proper academic citation formatting.

    Attributes:
        literature (list[ZoteroReference]): List of Zotero literature references.
        response (str): The synthesized response from the LLM.
    """

    literature: list[ZoteroReference] = pydantic.Field(
        ...,
        description="List of Zotero literature references cited or used in generating the response.",
    )


class RagZot(RagAgent):
    """A Retrieval Augmented Generation (RAG) agent using Zotero data via ChromaDB.

    This agent inherits from the generic RagAgent base class and adds Zotero-specific
    functionality including academic citation handling and research-focused formatting.

    The agent leverages a Zotero library that has been processed and embedded into a
    ChromaDB vector store. It provides specialized search and synthesis capabilities
    for academic literature and research contexts.

    Key Zotero-specific features:
    - Academic citation formatting
    - DOI and URI handling  
    - Research-focused result presentation
    - Zotero metadata integration

    Configuration inherits from RagAgent with the same parameters:
    - n_results: Number of search results (default: 20)
    - no_duplicates: Filter for unique documents (default: False)  
    - max_queries: Maximum concurrent queries (default: 5)
    """

    _output_model: type[pydantic.BaseModel] | None = ZoteroResearchResult
    _ref_result_class: type[RefResult] = ZoteroRefResult  # Use Zotero-specific result class
