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
            citation=item_metadata.get("citation", ""),
            doi_or_url=item_metadata.get("doi_or_url"),
            metadata=item_metadata,
        )


class ZoteroReference(Reference):
    """Zotero-specific reference with academic citation formatting.

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

    @pydantic.model_validator(mode="after")
    def _load_zotero_tools(self) -> Self:
        """Initialize Zotero-specific tools and configurations.

        Extends the base RagAgent initialization with Zotero-specific setup.

        Returns:
            Self: The initialized RagZot instance.
        """
        # Call parent initialization first
        super()._load_tools()

        # Override the search tool with Zotero-specific description
        if hasattr(self, "_tools_list") and self._tools_list:
            # Update the search tool description for Zotero context
            search_tool = self._tools_list[0]  # Should be the search tool from parent
            search_tool.description = (
                self.description or
                "Searches a Zotero-based academic knowledge base for relevant research literature. "
                "Returns formatted text chunks with academic citations and metadata."
            )
            search_tool.name = "search_zotero_knowledge_base"

        logger.info(f"RagZot '{self.agent_id}': Zotero-specific search tool configured.")
        return self

    async def _query_db(self, query: str) -> ToolOutput:
        """Perform a Zotero-specific query with academic formatting.

        Extends the base query method to use ZoteroRefResult objects
        for proper academic citation and metadata handling.

        Args:
            query (str): The natural language query string.

        Returns:
            ToolOutput: Tool output with Zotero-formatted results.
        """
        logger.info(f"RagZot '{self.agent_id}': Querying Zotero vector store with: '{query[:100]}...'")
        try:
            # Query ChromaDB
            num_to_fetch = self.n_results * 4 if self.no_duplicates else self.n_results
            chroma_results = self._vectorstore.query(
                query_texts=[query],
                n_results=num_to_fetch,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.error(f"RagZot '{self.agent_id}': Error querying ChromaDB: {e!s}", exc_info=True)
            return ToolOutput(
                name=self.agent_name, 
                call_id="", 
                content=f"Error querying database: {e!s}", 
                is_error=True, 
                args={"query": query}
            )

        if not chroma_results or not chroma_results.get("ids") or not chroma_results["ids"][0]:
            logger.info(f"RagZot '{self.agent_id}': No results from ChromaDB for query: '{query[:100]}...'")
            return ToolOutput(
                name=self.agent_name, 
                call_id="", 
                content="No results found.", 
                results=[], 
                args={"query": query}, 
                messages=[]
            )

        # Convert ChromaDB results to ZoteroRefResult objects
        parsed_records: list[ZoteroRefResult] = []
        num_ids_found = len(chroma_results["ids"][0])
        for i in range(num_ids_found):
            try:
                parsed_records.append(ZoteroRefResult.from_chroma(chroma_results, i))
            except Exception as e:
                logger.warning(f"RagZot '{self.agent_id}': Failed to parse ChromaDB item at index {i}: {e!s}")
                continue

        # Filter for unique documents if requested
        final_records: list[ZoteroRefResult]
        if self.no_duplicates:
            unique_document_ids: set[str] = set()
            filtered_records: list[ZoteroRefResult] = []
            for record in parsed_records:
                if record.document_id not in unique_document_ids:
                    filtered_records.append(record)
                    unique_document_ids.add(record.document_id)
                    if len(filtered_records) >= self.n_results:
                        break
            final_records = filtered_records
        else:
            final_records = parsed_records[:self.n_results]

        # Create context messages with token limit checking
        context_messages: list[UserMessage] = []
        if hasattr(self, "_model_client") and self._model_client and hasattr(self._model_client, "client"):
            for i, rec in enumerate(final_records):
                current_rec_message = rec.as_message()
                try:
                    if hasattr(self._model_client.client, "remaining_tokens") and self._model_client.client.remaining_tokens(messages=context_messages + [current_rec_message]) < 5000:
                        logger.warning(
                            f"RagZot '{self.agent_id}': Approaching token limit. "
                            f"Truncating results from {len(final_records)} to {i}."
                        )
                        final_records = final_records[:i]
                        break
                except (AttributeError, KeyError):
                    logger.debug(f"RagZot '{self.agent_id}': Token limit checking not available.")
                context_messages.append(current_rec_message)
        else:
            context_messages = [rec.as_message() for rec in final_records]

        # Construct ToolOutput with Zotero-specific formatting
        return ToolOutput(
            name=self.agent_name or "RagZot",
            call_id="",
            content="\n\n".join([str(r) for r in final_records]),
            results=final_records,
            args={"query": query},
            messages=context_messages,
            is_error=False,
        )
