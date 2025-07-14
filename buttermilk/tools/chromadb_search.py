"""Standalone ChromaDB search tool for vector database queries.

This module provides a modular, reusable tool for searching ChromaDB vector stores.
It can be configured with any ChromaDB instance and used by any agent.
"""

from typing import Any, Optional

from autogen_core.tools import FunctionTool
from pydantic import BaseModel, ConfigDict, Field

from buttermilk import logger
from buttermilk._core.config import ToolConfig
from buttermilk.data.vector import ChromaDBEmbeddings


class SearchResult(BaseModel):
    """Represents a single search result from ChromaDB."""

    id: str = Field(..., description="Unique ID of the retrieved chunk")
    content: str = Field(..., description="The actual text content")
    document_id: str = Field(..., description="ID of the parent document")
    document_title: Optional[str] = Field(None, description="Title of the parent document")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    score: Optional[float] = Field(None, description="Similarity score")


class ChromaDBSearchTool(ChromaDBEmbeddings, ToolConfig):
    """Standalone ChromaDB search tool.
    
    This tool provides vector search capabilities for any ChromaDB instance.
    It inherits all configuration from ChromaDBEmbeddings to ensure compatibility
    with the same YAML configs used for embedding creation.
    """

    # Allow extra fields from YAML config to be ignored and arbitrary types
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    # Search-specific parameters (in addition to those inherited from ChromaDBEmbeddings)
    n_results: int = Field(default=10, description="Number of results per search")
    no_duplicates: bool = Field(default=False, description="Filter for unique documents")

    # Override these as we're not creating a storage config
    description: str = Field(default="ChromaDB vector search tool", description="Tool description")
    tool_obj: str = Field(default="chromadb_search", description="Tool identifier")
    _initialized: bool = False

    async def initialize(self) -> None:
        """Initialize the ChromaDB connection."""
        if self._initialized:
            return

        try:
            # Use the inherited ChromaDBEmbeddings initialization
            await self.ensure_cache_initialized()
            self._initialized = True

            logger.info(f"ChromaDBSearchTool initialized with collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDBSearchTool: {e}")
            raise

    async def search(self, query: str, n_results: int = 10) -> list[SearchResult]:
        """Search the ChromaDB collection.
        
        Args:
            query: Natural language search query
            n_results: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        await self.initialize()

        # Use provided n_results or fall back to instance default
        num_results = n_results if n_results > 0 else self.n_results

        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=num_results * 3 if self.no_duplicates else num_results,
            include=["documents", "metadatas", "distances"]
        )

        # Parse results
        search_results = []
        if results["ids"] and results["ids"][0]:
            seen_docs = set()

            for i, (doc_id, doc, metadata, distance) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                # Filter duplicates if requested
                parent_doc_id = metadata.get("document_id", doc_id)
                if self.no_duplicates and parent_doc_id in seen_docs:
                    continue

                seen_docs.add(parent_doc_id)

                search_results.append(SearchResult(
                    id=doc_id,
                    content=doc,
                    document_id=parent_doc_id,
                    document_title=metadata.get("document_title"),
                    metadata=metadata,
                    score=1.0 - distance  # Convert distance to similarity
                ))

                if len(search_results) >= num_results:
                    break

        return search_results

    async def search_with_output(self, query: str) -> str:
        """Search and return formatted results as a string.
        
        Args:
            query: Natural language search query
            
        Returns:
            Formatted string with search results
        """
        # Use instance default for n_results
        results = await self.search(query, self.n_results)

        # Format results for display
        formatted_parts = []
        for i, result in enumerate(results):
            formatted_parts.append(
                f"**Result {i+1}** (Doc: {result.document_title or result.document_id})\n"
                f"{result.content}"
            )

        return "\n---\n".join(formatted_parts) if formatted_parts else "No results found."

    def get_tool(self) -> FunctionTool:
        """Get this as an autogen FunctionTool.
        
        Returns:
            FunctionTool that can be used by agents
        """
        return FunctionTool(
            name="search_vector_database",
            description=(
                f"Search the {self.collection_name} vector database for relevant information. "
                "Returns text chunks that match the query semantically."
            ),
            func=self.search_with_output,
            strict=True
        )

    @property
    def config(self) -> list[FunctionTool]:
        """Return tool configuration for ToolConfig interface."""
        return [self.get_tool()]
