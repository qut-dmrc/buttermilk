"""Standalone ChromaDB search tool for vector database queries.

This module provides a modular, reusable tool for searching ChromaDB vector stores.
It can be configured with any ChromaDB instance and used by any agent.
"""

import asyncio
from typing import Any, Optional

from autogen_core.tools import FunctionTool
from chromadb import Collection
from pydantic import BaseModel, Field

from buttermilk import logger
from buttermilk._core.config import ToolConfig
from buttermilk._core.contract import ToolOutput
from buttermilk._core.types import UserMessage
from buttermilk.data.vector import ChromaDBEmbeddings


class SearchResult(BaseModel):
    """Represents a single search result from ChromaDB."""
    
    id: str = Field(..., description="Unique ID of the retrieved chunk")
    content: str = Field(..., description="The actual text content")
    document_id: str = Field(..., description="ID of the parent document")
    document_title: Optional[str] = Field(None, description="Title of the parent document")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    score: Optional[float] = Field(None, description="Similarity score")


class ChromaDBSearchTool(ToolConfig):
    """Standalone ChromaDB search tool.
    
    This tool provides vector search capabilities for any ChromaDB instance.
    It can be configured with a data source and used by any agent.
    """
    
    # Configuration
    collection_name: str = Field(..., description="Name of the ChromaDB collection")
    persist_directory: str = Field(..., description="ChromaDB persistence directory")
    embedding_model: str = Field(default="vertex", description="Embedding model to use")
    dimensionality: int = Field(default=768, description="Embedding dimensionality")
    
    # Search parameters
    n_results: int = Field(default=10, description="Number of results per search")
    no_duplicates: bool = Field(default=False, description="Filter for unique documents")
    
    # Internal state
    _chromadb: Optional[ChromaDBEmbeddings] = None
    _collection: Optional[Collection] = None
    _initialized: bool = False
    
    class Config:
        arbitrary_types_allowed = True
    
    async def initialize(self) -> None:
        """Initialize the ChromaDB connection."""
        if self._initialized:
            return
            
        try:
            # Create ChromaDB config
            config = {
                "type": "chromadb",
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "embedding_model": self.embedding_model,
                "dimensionality": self.dimensionality
            }
            
            self._chromadb = ChromaDBEmbeddings(**config)
            await self._chromadb.ensure_cache_initialized()
            self._collection = self._chromadb.collection
            self._initialized = True
            
            logger.info(f"ChromaDBSearchTool initialized with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDBSearchTool: {e}")
            raise
    
    async def search(self, query: str, n_results: Optional[int] = None) -> list[SearchResult]:
        """Search the ChromaDB collection.
        
        Args:
            query: Natural language search query
            n_results: Number of results (overrides default)
            
        Returns:
            List of SearchResult objects
        """
        await self.initialize()
        
        num_results = n_results or self.n_results
        
        # Query ChromaDB
        results = self._collection.query(
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
    
    async def search_with_output(self, query: str, n_results: Optional[int] = None) -> ToolOutput:
        """Search and return results as ToolOutput for agent integration.
        
        Args:
            query: Natural language search query
            n_results: Number of results (overrides default)
            
        Returns:
            ToolOutput with formatted results
        """
        results = await self.search(query, n_results)
        
        # Format results for display
        formatted_content = []
        for i, result in enumerate(results):
            formatted_content.append(
                f"**Result {i+1}** (Doc: {result.document_title or result.document_id})\n"
                f"{result.content}\n"
            )
        
        return ToolOutput(
            name="chromadb_search",
            call_id="",
            content="\n---\n".join(formatted_content) if formatted_content else "No results found.",
            results=results,
            args={"query": query, "n_results": n_results or self.n_results},
            messages=[UserMessage(content=str(r.content), source="search_result") for r in results],
            is_error=False
        )
    
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
            strict=False
        )
    
    @property
    def config(self) -> list[FunctionTool]:
        """Return tool configuration for ToolConfig interface."""
        return [self.get_tool()]