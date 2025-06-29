"""Generic Retrieval Augmented Generation (RAG) agent base class.

This module defines the base `RagAgent` class which provides core RAG functionality
that can be inherited by specialized implementations like RagZot. It handles the
generic aspects of retrieval augmented generation including vector store queries,
result processing, and LLM synthesis.

The base class is designed to work with any ChromaDB vector store without making
assumptions about the specific domain or citation format.
"""

import asyncio
from typing import Any, Self

from autogen_core import FunctionCall
from autogen_core.models._types import UserMessage
from autogen_core.tools import FunctionTool
from chromadb import Collection
from pydantic import BaseModel, Field, PrivateAttr

from buttermilk import logger
from buttermilk._core.config import ToolConfig
from buttermilk._core.contract import ToolOutput
from buttermilk.agents.llm import LLMAgent, Tool
from buttermilk.data.vector import ChromaDBEmbeddings

import pydantic


class RefResult(BaseModel):
    """Represents a single retrieved reference (document chunk) from the vector store.

    This model structures the data retrieved from a ChromaDB query result,
    representing a chunk of text from a source document.

    Attributes:
        id (str): The unique ID of this specific chunk in the vector store.
        full_text (str): The actual text content of the retrieved chunk.
        uri (str | None): An optional URI pointing to the source document.
        chunk_index (int): The index or sequence number of this chunk within its
            parent document.
        document_id (str): The unique identifier of the parent document from which
            this chunk originates.
        document_title (str | None): The title of the parent document.
        metadata (dict[str, Any]): Additional metadata from the vector store.
    """

    id: str = Field(..., description="Unique ID of the retrieved text chunk from the vector store.")
    full_text: str = Field(..., description="The actual text content of the retrieved chunk.")
    uri: str | None = Field(default=None, description="Optional URI to the source document.")
    chunk_index: int = Field(..., description="Index of this chunk within its parent document.")
    document_id: str = Field(..., description="Unique ID of the parent document.")
    document_title: str | None = Field(default=None, description="Title of the parent document.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata from the vector store.")

    def __str__(self) -> str:
        """Return a formatted string representation of the reference result.

        Returns:
            str: A human-readable string summary of the reference.
        """
        return (
            f"**Title:** {self.document_title or 'N/A'}\n"
            f"**Document ID:** {self.document_id} (Chunk: {self.chunk_index})\n"
            f"**URI:** {self.uri or 'N/A'}\n"
            f"**Full Text:** {self.full_text[:2000] + '...' if len(self.full_text) > 2000 else self.full_text}\n"
        )

    def as_message(self, source: str = "search_result") -> UserMessage:
        """Convert this reference result into an Autogen `UserMessage`.

        Args:
            source (str): The source identifier for the `UserMessage`.

        Returns:
            UserMessage: An Autogen `UserMessage` object.
        """
        return UserMessage(content=str(self), source=source)

    @classmethod
    def from_chroma(cls, results: dict[str, list[list[Any]]], index: int) -> "RefResult":
        """Create a `RefResult` instance from ChromaDB query results.

        Args:
            results (dict): The raw query results dictionary from ChromaDB.
            index (int): The index within the inner lists to pick the specific item's data.

        Returns:
            RefResult: An instance of `RefResult` populated with data from ChromaDB results.

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

        # Extract standard fields from metadata
        return cls(
            id=item_id,
            full_text=item_document,
            uri=item_metadata.get("uri"),
            chunk_index=item_metadata.get("chunk_index", 0),
            document_id=item_metadata.get("document_id", item_id),
            document_title=item_metadata.get("document_title"),
            metadata=item_metadata,
        )


class Reference(BaseModel):
    """Represents a single cited reference within a research result.

    Attributes:
        summary (str): A brief summary of the key information from this reference.
        source (str): The source identifier or title for the reference.
    """

    summary: str = Field(..., description="Summary of the key information from this reference.")
    source: str = Field(..., description="Source identifier or title for the reference.")


class ResearchResult(BaseModel):
    """Represents the structured output of a RAG process.

    Attributes:
        literature (list[Reference]): List of references used in generating the response.
        response (str): The final synthesized response from the LLM.
    """

    literature: list[Reference] = Field(
        ...,
        description="List of literature references used in generating the response.",
    )
    response: str = Field(
        ...,
        description="The synthesized textual response from the LLM.",
    )


class RagAgent(LLMAgent, ToolConfig):
    """Base class for Retrieval Augmented Generation (RAG) agents.

    This agent provides core RAG functionality including:
    1. Connection to a ChromaDB vector store
    2. Semantic search capabilities  
    3. Result processing and filtering
    4. LLM synthesis with retrieved context

    The class is designed to be inherited by specialized implementations
    that add domain-specific functionality.

    Configuration Parameters:
        n_results (int): Number of search results to retrieve per query. Default: 20.
        no_duplicates (bool): Filter results for unique parent documents. Default: False.
        max_queries (int): Maximum concurrent search queries. Default: 5.
    """

    _chromadb: ChromaDBEmbeddings = PrivateAttr()
    _vectorstore: Collection = PrivateAttr()
    _output_model: type[BaseModel] | None = ResearchResult
    _ref_result_class: type[RefResult] = RefResult  # Subclasses can override this

    # RAG configuration
    n_results: int = Field(
        default=20,
        description="Number of search results to retrieve per query for context.",
    )
    no_duplicates: bool = Field(
        default=False,
        description="If True, filter search results to ensure unique parent documents.",
    )
    max_queries: int = Field(
        default=5,
        description="Maximum number of concurrent search queries when used as a tool.",
    )

    @pydantic.model_validator(mode="after")
    def _load_tools(self) -> Self:
        """Initialize ChromaDB connection and set up the search tool.

        Returns:
            Self: The initialized agent instance.

        Raises:
            ValueError: If no ChromaDB data source is configured.
        """
        # Initialize ChromaDB connection from self.data configuration
        chroma_config_found = False
        if self.data:
            for data_source_name, data_conf in self.data.items():
                if data_conf.type == "chromadb":
                    logger.info(f"RagAgent '{self.agent_id}': Initializing ChromaDB from data source '{data_source_name}'.")
                    try:
                        self._chromadb = ChromaDBEmbeddings(**data_conf.model_dump())
                        chroma_config_found = True
                        break
                    except Exception as e:
                        logger.error(f"RagAgent '{self.agent_id}': Failed to initialize ChromaDB from data source '{data_source_name}': {e!s}")
                        raise ValueError(f"ChromaDB initialization failed for RAG agent: {e!s}") from e

        if not chroma_config_found:
            raise ValueError("RAG agent requires a DataSourceConfig of type 'chromadb' but none was found.")

        # Define the search tool
        search_tool_description = (
            self.description or
            "Searches a vector knowledge base for relevant information based on natural language queries. "
            "Returns formatted text chunks from the knowledge base."
        )
        search_tool = FunctionTool(
            name="search_knowledge_base",
            description=search_tool_description,
            func=self.fetch,
            strict=False,
        )

        # Add search tool to available tools
        if not hasattr(self, "_tools_list") or not self._tools_list:
            self._tools_list = []
        self._tools_list.append(search_tool)

        logger.info(f"RagAgent '{self.agent_id}': Search tool '{search_tool.name}' loaded.")
        return self

    async def ensure_chromadb_ready(self) -> None:
        """Ensure ChromaDB is ready for use, handling remote caching if needed."""
        if not hasattr(self, "_chromadb") or not self._chromadb:
            raise ValueError("ChromaDB not initialized. Call _load_tools first.")

        # Initialize cache for remote persist_directory
        await self._chromadb.ensure_cache_initialized()

        # Access the collection
        self._vectorstore = self._chromadb.collection
        logger.info(f"RagAgent '{self.agent_id}': ChromaDB collection ready for queries.")

    def get_functions(self) -> list[FunctionTool]:
        """Return the list of available function tools for this agent.

        Returns:
            list[FunctionTool]: List of function tools.
        """
        return self._tools_list

    @property
    def config(self) -> list[Tool]:
        """Provides the tool configuration for this agent.

        Returns:
            list: List of tool definitions.
        """
        return self._tools_list

    async def fetch(self, queries: list[str]) -> list[ToolOutput]:
        """Execute search queries against the ChromaDB vector store.

        Args:
            queries (list[str]): List of natural language query strings.

        Returns:
            list[ToolOutput]: List of tool outputs with search results.

        Raises:
            ValueError: If no queries provided or vector store not initialized.
        """
        if not queries:
            raise ValueError("No queries provided to RAG agent search tool.")

        # Ensure ChromaDB is ready
        if not hasattr(self, "_vectorstore") or not self._vectorstore:
            await self.ensure_chromadb_ready()

        if len(queries) > self.max_queries:
            logger.warning(
                f"RagAgent '{self.agent_id}': Received {len(queries)} queries, limiting to {self.max_queries}."
            )
            queries = queries[:self.max_queries]

        # Execute queries concurrently
        search_tasks = [self._query_db(query=q_str) for q_str in queries]
        tool_outputs: list[ToolOutput] = await asyncio.gather(*search_tasks)

        return tool_outputs

    async def _query_db(self, query: str) -> ToolOutput:
        """Perform a single query against the ChromaDB vector store.

        Args:
            query (str): The natural language query string.

        Returns:
            ToolOutput: Tool output containing search results.
        """
        logger.info(f"RagAgent '{self.agent_id}': Querying vector store with: '{query[:100]}...'")
        try:
            # Query ChromaDB with expanded results if filtering duplicates
            num_to_fetch = self.n_results * 4 if self.no_duplicates else self.n_results
            chroma_results = self._vectorstore.query(
                query_texts=[query],
                n_results=num_to_fetch,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.error(f"RagAgent '{self.agent_id}': Error querying ChromaDB: {e!s}", exc_info=True)
            return ToolOutput(
                name=self.agent_name, 
                call_id="", 
                content=f"Error querying database: {e!s}", 
                is_error=True, 
                args={"query": query}
            )

        if not chroma_results or not chroma_results.get("ids") or not chroma_results["ids"][0]:
            logger.info(f"RagAgent '{self.agent_id}': No results from ChromaDB for query: '{query[:100]}...'")
            return ToolOutput(
                name=self.agent_name, 
                call_id="", 
                content="No results found.", 
                results=[], 
                args={"query": query}, 
                messages=[]
            )

        # Convert ChromaDB results to RefResult objects
        parsed_records: list[RefResult] = []
        num_ids_found = len(chroma_results["ids"][0])
        for i in range(num_ids_found):
            try:
                parsed_records.append(self._ref_result_class.from_chroma(chroma_results, i))
            except Exception as e:
                logger.warning(f"RagAgent '{self.agent_id}': Failed to parse ChromaDB item at index {i}: {e!s}")
                continue

        # Filter for unique documents if requested
        final_records = self._filter_unique_documents(parsed_records) if self.no_duplicates else parsed_records[:self.n_results]

        # Create context messages for LLM
        context_messages, final_records = self._create_context_messages(final_records)

        # Construct ToolOutput
        return ToolOutput(
            name=self.agent_name or "RagAgent",
            call_id="",
            content="\n\n".join([str(r) for r in final_records]),
            results=final_records,
            args={"query": query},
            messages=context_messages,
            is_error=False,
        )

    def _filter_unique_documents(self, records: list[RefResult]) -> list[RefResult]:
        """Filter records to ensure unique parent documents.
        
        Args:
            records: List of RefResult objects to filter
            
        Returns:
            list[RefResult]: Filtered list with unique document IDs
        """
        unique_document_ids: set[str] = set()
        filtered_records: list[RefResult] = []
        for record in records:
            if record.document_id not in unique_document_ids:
                filtered_records.append(record)
                unique_document_ids.add(record.document_id)
                if len(filtered_records) >= self.n_results:
                    break
        return filtered_records

    def _create_context_messages(self, records: list[RefResult]) -> tuple[list[UserMessage], list[RefResult]]:
        """Create context messages from records with optional token limit checking.
        
        Args:
            records: List of RefResult objects
            
        Returns:
            tuple: (context_messages, potentially_truncated_records)
        """
        context_messages: list[UserMessage] = []
        final_records = records
        
        if hasattr(self, "_model_client") and self._model_client and hasattr(self._model_client, "client"):
            for i, rec in enumerate(records):
                current_rec_message = rec.as_message()
                try:
                    # Check token limits if available
                    if hasattr(self._model_client.client, "remaining_tokens") and \
                       self._model_client.client.remaining_tokens(messages=context_messages + [current_rec_message]) < 5000:
                        logger.warning(
                            f"RagAgent '{self.agent_id}': Approaching token limit. "
                            f"Truncating results from {len(records)} to {i}."
                        )
                        final_records = records[:i]
                        break
                except (AttributeError, KeyError):
                    logger.debug(f"RagAgent '{self.agent_id}': Token limit checking not available.")
                context_messages.append(current_rec_message)
        else:
            context_messages = [rec.as_message() for rec in records]
            
        return context_messages, final_records
