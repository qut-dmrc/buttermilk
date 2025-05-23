"""Provides a Retrieval Augmented Generation (RAG) agent using Zotero and ChromaDB.

This module defines the `RagZot` agent, which leverages a Zotero library as its
knowledge base. It assumes Zotero data has been processed and embedded into a
ChromaDB vector store. The agent can then query this vector store to retrieve
relevant context for answering user prompts with an LLM.

It also defines Pydantic models (`RefResult`, `Reference`, `ResearchResult`)
for structuring retrieved references and the final LLM output.
"""

import asyncio
from typing import Any, Self  # For type hinting

from autogen_core import FunctionCall  # Autogen type for function calls
from autogen_core.models._types import UserMessage  # Autogen UserMessage type
from autogen_core.tools import FunctionTool  # Autogen FunctionTool for LLM tools
from chromadb import Collection  # ChromaDB Collection type

from buttermilk import logger  # Centralized logger
from buttermilk._core.config import ToolConfig  # Base ToolConfig
from buttermilk._core.contract import ToolOutput  # Buttermilk ToolOutput contract
from buttermilk.agents.llm import LLMAgent, Tool, ToolSchema  # Buttermilk LLMAgent and tool types
from buttermilk.data.vector import ChromaDBEmbeddings  # Buttermilk ChromaDB integration

TASK_FOR_QUERY = "RETRIEVAL_QUERY"
"""Constant string used possibly as a task type identifier for query embeddings.
(Note: Its specific usage context within the embedding process is not detailed here).
"""

import pydantic  # Pydantic core


class RefResult(pydantic.BaseModel):
    """Represents a single retrieved reference (document chunk) from the vector store.

    This model structures the data retrieved from a ChromaDB query result,
    typically representing a chunk of text from a Zotero document.

    Attributes:
        id (str): The unique ID of this specific chunk in the vector store.
        full_text (str): The actual text content of the retrieved chunk.
        uri (str | None): An optional URI pointing to the source document or item
            (e.g., a Zotero URI, a DOI link).
        chunk_index (int): The index or sequence number of this chunk within its
            parent document.
        citation (str | None): An optional citation string for the document.
        document_id (str): The unique identifier of the parent document from which
            this chunk originates (e.g., Zotero item key).
        document_title (str | None): The title of the parent document.
        doi_or_url (str | None): The DOI or a direct URL associated with the parent document.
        _extra_zotero_data (str): A private attribute intended to store additional
            raw Zotero metadata associated with this chunk or document.
            (Note: Its population is not shown in `from_chroma`, may need review).

    """

    id: str = Field(..., description="Unique ID of the retrieved text chunk from the vector store.")
    full_text: str = Field(..., description="The actual text content of the retrieved chunk.")
    uri: str | None = Field(default=None, description="Optional URI to the source document or item.")
    chunk_index: int = Field(..., description="Index of this chunk within its parent document.")
    citation: str | None = Field(default="", description="Optional citation string for the document.")
    document_id: str = Field(..., description="Unique ID of the parent document (e.g., Zotero item key).")
    document_title: str | None = Field(default=None, description="Title of the parent document.")
    doi_or_url: str | None = Field(default=None, description="DOI or URL of the parent document.")
    _extra_zotero_data: str = Field(default="", description="Additional raw Zotero metadata (usage may vary).")  # Added default

    def __str__(self) -> str:
        """Returns a formatted string representation of the reference result.

        Includes title, document ID, DOI/URL, chunk index, citation, and full text.

        Returns:
            str: A human-readable string summary of the reference.

        """
        return (
            f"**Title:** {self.document_title or 'N/A'}\n"
            f"**Document ID:** {self.document_id} (Source: {self.doi_or_url or 'N/A'}, Chunk: {self.chunk_index})\n"
            f"**Citation:** {self.citation or 'N/A'}\n"
            f"**Full Text Snippet:** {self.full_text[:200] + '...' if len(self.full_text) > 200 else self.full_text}\n"
        )

    def as_message(self, source: str = "search_result") -> UserMessage:
        """Converts this reference result into an Autogen `UserMessage`.

        The content of the message is the string representation of this `RefResult`.

        Args:
            source (str): The source identifier for the `UserMessage`.
                Defaults to "search_result".

        Returns:
            UserMessage: An Autogen `UserMessage` object.

        """
        return UserMessage(content=str(self), source=source)

    @classmethod
    def from_chroma(cls, results: dict[str, list[list[Any]]], index: int) -> "RefResult":
        """Creates a `RefResult` instance from ChromaDB query results.

        Assumes `results` is a dictionary as returned by `collection.query()`,
        containing keys like "ids", "metadatas", "documents", where each value
        is a list of lists (one inner list per query, though this method assumes
        results from a single query, hence `[0]`).

        Args:
            results (dict): The raw query results dictionary from ChromaDB.
                Expected format: `{'ids': [[id1, id2]], 'metadatas': [[meta1, meta2]], 'documents': [[doc1, doc2]]}`.
            index (int): The index within the inner lists to pick the specific item's
                data (e.g., 0 for the first retrieved chunk).

        Returns:
            RefResult: An instance of `RefResult` populated with data from the
            ChromaDB results at the specified index.

        Raises:
            IndexError: If `index` is out of bounds for the items in `results`.
            KeyError: If expected keys ("ids", "metadatas", "documents") are missing
                      or if `results[key][0]` is empty.

        """
        # Extract data for the item at the given index from the first (and assumed only) query result
        try:
            item_id = results["ids"][0][index]
            item_metadata = results["metadatas"][0][index] if results.get("metadatas") and results["metadatas"][0] else {}
            item_document = results["documents"][0][index] if results.get("documents") and results["documents"][0] else ""
        except (IndexError, KeyError) as e:
            logger.error(f"Error parsing ChromaDB result at index {index}: {e!s}. Results structure: {results}")
            # Depending on desired strictness, could raise error or return a default/empty RefResult
            raise ValueError(f"Could not parse ChromaDB result at index {index} due to missing data or structure mismatch.") from e

        # Filter metadata to only include fields defined in RefResult
        # This also handles renaming if aliases were used in ChromaDB metadata.
        valid_meta = {k: v for k, v in item_metadata.items() if k in cls.model_fields}

        # _extra_zotero_data is not typically part of ChromaDB metadata unless explicitly stored.
        # If it's stored under a specific key, it should be mapped here.
        # For now, it will default to "" as per field definition if not in valid_meta.

        return cls(
            id=item_id,
            full_text=item_document,
            uri=item_metadata.get("uri"),  # Safely get URI from metadata
            **valid_meta,  # Spread other valid metadata fields
        )


class Reference(pydantic.BaseModel):
    """Represents a single cited reference within a research result.

    Used by `ResearchResult` to structure the literature references that support
    the LLM's generated response.

    Attributes:
        summary (str): A brief summary of the key point or information derived
            from this specific reference that contributes to the overall response.
        citation (str): The citation string for the reference (e.g., author-date,
            numerical, or as provided by the `RefResult.citation`).

    """

    summary: str = pydantic.Field(..., description="Summary of the key point or information derived from this reference.")
    citation: str = pydantic.Field(..., description="Full citation for the reference.")


class ResearchResult(pydantic.BaseModel):
    """Represents the structured output of a RAG (Retrieval Augmented Generation) process.

    This model is typically used as the `_output_model` for the `RagZot` agent.
    It combines the LLM's synthesized textual `response` with a list of `Reference`
    objects that were used to generate that response.

    Attributes:
        literature (list[Reference]): A list of `Reference` objects, where each
            details a piece of retrieved literature (document chunk) that was
            cited or used in formulating the `response`.
        response (str): The final textual response generated by the LLM, synthesized
            from its knowledge and the information retrieved from the literature.

    """

    literature: list[Reference] = pydantic.Field(
        ...,
        description="List of literature references (document chunks) cited or used in generating the response.",
    )
    response: str = pydantic.Field(
        ...,
        description="The synthesized textual response from the LLM, based on the query and retrieved literature.",
    )


class RagZot(LLMAgent, ToolConfig):
    """A Retrieval Augmented Generation (RAG) agent using Zotero data via ChromaDB.

    This agent combines the capabilities of an `LLMAgent` (for interacting with
    a Language Model) and `ToolConfig` (allowing it to be used as a tool by
    other agents, specifically for its search functionality).

    Its primary workflow involves:
    1.  Initializing a connection to a ChromaDB vector store, which is assumed
        to contain embeddings of documents from a Zotero library. This is
        configured via `self.data` (an `AgentConfig` field).
    2.  Providing a `search` tool (implemented by its `_run` method) that can
        take a list of natural language queries.
    3.  For each query, the `_query_db` method searches the ChromaDB vector store
        to find relevant document chunks (`RefResult` instances).
    4.  These retrieved chunks are then formatted and can be used as context
        when this agent (acting as an `LLMAgent`) processes a primary prompt
        using its own LLM. The LLM is expected to synthesize this retrieved
        information to generate a final response, structured as `ResearchResult`.

    Key Configuration Parameters:
        - `data` (Mapping[str, DataSourceConfig]): In `AgentConfig`, one of the
          data sources must be of `type: "chromadb"` and provide connection
          details for `ChromaDBEmbeddings`.
        - `model` (str): (From `LLMAgent`) The LLM to use for synthesizing answers
          after retrieval.
        - `prompt_template` (str): (From `LLMAgent`) The prompt template that will
          receive the user's query and the retrieved context (formatted `RefResult`s)
          to guide the LLM in generating the final `ResearchResult`.
        - `n_results` (int): Number of search results to retrieve per query and
          ultimately use for context. Default: 20.
        - `no_duplicates` (bool): If True, filters search results to ensure unique
          parent documents. Default: True.
        - `max_queries` (int): Maximum number of concurrent search queries allowed
          when the `search` tool is called with multiple queries. Default: 3.

    Attributes:
        _chromadb (ChromaDBEmbeddings): Instance for interacting with ChromaDB.
        _vectorstore (Collection): The specific ChromaDB collection being queried.
        _output_model (Type[pydantic.BaseModel]): Set to `ResearchResult`, defining
            the expected Pydantic model for this agent's LLM's structured output.

    """

    _chromadb: ChromaDBEmbeddings = PrivateAttr()  # Default handled by _load_tools
    _vectorstore: Collection = PrivateAttr()  # Default handled by _load_tools
    _output_model: type[pydantic.BaseModel] | None = ResearchResult  # Expected output from this agent's LLM

    # RAG query settings
    n_results: int = pydantic.Field(
        default=20,
        description="Number of search results to retrieve per query for context.",
    )
    no_duplicates: bool = pydantic.Field(
        default=True,
        description="If True, filter search results to ensure unique parent documents.",
    )
    max_queries: int = pydantic.Field(
        default=3,
        description="Maximum number of concurrent search queries when used as a tool.",
    )

    @pydantic.model_validator(mode="after")
    def _load_tools(self) -> Self:
        """Initializes ChromaDB connection and sets up the search tool.

        This Pydantic validator runs after the model is created. It iterates
        through `self.data` configurations to find one of `type: "chromadb"`,
        initializes `ChromaDBEmbeddings` and `self._vectorstore` from it.
        It then creates a `FunctionTool` for the `_run` method (which performs
        searches) and adds it to `self._tools_list`, making it available if this
        agent's LLM is configured to use tools, or if this agent itself is
        used as a tool by another agent.

        Returns:
            Self: The initialized `RagZot` instance.

        Raises:
            ValueError: If no ChromaDB data source is configured or if the
                        vector store cannot be initialized.

        """
        # Initialize ChromaDB connection from self.data configuration
        chroma_config_found = False
        if self.data:  # self.data is from AgentConfig, a Mapping[str, DataSourceConfig]
            for data_source_name, data_conf in self.data.items():
                if data_conf.type == "chromadb":
                    logger.info(f"RagZot '{self.agent_id}': Initializing ChromaDB from data source '{data_source_name}'.")
                    try:
                        self._chromadb = ChromaDBEmbeddings(**data_conf.model_dump())
                        self._vectorstore = self._chromadb.collection
                        chroma_config_found = True
                        break  # Found and initialized ChromaDB, no need to check other data sources
                    except Exception as e:
                        logger.error(f"RagZot '{self.agent_id}': Failed to initialize ChromaDB from data source '{data_source_name}': {e!s}")
                        raise ValueError(f"ChromaDB initialization failed for RagZot agent: {e!s}") from e

        if not chroma_config_found:
            raise ValueError("RagZot agent requires a DataSourceConfig of type 'chromadb' but none was found or properly initialized.")

        # Define the search tool for this agent
        # The description should guide an LLM on how and when to use this search tool.
        search_tool_description = (
            self.description or  # Use agent's own description if provided
            "Searches a Zotero-based vector knowledge base for relevant information based on a list of natural language queries. "
            "Returns formatted text chunks and citations from the knowledge base."
        )
        search_tool = FunctionTool(
            name="search_zotero_knowledge_base",  # Clearer name for LLM
            description=search_tool_description,
            func=self.fetch,  # Points to the method that executes the search
            strict=False,  # Pydantic validation mode for arguments from LLM
        )

        # Add this search tool to the agent's list of available tools.
        # self._tools_list is from LLMAgent.
        if not hasattr(self, "_tools_list") or not self._tools_list:
            self._tools_list = []
        self._tools_list.append(search_tool)

        logger.info(f"RagZot '{self.agent_id}': Search tool '{search_tool.name}' loaded.")
        return self

    def get_functions(self) -> list[FunctionTool]:  # More specific return type
        """Returns the list of Autogen `FunctionTool` definitions for this agent.

        This method is typically used when the agent's tools need to be registered
        with another system or LLM that understands the Autogen tool format.
        For `RagZot`, this primarily includes its Zotero search capability.

        Returns:
            list[FunctionTool]: A list of `FunctionTool` instances available on this agent.

        """
        return self._tools_list  # type: ignore # Assuming _tools_list contains FunctionTool

    @property
    def config(self) -> list[FunctionCall | Tool | ToolSchema | FunctionTool]:  # Matches LLMAgent if this is an override
        """Provides the tool configuration for this agent.

        This property returns the list of tools (including the search tool)
        that this agent can use or expose.

        Returns:
            list[FunctionCall | Tool | ToolSchema | FunctionTool]: A list of
            Autogen-compatible tool definitions.

        """
        return self._tools_list

    async def fetch(self, queries: list[str]) -> list[ToolOutput]:  # This is the tool's callable method
        """Executes one or more search queries against the ChromaDB vector store concurrently.

        This method is registered as a tool function (e.g., for an LLM to call).
        It takes a list of query strings, runs them in parallel against the
        vector store using `_query_db`, and aggregates their `ToolOutput` results.

        Args:
            queries (list[str]): A list of natural language query strings to search for.

        Returns:
            list[ToolOutput]: A list of `ToolOutput` objects, where each object
            contains the formatted results (as `RefResult` instances) for one of
            the input queries.

        Raises:
            ValueError: If `queries` is empty or the vector store (`_vectorstore`)
                is not initialized.

        """
        if not queries:
            raise ValueError("No queries provided to RagZot search tool (_run).")
        if not hasattr(self, "_vectorstore") or not self._vectorstore:  # Check if vectorstore is initialized
            raise ValueError("Vector store not initialized for RagZot search tool (_run). Ensure _load_tools ran successfully.")

        if len(queries) > self.max_queries:
            logger.warning(
                f"RagZot '{self.agent_id}': Received {len(queries)} queries, but 'max_queries' is {self.max_queries}. "
                f"Limiting to the first {self.max_queries} queries.",
            )
            queries = queries[:self.max_queries]

        search_tasks = [self._query_db(query=q_str) for q_str in queries]
        # gather will run them concurrently and return list of results in order
        tool_outputs: list[ToolOutput] = await asyncio.gather(*search_tasks)

        return tool_outputs

    async def _query_db(self, query: str) -> ToolOutput:
        """Performs a single query against the ChromaDB vector store and formats results.

        Queries the `self._vectorstore` for `query`. Processes the raw ChromaDB
        results into a list of `RefResult` Pydantic models. If `self.no_duplicates`
        is True, it filters these results to ensure uniqueness based on `document_id`.
        It also truncates the number of results to `self.n_results` and checks
        for LLM token limits if a model client is available.

        Args:
            query (str): The natural language query string.

        Returns:
            ToolOutput: A `ToolOutput` object containing:
                - `name`: The name of this RagZot agent.
                - `call_id`: An empty string (as this is an internal query, not a direct LLM tool call response).
                           Consider generating a unique ID if needed for tracing sub-queries.
                - `content`: A string joining all formatted `RefResult` string representations.
                - `results`: The list of `RefResult` Pydantic model instances.
                - `args`: A dictionary containing the original `query`.
                - `messages`: A list of Autogen `UserMessage` objects, each created from a `RefResult`.
                - `is_error`: False, as errors here would typically raise exceptions.

        Note:
            Token limit checking for `self._model_client.client.remaining_tokens`
            might be fragile if `_model_client` or its `client` attribute isn't
            set up as expected, or if the client doesn't have `remaining_tokens`.

        """
        logger.info(f"RagZot '{self.agent_id}': Querying vector store with: '{query[:100]}...'")
        try:
            # Query ChromaDB. Multiply n_results if filtering duplicates to get enough candidates.
            num_to_fetch = self.n_results * 4 if self.no_duplicates else self.n_results
            chroma_results = self._vectorstore.query(
                query_texts=[query],  # ChromaDB expects a list of query texts
                n_results=num_to_fetch,
                include=["documents", "metadatas"],  # Specify fields to include
            )
        except Exception as e:
            logger.error(f"RagZot '{self.agent_id}': Error querying ChromaDB: {e!s}", exc_info=True)
            return ToolOutput(name=self.agent_name, call_id="", content=f"Error querying database: {e!s}", is_error=True, args={"query": query})

        if not chroma_results or not chroma_results.get("ids") or not chroma_results["ids"][0]:
            logger.info(f"RagZot '{self.agent_id}': No results from ChromaDB for query: '{query[:100]}...'")
            return ToolOutput(name=self.agent_name, call_id="", content="No results found.", results=[], args={"query": query}, messages=[])

        # Convert ChromaDB results to RefResult Pydantic models
        parsed_records: list[RefResult] = []
        num_ids_found = len(chroma_results["ids"][0])
        for i in range(num_ids_found):
            try:
                parsed_records.append(RefResult.from_chroma(chroma_results, i))
            except Exception as e:
                logger.warning(f"RagZot '{self.agent_id}': Failed to parse ChromaDB item at index {i} for query '{query[:50]}...'. Error: {e!s}")
                continue  # Skip this problematic record

        # Filter for unique documents if no_duplicates is True
        final_records_for_context: list[RefResult]
        if self.no_duplicates:
            unique_document_ids: set[str] = set()
            filtered_records: list[RefResult] = []
            for record_item in parsed_records:
                if record_item.document_id not in unique_document_ids:
                    filtered_records.append(record_item)
                    unique_document_ids.add(record_item.document_id)
                    if len(filtered_records) >= self.n_results:  # Stop once we have enough unique docs
                        break
            final_records_for_context = filtered_records
        else:
            final_records_for_context = parsed_records[:self.n_results]  # Simple truncation

        # Check token limits for the context to be sent to LLM (if model client is available)
        # This part is specific to when RagZot acts as an LLMAgent itself, not just a tool.
        # When RagZot is a tool, the calling LLMAgent would handle token limits of its own LLM.
        # For now, this token check remains, assuming it might be relevant for internal LLM use by RagZot.
        context_messages_for_llm: list[UserMessage] = []
        if hasattr(self, "_model_client") and self._model_client and hasattr(self._model_client, "client"):
            for i, rec in enumerate(final_records_for_context):
                current_rec_message = rec.as_message()
                # Check token count before adding this message.
                # This requires the LLM client to have a `remaining_tokens` method.
                try:
                    # This check might be too simplistic as it doesn't account for the main prompt/query tokens.
                    if self._model_client.client.remaining_tokens(messages=context_messages_for_llm + [current_rec_message]) < 5000:  # type: ignore
                        logger.warning(
                            f"RagZot '{self.agent_id}': Approaching token limit for LLM context. "
                            f"Truncating retrieved results from {len(final_records_for_context)} to {i} for query '{query[:50]}...'.",
                        )
                        final_records_for_context = final_records_for_context[:i]
                        break
                except AttributeError:  # If client doesn't have remaining_tokens
                    logger.debug(f"RagZot '{self.agent_id}': LLM client for model '{self._model_client.model_info.get('model')}' does not support remaining_tokens check.")
                except KeyError:  # If model is not known to autogen for token counting
                    logger.debug(f"RagZot '{self.agent_id}': Token usage could not be calculated for model '{self._model_client.model_info.get('model')}'.")
                context_messages_for_llm.append(current_rec_message)
        else:  # No model client, just use all records
            context_messages_for_llm = [rec.as_message() for rec in final_records_for_context]

        # Construct ToolOutput
        # `content` is a string concatenation for simple display or if LLM can't handle structured list.
        # `results` holds the structured list of RefResult objects.
        # `messages` holds Autogen UserMessages for potential inclusion in an LLM prompt.
        return ToolOutput(
            name=self.agent_name or "RagZotTool",  # Use agent_name if available
            call_id="",  # This is an internal query, not directly responding to an LLM tool_call_id
            content="\n\n".join([str(r) for r in final_records_for_context]),
            results=final_records_for_context,  # The list of RefResult objects
            args={"query": query},  # Arguments that led to this output
            messages=context_messages_for_llm,  # Autogen messages
            is_error=False,
        )
