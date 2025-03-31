# Add this to the first cell or a new cell before getting the collection

import asyncio
from typing import Self

from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool

from buttermilk import logger
from buttermilk._core.contract import AgentInput, AgentOutput
from buttermilk.agents.llm import LLMAgent
from buttermilk.data.vector import ChromaDBEmbeddings

PERSIST_DIR = "/home/nic/data/prosocial_zot/files"
COLLECTION_NAME = "prosocial_zot"
MODEL_NAME = "text-embedding-large-exp-03-07"  # From your vector.py
TASK_FOR_QUERY = "RETRIEVAL_QUERY"  # Use RETRIEVAL_QUERY for query embedding
DIMENSIONALITY = 3072  # From your vector.py

import pydantic


class RefResult(pydantic.BaseModel):
    id: str
    full_text: str
    uri: str | None
    chunk_index: int
    citation: str | None = ""
    document_id: str
    document_title: str | None
    doi_or_url: str | None
    _extra_zotero_data: str

    def __str__(self) -> str:
        formatted_output = f"**Title:** {self.document_title}\n"
        formatted_output += f"**Document ID:** {self.document_id} ({self.doi_or_url}, chunk {self.chunk_index})\n"
        formatted_output += f"**Citation:** {self.citation}\n"
        formatted_output += f"**Full Text:** {self.full_text}\n"

        return formatted_output

    @classmethod
    def from_chroma(cls, results, index) -> "RefResult":
        result = {
            x: results[x][0][index]
            for x in [
                "ids",
                "metadatas",
                "documents",
            ]
            if results[x]
        }
        meta = {
            k: v for k, v in result["metadatas"].items() if k in RefResult.model_fields
        }
        return RefResult(
            id=result["ids"],
            full_text=result["documents"],
            uri=result.get("uri", ""),
            **meta,
        )


class RagZot(LLMAgent):
    """Retrieval Augmented Generation Agent that queries a vector store
    and fills a template with the results.
    """

    # Vector store configuration
    persist_directory: str = pydantic.Field(...)
    collection_name: str = pydantic.Field(...)
    embedding_model: str = pydantic.Field(default="text-embedding-large-exp-03-07")
    dimensionality: int = pydantic.Field(default=3072)

    # RAG query settings
    n_results: int = pydantic.Field(default=20)
    no_duplicates: bool = pydantic.Field(default=True)
    max_queries: int = 3

    _vectorstore: ChromaDBEmbeddings

    @pydantic.model_validator(mode="after")
    def _load_vectorstore(self) -> Self:
        # Initialize the vector store
        self._vectorstore = ChromaDBEmbeddings(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
            dimensionality=self.dimensionality,
        )
        # Add search tool
        search_tool = FunctionTool(
            name="search",
            description="Search for information in the knowledge base about a specific topic or question",
            fn=self.search_knowledge_base,
            parameters={
                "query": {
                    "type": "string",
                    "description": "The search query for finding relevant information",
                },
            },
        )

        # Add concurrent search tool
        concurrent_search_tool = FunctionTool(
            name="concurrent_search",
            description="Search for multiple topics concurrently to find relevant information",
            fn=self.concurrent_search,
            parameters={
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of search queries to execute concurrently",
                },
            },
        )

        # Add tools to the agent
        self._tools.extend([search_tool, concurrent_search_tool])

        return self

    async def search_knowledge_base(self, query: str) -> str:
        """Search the vector store for information about a specific query."""
        if not query or not self._vectorstore:
            return "No query provided or vector store not initialized."

        try:
            results = await self._vectorstore.collection.query(
                query_texts=[query],
                n_results=self.n_results,
                include=["documents", "metadatas"],
            )

            # Format the results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                # Create RefResult for consistent formatting
                ref = RefResult(
                    id=results["ids"][0][i],
                    full_text=results["documents"][0][i],
                    uri=results["metadatas"][0][i].get("uri", ""),
                    chunk_index=results["metadatas"][0][i].get("chunk_index", 0),
                    citation=results["metadatas"][0][i].get("citation", ""),
                    document_id=results["metadatas"][0][i].get(
                        "document_id",
                        "Unknown",
                    ),
                    document_title=results["metadatas"][0][i].get("document_title", ""),
                    doi_or_url=results["metadatas"][0][i].get("doi_or_url", ""),
                    _extra_zotero_data="",
                )
                formatted_results.append(str(ref))

            return f"### Search Results for: '{query}'\n\n" + "\n\n".join(
                formatted_results,
            )

        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return f"Error searching for '{query}': {e!s}"

    async def concurrent_search(self, queries: list[str]) -> str:
        """Execute multiple search queries concurrently and return all results."""
        if not queries or not self._vectorstore:
            return "No queries provided or vector store not initialized."

        # Limit the number of concurrent queries
        if len(queries) > self.max_queries:
            logger.warning(
                f"Limiting concurrent searches from {len(queries)} to {self.max_queries}",
            )
            queries = queries[: self.max_queries]

        try:
            # Execute searches concurrently
            search_tasks = [self.search_knowledge_base(query) for query in queries]
            results = await asyncio.gather(*search_tasks)

            # Combine all results
            combined_results = "\n\n".join(results)
            return combined_results

        except Exception as e:
            logger.error(f"Error executing concurrent searches: {e}")
            return f"Error executing concurrent searches: {e!s}"

    async def _query_and_format_results(self, query: str) -> str:
        results = await self._query_db(query)

        # Format results for template
        formatted_results = "\n\n".join([str(r) for r in results])
        return formatted_results

    async def _query_db(self, query: str) -> list[RefResult]:
        results = self._vectorstore.collection.query(
            query_texts=[query],
            n_results=self.n_results if self.no_duplicates else self.n_results * 4,
            include=[
                "documents",
                "metadatas",
            ],
        )
        records = [
            RefResult.from_chroma(results, i) for i in range(len(results["ids"][0]))
        ]

        if self.no_duplicates:
            already_included = []
            output = []
            for record in records:
                if record.document_id in already_included:
                    continue
                already_included.append(record.document_id)
                output.append(record)
                if len(output) >= 20:
                    break
            records = output

        return records

    async def _process(
        self,
        input_data: AgentInput,
        cancellation_token: CancellationToken | None = None,
        **kwargs,
    ) -> AgentOutput:
        """Process the input by querying the vector store and filling the template."""
        # Get the query from input
        query = input_data.inputs.get("prompt", "")

        if query and self._vectorstore:
            try:
                # Query the vector store
                results = await self._query_and_format_results(query)
                input_data.inputs["rag"] = results
            except Exception as e:
                logger.error(f"Error querying vector store: {e}")
                input_data.inputs["rag"] = "Error retrieving relevant information."

        # Call the parent's _process method to handle the template filling
        return await super()._process(input_data, cancellation_token, **kwargs)
