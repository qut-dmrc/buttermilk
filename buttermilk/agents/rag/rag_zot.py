# Add this to the first cell or a new cell before getting the collection

import asyncio
from typing import Self

from autogen_core.tools import FunctionTool
from chromadb import Collection

from buttermilk import logger
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


class QueryResults(pydantic.BaseModel):
    query: str
    results: list[RefResult]


class RagZot(LLMAgent):
    """Retrieval Augmented Generation Agent that queries a vector store
    and fills a template with the results.
    """

    # Vector store configuration
    persist_directory: str = pydantic.Field(...)
    collection_name: str = pydantic.Field(...)
    embedding_model: str = pydantic.Field(default="text-embedding-large-exp-03-07")
    dimensionality: int = pydantic.Field(default=3072)

    _vectorstore: Collection

    # RAG query settings
    n_results: int = pydantic.Field(default=20)
    no_duplicates: bool = pydantic.Field(default=True)
    max_queries: int = 3

    @pydantic.model_validator(mode="after")
    def _load_tools(self) -> Self:
        # Initialize the vector store
        self._vectorstore = ChromaDBEmbeddings(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
            dimensionality=self.dimensionality,
        ).collection

        # Add concurrent search tool
        search_tool = FunctionTool(
            name="search",
            description="Search for information in the knowledge base about a specific topic or question",
            func=self.search,
        )

        # Add tools to the agent
        self._tools.extend([search_tool])

        return self

    async def search(self, queries: list[str]) -> list[RefResult]:
        """Execute multiple search queries concurrently and return all results."""
        if not queries or not self._vectorstore:
            return "No queries provided or vector store not initialized."

        # Limit the number of concurrent queries
        if len(queries) > self.max_queries:
            logger.warning(
                f"Limiting concurrent searches from {len(queries)} to {self.max_queries}",
            )
            queries = queries[: self.max_queries]

        # Execute searches concurrently
        search_tasks = [
            self._query_db(
                query=query,
            )
            for query in queries
        ]
        results = await asyncio.gather(*search_tasks)

        # Combine all results
        return results

    async def _query_db(self, query: str) -> QueryResults:
        results = self._vectorstore.query(
            query_texts=[query],
            n_results=self.n_results * 4 if self.no_duplicates else self.n_results,
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

        return QueryResults(query=query, results=records)
