# Add this to the first cell or a new cell before getting the collection

import asyncio
from typing import Any, Self

from autogen_core import FunctionCall
from autogen_core.models._types import UserMessage
from autogen_core.tools import FunctionTool
from chromadb import Collection

from buttermilk import logger
from buttermilk._core.config import ToolConfig
from buttermilk._core.contract import ToolOutput
from buttermilk.agents.llm import LLMAgent, Tool, ToolSchema
from buttermilk.data.vector import ChromaDBEmbeddings

TASK_FOR_QUERY = "RETRIEVAL_QUERY"  # Use RETRIEVAL_QUERY for query embedding

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

    def as_message(self, source="search") -> UserMessage:
        return UserMessage(content=str(self), source=source)

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
        meta = {k: v for k, v in result["metadatas"].items() if k in RefResult.model_fields}
        return RefResult(
            id=result["ids"],
            full_text=result["documents"],
            uri=result.get("uri", ""),
            **meta,
        )


class Reference(pydantic.BaseModel):
    summary: str
    citation: str


class ResearchResult(pydantic.BaseModel):
    literature: list[Reference]
    response: str


class RagZot(LLMAgent, ToolConfig):
    """Retrieval Augmented Generation Agent that queries a vector store
    and fills a template with the results.
    """

    _chromadb: ChromaDBEmbeddings
    _vectorstore: Collection
    _output_model: type[pydantic.BaseModel] | None = ResearchResult

    # RAG query settings
    n_results: int = pydantic.Field(default=20)
    no_duplicates: bool = pydantic.Field(default=True)
    max_queries: int = 3

    @pydantic.model_validator(mode="after")
    def _load_tools(self) -> Self:
        for data_conf in self.data.values():
            if data_conf.type == "chromadb":
                self._chromadb = ChromaDBEmbeddings(**data_conf.model_dump())
                self._vectorstore = self._chromadb.collection
                break

        # Add concurrent search tool
        search_tool = FunctionTool(
            name="search",
            description=self.description,
            func=self._run,
        )

        # Add tools to the agent
        self._tools_list.extend([search_tool])

        if not self._vectorstore:
            raise ValueError("No vector store found. Please check your configuration.")

        return self

    def get_functions(self) -> list[Any]:
        return self._tools_list

    @property
    def config(self) -> list[FunctionCall | Tool | ToolSchema | FunctionTool]:
        return self._tools_list

    async def _run(self, queries: list[str]) -> list[ToolOutput]:
        """Execute multiple search queries concurrently and return all results."""
        if not queries or not self._vectorstore:
            raise ValueError("No queries provided or vector store not initialized.")

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

        return results

    async def _query_db(self, query: str) -> ToolOutput:
        results = self._vectorstore.query(
            query_texts=[query],
            n_results=self.n_results * 4 if self.no_duplicates else self.n_results,
            include=[
                "documents",
                "metadatas",
            ],
        )
        records = [RefResult.from_chroma(results, i) for i in range(len(results["ids"][0]))]

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

        messages = []
        for i, rec in enumerate(records):
            messages.append(rec.as_message())
            try:
                if self._model_client.client.remaining_tokens(messages=messages) < 5000:
                    logger.warning(
                        f"RAG search query exceeded token limit. Truncating to {i - 1} results.",
                    )
                    records = records[: i - 1]
                    break
            except KeyError:
                # This happens when the model we are using is not known to autogen.
                logger.debug(f"Tried to calculate token usage but couldn't: {self._model_client.model_info}")

        records = records[: self.n_results]
        return ToolOutput(
            name=self.name,
            call_id="",
            content="\n\n".join([str(r) for r in records]),
            results=records,
            args=dict(query=query),
            messages=[r.as_message() for r in records],
            is_error=False,
        )
