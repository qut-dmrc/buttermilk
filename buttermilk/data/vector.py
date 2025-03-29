import asyncio
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, Self, TypeVar

import chromadb
import hydra
import pydantic
from chromadb import Collection, Documents, Embeddings
from chromadb.api import ClientAPI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, PrivateAttr
from vertexai.language_models import (
    TextEmbedding,
    TextEmbeddingInput,
    TextEmbeddingModel,
)

from buttermilk import logger
from buttermilk.bm import BM

MODEL_NAME = "text-embedding-large-exp-03-07"
DEFAULT_UPSERT_BATCH_SIZE = 100

T = TypeVar("T")


class InputDocument(BaseModel):
    """Represents a single input document with its text content."""

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for the original document",
    )
    file_path: str = Field(..., description="Path to the input document")
    full_text: str
    title: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkedDocument(BaseModel):
    """Represents a single chunk derived from an InputDocument."""

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_title: str
    chunk_index: int
    chunk_text: str
    document_id: str
    embedding: Sequence[float] | Sequence[int] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def chunk_title(self) -> str:
        """Generates a title hint for the embedding model."""
        return f"{self.document_title}_{self.chunk_index}"


async def _batch_iterator(
    aiter: AsyncIterator[T],
    batch_size: int,
) -> AsyncIterator[list[T]]:
    """Batches items from an async iterator."""
    batch = []
    async for item in aiter:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class ChromaDBEmbeddings(BaseModel):
    """Handles chunking using LangChain, async embedding generation using Google Vertex AI,
    and storage in a ChromaDB vector store via an async pipeline.
    """

    embedding_model: str = MODEL_NAME
    task: str = "RETRIEVAL_DOCUMENT"
    collection_name: str
    dimensionality: int | None = Field(default=3072)
    chunk_size: int = Field(default=9000)
    chunk_overlap: int = Field(default=1000)
    persist_directory: str
    concurrency: int = Field(default=20)
    upsert_batch_size: int = DEFAULT_UPSERT_BATCH_SIZE
    embedding_batch_size: int = Field(default=1)

    _semaphore: asyncio.Semaphore = PrivateAttr()
    _collection: Collection = PrivateAttr()
    _embedding_model: TextEmbeddingModel = PrivateAttr()
    _client: ClientAPI = PrivateAttr()
    _text_splitter: RecursiveCharacterTextSplitter = PrivateAttr()

    @pydantic.model_validator(mode="after")
    def load_models(self) -> Self:
        """Initializes the embedding model, ChromaDB client, and text splitter."""
        logger.info(f"Loading embedding model: {self.embedding_model}")
        self._embedding_model = TextEmbeddingModel.from_pretrained(self.embedding_model)
        logger.info(f"Initializing ChromaDB client at: {self.persist_directory}")
        self._client = chromadb.PersistentClient(path=self.persist_directory)
        self._collection = self._client.get_or_create_collection(self.collection_name)
        logger.info(f"Using ChromaDB collection: {self.collection_name}")
        self._semaphore = asyncio.Semaphore(self.concurrency)
        # Initialize the text splitter here
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,  # Use character length by default
            # You can customize separators if needed:
            # separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=False,  # Don't need start index metadata
        )
        logger.info(
            f"Initialized RecursiveCharacterTextSplitter (chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})",
        )
        return self

    @property
    def collection(self) -> Collection:
        """Provides access to the ChromaDB collection."""
        if not hasattr(self, "_collection") or not self._collection:
            logger.warning("ChromaDB collection accessed before initialization.")
            if not hasattr(self, "_client") or not self._client:
                self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = self._client.get_or_create_collection(
                self.collection_name,
            )
        return self._collection

    async def embed_records(
        self,
        chunked_documents: Sequence[ChunkedDocument],
    ) -> list[list[float | int] | None]:
        """Generates embeddings asynchronously for a sequence of ChunkedDocument objects."""
        if not chunked_documents:
            return []
        logger.debug(
            f"Generating embeddings asynchronously for {len(chunked_documents)} chunks.",
        )
        inputs = [
            TextEmbeddingInput(
                text=chunk.chunk_text,
                task_type=self.task,
                title=chunk.chunk_title,
            )
            for chunk in chunked_documents
        ]
        return await self._embed(inputs)

    async def embed_documents(self, texts: Documents) -> list[list[float | int] | None]:
        """Generates embeddings asynchronously for a list of raw text strings."""
        if not texts:
            return []
        logger.info(
            f"Generating embeddings asynchronously for {len(texts)} raw text documents.",
        )
        inputs = [TextEmbeddingInput(text=text, task_type=self.task) for text in texts]
        return await self._embed(inputs)

    async def embed_query(self, query: str) -> list[float | int] | None:
        """Generates an embedding asynchronously for a single query string."""
        logger.debug(f"Generating embedding for query: '{query[:50]}...'")
        inputs = [TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")]
        results = await self._embed(inputs)
        return results[0] if results else None

    async def _run_embedding_task(
        self,
        chunk_input: TextEmbeddingInput,
        index: int,
    ) -> list[float | int] | None:
        """Helper coroutine to run a single embedding task and handle errors."""
        async with self._semaphore:
            kwargs = dict(
                output_dimensionality=self.dimensionality,
                auto_truncate=False,
            )
            try:
                embeddings_result: list[
                    TextEmbedding
                ] = await self._embedding_model.get_embeddings_async(
                    [chunk_input],
                    **kwargs,
                )
                if embeddings_result:
                    return embeddings_result[0].values
                logger.warning(f"No embedding result returned for input {index}.")
                return None
            except Exception as exc:
                logger.error(
                    f"Error getting embedding for input {index}: {exc}",
                    exc_info=True,
                )
                return None

    async def _embed(
        self,
        inputs: Sequence[TextEmbeddingInput],
    ) -> list[list[float | int] | None]:
        """Internal async method to call the Vertex AI embedding model concurrently."""
        if not inputs:
            return []
        tasks = [
            self._run_embedding_task(chunk_input, i)
            for i, chunk_input in enumerate(inputs)
        ]
        results: list[list[float | int] | None] = await asyncio.gather(*tasks)
        return results

    async def prepare_docs(
        self,
        input_docs: AsyncIterator[InputDocument],
        processor: Callable,
    ) -> AsyncIterator[ChunkedDocument]:
        """Chunks text from InputDocuments using LangChain's RecursiveCharacterTextSplitter,
        copies metadata, and yields ChunkedDocument objects asynchronously.
        """
        processed_doc_count = 0
        total_chunks_yielded = 0
        logger.info(
            "Starting preparation (chunking with LangChain splitter) of input documents...",
        )
        async for input_doc in input_docs:
            processed_doc_count += 1
            logger.debug(
                f"Processing document #{processed_doc_count}: {input_doc.record_id} ({input_doc.title})",
            )

            full_text = input_doc.full_text
            if not full_text:
                logger.warning(
                    f"Skipping record {input_doc.record_id} (document #{processed_doc_count}): missing full_text.",
                )
                continue

            # process (get citation or whatever)
            input_doc = await processor(input_doc)

            # Use the initialized text splitter
            # Note: split_text is synchronous. If it becomes a bottleneck for very large
            # documents or complex splitting, consider running it in a thread pool executor.
            # For typical document sizes, this should be acceptable within an async loop.
            try:
                text_chunks = self._text_splitter.split_text(full_text)
            except Exception as e:
                logger.error(
                    f"Error splitting text for doc {input_doc.record_id}: {e}",
                    exc_info=True,
                )
                continue  # Skip this document if splitting fails

            doc_chunk_count = 0
            for i, text_chunk in enumerate(text_chunks):
                # Ensure the chunk is not just whitespace
                if not text_chunk.strip():
                    continue

                chunk = ChunkedDocument(
                    document_title=input_doc.title,
                    chunk_index=i,  # Use the index from the splitter's output list
                    chunk_text=text_chunk.strip(),  # Store the stripped chunk
                    document_id=input_doc.record_id,
                    metadata=input_doc.metadata.copy(),
                )
                yield chunk
                total_chunks_yielded += 1
                doc_chunk_count += 1

            logger.debug(
                f"Finished processing doc {input_doc.record_id}, yielded {doc_chunk_count} chunks.",
            )

        logger.info(
            f"Finished preparation. Yielded a total of {total_chunks_yielded} chunks from {processed_doc_count} processed documents.",
        )

    async def get_embedded_records(
        self,
        chunked_documents_iter: AsyncIterator[ChunkedDocument],
        batch_size: int | None = None,
    ) -> AsyncIterator[ChunkedDocument]:
        """Generates embeddings asynchronously for chunks from an iterator and yields
        ChunkedDocument objects with embeddings assigned. Processes in batches.
        """
        embed_batch_size = batch_size or self.embedding_batch_size
        total_processed = 0
        total_succeeded = 0
        total_failed = 0
        logger.info(
            f"Starting embedding process for chunk iterator (batch size: {embed_batch_size})...",
        )

        async for batch in _batch_iterator(chunked_documents_iter, embed_batch_size):
            if not batch:
                continue

            batch_start_index = total_processed
            total_processed += len(batch)
            logger.debug(
                f"Processing embedding batch #{batch_start_index // embed_batch_size + 1} (chunks {batch_start_index + 1}-{total_processed})...",
            )

            embedding_results = await self.embed_records(batch)
            batch_succeeded = 0
            batch_failed = 0

            for i, embedding in enumerate(embedding_results):
                chunk = batch[i]
                if embedding is not None:
                    chunk.embedding = embedding
                    yield chunk
                    batch_succeeded += 1
                else:
                    batch_failed += 1
                    logger.warning(
                        f"Embedding failed for chunk {i + batch_start_index} (ID: {chunk.chunk_id}), skipping.",
                    )
            total_succeeded += batch_succeeded
            total_failed += batch_failed
            logger.debug(
                f"Finished embedding batch. Success: {batch_succeeded}, Failed: {batch_failed}.",
            )

        logger.info(
            f"Finished embedding process. Total Chunks Processed: {total_processed}, Succeeded: {total_succeeded}, Failed: {total_failed}.",
        )

    async def create_vectorstore_chromadb(
        self,
        input_docs_iter: Any,
        processor: Any,
    ) -> int:
        """Processes input documents from an async iterator through a pipeline:
        chunking -> embedding -> batch upserting into ChromaDB.

        Args:
            input_docs_iter: An async iterator yielding InputDocument objects.

        Returns:
            The total number of chunked records successfully upserted into ChromaDB.

        """
        logger.info(
            f"Starting vector store creation pipeline for collection '{self.collection_name}' (upsert batch size: {self.upsert_batch_size}).",
        )

        # 1. Prepare (Chunk) Documents using LangChain splitter
        chunked_docs_iter = self.prepare_docs(input_docs_iter, processor)

        # 2. Get Embeddings (in batches)
        embedded_docs_iter = self.get_embedded_records(
            chunked_docs_iter,
            batch_size=self.embedding_batch_size,
        )

        # 3. Batch Upsert to ChromaDB
        total_upserted_count = 0
        batch_num = 0
        async for batch in _batch_iterator(embedded_docs_iter, self.upsert_batch_size):
            batch_num += 1
            if not batch:
                continue

            records_to_upsert = [rec for rec in batch if rec.embedding is not None]

            if not records_to_upsert:
                logger.warning(
                    f"Upsert batch #{batch_num} was empty or contained only records with failed embeddings.",
                )
                continue

            ids = [rec.chunk_id for rec in records_to_upsert]
            documents = [rec.chunk_text for rec in records_to_upsert]
            embeddings_list: Embeddings = [
                list(rec.embedding) for rec in records_to_upsert if rec.embedding
            ]

            metadatas = []
            for rec in records_to_upsert:
                meta = {
                    "document_title": rec.document_title,
                    "chunk_index": rec.chunk_index,
                    "document_id": rec.document_id,
                }
                meta.update({k: v for k, v in rec.metadata.items() if v is not None})
                metadatas.append(meta)

            logger.info(
                f"Upserting batch #{batch_num} ({len(ids)} chunks) into ChromaDB collection '{self.collection_name}'...",
            )
            try:
                await asyncio.to_thread(
                    self.collection.upsert,
                    ids=ids,
                    embeddings=embeddings_list,
                    metadatas=metadatas,
                    documents=documents,
                )
                upserted_in_batch = len(ids)
                total_upserted_count += upserted_in_batch
                logger.debug(
                    f"Upsert batch #{batch_num} complete. Added/Updated {upserted_in_batch} items.",
                )
            except Exception as e:
                logger.error(
                    f"Failed to upsert batch #{batch_num} into ChromaDB: {e}",
                    exc_info=True,
                )

        logger.info(
            f"Vector store creation pipeline finished. Total chunks successfully upserted: {total_upserted_count}.",
        )
        return total_upserted_count


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg) -> None:
    # Hydra will automatically instantiate the objects
    objs = hydra.utils.instantiate(cfg)
    bm: BM = objs.bm
    vectoriser = objs.vectoriser
    input_docs = objs.input_docs
    processor = objs.processor

    # give our flows a little longer to set up
    loop = asyncio.get_event_loop()
    loop.slow_callback_duration = 35.0
    input_docs_iter = objs.input_docs
    loop.run_until_complete(
        objs.vectoriser.create_vectorstore_chromadb(
            input_docs_iter.get_all_records(),
            processor.process,
        ),
    )


if __name__ == "__main__":
    main()
