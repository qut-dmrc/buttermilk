import asyncio
import uuid
from collections.abc import AsyncIterator, Sequence  # Added AsyncIterator
from typing import Any, Self

import chromadb
import pydantic
from chromadb import Collection, Documents, Embeddings
from chromadb.api import ClientAPI

# Add pdfminer import
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams  # For potential finer control if needed
from pydantic import BaseModel, Field, PrivateAttr
from vertexai.language_models import (
    TextEmbedding,
    TextEmbeddingInput,
    TextEmbeddingModel,
)

from buttermilk import logger

MODEL_NAME = "text-embedding-large-exp-03-07"


class InputDocument(BaseModel):
    """Represents a single input document to be processed."""

    file_path: str
    record_id: str  # Unique ID for the original document
    title: str  # Title of the original document
    metadata: dict[str, Any] = Field(default_factory=dict)  # Store arbitrary metadata


class ChunkedDocument(BaseModel):
    """Represents a single chunk derived from an InputDocument."""

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_title: str
    chunk_index: int
    chunk_text: str
    document_id: str  # References InputDocument.record_id
    embedding: Sequence[float] | Sequence[int] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def chunk_title(self) -> str:
        """Generates a title hint for the embedding model."""
        return f"{self.document_title}_{self.chunk_index}"


class ChromaDBEmbeddings(BaseModel):
    """Handles PDF text extraction, chunking, async embedding generation using Google Vertex AI,
    and storage in a ChromaDB vector store. Ensures data integrity on embedding failures.
    Includes async citation generation from initial document text.
    """

    embedding_model: str = MODEL_NAME
    task: str = "RETRIEVAL_DOCUMENT"
    collection_name: str
    dimensionality: int | None = None
    chunk_size: int = 9000
    chunk_overlap: int = 1000
    persist_directory: str
    concurrency: int = 20

    _semaphore: asyncio.Semaphore = PrivateAttr()
    _collection: Collection = PrivateAttr()
    _embedding_model: TextEmbeddingModel = PrivateAttr()
    _client: ClientAPI = PrivateAttr()

    @pydantic.model_validator(mode="after")
    def load_models(self) -> Self:
        """Initializes the embedding model and ChromaDB client."""
        logger.info(f"Loading embedding model: {self.embedding_model}")
        self._embedding_model = TextEmbeddingModel.from_pretrained(self.embedding_model)
        logger.info(f"Initializing ChromaDB client at: {self.persist_directory}")
        self._client = chromadb.PersistentClient(path=self.persist_directory)
        self._collection = self._client.get_or_create_collection(self.collection_name)
        logger.info(f"Using ChromaDB collection: {self.collection_name}")

        self._semaphore = asyncio.Semaphore(
            self.concurrency,
        )
        return self

    @property
    def collection(self) -> Collection:
        """Provides access to the ChromaDB collection."""
        if not hasattr(self, "_collection") or not self._collection:
            logger.warning("ChromaDB collection accessed before initialization.")
            # Attempt re-initialization (might be redundant if validator failed)
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
        logger.info(
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
                ] = await self._embedding_model.get_embeddings(
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
        successful_count = sum(1 for r in results if r is not None)
        failed_count = len(results) - successful_count
        logger.info(
            f"Embedding process completed. Success: {successful_count}, Failed: {failed_count}.",
        )
        return results

    async def prepare_docs(
        self,
        input_docs: AsyncIterator[InputDocument],  # Changed type hint
    ) -> Sequence[ChunkedDocument]:
        """Extracts text from PDFs, chunks by paragraphs, copies metadata,
        and handles overlaps/size limits. Accepts an async iterator of InputDocuments.
        """
        chunked_documents = []
        processed_doc_count = 0
        logger.info("Starting preparation of input documents from async iterator...")
        async for input_doc in input_docs:  # Changed to async for
            processed_doc_count += 1
            if not input_doc.file_path:
                logger.warning(
                    f"Skipping record {input_doc.record_id} (document #{processed_doc_count}): missing file_path.",
                )
                continue

            logger.debug(
                f"Processing document #{processed_doc_count}: {input_doc.file_path}",
            )
            try:
                # Consider making extract_text async if it becomes a bottleneck
                # For now, assume it's acceptable to run synchronously within the async loop
                full_text = extract_text(input_doc.file_path, laparams=LAParams())
            except Exception as e:
                logger.error(
                    f"Error extracting text from PDF {input_doc.file_path}: {e}",
                    exc_info=True,
                )
                continue

            paragraphs = full_text.split("\n\n")
            paragraphs = [p.strip() for p in paragraphs if p.strip()]

            current_chunk_index = 0
            doc_chunk_count = 0  # Count chunks for this specific document
            for paragraph in paragraphs:
                if len(paragraph) > self.chunk_size:
                    logger.debug(
                        f"Paragraph in doc {input_doc.record_id} exceeds chunk size ({len(paragraph)} > {self.chunk_size}), splitting...",
                    )
                    start = 0
                    while start < len(paragraph):
                        end = start + self.chunk_size
                        chunk_text_part = paragraph[start:end].strip()
                        if chunk_text_part:
                            chunk = ChunkedDocument(
                                document_title=input_doc.title,
                                chunk_index=current_chunk_index,
                                chunk_text=chunk_text_part,
                                document_id=input_doc.record_id,
                                metadata=input_doc.metadata.copy(),  # Copies metadata including the generated citation
                            )
                            chunked_documents.append(chunk)
                            current_chunk_index += 1
                            doc_chunk_count += 1
                        # Adjust start for overlap, ensuring it moves forward
                        start += self.chunk_size - self.chunk_overlap
                        # Ensure start doesn't get stuck if overlap is large or chunk_size is small
                        start = max(
                            start,
                            end - self.chunk_overlap + 1
                            if self.chunk_overlap < self.chunk_size
                            else end,
                        )

                elif paragraph:  # Ensure paragraph is not empty after strip
                    chunk = ChunkedDocument(
                        document_title=input_doc.title,
                        chunk_index=current_chunk_index,
                        chunk_text=paragraph,
                        document_id=input_doc.record_id,
                        metadata=input_doc.metadata.copy(),  # Copies metadata including the generated citation
                    )
                    chunked_documents.append(chunk)
                    current_chunk_index += 1
                    doc_chunk_count += 1
            logger.debug(
                f"Finished processing doc {input_doc.record_id}, generated {doc_chunk_count} chunks.",
            )

        logger.info(
            f"Generated a total of {len(chunked_documents)} chunks from {processed_doc_count} processed documents.",
        )
        return chunked_documents

    async def get_embedded_records(
        self,
        chunked_documents: Sequence[ChunkedDocument],
    ) -> Sequence[ChunkedDocument]:
        """Generates embeddings asynchronously and assigns them back to the ChunkedDocument objects."""
        if not chunked_documents:
            logger.info("No chunked documents provided to embed.")
            return []
        embedding_results = await self.embed_records(chunked_documents)
        assigned_count = 0
        failed_count = 0
        for i, embedding in enumerate(embedding_results):
            if embedding is not None:
                if i < len(chunked_documents):
                    chunked_documents[i].embedding = embedding
                    assigned_count += 1
                else:
                    logger.error(
                        f"Index mismatch: embedding index {i} out of bounds for chunked_documents (len {len(chunked_documents)})",
                    )
            else:
                failed_count += 1
                logger.warning(
                    f"Embedding failed for chunk {i} (ID: {chunked_documents[i].chunk_id if i < len(chunked_documents) else 'N/A'}), skipping assignment.",
                )
        logger.info(
            f"Finished assigning embeddings. Assigned: {assigned_count}, Failed/Skipped: {failed_count}.",
        )
        return chunked_documents

    # create_vectorstore_chromadb remains async
    async def create_vectorstore_chromadb(
        self,
        input_data: list[InputDocument | ChunkedDocument]
        | AsyncIterator[InputDocument],  # Allow AsyncIterator
    ) -> int:
        """Processes input data (Sequence or AsyncIterator), generates embeddings asynchronously if needed,
        and upserts the data into the ChromaDB collection. Ensures only records
        with successful embeddings are stored. (Async)
        """
        if not input_data:
            logger.warning("No input data provided to create_vectorstore_chromadb.")
            return 0

        processed_records: Sequence[ChunkedDocument] = []

        # Determine input type and process accordingly
        if isinstance(input_data, AsyncIterator):
            logger.info(
                "Processing input documents from async iterator: Chunking, citation generation, and embedding asynchronously...",
            )
            # Pass the async iterator directly to prepare_docs
            docs = await self.prepare_docs(input_docs=input_data)
            if not docs:
                logger.warning(
                    "No documents could be prepared from the input iterator."
                )
                return 0
            processed_records = await self.get_embedded_records(docs)

        elif isinstance(input_data, list) and input_data:
            first_item = input_data[0]
            if isinstance(first_item, ChunkedDocument):
                chunked_docs: Sequence[ChunkedDocument] = [
                    doc for doc in input_data if isinstance(doc, ChunkedDocument)
                ]
                if hasattr(first_item, "embedding") and first_item.embedding:
                    logger.info(
                        f"Using {len(chunked_docs)} pre-chunked and pre-embedded documents.",
                    )
                    processed_records = chunked_docs
                else:
                    logger.info(
                        f"Using {len(chunked_docs)} pre-chunked documents, generating embeddings asynchronously...",
                    )
                    processed_records = await self.get_embedded_records(chunked_docs)
            elif isinstance(first_item, InputDocument):
                # This case now requires collecting the list into an async iterator
                # or handling it differently. For simplicity, let's assume
                # if a list of InputDocument is passed, it's small enough to collect.
                # A more robust solution might involve a separate pathway or
                # converting the list to an async iterator internally.
                logger.warning(
                    "Processing a list of InputDocument. Consider passing an AsyncIterator for large datasets."
                )

                async def _list_to_async_iter(items):
                    for item in items:
                        yield item
                        await asyncio.sleep(0)  # Yield control briefly

                input_docs_iter = _list_to_async_iter([
                    doc for doc in input_data if isinstance(doc, InputDocument)
                ])
                docs = await self.prepare_docs(input_docs=input_docs_iter)
                if not docs:
                    logger.warning(
                        "No documents could be prepared from the input list."
                    )
                    return 0
                processed_records = await self.get_embedded_records(docs)

            else:
                logger.error(
                    f"Invalid input type in list: {type(first_item)}. Expected InputDocument or ChunkedDocument.",
                )
                return 0
        else:
            logger.error(
                f"Invalid input type: {type(input_data)}. Expected list or AsyncIterator.",
            )
            return 0

        records_to_upsert = [
            rec for rec in processed_records if rec.embedding is not None
        ]

        if len(records_to_upsert) < len(processed_records):
            logger.warning(
                f"Excluded {len(processed_records) - len(records_to_upsert)} records due to missing/failed embeddings before upserting.",
            )

        if not records_to_upsert:
            logger.warning(
                "No valid records with embeddings available to upsert into ChromaDB.",
            )
            return 0

        ids = [rec.chunk_id for rec in records_to_upsert]
        documents = [rec.chunk_text for rec in records_to_upsert]
        # Ensure embeddings are lists, not tuples or other sequences if ChromaDB requires lists
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
            # Ensure the generated citation (if present) is included
            meta.update({k: v for k, v in rec.metadata.items() if v is not None})
            metadatas.append(meta)

        logger.info(
            f"Upserting {len(ids)} chunks into ChromaDB collection '{self.collection_name}'...",
        )
        try:
            # Consider wrapping in asyncio.to_thread if upsert is blocking
            # ChromaDB client operations might be network-bound or CPU-bound depending on setup
            await asyncio.to_thread(
                self.collection.upsert,
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents,
            )
            logger.info(f"Upsert complete. Added/Updated {len(ids)} items.")
            return len(ids)
        except Exception as e:
            logger.error(f"Failed to upsert data into ChromaDB: {e}", exc_info=True)
            return 0
