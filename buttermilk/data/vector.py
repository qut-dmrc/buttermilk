import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Self, TypeVar  # Corrected import for Tuple

import chromadb
import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import pydantic
from chromadb import Collection, Embeddings
from chromadb.api import ClientAPI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, PrivateAttr
from vertexai.language_models import (
    TextEmbedding,
    TextEmbeddingInput,
    TextEmbeddingModel,
)

from buttermilk import logger
from buttermilk.bm import BM, bm

MODEL_NAME = "text-embedding-large-exp-03-07"
DEFAULT_UPSERT_BATCH_SIZE = 10  # Still used for failed batch saving logic if needed
FAILED_BATCH_DIR = "failed_upsert_batches"

T = TypeVar("T")


# --- Pydantic Models ---
class ChunkedDocument(BaseModel):
    """Represents a single chunk derived from an InputDocument."""

    model_config = pydantic.ConfigDict(extra="ignore")

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


class InputDocument(BaseModel):
    """Represents a single input document with its text content."""

    model_config = pydantic.ConfigDict(extra="ignore")

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str = Field(...)
    record_path: str = Field(default="")
    chunks_path: str = Field(
        default="",
        description="Path to PyArrow file with chunks and embeddings.",
    )
    full_text: str = Field(default="")
    chunks: list[ChunkedDocument] = Field(default_factory=list)
    title: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# --- Type Aliases ---
ProcessorCallable = Callable[[InputDocument], Awaitable[InputDocument]]


# --- Helper Functions ---
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


def _sanitize_metadata_for_chroma(
    metadata: dict[str, Any],
) -> dict[str, str | int | float | bool]:
    """Converts metadata values to types supported by ChromaDB."""
    sanitized = {}
    if not isinstance(metadata, dict):
        logger.warning(f"Metadata is not a dict: {metadata}. Skipping sanitization.")
        return {}

    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            sanitized[k] = v
        elif v is None:
            continue
        elif isinstance(v, (list, dict, BaseModel)):
            try:
                if isinstance(v, BaseModel):
                    json_str = v.model_dump_json()
                else:
                    json_str = json.dumps(v, ensure_ascii=False)
                sanitized[k] = json_str
            except (TypeError, Exception) as e:
                logger.warning(
                    f"Could not JSON serialize metadata value for key '{k}': {type(v)}. Error: {e}. Skipping key.",
                )
        else:
            try:
                sanitized[k] = str(v)
            except Exception as e:
                logger.warning(
                    f"Could not convert metadata value for key '{k}' to string: {type(v)}. Error: {e}. Skipping key.",
                )
    return sanitized


# --- Add list_to_async_iterator helper ---
async def list_to_async_iterator(items: list[T]) -> AsyncIterator[T]:
    """Converts a list into an asynchronous iterator."""
    for item in items:
        yield item
        await asyncio.sleep(0)  # Yield control briefly


# --- Core Embedding and DB Interaction Class ---
class ChromaDBEmbeddings(BaseModel):
    """Handles configuration, embedding model interaction, and ChromaDB connection."""

    model_config = pydantic.ConfigDict(extra="ignore")

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
    arrow_save_dir: str = Field(...)

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
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=False,
        )
        logger.info(
            f"Initialized RecursiveCharacterTextSplitter (chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})",
        )
        Path(FAILED_BATCH_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.arrow_save_dir).mkdir(parents=True, exist_ok=True)
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

    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Provides access to the text splitter instance."""
        return self._text_splitter

    async def embed_document(
        self,
        input_doc: InputDocument,
    ) -> InputDocument | None:
        """Generates embeddings asynchronously for chunks within an InputDocument,
        assigns them, and saves the result to a Parquet file.
        """
        if not input_doc.chunks:
            logger.warning(
                f"No chunks found for document {input_doc.record_id}, cannot embed or save.",
            )
            return None

        logger.debug(
            f"Generating embeddings asynchronously for doc {input_doc.record_id} with {len(input_doc.chunks)} chunks.",
        )

        embeddings_input: list[tuple[int, TextEmbeddingInput]] = []
        for chunk in input_doc.chunks:
            if chunk.chunk_index is not None:
                embeddings_input.append((
                    chunk.chunk_index,
                    TextEmbeddingInput(
                        text=chunk.chunk_text,
                        task_type=self.task,
                        title=chunk.chunk_title,
                    ),
                ))
            else:
                logger.warning(
                    f"Chunk missing index in doc {input_doc.record_id}, skipping embedding for this chunk.",
                )

        embedding_results = await self._embed(embeddings_input)

        successful_embeddings = 0
        for idx, embedding in embedding_results:
            try:
                list_index = next(
                    i
                    for i, chk in enumerate(input_doc.chunks)
                    if chk.chunk_index == idx
                )
                if embedding is not None:
                    input_doc.chunks[list_index].embedding = embedding
                    successful_embeddings += 1
                else:
                    logger.warning(
                        f"Embedding failed for chunk index {idx} in doc {input_doc.record_id}",
                    )
            except StopIteration:
                logger.error(
                    f"Could not find chunk with index {idx} in doc {input_doc.record_id} to assign embedding.",
                )

        if successful_embeddings == 0 and len(input_doc.chunks) > 0:
            logger.error(
                f"All embeddings failed for document {input_doc.record_id}. Skipping save.",
            )
            return None

        arrow_file_path = Path(self.arrow_save_dir) / f"{input_doc.record_id}.parquet"
        input_doc.chunks_path = arrow_file_path.as_posix()

        try:
            await asyncio.to_thread(
                self._write_document_to_parquet,
                input_doc,
                arrow_file_path,
            )
            logger.info(
                f"Successfully saved document chunks and embeddings to {arrow_file_path}",
            )
            return input_doc
        except Exception as e:
            logger.error(
                f"Failed to save document {input_doc.record_id} to Parquet file {arrow_file_path}: {e} {e.args=}",
            )
            input_doc.chunks_path = ""
            return None

    def _write_document_to_parquet(self, doc: InputDocument, file_path: Path):
        """Synchronous helper to write InputDocument chunks to a Parquet file."""
        if not doc.chunks:
            logger.warning(
                f"Attempted to write empty chunks for doc {doc.record_id} to {file_path}. Skipping.",
            )
            return

        data = {
            "chunk_id": [c.chunk_id for c in doc.chunks],
            "document_id": [c.document_id for c in doc.chunks],
            "document_title": [c.document_title for c in doc.chunks],
            "chunk_index": [c.chunk_index for c in doc.chunks],
            "chunk_text": [c.chunk_text for c in doc.chunks],
            "embedding": [
                list(c.embedding) if c.embedding is not None else None
                for c in doc.chunks
            ],
            "chunk_metadata": [
                json.dumps(c.metadata) if c.metadata else None for c in doc.chunks
            ],
        }

        embedding_type = pa.list_(pa.float32())
        if self.dimensionality:
            embedding_type = pa.list_(pa.float32(), self.dimensionality)

        schema = pa.schema([
            pa.field("chunk_id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("document_title", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("chunk_text", pa.string()),
            pa.field("embedding", embedding_type),
            pa.field("chunk_metadata", pa.string()),
        ])

        table = pa.Table.from_pydict(data, schema=schema)

        doc_meta_serializable = {
            "record_id": doc.record_id,
            "title": doc.title,
            "file_path": doc.file_path,
            "record_path": doc.record_path,
            "metadata": json.dumps(doc.metadata),
        }
        arrow_metadata = {
            k.encode("utf-8"): str(v).encode("utf-8")
            for k, v in doc_meta_serializable.items()
        }

        final_schema = table.schema.with_metadata(arrow_metadata)
        table = table.cast(final_schema)

        pq.write_table(table, file_path, compression="snappy")

    async def _run_embedding_task(
        self,
        chunk_input: TextEmbeddingInput,
        index: int,
    ) -> tuple[int, list[float | int] | None]:
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
                    return index, embeddings_result[0].values
                logger.warning(f"No embedding result returned for input {index}.")
                return index, None
            except Exception as exc:
                logger.error(
                    f"Error getting embedding for input {index}: {exc} {exc.args=}",
                )
                return index, None

    async def _embed(
        self,
        inputs: Sequence[tuple[int, TextEmbeddingInput]],
    ) -> list[tuple[int, list[float | int] | None]]:
        """Internal async method to call the Vertex AI embedding model concurrently."""
        if not inputs:
            return []
        tasks = [
            self._run_embedding_task(chunk_input=chunk_input, index=idx)
            for idx, chunk_input in inputs
        ]
        results: list[tuple[int, list[float | int] | None]] = await asyncio.gather(
            *tasks,
        )
        return results

    # --- DB Interaction ---
    def check_document_exists(self, document_id: str) -> bool:
        """Checks if a document with the given ID already exists in the collection."""
        if not document_id:
            return False
        try:
            results = self.collection.get(
                where={"document_id": document_id},
                limit=1,
                include=[],
            )
            exists = len(results.get("ids", [])) > 0
            if exists:
                logger.debug(f"Document ID '{document_id}' found in ChromaDB.")
            return exists
        except Exception as e:
            logger.error(
                f"Error checking existence of document ID '{document_id}' in ChromaDB: {e} {e.args=}",
            )
            return False


# --- Async Pipeline Stages ---
async def preprocess_documents(
    doc_iterator: AsyncIterator[InputDocument],
    extractor: Callable[[str], Awaitable[str | None]],
) -> AsyncIterator[InputDocument]:
    """Extracts full text for documents that don't have it."""
    processed_count = 0
    async for doc in doc_iterator:
        if doc.full_text:
            yield doc
            continue
        try:
            full_text = await asyncio.to_thread(extractor, doc.file_path)
            if full_text is None:
                logger.warning(
                    f"Text extraction failed for {doc.record_id} ({doc.file_path}). Skipping doc.",
                )
                continue

            doc.full_text = full_text
            processed_count += 1
            yield doc
        except Exception as e:
            logger.error(
                f"Text extraction processor failed for doc {doc.record_id}: {e} {e.args=}",
            )
            logger.warning(
                f"Skipping document {doc.record_id} due to extraction processor error.",
            )
    logger.info(
        f"Text Extractor finished. Processed {processed_count} documents needing extraction.",
    )


async def process_documents(
    doc_iterator: AsyncIterator[InputDocument],
    processor: ProcessorCallable,
) -> AsyncIterator[InputDocument]:
    """Applies an async processor to each document."""
    processed_count = 0
    async for doc in doc_iterator:
        try:
            processed_doc = await processor(doc)
            processed_count += 1
            yield processed_doc
        except Exception as e:
            logger.error(
                f"Processor failed for doc {doc.record_id}: {e} {e.args=}",
            )
            logger.warning(
                f"Skipping document {doc.record_id} due to processor error, yielding original.",
            )
            yield doc
    logger.info(f"Processor finished. Processed {processed_count} documents.")


async def chunk_documents(
    doc_iterator: AsyncIterator[InputDocument],
    text_splitter: RecursiveCharacterTextSplitter,
) -> AsyncIterator[InputDocument]:
    """Chunks documents and adds the chunks list to the InputDocument."""
    processed_doc_count = 0
    async for doc in doc_iterator:
        processed_doc_count += 1
        if not doc.full_text:
            logger.warning(
                f"Skipping chunking for record {doc.record_id} (doc #{processed_doc_count}) due to missing full_text.",
            )
            continue
        try:
            text_chunks = await asyncio.to_thread(
                text_splitter.split_text,
                doc.full_text,
            )
            doc.chunks = []
            doc_chunk_count = 0
            for i, text_chunk in enumerate(text_chunks):
                if not text_chunk.strip():
                    continue
                doc.chunks.append(
                    ChunkedDocument(
                        document_title=doc.title,
                        chunk_index=i,
                        chunk_text=text_chunk.strip(),
                        document_id=doc.record_id,
                        metadata=doc.metadata.copy(),
                    ),
                )
                doc_chunk_count += 1

            if doc_chunk_count > 0:
                logger.debug(
                    f"Finished chunking doc {doc.record_id}, created {doc_chunk_count} chunks.",
                )
                yield doc
            else:
                logger.warning(
                    f"No chunks generated for doc {doc.record_id} after splitting.",
                )

        except Exception as e:
            logger.error(
                f"Error splitting text for doc {doc.record_id}: {e} {e.args=}",
            )
    logger.info(f"Chunking finished processing {processed_doc_count} documents.")


# --- Upsert Function (can still be used per document) ---
async def upsert_document_chunks(
    doc_iterator: AsyncIterator[InputDocument],
    db_handler: ChromaDBEmbeddings,
) -> tuple[int, int]:
    """Upserts all chunks for each InputDocument from the iterator into ChromaDB."""
    total_docs_processed = 0
    successful_docs_upserted = 0
    failed_docs_upserted = 0

    async for doc in doc_iterator:
        total_docs_processed += 1
        if not doc.chunks:
            logger.warning(f"Document {doc.record_id} has no chunks, skipping upsert.")
            continue

        chunks_to_upsert = [c for c in doc.chunks if c.embedding is not None]

        if not chunks_to_upsert:
            logger.warning(
                f"Document {doc.record_id} has no chunks with successful embeddings, skipping upsert.",
            )
            continue

        ids = []
        documents = []
        embeddings_list = []
        metadatas = []

        for rec in chunks_to_upsert:
            ids.append(rec.chunk_id)
            documents.append(rec.chunk_text)
            embeddings_list.append(list(rec.embedding))  # type: ignore
            base_meta = {
                "document_title": rec.document_title,
                "chunk_index": rec.chunk_index,
                "document_id": rec.document_id,
            }
            combined_meta = {**rec.metadata, **base_meta}
            metadatas.append(_sanitize_metadata_for_chroma(combined_meta))

        chroma_embeddings: Embeddings = embeddings_list

        logger.info(
            f"Upserting {len(ids)} chunks for document {doc.record_id} into collection '{db_handler.collection_name}'...",
        )
        try:
            await asyncio.to_thread(
                db_handler.collection.upsert,
                ids=ids,
                embeddings=chroma_embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            successful_docs_upserted += 1
            logger.debug(f"Successfully upserted chunks for document {doc.record_id}.")
        except Exception as e:
            failed_docs_upserted += 1
            logger.error(
                f"Failed to upsert chunks for document {doc.record_id} into ChromaDB: {e} {e.args=}",
            )
            try:
                failed_doc_filename = (
                    Path(FAILED_BATCH_DIR)
                    / f"failed_upsert_doc_{doc.record_id}_{uuid.uuid4()}.pkl"
                )
                logger.info(
                    f"Saving failed document {doc.record_id} to {failed_doc_filename}",
                )
                bm.save(doc, failed_doc_filename)
            except Exception as save_e:
                logger.error(
                    f"Could not save failed document {doc.record_id} to disk: {save_e} {save_e.args=}",
                )

    # This logging might be less useful now as it runs per document in the main loop
    # logger.info(f"Upsert process finished. Total documents processed: {total_docs_processed}. Successfully upserted: {successful_docs_upserted}. Failed: {failed_docs_upserted}.")
    return successful_docs_upserted, failed_docs_upserted


# --- Main Execution ---

@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg) -> None:
    objs = hydra.utils.instantiate(cfg)
    bm_instance: BM = objs.bm
    vectoriser: ChromaDBEmbeddings = objs.vectoriser
    input_docs_source = objs.input_docs
    preprocessor_instance = objs.preprocessor
    processor_instance = objs.processor

    logger.info("Setting vector store instance on input document source.")
    input_docs_source.set_vector_store(vectoriser)

    loop = asyncio.get_event_loop()
    loop.slow_callback_duration = 35.0

    async def run_pipeline():
        logger.info("Starting data processing pipeline...")
        total_processed_docs = 0
        total_saved_docs = 0
        total_upserted_docs = 0
        total_failed_upsert_docs = 0

        # 1. Source Documents
        doc_iterator = input_docs_source.get_all_records()

        # 2. Pre-process (Extract Text if needed)
        pre_processed_iterator = preprocess_documents(
            doc_iterator,
            preprocessor_instance.extract,
        )

        # 3. Process Documents (e.g., add citations)
        processed_doc_iterator = process_documents(
            pre_processed_iterator,
            processor_instance.process,
        )

        # 4. Chunk Documents (Adds chunks to InputDocument)
        chunked_doc_iterator = chunk_documents(
            processed_doc_iterator,
            vectoriser.text_splitter,
        )

        # 5. Embed, Save, and Upsert each document
        async for doc_with_chunks in chunked_doc_iterator:
            total_processed_docs += 1
            logger.debug(f"Processing document {doc_with_chunks.record_id}...")

            # Embed chunks and save the result to Parquet
            saved_doc = await vectoriser.embed_document(doc_with_chunks)

            if saved_doc:
                total_saved_docs += 1
                logger.debug(
                    f"Document {saved_doc.record_id} saved to Parquet. Attempting upsert...",
                )

                # Upsert this single document immediately
                # Wrap the single saved_doc in an async iterator
                single_doc_iterator = list_to_async_iterator([saved_doc])
                upserted_count, failed_count = await upsert_document_chunks(
                    single_doc_iterator,
                    vectoriser,
                )
                total_upserted_docs += upserted_count
                total_failed_upsert_docs += failed_count
                if failed_count > 0:
                    logger.warning(f"Upsert failed for document {saved_doc.record_id}.")
                else:
                    logger.debug(
                        f"Upsert successful for document {saved_doc.record_id}.",
                    )
            else:
                logger.warning(
                    f"Embedding/Saving failed for document {doc_with_chunks.record_id}, skipping upsert.",
                )

        # Removed separate Step 6

        logger.info("Data processing pipeline finished.")
        logger.info(
            f"Summary: Total Docs Processed: {total_processed_docs}, Saved to Parquet: {total_saved_docs}, Successfully Upserted: {total_upserted_docs}, Failed Upsert: {total_failed_upsert_docs}",
        )

    loop.run_until_complete(run_pipeline())


if __name__ == "__main__":
    main()
