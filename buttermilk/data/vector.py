import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Self, TypeVar, cast  # Corrected import for Tuple

import chromadb
import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import pydantic
from chromadb import Collection, EmbeddingFunction, Embeddings
from chromadb.api import ClientAPI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, PrivateAttr
from tqdm.asyncio import tqdm
from vertexai.language_models import (
    TextEmbedding,
    TextEmbeddingInput,
    TextEmbeddingModel,
)

from buttermilk import logger
from buttermilk._core.config import DataSouce
from buttermilk._core.log import logger  # noqa, bm, logger  # no-qa

MODEL_NAME = "text-embedding-large-exp-03-07"
DEFAULT_UPSERT_BATCH_SIZE = 10  # Still used for failed batch saving logic if needed
FAILED_BATCH_DIR = "failed_upsert_batches"
MAX_TOTAL_TASKS_PER_RUN = 500

T = TypeVar("T")


_db_registry = {}

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


class DefaultTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, chunk_size: int = 9000, chunk_overlap: int = 1000):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            add_start_index=False,
        )
        logger.info(
            f"Initialized RecursiveCharacterTextSplitter (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})",
        )

    async def process(self, doc: InputDocument, **kwargs) -> InputDocument | None:
        """Chunks documents and adds the chunks list to the InputDocument."""
        if not doc.full_text:
            logger.warning(
                f"Skipping chunking for record {doc.record_id} due to missing full_text.",
            )
            return None
        try:
            text_chunks = await asyncio.to_thread(
                self.split_text,
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
                return doc
            logger.warning(
                f"No chunks generated for doc {doc.record_id} after splitting.",
            )

        except Exception as e:
            logger.error(
                f"Error splitting text for doc {doc.record_id}: {e} {e.args=}",
            )
        return None


class VertexAIEmbeddingFunction(EmbeddingFunction):
    def __init__(
        self,
        embedding_model: str,
        dimensionality: int = 3072,
    ):
        self.dimensionality = dimensionality
        self._embedding_model = TextEmbeddingModel.from_pretrained(embedding_model)

    def __call__(self, input) -> Embeddings:
        kwargs = dict(
            auto_truncate=False,
            output_dimensionality=self.dimensionality,
        )

        # Vertex batch_size is 1
        results = self._embedding_model.get_embeddings(texts=input, **kwargs)

        # convert from numpy
        return [cast("list[float]", r.values) for r in results]


# --- Core Embedding and DB Interaction Class ---
class ChromaDBEmbeddings(DataSouce):
    """Handles configuration, embedding model interaction, and ChromaDB connection."""

    model_config = pydantic.ConfigDict(extra="ignore")

    embedding_model: str = Field(default=MODEL_NAME)
    task: str = "RETRIEVAL_DOCUMENT"
    collection_name: str = Field(default=...)
    dimensionality: int = Field(default=3072)
    persist_directory: str = Field(default=...)
    concurrency: int = Field(default=20)
    upsert_batch_size: int = DEFAULT_UPSERT_BATCH_SIZE
    embedding_batch_size: int = Field(default=1)
    arrow_save_dir: str = Field(default="")
    embedding_model: str = Field(default="text-embedding-large-exp-03-07")

    _embedding_semaphore: asyncio.Semaphore = PrivateAttr()
    _collection: Collection = PrivateAttr()
    _embedding_model: TextEmbeddingModel = PrivateAttr()
    _embedding_function: Callable = PrivateAttr()
    _client: ClientAPI = PrivateAttr()

    @pydantic.model_validator(mode="after")
    def load_models(self) -> Self:
        """Initializes the embedding model, ChromaDB client, and text splitter."""
        logger.info(f"Loading embedding model: {self.embedding_model}")
        self._embedding_model = TextEmbeddingModel.from_pretrained(self.embedding_model)
        self._embedding_function = VertexAIEmbeddingFunction(
            embedding_model=self.embedding_model,
            dimensionality=self.dimensionality,
        )
        logger.info(f"Initializing ChromaDB client at: {self.persist_directory}")
        self._client = chromadb.PersistentClient(path=self.persist_directory)
        logger.info(f"Using ChromaDB collection: {self.collection_name}")

        self._embedding_semaphore = asyncio.Semaphore(self.concurrency)

        Path(FAILED_BATCH_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.arrow_save_dir).mkdir(parents=True, exist_ok=True)
        return self

    @property
    def collection(self) -> Collection:
        """Provides access to the ChromaDB collection."""
        # This vector store is single-threaded, so we're only keeping one instance around
        if _db_registry.get("vectorstore") is None:
            # Initialize the vector store.
            if not hasattr(self, "_client") or not self._client:
                self._client = chromadb.PersistentClient(path=self.persist_directory)

            _db_registry["vectorstore"] = self._client.get_or_create_collection(
                self.collection_name,
                embedding_function=self._embedding_function,
            )
        return _db_registry["vectorstore"]

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
                embeddings_input.append(
                    (
                        chunk.chunk_index,
                        TextEmbeddingInput(
                            text=chunk.chunk_text,
                            task_type=self.task,
                            title=chunk.chunk_title,
                        ),
                    ),
                )
            else:
                logger.warning(
                    f"Chunk missing index in doc {input_doc.record_id}, skipping embedding for this chunk.",
                )

        embedding_results = await self._embed(embeddings_input)

        successful_embeddings = 0
        for idx, embedding in embedding_results:
            try:
                list_index = next(i for i, chk in enumerate(input_doc.chunks) if chk.chunk_index == idx)
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
            "embedding": [list(c.embedding) if c.embedding is not None else None for c in doc.chunks],
            "chunk_metadata": [json.dumps(c.metadata) if c.metadata else None for c in doc.chunks],
        }

        embedding_type = pa.list_(pa.float32())
        if self.dimensionality:
            embedding_type = pa.list_(pa.float32(), self.dimensionality)

        schema = pa.schema(
            [
                pa.field("chunk_id", pa.string()),
                pa.field("document_id", pa.string()),
                pa.field("document_title", pa.string()),
                pa.field("chunk_index", pa.int32()),
                pa.field("chunk_text", pa.string()),
                pa.field("embedding", embedding_type),
                pa.field("chunk_metadata", pa.string()),
            ],
        )

        table = pa.Table.from_pydict(data, schema=schema)

        doc_meta_serializable = {
            "record_id": doc.record_id,
            "title": doc.title,
            "file_path": doc.file_path,
            "record_path": doc.record_path,
            "metadata": json.dumps(doc.metadata),
        }
        arrow_metadata = {k.encode("utf-8"): str(v).encode("utf-8") for k, v in doc_meta_serializable.items()}

        final_schema = table.schema.with_metadata(arrow_metadata)
        table = table.cast(final_schema)

        pq.write_table(table, file_path, compression="snappy")

    async def _run_embedding_task(
        self,
        chunk_input: TextEmbeddingInput,
        index: int,
    ) -> tuple[int, list[float | int] | None]:
        """Helper coroutine to run a single embedding task and handle errors."""
        async with self._embedding_semaphore:
            kwargs = dict(
                output_dimensionality=self.dimensionality,
                auto_truncate=False,
            )
            try:
                embeddings_result: list[TextEmbedding] = await self._embedding_model.get_embeddings_async(
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
        tasks = [self._run_embedding_task(chunk_input=chunk_input, index=idx) for idx, chunk_input in inputs]
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

    async def upsert_document_chunks(
        self,
        doc_iterator: AsyncIterator[InputDocument],
    ) -> tuple[int, int]:
        """Upserts all chunks for each InputDocument from the iterator into ChromaDB."""
        total_docs_processed = 0
        successful_docs_upserted = 0
        failed_docs_upserted = 0

        async for doc in doc_iterator:
            total_docs_processed += 1
            if not doc.chunks:
                logger.warning(
                    f"Document {doc.record_id} has no chunks, skipping upsert.",
                )
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
                f"Upserting {len(ids)} chunks for document {doc.record_id} into collection '{self.collection_name}'...",
            )
            try:
                await asyncio.to_thread(
                    self.collection.upsert,
                    ids=ids,
                    embeddings=chroma_embeddings,
                    metadatas=metadatas,
                    documents=documents,
                )
                successful_docs_upserted += 1
                logger.debug(
                    f"Successfully upserted chunks for document {doc.record_id}.",
                )
            except Exception as e:
                failed_docs_upserted += 1
                logger.error(
                    f"Failed to upsert chunks for document {doc.record_id} into ChromaDB: {e} {e.args=}",
                )
                try:
                    failed_doc_filename = bm.save_dir / Path(FAILED_BATCH_DIR) / f"failed_upsert_doc_{doc.record_id}_{uuid.uuid4()}.pkl"
                    logger.info(
                        f"Saving failed document {doc.record_id} to {failed_doc_filename}",
                    )
                    bm.save(doc, failed_doc_filename)
                except Exception as save_e:
                    logger.error(
                        f"Could not save failed document {doc.record_id} to disk: {save_e} {save_e.args=}",
                    )

        return successful_docs_upserted, failed_docs_upserted

    async def process(self, doc: InputDocument) -> InputDocument | None:
        """Process a document by embedding, saving to parquet, and upserting to ChromaDB.

        Follows the ProcessorCallable signature for compatibility with DocProcessor.
        """
        if not doc.chunks:
            logger.warning(
                f"Document {doc.record_id} has no chunks, skipping embedding and upsert.",
            )
            return None

        # 1. Generate embeddings and save to Parquet
        doc_with_embeddings = await self.embed_document(doc)
        if not doc_with_embeddings:
            logger.warning(
                f"Embedding failed for document {doc.record_id}, skipping upsert.",
            )
            return None

        # 2. Upsert to ChromaDB
        try:
            ids = []
            documents = []
            embeddings_list = []
            metadatas = []

            chunks_to_upsert = [c for c in doc_with_embeddings.chunks if c.embedding is not None]

            for chunk in chunks_to_upsert:
                ids.append(chunk.chunk_id)
                documents.append(chunk.chunk_text)
                embeddings_list.append(list(chunk.embedding))  # type: ignore

                base_meta = {
                    "document_title": chunk.document_title,
                    "chunk_index": chunk.chunk_index,
                    "document_id": chunk.document_id,
                }
                combined_meta = {**chunk.metadata, **base_meta}
                metadatas.append(_sanitize_metadata_for_chroma(combined_meta))

            logger.info(f"Upserting {len(ids)} chunks for document {doc.record_id}...")

            # Execute the upsert operation
            await asyncio.to_thread(
                self.collection.upsert,
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents,
            )

            logger.info(
                f"Successfully processed document {doc.record_id} - embedded, saved, and upserted.",
            )
            return doc_with_embeddings

        except Exception as e:
            logger.error(f"Failed to upsert document {doc.record_id}: {e}")
            # Save the failed document for retry
            try:
                failed_doc_filename = bm.save_dir / Path(FAILED_BATCH_DIR) / f"failed_upsert_doc_{doc.record_id}_{uuid.uuid4()}.pkl"
                logger.info(
                    f"Saving failed document {doc.record_id} to {failed_doc_filename}",
                )
                bm.save(doc_with_embeddings, failed_doc_filename)
            except Exception as save_e:
                logger.error(
                    f"Could not save failed document {doc.record_id}: {save_e}",
                )

            return None


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


class DocProcessor(BaseModel):
    """Callable class for processing documents from an iterator."""

    concurrency: int = Field(default=20)
    _semaphore: asyncio.Semaphore = PrivateAttr()
    doc_iterator: AsyncIterator[InputDocument] = Field(default=None, exclude=True)
    processor: ProcessorCallable = Field(default=None, exclude=True)
    _name: str = PrivateAttr(default="")

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
    )

    @pydantic.model_validator(mode="after")
    def _init(self) -> Self:
        self._semaphore = asyncio.Semaphore(self.concurrency)
        if hasattr(self._processor, "__name__"):
            self._name = self._processor.__name__
        else:
            self._name = self._processor.__class__.__name__
        return self

    async def _process(self, doc: InputDocument) -> InputDocument | None:
        async with self._semaphore:
            try:
                processed_doc = await self._processor(doc)
                return processed_doc
            except Exception as e:
                logger.error(
                    f"Error processing document {doc.record_id}: {e} {e.args=}",
                )
                logger.warning(
                    f"Skipping document {doc.record_id} due to processor error.",
                )
                return None

    async def __call__(self) -> AsyncIterator[InputDocument]:
        """Processes documents from the iterator, yielding them as they complete."""
        processed_count = 0
        pending_tasks = set()
        max_pending = self.concurrency * 2  # Ensure we don't accumulate too many tasks

        try:
            # Start initial tasks up to our limit
            iterator_exhausted = False
            while not iterator_exhausted:
                # Add new tasks up to our max_pending limit
                while len(pending_tasks) < max_pending and not iterator_exhausted:
                    try:
                        doc = await anext(self._doc_iterator)
                        task = asyncio.create_task(self._process(doc))
                        pending_tasks.add(task)
                        # Set up task completion callback to remove it from pending set
                        task.add_done_callback(pending_tasks.discard)
                    except StopAsyncIteration:
                        iterator_exhausted = True
                        break

                if not pending_tasks:
                    break

                # Wait for at least one task to complete
                done, _ = await asyncio.wait(
                    pending_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Process completed tasks
                for task in done:
                    try:
                        result = task.result()
                        if result is not None:
                            processed_count += 1
                            yield result
                    except Exception as e:
                        logger.error(f"Task raised an exception: {e}")

            logger.info(
                f"Finished processing {processed_count} documents with {self._name}.",
            )

        except Exception as e:
            logger.error(f"Error in DocProcessor: {e}")
            # Cancel any pending tasks
            for task in pending_tasks:
                if not task.done():
                    task.cancel()


# --- Main Execution ---


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg) -> None:
    objs = hydra.utils.instantiate(cfg)
    bm_instance: BM = objs.bm
    vectoriser: ChromaDBEmbeddings = objs.vectoriser
    input_docs_source = objs.input_docs
    preprocessor_instance = objs.preprocessor
    processor_instance = objs.processor
    text_splitter_instance = objs.chunker

    logger.info("Setting vector store instance on input document source.")
    input_docs_source.set_vector_store(vectoriser)

    loop = asyncio.get_event_loop()
    loop.slow_callback_duration = 35.0

    async def run_pipeline():
        logger.info("Starting data processing pipeline...")
        total_saved_docs = 0
        total_upserted_docs = 0

        # 1. Source Documents
        doc_iterator = input_docs_source.get_all_records(start=objs.start_from)

        # 2. Pre-process (Extract Text if needed)
        pre_processed_iterator = DocProcessor(
            _doc_iterator=doc_iterator,
            _processor=preprocessor_instance.process,
        )

        # 3. Process Documents (e.g., add citations)
        processed_doc_iterator = DocProcessor(
            _doc_iterator=pre_processed_iterator(),
            _processor=processor_instance.process,
        )

        # 4. Chunk Documents (Adds chunks to InputDocument)
        chunked_doc_iterator = DocProcessor(
            _doc_iterator=processed_doc_iterator(),
            _processor=text_splitter_instance.process,
        )

        # 5. Vectorize and Upsert - now using the same DocProcessor pattern
        vectorizer_processor = DocProcessor(
            _doc_iterator=chunked_doc_iterator(),
            _processor=vectoriser.process,
            concurrency=vectoriser.concurrency,
        )

        # Process documents through the complete pipeline with a limit
        max_docs = MAX_TOTAL_TASKS_PER_RUN
        pbar = tqdm(total=max_docs, desc="Processing documents")
        stats = {
            "preprocessed": 0,
            "processed": 0,
            "chunked": 0,
            "embedded": 0,
        }

        # Use the pipeline to process documents

        async for doc in vectorizer_processor():
            if doc is not None:
                stats["embedded"] += 1
                pbar.update(1)
                pbar.set_postfix(stats)

            if stats["embedded"] >= max_docs:
                logger.info(f"Reached document limit: {max_docs}")
                break

        pbar.close()

        # Log final stats
        logger.info(
            f"Pipeline complete! Documents successfully processed: {stats['embedded']}",
        )

    loop.run_until_complete(run_pipeline())


if __name__ == "__main__":
    main()
