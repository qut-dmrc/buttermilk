"""Modular embedding generator processor for pipeline.

This processor generates embeddings for chunked documents, decoupled from storage.
It can be swapped out for different embedding models or strategies.
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any, cast

import pydantic
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai.types import ContentListUnion
from pydantic import BaseModel, Field, PrivateAttr
from vertexai.language_models import (
    TextEmbeddingInput,
)

from buttermilk import bm, logger
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.types import BaseRecord
from buttermilk.utils.utils import scrub_serializable


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(
        self,
        embedding_model: str,
        dimensionality: int = 3072,
    ):
        self.dimensionality = dimensionality
        self._client: genai.Client | None = None
        self._embedding_model = embedding_model

    @property
    def client(self) -> genai.Client:
        """Lazily initialize the Gemini client on first access."""
        if self._client is None:
            self._client = bm.genai
        return self._client

    def __call__(self, input: Documents) -> Embeddings:
        response = self.client.models.embed_content(
            model=self._embedding_model,
            contents=cast("ContentListUnion", input),
            config={
                "output_dimensionality": self.dimensionality,
                "auto_truncate": False,
            },
        )

        # Extract embeddings from response
        embeddings: list[Any] = []
        if response.embeddings is None:
            raise ValueError("Gemini embed_content returned no embeddings")
        for embedding in response.embeddings:
            # Convert to list if it's a numpy array
            embeddings.append(scrub_serializable(embedding.values))

        return embeddings


class EmbeddingGenerator(BaseModel):
    """Generate embeddings for chunked documents.

    This processor expects records with a 'chunks' field containing
    ChunkedDocument objects, and adds embeddings to those chunks in-place.

    Features:
    - Configurable embedding model and dimensionality
    - Retry logic with exponential backoff for rate limits
    - Batch processing with configurable batch size
    - Partial failure handling
    """

    model_config = pydantic.ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    # Embedding configuration
    embedding_model: str = Field(default="gemini-embedding-001", description="Embedding model to use")
    dimensionality: int = Field(default=3072, description="Embedding vector dimensionality")
    task: str = Field(default="RETRIEVAL_DOCUMENT", description="Task type for embeddings")

    # Batch and retry configuration
    embedding_batch_size: int = Field(default=100, description="Batch size for embedding API calls")
    embedding_max_retries: int = Field(default=5, description="Max retries for embedding API calls")
    embedding_min_wait_seconds: float = Field(default=1.0, description="Min wait between embedding retries")
    embedding_max_wait_seconds: float = Field(default=120.0, description="Max wait between embedding retries")
    embedding_cooldown_seconds: float = Field(default=0.1, description="Cooldown between successful embedding calls")

    # Private attributes
    _embedding_semaphore: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Initialize semaphore for concurrent embedding calls."""
        self._embedding_semaphore = asyncio.Semaphore(20)
        logger.info(
            "Initialized EmbeddingGenerator",
            embedding_model=self.embedding_model,
            dimensionality=self.dimensionality,
            batch_size=self.embedding_batch_size,
        )

    async def process(self, context: ProcessingContext) -> AsyncGenerator[BaseRecord, None]:
        """Process a record by generating embeddings for its chunks.

        Args:
            context: Unified processing context

        Yields:
            BaseRecord with embeddings added to chunks
        """
        record = context.record
        processor_stage = context.session_id
        # Check if record has chunks
        chunks_count = len(getattr(record, "chunks", []))
        logger.debug(
            "EmbeddingGenerator received record",
            record_id=record.record_id,
            has_chunks=hasattr(record, "chunks"),
            chunks_count=chunks_count,
            processor_stage=processor_stage,
        )

        if not hasattr(record, "chunks") or not record.chunks:
            logger.warning(
                "Record has no chunks to embed",
                record_id=record.record_id,
                processor_stage=processor_stage,
            )
            yield record
            return

        # Generate embeddings
        start_time = time.time()
        success = await self._embed_chunks(record.chunks, record.record_id)
        processing_time_ms = (time.time() - start_time) * 1000

        if success:
            logger.info(
                "Successfully generated embeddings",
                record_id=record.record_id,
                chunks_count=len(record.chunks),
                processing_time_ms=processing_time_ms,
                processor_stage=processor_stage,
            )

            # Add metadata about embedding
            metadata = record.metadata.copy() if record.metadata else {}
            metadata[processor_stage] = {
                "status": "embedded",
                "timestamp": time.time(),
                "processor": "EmbeddingGenerator",
                "chunks_embedded": len(record.chunks),
                "embedding_model": self.embedding_model,
                "dimensionality": self.dimensionality,
                "processing_time_ms": processing_time_ms,
            }

            # Yield record with updated metadata
            processed_record = record.model_copy(update={"metadata": metadata})
            yield processed_record
        else:
            logger.error(
                "Failed to generate embeddings",
                record_id=record.record_id,
                chunks_count=len(record.chunks),
                processor_stage=processor_stage,
            )
            # Don't yield the record if embedding failed
            return

    async def _embed_chunks(self, chunks: list[Any], record_id: str) -> bool:
        """Generate embeddings for a list of chunks in place.

        Returns:
            bool: True if all embeddings succeeded
        """
        if not chunks:
            return False

        # Work with ChunkedDocument objects directly - don't convert to dicts
        embeddings_input: list[tuple[int, TextEmbeddingInput]] = []
        for i, chunk in enumerate(chunks):
            # Support both ChunkedDocument objects and dicts
            if hasattr(chunk, "chunk_text"):
                # ChunkedDocument object
                embeddings_input.append(
                    (
                        i,
                        TextEmbeddingInput(
                            text=chunk.chunk_text,
                            task_type=self.task,
                            title=f"{chunk.document_title}_{chunk.chunk_index}",
                        ),
                    ),
                )
            elif isinstance(chunk, dict):
                # Dict representation
                embeddings_input.append(
                    (
                        i,
                        TextEmbeddingInput(
                            text=chunk["chunk_text"],
                            task_type=self.task,
                            title=f"{chunk['document_title']}_{chunk['chunk_index']}",
                        ),
                    ),
                )
            else:
                logger.warning(f"Unsupported chunk type: {type(chunk)}")
                continue

        embedding_results = await self._embed(embeddings_input)

        success_count = 0
        for idx, embedding in embedding_results:
            if embedding is not None and idx < len(chunks):
                chunk = chunks[idx]
                # Set embedding based on chunk type
                if hasattr(chunk, "embedding"):
                    # ChunkedDocument object - set attribute
                    chunk.embedding = embedding
                    success_count += 1
                elif isinstance(chunk, dict):
                    # Dict - set key
                    chunk["embedding"] = embedding
                    success_count += 1

        if success_count == 0:
            logger.error(f"All embeddings failed for record {record_id} after {self.embedding_max_retries} retries")
            return False

        if success_count < len(chunks):
            logger.warning("Partial embedding failure", succeeded=success_count, total=len(chunks))
            # Clear embeddings so we don't have partial state
            for c in chunks:
                if hasattr(c, "embedding"):
                    c.embedding = None
                elif isinstance(c, dict):
                    c["embedding"] = None
            return False

        logger.debug("Generated embeddings", count=success_count)
        return True

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        """Check if an exception is a rate limit error."""
        msg = str(exc).lower()
        return any(k in msg for k in ["rate limit", "quota", "too many requests", "429"])

    async def _embed(self, embeddings_input: list[tuple[int, Any]]) -> list[tuple[int, list[float] | None]]:
        """Generate embeddings with retry logic.

        Args:
            embeddings_input: List of (index, TextEmbeddingInput) tuples

        Returns:
            List of (index, embedding) tuples where embedding can be None on failure
        """
        embedding_function = GeminiEmbeddingFunction(self.embedding_model, self.dimensionality)

        async def _run_embed_batch(batch_docs: list[Any], attempt: int = 0) -> list[Any]:
            """Run embedding for a batch with semaphore."""
            async with self._embedding_semaphore:
                try:
                    embeddings = embedding_function(batch_docs)

                    logger.debug(
                        "Embedding batch",
                        batch_size=len(docs),
                        attempt=attempt + 1,
                        model=self.embedding_model,
                        dimensionality=self.dimensionality,
                        auto_truncate=False,
                        embeddings_count=len(embeddings),
                        cooldown_seconds=self.embedding_cooldown_seconds,
                    )
                    # Add cooldown to avoid rate limits
                    if self.embedding_cooldown_seconds > 0:
                        await asyncio.sleep(self.embedding_cooldown_seconds)

                    return embeddings

                except Exception as e:
                    logger.exception(
                        "Embedding API error",
                        error=str(e),
                        args=e.args,
                        batch_size=len(batch_docs),
                        attempt=attempt + 1,
                        model=self.embedding_model,
                        dimensionality=self.dimensionality,
                        auto_truncate=False,
                        cooldown_seconds=self.embedding_cooldown_seconds,
                    )
                    raise

        # Process in batches
        results: list[tuple[int, list[float] | None]] = []
        batch_size = self.embedding_batch_size

        for i in range(0, len(embeddings_input), batch_size):
            batch = embeddings_input[i : i + batch_size]
            indices = [idx for idx, _ in batch]
            docs = [doc for _, doc in batch]

            # Retry logic
            for attempt in range(self.embedding_max_retries):
                try:
                    embeddings = await _run_embed_batch(docs, attempt=attempt)

                    # Pair indices with embeddings
                    for idx, embedding in zip(indices, embeddings):
                        results.append((idx, embedding))
                    break

                except Exception as exc:
                    if self._is_rate_limit_error(exc):
                        # Exponential backoff for rate limits
                        wait_time = min(
                            self.embedding_min_wait_seconds * (2**attempt),
                            self.embedding_max_wait_seconds,
                        )
                        logger.warning(
                            "Rate limit hit, retrying",
                            attempt=attempt + 1,
                            wait_time=wait_time,
                        )
                        await asyncio.sleep(wait_time)
                    elif attempt == self.embedding_max_retries - 1:
                        # Last attempt failed
                        logger.error(
                            "Embedding batch failed after retries",
                            batch_size=len(docs),
                            error=str(exc),
                        )
                        # Return None for failed embeddings
                        for idx in indices:
                            results.append((idx, None))
                        break
                    else:
                        # Non-rate limit error, retry immediately
                        logger.warning(
                            "Embedding error, retrying",
                            attempt=attempt + 1,
                            error=str(exc),
                        )

        return results
