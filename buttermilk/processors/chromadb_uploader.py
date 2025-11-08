"""ChromaDB uploader processor for pipeline.

This processor handles uploading records with embeddings to ChromaDB,
separated from the embedding generation logic.
"""

import asyncio
import time
from pathlib import Path
from typing import Any, AsyncGenerator

import chromadb
import pydantic
from chromadb.api import ClientAPI
from pydantic import BaseModel, Field, PrivateAttr

from buttermilk import bm, logger
from buttermilk._core.types import BaseRecord
from buttermilk.data.vector import _sanitize_metadata_for_chroma
from buttermilk.utils.utils import scrub_serializable, upload_chromadb_cache


class ChromaDBUploader(BaseModel):
    """Upload records with embeddings to ChromaDB.

    This processor expects records with embedded chunks and uploads them
    to a ChromaDB collection. It handles remote storage, caching, and syncing.

    This is typically used for batch updates to the vector database and
    wouldn't be included in production RAG pipelines.

    Note: This processor disables pipeline caching since ChromaDB operations
    are upserts that need to run every time to check for data changes.
    """

    model_config = pydantic.ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    # Pipeline configuration
    skip_cache: bool = Field(
        default=True,
        description="Skip pipeline caching for this processor (recommended for upserts)",
    )

    # ChromaDB configuration
    collection_name: str = Field(..., description="ChromaDB collection name")
    persist_directory: str = Field(
        ..., description="ChromaDB persist directory (can be remote)"
    )

    # Sync configuration
    sync_batch_size: int = Field(
        default=50, description="Sync to remote every N records"
    )
    sync_interval_minutes: int = Field(
        default=10, description="Sync to remote every N minutes"
    )
    disable_auto_sync: bool = Field(
        default=False, description="Disable automatic syncing (manual only)"
    )

    # Batch configuration
    upsert_batch_size: int = Field(
        default=1000, description="Batch size for ChromaDB upserts"
    )
    # Private attributes
    _client: ClientAPI | None = PrivateAttr(default=None)
    _collection: chromadb.Collection | None = PrivateAttr(default=None)
    _original_remote_path: str | None = PrivateAttr(default=None)
    _processed_count: int = PrivateAttr(default=0)
    _last_sync_time: float = PrivateAttr(default=0)
    _cache_initialized: bool = PrivateAttr(default=False)

    def model_post_init(self, __context: Any) -> None:
        """Initialize uploader."""
        logger.info(
            "Initialized ChromaDBUploader",
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
        )
        self._last_sync_time = time.time()

    async def process(
        self, record: BaseRecord, *, processor_stage: str = "chromadb_upload", **kwargs
    ) -> AsyncGenerator[BaseRecord, None]:
        """Process a record by uploading its embedded chunks to ChromaDB.

        Args:
            record: BaseRecord with embedded chunks
            processor_stage: Stage name for metadata tracking

        Yields:
            BaseRecord unchanged (passthrough after upload)
        """
        # Ensure cache is initialized for remote storage
        if not self._cache_initialized:
            await self._ensure_cache_initialized()

        # Check if record has embedded chunks
        chunks_count = len(getattr(record, "chunks", []))
        logger.debug(
            "ChromaDBUploader received record",
            record_id=record.record_id,
            has_chunks=hasattr(record, "chunks"),
            chunks_count=chunks_count,
            processor_stage=processor_stage,
        )

        if not hasattr(record, "chunks") or not record.chunks:
            logger.warning(
                "Record has no chunks to upload",
                record_id=record.record_id,
                processor_stage=processor_stage,
            )
            yield record
            return

        # Check if chunks have embeddings
        chunks_with_embeddings = [
            c
            for c in record.chunks
            if (
                c.get("embedding")
                if isinstance(c, dict)
                else getattr(c, "embedding", None)
            )
            is not None
        ]
        logger.debug(
            "ChromaDBUploader chunk embedding status",
            record_id=record.record_id,
            total_chunks=len(record.chunks),
            chunks_with_embeddings=len(chunks_with_embeddings),
            processor_stage=processor_stage,
        )

        if not chunks_with_embeddings:
            logger.warning(
                "Record chunks have no embeddings",
                record_id=record.record_id,
                chunks_count=len(record.chunks),
                processor_stage=processor_stage,
            )
            yield record
            return

        # Upload chunks to ChromaDB
        start_time = time.time()
        try:
            await self._store_chunks_for_record(record)
            processing_time_ms = (time.time() - start_time) * 1000

            logger.info(
                "Successfully uploaded to ChromaDB",
                record_id=record.record_id,
                chunks_uploaded=len(chunks_with_embeddings),
                processing_time_ms=processing_time_ms,
                processor_stage=processor_stage,
            )

            # Update processed count
            self._processed_count += 1

            # Check if we need to sync
            await self._maybe_sync()

            # Add metadata about upload
            metadata = record.metadata.copy() if record.metadata else {}
            metadata[processor_stage] = {
                "status": "uploaded",
                "timestamp": time.time(),
                "processor": "ChromaDBUploader",
                "chunks_uploaded": len(chunks_with_embeddings),
                "collection": self.collection_name,
                "processing_time_ms": processing_time_ms,
            }

            # Yield record with updated metadata
            processed_record = record.model_copy(update={"metadata": metadata})
            yield processed_record

        except Exception as e:
            logger.error(
                "Failed to upload to ChromaDB",
                record_id=record.record_id,
                error=str(e),
                processor_stage=processor_stage,
            )
            # Still yield the record even if upload failed
            # (could be retried later or handled differently)
            yield record

    async def _ensure_cache_initialized(self) -> None:
        """Ensure ChromaDB cache and collection are ready for use."""
        persist_dir = self.persist_directory

        # Handle remote storage by downloading to local cache
        if persist_dir.startswith(("gs://", "s3://", "azure://", "gcs://")):
            self._original_remote_path = persist_dir
            local_cache_path = await self._setup_local_cache(persist_dir)
            persist_dir = str(local_cache_path)

        # Initialize ChromaDB client
        if not self._client:
            self._client = chromadb.PersistentClient(
                path=persist_dir, settings=chromadb.Settings(anonymized_telemetry=False)
            )
            logger.info("ChromaDB client initialized", persist_directory=persist_dir)

        # Get or create collection
        if not self._collection:
            try:
                self._collection = self._client.get_collection(
                    name=self.collection_name
                )
                logger.info(
                    "Using existing collection",
                    collection_name=self.collection_name,
                    count=self._collection.count(),
                )
            except Exception:
                self._collection = self._client.create_collection(
                    name=self.collection_name
                )
                logger.info(
                    "Created new collection", collection_name=self.collection_name
                )

        self._cache_initialized = True

    async def _setup_local_cache(self, remote_path: str) -> Path:
        """Setup local cache for remote ChromaDB."""
        # Use SessionInfo for consistent cache key generation
        cache_key = bm.session_info.generate_cache_key(remote_path)
        local_cache_path = bm.session_info.get_chromadb_cache_dir() / cache_key
        local_cache_path.mkdir(parents=True, exist_ok=True)

        # For production, you'd implement proper sync logic here
        # For now, we'll just use the local cache
        logger.info(
            "Using local cache for remote ChromaDB",
            remote_path=remote_path,
            local_cache=str(local_cache_path),
        )

        return local_cache_path

    async def _store_chunks_for_record(self, record: BaseRecord) -> None:
        """Store record chunks with metadata in ChromaDB."""
        if not self._collection:
            raise ValueError("Collection not initialized")

        chunks_to_upsert = []

        # Convert chunks to dicts if needed and filter for embeddings
        for c in record.chunks:
            if hasattr(c, "model_dump"):
                # Convert ChunkedDocument to dict
                chunk_dict = c.model_dump()
            elif isinstance(c, dict):
                chunk_dict = c
            else:
                # Skip unsupported chunk types
                continue

            if chunk_dict.get("embedding") is not None:
                chunks_to_upsert.append(chunk_dict)

        if not chunks_to_upsert:
            return

        ids = []
        documents = []
        embeddings_list = []
        metadatas = []

        for chunk in chunks_to_upsert:
            ids.append(chunk["chunk_id"])
            documents.append(chunk["chunk_text"])

            # Convert numpy array or list to regular Python floats
            embedding = chunk["embedding"]
            if hasattr(embedding, "tolist"):
                embeddings_list.append(embedding.tolist())
            else:
                embeddings_list.append([float(x) for x in embedding])

            # Build metadata
            chunk_metadata = chunk.get("metadata", {})
            enhanced_metadata = {
                "document_title": chunk["document_title"],
                "chunk_index": chunk["chunk_index"],
                "document_id": chunk["document_id"],
                "content_type": chunk_metadata.get("content_type", "unknown"),
                "chunk_type": chunk_metadata.get("chunk_type", "unknown"),
                **{
                    k: v
                    for k, v in chunk_metadata.items()
                    if k not in ["content_type", "chunk_type"]
                },
            }
            # Ensure metadata is serializable and ChromaDB-compatible
            enhanced_metadata = scrub_serializable(enhanced_metadata)
            metadatas.append(_sanitize_metadata_for_chroma(enhanced_metadata))

        # Batch upsert to ChromaDB
        for i in range(0, len(ids), self.upsert_batch_size):
            batch_end = min(i + self.upsert_batch_size, len(ids))

            await asyncio.to_thread(
                self._collection.upsert,
                ids=ids[i:batch_end],
                documents=documents[i:batch_end],
                embeddings=embeddings_list[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )

            logger.debug(
                "Upserted batch to ChromaDB",
                record_id=record.record_id,
                batch_start=i,
                batch_end=batch_end,
                total=len(ids),
            )

    async def _maybe_sync(self) -> None:
        """Sync to remote if conditions are met."""
        if self.disable_auto_sync:
            return

        # Check if we should sync based on count or time
        should_sync = False
        current_time = time.time()

        if self._processed_count >= self.sync_batch_size:
            should_sync = True
            reason = f"batch size ({self._processed_count} records)"
        elif (current_time - self._last_sync_time) >= (self.sync_interval_minutes * 60):
            should_sync = True
            reason = f"time interval ({self.sync_interval_minutes} minutes)"

        if should_sync and self._original_remote_path:
            logger.debug(
                "Syncing to remote storage",
                reason=reason,
                processed_count=self._processed_count,
            )
            try:
                # Get the local cache path
                local_cache_path = await self._get_local_cache_path()
                if local_cache_path:
                    await upload_chromadb_cache(
                        str(local_cache_path), self._original_remote_path
                    )
                    logger.info(
                        "Successfully synced ChromaDB to remote storage",
                        processed_count=self._processed_count,
                    )
                else:
                    logger.warning("Could not determine local cache path for sync")
            except Exception as e:
                logger.error(
                    "Failed to sync to remote storage",
                    error=str(e),
                    processed_count=self._processed_count,
                )
            finally:
                # Reset counters regardless of sync success/failure
                self._processed_count = 0
                self._last_sync_time = current_time

    async def finalize_processing(self) -> bool:
        """Finalize processing by syncing to remote."""
        if self._original_remote_path:
            logger.info(
                "Final sync to remote storage", processed_count=self._processed_count
            )
            try:
                # Get the local cache path
                local_cache_path = await self._get_local_cache_path()
                if local_cache_path:
                    await upload_chromadb_cache(
                        str(local_cache_path), self._original_remote_path
                    )
                    logger.info(
                        "Successfully completed final sync to remote storage",
                        processed_count=self._processed_count,
                    )
                    return True
                else:
                    logger.warning(
                        "Could not determine local cache path for final sync"
                    )
                    return False
            except Exception as e:
                logger.error(
                    "Failed final sync to remote storage",
                    error=str(e),
                    processed_count=self._processed_count,
                )
                return False
        return True

    async def _get_local_cache_path(self) -> Path | None:
        """Get the local cache path for the ChromaDB instance."""
        if not self._original_remote_path:
            return None

        # Recreate the cache path logic from _setup_local_cache (must match SessionInfo)
        cache_key = bm.session_info.generate_cache_key(self._original_remote_path)
        local_cache_path = bm.session_info.get_chromadb_cache_dir() / cache_key

        if local_cache_path.exists():
            return local_cache_path
        else:
            logger.warning(
                "Local cache path does not exist", cache_path=str(local_cache_path)
            )
            return None
