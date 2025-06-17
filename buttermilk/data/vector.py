import asyncio
import hashlib
import json
import shortuuid
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self, TypeVar, cast  # Corrected import for Tuple

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

from buttermilk import (
    buttermilk as bm,  # Global Buttermilk instance
    logger,
)
from buttermilk._core.config import DataSouce
from buttermilk._core.storage_config import MultiFieldEmbeddingConfig
from buttermilk._core.types import Record
from buttermilk._core.log import logger  # noqa, bm, logger  # no-qa
from buttermilk.utils.utils import ensure_chromadb_cache

MODEL_NAME = "gemini-embedding-001"
DEFAULT_UPSERT_BATCH_SIZE = 10  # Still used for failed batch saving logic if needed
FAILED_BATCH_DIR = "failed_upsert_batches"
MAX_TOTAL_TASKS_PER_RUN = 500

T = TypeVar("T")


_db_registry = {}

# --- New Result Types and Configuration (Breaking Changes) ---

@dataclass
class ExistenceCheck:
    """Detailed result from checking if a record+model combination exists."""
    exists: bool
    embedding_model: str
    chunk_count: int
    last_processed: datetime | None
    metadata_hash: str | None

@dataclass
class ProcessingResult:
    """Comprehensive result from record processing."""
    record: Record | None
    status: Literal["processed", "skipped", "failed"]
    reason: str
    chunks_created: int
    embedding_model: str
    processing_time_ms: float
    metadata: dict[str, Any]

@dataclass
class BatchProcessingResult:
    """Result from batch processing operations."""
    total_records: int
    successful_count: int
    skipped_count: int
    failed_count: int
    processing_time_ms: float
    validation_result: dict[str, Any] | None
    failed_records: list[tuple[str, str]]  # (record_id, error_message)
    metadata: dict[str, Any]

class ChromaDBConfig(BaseModel):
    """Strict configuration with required fields for ChromaDB."""
    
    # Required fields (no defaults)
    persist_directory: str
    collection_name: str
    embedding_model: str
    
    # Required deduplication strategy
    deduplication_strategy: Literal["record_id", "content_hash", "both"] = "both"
    
    # Optional fields with reasonable defaults
    dimensionality: int = 3072
    concurrency: int = 20
    sync_batch_size: int = 50
    sync_interval_minutes: int = 10
    disable_auto_sync: bool = False
    multi_field_config: MultiFieldEmbeddingConfig | None = None

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


# --- Compatibility Functions ---
# REMOVED: Deprecated compatibility functions
# - create_input_document_from_record: Use Record directly
# - InputDocument_compat: Use Record class directly


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
    type: Literal["chromadb"] = "chromadb"
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
    multi_field_config: MultiFieldEmbeddingConfig | None = Field(default=None)

    # New sync configuration options
    sync_batch_size: int = Field(default=50, description="Sync every N records")
    sync_interval_minutes: int = Field(default=10, description="Sync every N minutes")
    disable_auto_sync: bool = Field(default=False, description="Disable automatic syncing (manual only)")
    
    # New deduplication configuration (Breaking Change)
    deduplication_strategy: Literal["record_id", "content_hash", "both"] = Field(default="both")
    

    _embedding_semaphore: asyncio.Semaphore = PrivateAttr()
    _collection: Collection = PrivateAttr()
    _embedding_model: TextEmbeddingModel = PrivateAttr()
    _embedding_function: Callable = PrivateAttr()
    _client: ClientAPI = PrivateAttr()
    _original_remote_path: str | None = PrivateAttr(default=None)
    _processed_records_count: int = PrivateAttr(default=0)
    
    # New private attributes for deduplication
    _processed_combinations_cache: set[str] = PrivateAttr(default_factory=set)
    _last_sync_time: float = PrivateAttr(default=0.0)
    _sync_batch_size: int = PrivateAttr(default=50)  # Sync every 50 records
    _sync_interval_seconds: int = PrivateAttr(default=600)  # Sync every 10 minutes

    @pydantic.model_validator(mode="after")
    def load_models(self) -> Self:
        """Initializes the embedding model, ChromaDB client, and text splitter."""
        import time

        # Initialize sync timing and configure sync behavior
        self._last_sync_time = time.time()
        self._sync_batch_size = self.sync_batch_size
        self._sync_interval_seconds = self.sync_interval_minutes * 60

        logger.info(f"Loading embedding model: {self.embedding_model}")
        self._embedding_model = TextEmbeddingModel.from_pretrained(self.embedding_model)
        self._embedding_function = VertexAIEmbeddingFunction(
            embedding_model=self.embedding_model,
            dimensionality=self.dimensionality,
        )

        # Handle remote persist_directory by caching locally
        logger.info(f"Initializing ChromaDB client at: {self.persist_directory}")

        # For remote persist_directory, we'll cache it during collection access
        # For local paths, use directly
        self._client = None  # Will be initialized lazily in collection property
        logger.info(f"Using ChromaDB collection: {self.collection_name}")

        self._embedding_semaphore = asyncio.Semaphore(self.concurrency)

        # Log sync configuration
        if not self.disable_auto_sync:
            logger.info(f"🔄 Auto-sync enabled: every {self.sync_batch_size} records OR every {self.sync_interval_minutes} minutes")
        else:
            logger.info("🔒 Auto-sync disabled - manual sync only")
        
        # Log deduplication strategy
        logger.info(f"🔍 Deduplication strategy: {self.deduplication_strategy}")

        Path(FAILED_BATCH_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.arrow_save_dir).mkdir(parents=True, exist_ok=True)
        return self

    async def ensure_cache_initialized(self) -> None:
        """Ensure ChromaDB cache and collection are ready for use.
        
        This method handles both creation and reading scenarios:
        - Downloads remote ChromaDB to local cache if needed (with smart caching)
        - Initializes ChromaDB client
        - Creates collection if it doesn't exist
        - Validates existing collection compatibility
        """
        # Step 1: Handle remote ChromaDB caching with smart cache management
        if self.persist_directory.startswith(("gs://", "s3://", "azure://", "gcs://")):
            self._original_remote_path = self.persist_directory  # Store original remote path
            local_cache_path = await self._smart_cache_management(self.persist_directory)

            # Update persist_directory to use local cache
            self.persist_directory = str(local_cache_path)
            logger.info(f"✅ ChromaDB cache ready at: {local_cache_path}")

        # Step 2: Initialize ChromaDB client
        if not hasattr(self, "_client") or not self._client:
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            logger.debug(f"📁 ChromaDB client initialized: {self.persist_directory}")

        # Step 3: Ensure collection is ready (create or validate)
        await self._ensure_collection_ready()

    async def _smart_cache_management(self, remote_path: str) -> Path:
        """
        Smart cache management that prevents overwriting newer local changes.
        
        Args:
            remote_path: Remote GCS/S3 path to ChromaDB
            
        Returns:
            Path to local cache directory
        """
        from buttermilk.utils.utils import ensure_chromadb_cache
        import os
        import time

        # Get local cache path
        cache_path = Path.home() / ".cache" / "buttermilk" / "chromadb" / remote_path.replace("://", "___").replace("/", "_")

        # Check if local cache exists and has recent modifications
        local_exists = cache_path.exists() and (cache_path / "chroma.sqlite3").exists()

        if local_exists:
            # Check modification time of local cache
            local_mtime = os.path.getmtime(cache_path / "chroma.sqlite3")
            time_since_modified = time.time() - local_mtime

            # If modified within last hour, don't re-download
            if time_since_modified < 3600:  # 1 hour
                logger.info(f"📋 Using existing local cache (modified {time_since_modified/60:.1f} minutes ago)")
                logger.info(f"🔒 Skipping download to preserve local changes")
                return cache_path
            else:
                logger.info(f"⏰ Local cache is {time_since_modified/3600:.1f} hours old, checking for updates...")

        # Download remote ChromaDB (will skip if already up to date)
        logger.info(f"🔄 Syncing remote ChromaDB: {remote_path}")
        local_cache_path = await ensure_chromadb_cache(remote_path)

        return local_cache_path

    async def _sync_local_changes_to_remote(self) -> None:
        """
        Sync local ChromaDB changes back to remote storage.
        
        This method should be called after embedding operations to ensure
        local changes are persisted to the remote storage.
        """
        # Use original remote path if available, otherwise check current persist_directory
        remote_path = self._original_remote_path or self.persist_directory

        if not remote_path.startswith(("gs://", "gcs://", "s3://", "azure://")):
            # Not a remote storage, no sync needed
            logger.debug("Local storage detected, no remote sync needed")
            return

        try:
            from buttermilk.utils.utils import upload_chromadb_cache
            import os
            import time

            # Use actual persist_directory as the local cache path (it's been set to local cache)
            cache_path = Path(self.persist_directory)

            # Check if local cache exists and has been modified
            if not cache_path.exists() or not (cache_path / "chroma.sqlite3").exists():
                logger.error(f"Local cache not found at expected location: {cache_path}")
                return

            # Check if local cache has been recently modified
            local_mtime = os.path.getmtime(cache_path / "chroma.sqlite3")
            time_since_modified = time.time() - local_mtime

            # Only sync if modified within last 6 hours (indicates recent embedding work)
            if time_since_modified > 21600:  # 6 hours
                logger.debug(f"Local cache not recently modified ({time_since_modified/3600:.1f}h ago), skipping sync")
                return

            logger.info(f"🔄 Syncing local changes back to remote: {cache_path} → {remote_path}")

            # Upload local cache to remote storage
            await upload_chromadb_cache(str(cache_path), remote_path)
            logger.info("✅ Successfully synced local changes to remote storage")

        except Exception as e:
            logger.error(f"❌ CRITICAL: Failed to sync local changes to remote storage: {e}")
            logger.error(f"Local cache with unsaved data: {cache_path}")
            logger.error(f"Target remote path: {remote_path}")
            raise RuntimeError(f"ChromaDB sync failed: {e}") from e

    async def _should_sync_now(self, force: bool = False) -> bool:
        """Determine if we should sync to remote storage now.

        Args:
            force: Force sync regardless of batch/time thresholds

        Returns:
            bool: True if sync should happen now
        """
        if force:
            return True

        if not self._original_remote_path:
            return False

        import time

        current_time = time.time()

        # Check batch threshold
        batch_threshold_met = self._processed_records_count >= self._sync_batch_size

        # Check time threshold (sync every 10 minutes)
        time_threshold_met = (current_time - self._last_sync_time) >= self._sync_interval_seconds

        return batch_threshold_met or time_threshold_met

    async def _conditional_sync_to_remote(self, force: bool = False) -> bool:
        """Conditionally sync to remote storage based on batch/time thresholds.

        Args:
            force: Force sync regardless of thresholds and auto_sync setting

        Returns:
            bool: True if sync was performed, False if skipped
        """
        # Respect auto-sync setting unless forced
        if not force and self.disable_auto_sync:
            return False

        if not await self._should_sync_now(force):
            return False

        try:
            await self._sync_local_changes_to_remote()

            # Reset counters after successful sync
            import time

            self._processed_records_count = 0
            self._last_sync_time = time.time()

            logger.info(f"✅ Batch sync completed (reset counter to 0)")
            return True

        except Exception as e:
            logger.error(f"Conditional sync failed: {e}")
            return False

    async def sync_to_remote(self, force: bool = False) -> bool:
        """Manually sync local changes to remote storage.
        
        Args:
            force: Sync even if no recent changes detected
            
        Returns:
            bool: True if sync succeeded, False otherwise
        """
        if not self._original_remote_path:
            logger.info("No remote storage configured, nothing to sync")
            return True

        try:
            # Temporarily override time check for forced sync
            if force:
                # Backup original method and replace with forced version
                original_method = self._sync_local_changes_to_remote
                async def forced_sync():
                    cache_path = Path(self.persist_directory)
                    remote_path = self._original_remote_path

                    if not cache_path.exists() or not (cache_path / "chroma.sqlite3").exists():
                        logger.error(f"Local cache not found at: {cache_path}")
                        return False

                    logger.info(f"🔄 Force syncing: {cache_path} → {remote_path}")

                    from buttermilk.utils.utils import upload_chromadb_cache
                    await upload_chromadb_cache(str(cache_path), remote_path)
                    logger.info("✅ Force sync completed successfully")
                    return True

                await forced_sync()
            else:
                await self._sync_local_changes_to_remote()

            return True

        except Exception as e:
            logger.error(f"Manual sync failed: {e}")
            return False

    async def finalize_processing(self) -> bool:
        """Perform final sync at the end of processing session.

        Uses existing BM logging infrastructure for run metadata.

        Returns:
            bool: True if final sync succeeded, False otherwise
        """
        try:
            if self._processed_records_count > 0:
                logger.info(f"🔄 Performing final sync after processing {self._processed_records_count} records...")
                sync_success = await self._conditional_sync_to_remote(force=True)
                if sync_success:
                    logger.info("✅ Final sync completed successfully")
                    
                    # Log processing summary using existing BM logger
                    logger.info(f"📊 Processing session complete:")
                    logger.info(f"   📦 Records processed: {self._processed_records_count}")
                    logger.info(f"   🔢 Total embeddings: {self.collection.count()}")
                    logger.info(f"   🔍 Deduplication strategy: {self.deduplication_strategy}")
                    logger.info(f"   📦 Cache size: {len(self._processed_combinations_cache)} combinations")
                    
                    return True
                else:
                    logger.error("❌ Final sync failed")
                    return False
            else:
                logger.info("No records processed, no final sync needed")
                return True

        except Exception as e:
            logger.error(f"❌ Finalization failed: {e}")
            return False

    async def _ensure_collection_ready(self) -> None:
        """Ensure the collection exists and is compatible with current configuration.
        
        Handles both creation (if missing) and validation (if exists) scenarios.
        """
        if not self._client:
            raise RuntimeError("ChromaDB client must be initialized before ensuring collection")

        # Check if collection already exists
        existing_collections = self._client.list_collections()
        collection_names = [col.name for col in existing_collections]

        if self.collection_name in collection_names:
            logger.info(f"📖 Found existing collection '{self.collection_name}'")
            await self._validate_existing_collection()
        else:
            logger.info(f"🆕 Creating new collection '{self.collection_name}'")
            await self._create_new_collection()

    async def _validate_existing_collection(self) -> None:
        """Validate that existing collection is compatible with current config."""
        try:
            # Get existing collection to check its properties
            existing_collection = self._client.get_collection(
                name=self.collection_name
            )

            # Ensure embedding function is set on existing collection
            if hasattr(self, '_embedding_function') and self._embedding_function is not None:
                existing_collection._embedding_function = self._embedding_function

            # Get some basic stats
            count = existing_collection.count()
            logger.info(f"✅ Collection '{self.collection_name}' ready ({count} embeddings)")

            # TODO: Could add more sophisticated validation here:
            # - Check embedding dimensionality by sampling
            # - Verify metadata schema compatibility
            # - Check embedding model consistency

        except Exception as e:
            logger.warning(f"⚠️  Could not fully validate collection '{self.collection_name}': {e}")
            logger.info("Proceeding with existing collection...")

    async def _create_new_collection(self) -> None:
        """Create a new collection with proper configuration."""
        try:
            # Create collection with metadata and embedding function
            new_collection = self._client.create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_function,
                metadata={
                    "embedding_model": self.embedding_model,
                    "dimensionality": self.dimensionality,
                    "created_by": "buttermilk",
                    "task_type": self.task
                }
            )

            logger.info(f"✅ Created collection '{self.collection_name}' with {self.embedding_model} embeddings")
            logger.debug(f"   Dimensionality: {self.dimensionality}, Task: {self.task}")

        except Exception as e:
            # If creation fails, try get_or_create as fallback
            logger.warning(f"Direct creation failed, using get_or_create fallback: {e}")
            fallback_collection = self._client.get_or_create_collection(
                name=self.collection_name
            )
            # Ensure embedding function is set on fallback collection
            if hasattr(self, '_embedding_function') and self._embedding_function is not None:
                fallback_collection._embedding_function = self._embedding_function
            logger.info(f"✅ Collection '{self.collection_name}' ready via fallback")

    @property
    def collection(self) -> Collection:
        """Provides access to the ChromaDB collection.
        
        Note: Call ensure_cache_initialized() first for proper setup.
        """
        if not hasattr(self, "_client") or not self._client:
            # Provide helpful error message about initialization
            if self.persist_directory.startswith(("gs://", "s3://", "azure://", "gcs://")):
                raise ValueError(
                    f"Remote persist_directory '{self.persist_directory}' detected. "
                    "Please call ensure_cache_initialized() asynchronously before "
                    "accessing the collection."
                )
            else:
                raise ValueError(
                    "ChromaDB client not initialized. "
                    "Please call ensure_cache_initialized() before accessing the collection."
                )

        # Use cached collection or get it from client
        cache_key = f"vectorstore_{id(self)}"
        if _db_registry.get(cache_key) is None:
            # Get the collection (should exist after ensure_cache_initialized)
            try:
                _db_registry[cache_key] = self._client.get_collection(
                    name=self.collection_name
                )
            except Exception as e:
                # Fallback to get_or_create if get fails
                logger.warning(f"Failed to get collection, falling back to get_or_create: {e}")
                _db_registry[cache_key] = self._client.get_or_create_collection(
                    name=self.collection_name
                )

        # Ensure collection._embedding_function is synchronized with vectorstore._embedding_function
        collection = _db_registry[cache_key]
        if hasattr(self, '_embedding_function') and self._embedding_function is not None:
            collection._embedding_function = self._embedding_function

        return collection

    # REMOVED: record_to_input_document (deprecated)
    # Use process_record() directly with Record for better performance

    # REMOVED: create_multi_field_chunks (redundant with create_multi_field_chunks_for_record)
    # Use create_multi_field_chunks_for_record for Record objects

    def create_multi_field_chunks_for_record(self, record: Record) -> list[ChunkedDocument]:
        """Create chunks for multiple content types directly from Record.
        
        Uses multi_field_config to determine which fields to embed.
        Works directly with Record without conversion overhead.
        
        Args:
            record: Record instance to chunk
            
        Returns:
            list[ChunkedDocument]: List of chunks created from the record
        """
        chunks = []

        # If no multi-field config, use traditional single-field chunking
        if not self.multi_field_config:
            content_text = record.text_content
            if content_text:
                text_splitter = DefaultTextSplitter(chunk_size=2000, chunk_overlap=500)
                content_chunks = text_splitter.split_text(content_text)

                for i, chunk_text in enumerate(content_chunks):
                    if chunk_text.strip():
                        chunks.append(ChunkedDocument(
                            document_title=record.title or f"Record {record.record_id}",
                            chunk_index=len(chunks),
                            chunk_text=chunk_text.strip(),
                            document_id=record.record_id,
                            metadata=record.metadata
                        ))
            return chunks

        # Multi-field chunking based on configuration
        config = self.multi_field_config

        # 1. Main content field (chunked) - use content field configured in multi_field_config
        content_text = getattr(record, config.content_field, record.text_content)
        if isinstance(content_text, str) and content_text:
            logger.debug(f"Content text length for {record.record_id}: {len(content_text)} chars (from {config.content_field})")
            text_splitter = DefaultTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
            content_chunks = text_splitter.split_text(content_text)
            logger.debug(f"Text splitter with chunk_size={config.chunk_size} created {len(content_chunks)} content chunks for {record.record_id}")

            for i, chunk_text in enumerate(content_chunks):
                if chunk_text.strip():
                    chunks.append(ChunkedDocument(
                        document_title=record.title or f"Record {record.record_id}",
                        chunk_index=len(chunks),
                        chunk_text=chunk_text.strip(),
                        document_id=record.record_id,
                        metadata={
                            **record.metadata,
                            "content_type": config.content_field,
                            "chunk_type": "content"
                        }
                    ))

        # 2. Additional fields (single chunks each)
        for field_config in config.additional_fields:
            field_value = record.metadata.get(field_config.source_field, '')

            # Convert structured data to readable text
            if field_value:
                if isinstance(field_value, list):
                    # Join list items with newlines for better readability
                    chunk_text = '\n'.join(str(item).strip() for item in field_value if str(item).strip())
                elif isinstance(field_value, dict):
                    # Convert dict to key-value pairs
                    chunk_text = '\n'.join(f"{k}: {v}" for k, v in field_value.items() if v)
                else:
                    # Handle simple strings and other types
                    chunk_text = str(field_value).strip()

                if chunk_text and len(chunk_text) >= field_config.min_length:
                    chunks.append(ChunkedDocument(
                        document_title=record.title or f"Record {record.record_id}",
                        chunk_index=len(chunks),
                        chunk_text=chunk_text,
                        document_id=record.record_id,
                        metadata={
                            **record.metadata,
                            "content_type": field_config.source_field,
                            "chunk_type": field_config.chunk_type,
                            "original_type": type(field_value).__name__  # Track original data type
                        }
                    ))

        return chunks

    def _get_record_model_key(self, record_id: str, embedding_model: str) -> str:
        """Generate unique key for record+model combination."""
        return f"{record_id}:{embedding_model}"
    
    def _get_content_hash(self, record: Record) -> str:
        """Generate content hash for a record based on text content."""
        content = record.text_content or ""
        # Include metadata that affects embeddings
        metadata_str = json.dumps({
            k: v for k, v in sorted(record.metadata.items()) 
            if k in ["title", "summary", "description"]  # Only include fields that affect embedding
        }, sort_keys=True)
        combined_content = f"{content}|{metadata_str}"
        return hashlib.sha256(combined_content.encode()).hexdigest()

    async def _check_record_exists(self, record: Record) -> ExistenceCheck:
        """Check if record+model combination already has embeddings.
        
        Args:
            record: Record to check
            
        Returns:
            ExistenceCheck: Detailed information about existence
        """
        cache_key = self._get_record_model_key(record.record_id, self.embedding_model)
        
        # Check in-memory cache first
        if cache_key in self._processed_combinations_cache:
            return ExistenceCheck(
                exists=True,
                embedding_model=self.embedding_model,
                chunk_count=0,  # Not available from cache
                last_processed=None,  # Not available from cache
                metadata_hash=None
            )
            
        # Check ChromaDB for existing chunks with this record+model
        try:
            results = self.collection.get(
                where={
                    "$and": [
                        {"document_id": record.record_id},
                        {"embedding_model": self.embedding_model}
                    ]
                },
                limit=10,  # Get a few to count and check metadata
                include=["metadatas"]
            )
            
            exists = len(results.get("ids", [])) > 0
            chunk_count = len(results.get("ids", []))
            
            # Try to extract timestamp and content hash from metadata
            last_processed = None
            metadata_hash = None
            
            if exists and results.get("metadatas"):
                for metadata in results["metadatas"]:
                    if metadata:
                        # Try to parse timestamp
                        if "created_timestamp" in metadata:
                            try:
                                last_processed = datetime.fromisoformat(metadata["created_timestamp"])
                            except:
                                pass
                        
                        # Get content hash if available
                        if "content_hash" in metadata:
                            metadata_hash = metadata["content_hash"]
                            break
            
            if exists:
                # Add to cache for future checks
                self._processed_combinations_cache.add(cache_key)
                logger.debug(f"Found existing record {record.record_id} with {self.embedding_model}: {chunk_count} chunks")
            
            return ExistenceCheck(
                exists=exists,
                embedding_model=self.embedding_model,
                chunk_count=chunk_count,
                last_processed=last_processed,
                metadata_hash=metadata_hash
            )
            
        except Exception as e:
            logger.error(f"Error checking record+model existence: {e}")
            # Default to not exists to allow processing
            return ExistenceCheck(
                exists=False,
                embedding_model=self.embedding_model,
                chunk_count=0,
                last_processed=None,
                metadata_hash=None
            )

    async def _should_skip_record(self, record: Record, force_reprocess: bool = False) -> tuple[bool, str]:
        """Determine if a record should be skipped based on deduplication strategy.
        
        Args:
            record: Record to check
            force_reprocess: Force reprocessing even if exists
            
        Returns:
            tuple: (should_skip, reason)
        """
        if force_reprocess:
            return False, "forced reprocessing"
        
        existence_check = await self._check_record_exists(record)
        
        if not existence_check.exists:
            return False, "new record"
        
        # Handle different deduplication strategies
        if self.deduplication_strategy == "record_id":
            return True, f"record_id already exists with {self.embedding_model}"
        
        elif self.deduplication_strategy == "content_hash":
            current_hash = self._get_content_hash(record)
            if existence_check.metadata_hash and existence_check.metadata_hash == current_hash:
                return True, f"content unchanged (hash: {current_hash[:8]}...)"
            else:
                return False, f"content changed (old: {existence_check.metadata_hash[:8] if existence_check.metadata_hash else 'unknown'}..., new: {current_hash[:8]}...)"
        
        elif self.deduplication_strategy == "both":
            # More conservative - skip only if record_id exists AND content is the same
            current_hash = self._get_content_hash(record)
            if existence_check.metadata_hash and existence_check.metadata_hash == current_hash:
                return True, f"record_id and content both unchanged"
            else:
                return False, f"record exists but content may have changed"
        
        return False, "unknown deduplication strategy"

    async def process_record(
        self, 
        record: Record, 
        *,
        skip_existing: bool = True,
        validate_before_process: bool = True,
        embedding_model_override: str | None = None,
        force_reprocess: bool = False
    ) -> ProcessingResult:
        """Process a Record object with comprehensive deduplication and validation.
        
        Breaking Change: Now returns ProcessingResult instead of Record | None.
        
        Args:
            record: Record instance to process
            skip_existing: Whether to skip existing records (default: True)
            validate_before_process: Whether to validate before processing (default: True)
            embedding_model_override: Override embedding model for this record
            force_reprocess: Force reprocessing even if exists (default: False)
            
        Returns:
            ProcessingResult: Comprehensive result with status and metadata
        """
        start_time = time.time()
        
        # Override embedding model if specified
        effective_embedding_model = embedding_model_override or self.embedding_model
        
        try:
            # Step 1: Check if we should skip this record
            if skip_existing and not force_reprocess:
                should_skip, skip_reason = await self._should_skip_record(record, force_reprocess)
                if should_skip:
                    processing_time_ms = (time.time() - start_time) * 1000
                    return ProcessingResult(
                        record=None,
                        status="skipped",
                        reason=skip_reason,
                        chunks_created=0,
                        embedding_model=effective_embedding_model,
                        processing_time_ms=processing_time_ms,
                        metadata={"skip_validation": True}
                    )
            
            # Step 2: Validate record if requested
            if validate_before_process:
                if not record.text_content and not any(hasattr(record, field) for field in ["title", "summary", "description"]):
                    processing_time_ms = (time.time() - start_time) * 1000
                    return ProcessingResult(
                        record=None,
                        status="failed",
                        reason="no processable content found",
                        chunks_created=0,
                        embedding_model=effective_embedding_model,
                        processing_time_ms=processing_time_ms,
                        metadata={"validation_failed": True}
                    )
            
            # Step 3: Create chunks using configuration
            record.chunks = self.create_multi_field_chunks_for_record(record)

            if not record.chunks:
                processing_time_ms = (time.time() - start_time) * 1000
                return ProcessingResult(
                    record=None,
                    status="failed",
                    reason="no chunks created",
                    chunks_created=0,
                    embedding_model=effective_embedding_model,
                    processing_time_ms=processing_time_ms,
                    metadata={"chunking_failed": True}
                )

            # Step 4: Generate embeddings for all chunks
            await self._embed_chunks(record.chunks)

            # Step 5: Enhance chunk metadata with provenance tracking
            content_hash = self._get_content_hash(record)
            current_timestamp = datetime.now().isoformat()
            
            # Get BM run info if available
            try:
                from buttermilk._core.dmrc import get_bm
                bm = get_bm()
                run_id = bm.run_info.run_id if bm and bm.run_info else None
            except:
                run_id = None
            
            for chunk in record.chunks:
                chunk.metadata.update({
                    "embedding_model": effective_embedding_model,
                    "content_hash": content_hash,
                    "created_timestamp": current_timestamp,
                    "deduplication_strategy": self.deduplication_strategy
                })
                
                # Only add run_id if available from BM
                if run_id:
                    chunk.metadata["processing_run_id"] = run_id

            # Step 6: Store chunks in ChromaDB
            await self._store_chunks_for_record(record)

            # Step 7: Track successful processing
            cache_key = self._get_record_model_key(record.record_id, effective_embedding_model)
            self._processed_combinations_cache.add(cache_key)

            processing_time_ms = (time.time() - start_time) * 1000
            
            # Log successful processing
            chunk_types = {}
            for chunk in record.chunks:
                chunk_type = chunk.metadata.get('chunk_type', 'content')
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            logger.info(f"✅ Processed record {record.record_id}: {len(record.chunks)} chunks ({chunk_types}) in {processing_time_ms:.1f}ms")

            return ProcessingResult(
                record=record,
                status="processed",
                reason="successfully processed",
                chunks_created=len(record.chunks),
                embedding_model=effective_embedding_model,
                processing_time_ms=processing_time_ms,
                metadata={
                    "chunk_types": chunk_types,
                    "content_hash": content_hash,
                    "run_id": run_id
                }
            )

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Failed to process record {record.record_id}: {e}")
            return ProcessingResult(
                record=None,
                status="failed",
                reason=f"processing error: {str(e)}",
                chunks_created=0,
                embedding_model=effective_embedding_model,
                processing_time_ms=processing_time_ms,
                metadata={"error": str(e)}
            )

    async def _store_chunks_for_record(self, record: Record) -> None:
        """Store record chunks with metadata in ChromaDB.
        
        Args:
            record: Record with chunks and embeddings to store
        """
        try:
            if not record.chunks:
                logger.warning(f"No chunks to store for record {record.record_id}")
                return

            ids = []
            documents = []
            embeddings_list = []
            metadatas = []

            chunks_to_upsert = [c for c in record.chunks if c.embedding is not None]

            if not chunks_to_upsert:
                logger.warning(f"No chunks with embeddings to store for record {record.record_id}")
                return

            for chunk in chunks_to_upsert:
                ids.append(chunk.chunk_id)
                documents.append(chunk.chunk_text)
                embeddings_list.append(list(chunk.embedding))  # type: ignore

                # Enhanced metadata with content type tagging
                enhanced_metadata = {
                    "document_title": chunk.document_title,
                    "chunk_index": chunk.chunk_index,
                    "document_id": chunk.document_id,
                    "content_type": chunk.metadata.get("content_type", "unknown"),
                    "chunk_type": chunk.metadata.get("chunk_type", "unknown"),
                    **{k: v for k, v in chunk.metadata.items() if k not in ["content_type", "chunk_type"]}
                }
                metadatas.append(_sanitize_metadata_for_chroma(enhanced_metadata))

            logger.info(f"Upserting {len(ids)} chunks for record {record.record_id}...")

            # Execute the upsert operation
            await asyncio.to_thread(
                self.collection.upsert,
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents,
            )

            logger.info(f"Successfully stored {len(ids)} chunks for record {record.record_id}")

            # Increment processed records counter
            self._processed_records_count += 1

            # Conditionally sync based on batch/time thresholds (not after every record!)
            sync_performed = await self._conditional_sync_to_remote()
            if sync_performed:
                logger.info(f"🔄 Performed batch sync after processing record {record.record_id}")

        except Exception as e:
            logger.error(f"Failed to store chunks for record {record.record_id}: {e}")
            raise

    async def validate_incremental_update(self, new_records: list[Record]) -> dict[str, Any]:
        """Validate that new records can be safely added to existing collection.
        
        Args:
            new_records: List of records to validate
            
        Returns:
            dict: Validation results with safety assessment
        """
        validation_results = {
            "safe_to_add": True,
            "warnings": [],
            "conflicts": [],
            "stats": {
                "new_records": len(new_records),
                "existing_count": self.collection.count(),
                "would_skip": 0,
                "would_process": 0
            }
        }
        
        logger.info(f"🔍 Validating {len(new_records)} records for incremental update...")
        
        for record in new_records:
            try:
                should_skip, reason = await self._should_skip_record(record)
                if should_skip:
                    validation_results["stats"]["would_skip"] += 1
                    validation_results["warnings"].append(
                        f"Record {record.record_id}: {reason}"
                    )
                else:
                    validation_results["stats"]["would_process"] += 1
            except Exception as e:
                validation_results["conflicts"].append(
                    f"Record {record.record_id}: validation error - {str(e)}"
                )
                validation_results["safe_to_add"] = False
        
        # Check for potential issues
        if validation_results["stats"]["would_skip"] == len(new_records):
            validation_results["warnings"].append(
                "All records already exist - no new embeddings would be created"
            )
            
        if validation_results["conflicts"]:
            validation_results["safe_to_add"] = False
            
        logger.info(f"📋 Validation complete: {validation_results['stats']['would_process']} new, {validation_results['stats']['would_skip']} existing, {len(validation_results['conflicts'])} conflicts")
        
        return validation_results

    async def process_batch(
        self,
        records: list[Record],
        *,
        mode: Literal["safe", "force", "validate_only"] = "safe",
        max_failures: int = 0,
        require_all_new: bool = False,
    ) -> BatchProcessingResult:
        """Process records with mandatory validation and detailed results.
        
        Breaking Change: New batch-first API with comprehensive validation.
        
        Args:
            records: List of records to process
            mode: Processing mode - "safe" (default), "force", or "validate_only"
            max_failures: Maximum failures before stopping (0 = fail fast)
            require_all_new: Require all records to be new (fail if any exist)
            
        Returns:
            BatchProcessingResult: Comprehensive batch processing results
        """
        start_time = time.time()
        
        # Step 1: Validate the batch
        validation_result = await self.validate_incremental_update(records)
        
        if mode == "validate_only":
            processing_time_ms = (time.time() - start_time) * 1000
            return BatchProcessingResult(
                total_records=len(records),
                successful_count=0,
                skipped_count=validation_result["stats"]["would_skip"],
                failed_count=0,
                processing_time_ms=processing_time_ms,
                validation_result=validation_result,
                failed_records=[],
                metadata={"mode": "validate_only"}
            )
        
        # Check if validation passed for strict modes
        if require_all_new and validation_result["stats"]["would_skip"] > 0:
            processing_time_ms = (time.time() - start_time) * 1000
            return BatchProcessingResult(
                total_records=len(records),
                successful_count=0,
                skipped_count=0,
                failed_count=len(records),
                processing_time_ms=processing_time_ms,
                validation_result=validation_result,
                failed_records=[(r.record_id, "require_all_new failed") for r in records],
                metadata={"mode": mode, "require_all_new": True}
            )
        
        # Step 2: Process records
        successful_count = 0
        skipped_count = 0
        failed_count = 0
        failed_records = []
        
        force_reprocess = (mode == "force")
        
        logger.info(f"🏭 Processing batch of {len(records)} records (mode: {mode})")
        
        for i, record in enumerate(records):
            try:
                result = await self.process_record(
                    record,
                    skip_existing=(mode != "force"),
                    validate_before_process=True,
                    force_reprocess=force_reprocess
                )
                
                if result.status == "processed":
                    successful_count += 1
                elif result.status == "skipped":
                    skipped_count += 1
                elif result.status == "failed":
                    failed_count += 1
                    failed_records.append((record.record_id, result.reason))
                    
                    # Check failure threshold
                    if failed_count > max_failures:
                        logger.error(f"❌ Stopping batch processing: {failed_count} failures exceed max_failures={max_failures}")
                        # Mark remaining records as failed
                        remaining = len(records) - (i + 1)
                        failed_count += remaining
                        failed_records.extend([
                            (records[j].record_id, "batch stopped due to failures") 
                            for j in range(i + 1, len(records))
                        ])
                        break
                        
            except Exception as e:
                failed_count += 1
                failed_records.append((record.record_id, f"processing exception: {str(e)}"))
                logger.error(f"❌ Exception processing record {record.record_id}: {e}")
                
                # Check failure threshold
                if failed_count > max_failures:
                    logger.error(f"❌ Stopping batch processing: {failed_count} failures exceed max_failures={max_failures}")
                    break
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"✅ Batch processing complete: {successful_count} processed, {skipped_count} skipped, {failed_count} failed in {processing_time_ms:.1f}ms")
        
        return BatchProcessingResult(
            total_records=len(records),
            successful_count=successful_count,
            skipped_count=skipped_count,
            failed_count=failed_count,
            processing_time_ms=processing_time_ms,
            validation_result=validation_result,
            failed_records=failed_records,
            metadata={
                "mode": mode,
                "max_failures": max_failures,
                "require_all_new": require_all_new
            }
        )

    async def _embed_chunks(self, chunks: list[ChunkedDocument]) -> None:
        """Generate embeddings for a list of chunks in place.
        
        Args:
            chunks: List of ChunkedDocument objects to embed
        """
        if not chunks:
            return

        # Prepare embedding inputs
        embeddings_input: list[tuple[int, TextEmbeddingInput]] = []
        for i, chunk in enumerate(chunks):
            embeddings_input.append(
                (
                    i,  # Use list index as identifier
                    TextEmbeddingInput(
                        text=chunk.chunk_text,
                        task_type=self.task,
                        title=chunk.chunk_title,
                    ),
                ),
            )

        # Generate embeddings
        embedding_results = await self._embed(embeddings_input)

        # Assign embeddings back to chunks
        for idx, embedding in embedding_results:
            if embedding is not None and idx < len(chunks):
                chunks[idx].embedding = embedding
            elif embedding is None:
                logger.warning(f"Failed to generate embedding for chunk {idx}")

        logger.debug(f"Generated embeddings for {len([c for c in chunks if c.embedding is not None])} out of {len(chunks)} chunks")

    async def process_with_multi_field_chunks(self, doc: InputDocument) -> InputDocument | None:
        """Process document with configuration-driven multi-field chunking."""

        # Create chunks using configuration (generic approach)
        doc.chunks = self.create_multi_field_chunks(doc)

        if not doc.chunks:
            logger.warning(f"No chunks created for document {doc.record_id}")
            return None

        # Log chunk breakdown if multi-field config is used
        if self.multi_field_config:
            chunk_types = {}
            for chunk in doc.chunks:
                content_type = chunk.metadata.get('content_type', 'unknown')
                chunk_types[content_type] = chunk_types.get(content_type, 0) + 1

            breakdown = ", ".join([f"{count} {ctype}" for ctype, count in chunk_types.items()])
            logger.info(f"Multi-field chunks for {doc.record_id}: {breakdown}")
        else:
            logger.info(f"Single-field chunks for {doc.record_id}: {len(doc.chunks)} content")

        # Continue with normal embedding and storage process
        doc_with_embeddings = await self.embed_document(doc)
        if not doc_with_embeddings:
            return None

        # Store in ChromaDB
        return await self._store_chunks(doc_with_embeddings)

    # REMOVED: process_with_enhanced_metadata (legacy redirect)
    # Use process_record() for Record objects directly

    async def _store_chunks(self, doc: InputDocument) -> InputDocument:
        """Store document chunks with metadata in ChromaDB."""
        try:
            ids = []
            documents = []
            embeddings_list = []
            metadatas = []

            chunks_to_upsert = [c for c in doc.chunks if c.embedding is not None]

            for chunk in chunks_to_upsert:
                ids.append(chunk.chunk_id)
                documents.append(chunk.chunk_text)
                embeddings_list.append(list(chunk.embedding))  # type: ignore

                # Enhanced metadata with content type tagging
                enhanced_metadata = {
                    "document_title": chunk.document_title,
                    "chunk_index": chunk.chunk_index,
                    "document_id": chunk.document_id,
                    "content_type": chunk.metadata.get("content_type", "unknown"),
                    "chunk_type": chunk.metadata.get("chunk_type", "unknown"),
                    **{k: v for k, v in chunk.metadata.items() if k not in ["content_type", "chunk_type"]}
                }
                metadatas.append(_sanitize_metadata_for_chroma(enhanced_metadata))

            logger.info(f"Upserting {len(ids)} enhanced chunks for document {doc.record_id}...")

            # Execute the upsert operation
            await asyncio.to_thread(
                self.collection.upsert,
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents,
            )

            logger.info(f"Successfully stored enhanced chunks for document {doc.record_id}")

            # Increment processed records counter
            self._processed_records_count += 1

            # Conditionally sync based on batch/time thresholds (not after every document!)
            sync_performed = await self._conditional_sync_to_remote()
            if sync_performed:
                logger.info(f"🔄 Performed batch sync after processing document {doc.record_id}")

            return doc

        except Exception as e:
            logger.error(f"Failed to store enhanced chunks for document {doc.record_id}: {e}")
            return None

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

        # Update processed records counter for batch operations
        self._processed_records_count += successful_docs_upserted

        # For batch operations, always sync if we processed any records successfully
        # This ensures batch operations don't lose data
        if successful_docs_upserted > 0:
            sync_performed = await self._conditional_sync_to_remote(force=True)
            if sync_performed:
                logger.info(f"🔄 Performed batch sync after processing {successful_docs_upserted} documents")

        return successful_docs_upserted, failed_docs_upserted

    async def process(self, doc: InputDocument) -> InputDocument | None:
        """Process a document by embedding, saving to parquet, and upserting to ChromaDB.

        Follows the ProcessorCallable signature for compatibility with DocProcessor.
        
        If multi_field_config is provided and doc has no chunks, will create chunks using
        the multi-field approach. Otherwise expects chunks to already exist.
        """
        # If no chunks exist and we have multi-field config, create chunks automatically
        if not doc.chunks and self.multi_field_config:
            doc.chunks = self.create_multi_field_chunks(doc)

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

            # Increment processed records counter
            self._processed_records_count += 1

            # Conditionally sync based on batch/time thresholds (not after every document!)
            sync_performed = await self._conditional_sync_to_remote()
            if sync_performed:
                logger.info(f"🔄 Performed batch sync after processing document {doc.record_id}")

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
