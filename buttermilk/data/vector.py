import asyncio
import json
import random
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self, TypeVar  # Corrected import for Tuple

import chromadb
import pydantic
import semchunk
from chromadb import Collection, Documents, EmbeddingFunction, Embeddings
from chromadb.api import ClientAPI
from google import genai
from pydantic import BaseModel, Field, PrivateAttr
from vertexai.language_models import (
    TextEmbeddingInput,
)

from buttermilk import (
    logger,
)
from buttermilk._core.log import logger  # noqa # Import logger from Buttermilk core
from buttermilk._core.retry import RetryWrapper  # Add retry functionality
from buttermilk._core.storage_config import VectorStorageConfig
from buttermilk._core.types import Record

ProcessingStatus = Literal["processed", "skipped", "failed"]
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
    # (multi-field embedding removed)


# --- Pydantic Models ---


class ChunkedDocument(BaseModel):
    """Represents a single chunk derived from a Record."""

    model_config = pydantic.ConfigDict(extra="ignore")

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_title: str
    chunk_index: int
    chunk_text: str
    offset: str | None | tuple[int, int] = Field(default=None, description="Offset of chunk in the original text")
    document_id: str  # References Record.record_id
    embedding: Sequence[float] | Sequence[int] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def chunk_title(self) -> str:
        """Generates a title hint for the embedding model."""
        return f"{self.document_title}_{self.chunk_index}"


# --- Type Aliases ---
ProcessorCallable = Callable[[Record], Awaitable[Record | ProcessingResult | None]]


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


class SemanticSplitter(BaseModel):
    # Defaults are roughly 750 words per chunk, with 250 word overlap
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=250)
    _chunker: semchunk.Chunker = PrivateAttr()

    model_config = pydantic.ConfigDict(extra="ignore")

    @pydantic.model_validator(mode="after")
    def initialize_chunker(self) -> Self:
        """Initializes the semantic chunker with the specified parameters."""
        self._chunker = semchunk.chunkerify("cl100k_base", self.chunk_size)
        logger.info(
            f"Initialized SemanticSplitter (chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})",
        )
        return self

    def _create_chunks(self, text) -> tuple[list[str], list[tuple[int, int]]]:
        # Pass an `offsets` argument to return the offsets of chunks, as well as an `overlap`
        # argument to overlap chunks by a ratio (if < 1) or an absolute number of tokens (if >= 1).
        chunks, offsets = self._chunker(text, offsets=True, overlap=self.chunk_overlap)

        return chunks, offsets

    async def process(self, doc: Record, **kwargs) -> Record | None:
        """Chunks documents and adds the chunks list to the Record."""
        # Extract text content from Record
        if hasattr(doc, "content"):
            text_content = doc.content if isinstance(doc.content, str) else str(doc.content)
        else:
            logger.warning(
                f"Skipping chunking for record {doc.record_id} due to missing content.",
            )
            return None

        if not text_content:
            logger.warning(
                f"Skipping chunking for record {doc.record_id} due to empty content.",
            )
            return None

        try:
            text_chunks, offsets = self._create_chunks(text_content)

            doc.chunks = []
            doc_chunk_count = 0
            for text_chunk, offset in zip(text_chunks, offsets, strict=True):
                if not text_chunk.strip():
                    continue
                doc.chunks.append(
                    ChunkedDocument(
                        document_title=doc.title,
                        chunk_index=doc_chunk_count,
                        chunk_text=text_chunk.strip(),
                        offset=offset,
                        document_id=doc.record_id,
                        chunk_id=f"{doc.record_id}_{doc_chunk_count}",
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


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(
        self,
        embedding_model: str,
        dimensionality: int = 3072,
    ):
        self.dimensionality = dimensionality
        self.client = genai.Client()
        self._embedding_model = embedding_model
        self._current_title = None  # Store title for current batch

    def set_title(self, title: str) -> None:
        """Set the title to use for the next embedding batch."""
        self._current_title = title

    def __call__(self, input: Documents) -> Embeddings:
        config_params = {
            "task_type": "retrieval_document",
            "output_dimensionality": self.dimensionality,
        }

        # Add title if available
        if self._current_title:
            config_params["title"] = self._current_title

        response = self.client.models.embed_content(
            model=self._embedding_model,
            contents=input,
            config=genai.types.EmbedContentConfig(**config_params),
        )

        # Extract embeddings from response
        embeddings = []
        for embedding in response.embeddings:
            embeddings.append(embedding.values)

        return embeddings


# --- Core Embedding and DB Interaction Class ---
class ChromaDBEmbeddings(VectorStorageConfig):
    """Handles configuration, embedding model interaction, and ChromaDB connection."""

    type: Literal["chromadb"] = "chromadb"
    model_config = pydantic.ConfigDict(extra="ignore")

    task: str = "RETRIEVAL_DOCUMENT"
    collection_name: str = Field(default=...)
    dimensionality: int = Field(default=3072)
    persist_directory: str = Field(default=...)
    concurrency: int = Field(default=20)
    upsert_batch_size: int = DEFAULT_UPSERT_BATCH_SIZE
    embedding_batch_size: int = Field(default=1)
    arrow_save_dir: str = Field(default="")

    # New sync configuration options
    sync_batch_size: int = Field(default=50, description="Sync every N records")
    sync_interval_minutes: int = Field(default=10, description="Sync every N minutes")
    disable_auto_sync: bool = Field(default=False, description="Disable automatic syncing (manual only)")

    # New deduplication configuration (Breaking Change)
    deduplication_strategy: Literal["record_id", "content_hash", "both"] = Field(default="both")

    # Retry configuration for embedding API calls
    embedding_max_retries: int = Field(default=5, description="Max retries for embedding API calls")
    embedding_min_wait_seconds: float = Field(default=1.0, description="Min wait between embedding retries")
    embedding_max_wait_seconds: float = Field(default=120.0, description="Max wait between embedding retries")
    embedding_cooldown_seconds: float = Field(default=0.1, description="Cooldown between successful embedding calls")

    _embedding_semaphore: asyncio.Semaphore = PrivateAttr()
    _collection: Collection = PrivateAttr()
    _embedding_model: str = PrivateAttr()
    _retry_wrapper: RetryWrapper = PrivateAttr()
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
        self._embedding_model = self.embedding_model  # Store the model name

        self._embedding_function = GeminiEmbeddingFunction(
            embedding_model=self.embedding_model,
            dimensionality=self.dimensionality,
        )
        # Wrap embedding model with retry logic
        self._retry_wrapper = RetryWrapper(
            client=self._embedding_function,
            max_retries=self.embedding_max_retries,
            min_wait_seconds=self.embedding_min_wait_seconds,
            max_wait_seconds=self.embedding_max_wait_seconds,
            cooldown_seconds=self.embedding_cooldown_seconds,
            jitter_seconds=2.0,  # Add some jitter for quota management
        )
        logger.info(f"🔄 Embedding retry configured: {self.embedding_max_retries} retries, {self.embedding_min_wait_seconds}-{self.embedding_max_wait_seconds}s backoff")

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
        """Smart cache management that prevents overwriting newer local changes.

        Args:
            remote_path: Remote GCS/S3 path to ChromaDB

        Returns:
            Path to local cache directory

        """
        import time

        # Get local cache path
        cache_path = Path.home() / ".cache" / "buttermilk" / "chromadb" / remote_path.replace("://", "___").replace("/", "_")

        # Check if local cache exists and has recent modifications
        local_exists = cache_path.exists() and (cache_path / "chroma.sqlite3").exists()

        if local_exists:
            # Check modification time of local cache
            local_mtime = Path(cache_path / "chroma.sqlite3").stat().st_mtime
            time_since_modified = time.time() - local_mtime

            # If modified within last hour, don't re-download
            if time_since_modified < 3600:  # 1 hour
                logger.info(f"📋 Using existing local cache (modified {time_since_modified / 60:.1f} minutes ago)")
                logger.info("🔒 Skipping download to preserve local changes")
                return cache_path
            logger.info(f"⏰ Local cache is {time_since_modified / 3600:.1f} hours old, checking for updates...")

        # Download remote ChromaDB (will skip if already up to date)
        logger.info(f"🔄 Syncing remote ChromaDB: {remote_path}")
        local_cache_path = await ensure_chromadb_cache(remote_path)

        return local_cache_path

    async def _sync_local_changes_to_remote(self) -> None:
        """Sync local ChromaDB changes back to remote storage.

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
            import time

            from buttermilk.utils.utils import upload_chromadb_cache

            # Use actual persist_directory as the local cache path (it's been set to local cache)
            cache_path = Path(self.persist_directory)

            # Check if local cache exists and has been modified
            if not cache_path.exists() or not (cache_path / "chroma.sqlite3").exists():
                logger.error(f"Local cache not found at expected location: {cache_path}")
                return

            # Check if local cache has been recently modified
            local_mtime = Path(cache_path / "chroma.sqlite3").stat().st_mtime
            time_since_modified = time.time() - local_mtime

            # Only sync if modified within last 6 hours (indicates recent embedding work)
            if time_since_modified > 21600:  # 6 hours
                logger.debug(f"Local cache not recently modified ({time_since_modified / 3600:.1f}h ago), skipping sync")
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

            logger.info("✅ Batch sync completed (reset counter to 0)")
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
                    logger.info("📊 Processing session complete:")
                    logger.info(f"   📦 Records processed: {self._processed_records_count}")
                    logger.info(f"   🔢 Total embeddings: {self.collection.count()}")
                    logger.info(f"   🔍 Deduplication strategy: {self.deduplication_strategy}")
                    logger.info(f"   📦 Cache size: {len(self._processed_combinations_cache)} combinations")

                    return True
                logger.error("❌ Final sync failed")
                return False
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
                name=self.collection_name,
            )

            # Ensure embedding function is set on existing collection
            if hasattr(self, "_embedding_function") and self._embedding_function is not None:
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
                    "task_type": self.task,
                },
            )

            logger.info(f"✅ Created collection '{self.collection_name}' with {self.embedding_model} embeddings")
            logger.debug(f"   Dimensionality: {self.dimensionality}, Task: {self.task}")

        except Exception as e:
            # If creation fails, try get_or_create as fallback
            logger.warning(f"Direct creation failed, using get_or_create fallback: {e}")
            fallback_collection = self._client.get_or_create_collection(
                name=self.collection_name,
            )
            # Ensure embedding function is set on fallback collection
            if hasattr(self, "_embedding_function") and self._embedding_function is not None:
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
                    "accessing the collection.",
                )
            raise ValueError(
                "ChromaDB client not initialized. Please call ensure_cache_initialized() before accessing the collection.",
            )

        # Use cached collection or get it from client
        cache_key = f"vectorstore_{id(self)}"
        if _db_registry.get(cache_key) is None:
            # Get the collection (should exist after ensure_cache_initialized)
            try:
                _db_registry[cache_key] = self._client.get_collection(
                    name=self.collection_name,
                )
            except Exception as e:
                # Fallback to get_or_create if get fails
                logger.warning(f"Failed to get collection, falling back to get_or_create: {e}")
                _db_registry[cache_key] = self._client.get_or_create_collection(
                    name=self.collection_name,
                )

        # Ensure collection._embedding_function is synchronized with vectorstore._embedding_function
        collection = _db_registry[cache_key]
        if hasattr(self, "_embedding_function") and self._embedding_function is not None:
            collection._embedding_function = self._embedding_function

        return collection

    async def process_record(
        self,
        record: Record,
        *,
        skip_existing: bool = True,
        validate_before_process: bool = True,
        embedding_model_override: str | None = None,
        force_reprocess: bool = False,
    ) -> ProcessingResult:
        """Process a Record object with deduplication & embedding.

        NOTE: Assumes prior pipeline stage already chunked (SemanticSplitter).
        Falls back to simple semantic splitting only if no chunks are present.
        """
        start_time = time.time()
        effective_embedding_model = embedding_model_override or self._embedding_model
        logger.info(f"🟣 [VECTORIZER-{record.record_id}] Starting to process record '{record.title[:50] if record.title else 'Unknown'}'")

        try:
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
                        metadata={"skip_validation": True},
                    )

            if validate_before_process:
                if not getattr(record, "chunks", None) and not record.text_content:
                    processing_time_ms = (time.time() - start_time) * 1000
                    return ProcessingResult(
                        record=None,
                        status="failed",
                        reason="no processable content found",
                        chunks_created=0,
                        embedding_model=effective_embedding_model,
                        processing_time_ms=processing_time_ms,
                        metadata={"validation_failed": True},
                    )

            if getattr(record, "chunks", None):
                logger.debug(f"🧩 [VECTORIZER-{record.record_id}] Using pre-existing {len(record.chunks)} chunks")
            else:
                logger.debug(f"🔪 [VECTORIZER-{record.record_id}] No chunks present, performing fallback semantic split")
                fallback_splitter = SemanticSplitter(chunk_size=1000, chunk_overlap=250)
                processed = await fallback_splitter.process(record)
                if processed:
                    record = processed
                if not getattr(record, "chunks", None):
                    processing_time_ms = (time.time() - start_time) * 1000
                    return ProcessingResult(
                        record=None,
                        status="failed",
                        reason="no chunks created (fallback)",
                        chunks_created=0,
                        embedding_model=effective_embedding_model,
                        processing_time_ms=processing_time_ms,
                        metadata={"chunking_failed": True},
                    )

            # --- Embeddings (now with robust retry) ---
            logger.debug(f"🧬 [VECTORIZER-{record.record_id}] Generating embeddings for {len(record.chunks)} chunks...")
            embedding_ok = await self._embed_chunks(record.chunks, record_title=record.title)

            if not embedding_ok:
                # Persist failed record for later retry BEFORE returning
                try:
                    failed_path = Path(FAILED_BATCH_DIR) / f"failed_embedding_record_{record.record_id}_{uuid.uuid4()}.json"
                    failed_payload = {
                        "record_id": record.record_id,
                        "title": record.title,
                        "reason": "embedding_failed",
                        "chunk_count": len(record.chunks),
                        "metadata": record.metadata,
                        "content_hash": self._get_content_hash(record),
                        "created_at": datetime.utcnow().isoformat(),
                    }
                    failed_path.write_text(json.dumps(failed_payload, ensure_ascii=False, indent=2))
                    logger.warning(f"💾 Saved failed embedding record for retry: {failed_path}")
                except Exception as save_e:
                    logger.error(f"Could not persist failed embedding record {record.record_id}: {save_e}")

                processing_time_ms = (time.time() - start_time) * 1000
                return ProcessingResult(
                    record=None,
                    status="failed",
                    reason="embedding failed after retries",
                    chunks_created=0,
                    embedding_model=effective_embedding_model,
                    processing_time_ms=processing_time_ms,
                    metadata={"embedding_failed": True},
                )

            # --- Metadata enhancement ---
            content_hash = self._get_content_hash(record)
            current_timestamp = datetime.now().isoformat()
            try:
                from buttermilk._core.dmrc import get_bm
                bm = get_bm()
                run_id = bm.run_info.run_id if bm and bm.run_info else None
            except:
                run_id = None

            for chunk in record.chunks:
                chunk.metadata.update(
                    {
                        "embedding_model": effective_embedding_model,
                        "content_hash": content_hash,
                        "created_timestamp": current_timestamp,
                        "deduplication_strategy": self.deduplication_strategy,
                    },
                )
                if run_id:
                    chunk.metadata["processing_run_id"] = run_id

            logger.debug(f"💾 [VECTORIZER-{record.record_id}] Storing chunks in ChromaDB...")
            await self._store_chunks_for_record(record)

            cache_key = self._get_record_model_key(record.record_id, effective_embedding_model)
            self._processed_combinations_cache.add(cache_key)

            processing_time_ms = (time.time() - start_time) * 1000

            chunk_types = {}
            for chunk in record.chunks:
                chunk_type = chunk.metadata.get("chunk_type", "content")
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

            logger.info(f"✅ [VECTORIZER-{record.record_id}] Successfully processed: {len(record.chunks)} chunks ({chunk_types}) in {processing_time_ms:.1f}ms")

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
                    "run_id": run_id,
                },
            )

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Failed to process record {record.record_id}: {e}")
            return ProcessingResult(
                record=None,
                status="failed",
                reason=f"processing error: {e!s}",
                chunks_created=0,
                embedding_model=effective_embedding_model,
                processing_time_ms=processing_time_ms,
                metadata={"error": str(e)},
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
                    **{k: v for k, v in chunk.metadata.items() if k not in ["content_type", "chunk_type"]},
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
                "would_process": 0,
            },
        }

        logger.info(f"🔍 Validating {len(new_records)} records for incremental update...")

        for record in new_records:
            try:
                should_skip, reason = await self._should_skip_record(record)
                if should_skip:
                    validation_results["stats"]["would_skip"] += 1
                    validation_results["warnings"].append(
                        f"Record {record.record_id}: {reason}",
                    )
                else:
                    validation_results["stats"]["would_process"] += 1
            except Exception as e:
                validation_results["conflicts"].append(
                    f"Record {record.record_id}: validation error - {e!s}",
                )
                validation_results["safe_to_add"] = False

        # Check for potential issues
        if validation_results["stats"]["would_skip"] == len(new_records):
            validation_results["warnings"].append(
                "All records already exist - no new embeddings would be created",
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
                metadata={"mode": "validate_only"},
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
                metadata={"mode": mode, "require_all_new": True},
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
                    force_reprocess=force_reprocess,
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
                failed_records.append((record.record_id, f"processing exception: {e!s}"))
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
                "require_all_new": require_all_new,
            },
        )

    async def _embed_chunks(self, chunks: list[ChunkedDocument], record_title: str | None = None) -> bool:
        """Generate embeddings for a list of chunks in place.
        
        Returns:
            bool: True if at least one embedding succeeded AND no hard failure.

        """
        if not chunks:
            return False

        if record_title and hasattr(self._embedding_function, "set_title"):
            self._embedding_function.set_title(record_title)

        embeddings_input: list[tuple[int, TextEmbeddingInput]] = []
        for i, chunk in enumerate(chunks):
            embeddings_input.append(
                (
                    i,
                    TextEmbeddingInput(
                        text=chunk.chunk_text,
                        task_type=self.task,
                        title=chunk.chunk_title,
                    ),
                ),
            )

        embedding_results = await self._embed(embeddings_input)

        success_count = 0
        for idx, embedding in embedding_results:
            if embedding is not None and idx < len(chunks):
                chunks[idx].embedding = embedding
                success_count += 1

        if success_count == 0:
            logger.error("All embeddings failed for this record")
            return False

        if success_count < len(chunks):
            logger.warning(f"Partial embedding: {success_count}/{len(chunks)} succeeded; failing record to retry later")
            # Clear embeddings so we don't upsert partials
            for c in chunks:
                c.embedding = None
            return False

        logger.debug(f"Generated embeddings for {success_count} chunks")
        return True

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(k in msg for k in ["rate limit", "quota", "too many requests", "429"])

    async def _embed(self, embeddings_input: list[tuple[int, Any]]) -> list[tuple[int, list[float] | None]]:
        """Generate embeddings with retry/backoff on rate limits."""
        if not embeddings_input:
            return []

        texts = []
        indices = []
        for idx, text_input in embeddings_input:
            indices.append(idx)
            if hasattr(text_input, "text"):
                texts.append(text_input.text)
            elif isinstance(text_input, str):
                texts.append(text_input)
            else:
                texts.append(str(text_input))

        max_retries = self.embedding_max_retries
        min_wait = self.embedding_min_wait_seconds
        max_wait = self.embedding_max_wait_seconds
        cooldown = self.embedding_cooldown_seconds

        for attempt in range(1, max_retries + 1):
            try:
                embeddings = await asyncio.to_thread(self._embedding_function, texts)
                # Optional cooldown after success
                if cooldown:
                    await asyncio.sleep(cooldown)
                results = []
                for i, idx in enumerate(indices):
                    if i < len(embeddings):
                        results.append((idx, embeddings[i]))
                    else:
                        results.append((idx, None))
                return results
            except Exception as e:
                if self._is_rate_limit_error(e):
                    if attempt == max_retries:
                        logger.error(f"Rate limit persists after {attempt} attempts: {e}")
                        break
                    wait = min(max_wait, min_wait * (2 ** (attempt - 1)))
                    jitter = random.uniform(0, min(2.0, wait * 0.25))
                    logger.warning(f"Rate limit (attempt {attempt}/{max_retries}) - backing off {wait + jitter:.1f}s")
                    await asyncio.sleep(wait + jitter)
                    continue
                # Non-rate-limit error: do not retry
                logger.error(f"Embedding failed (non-retryable): {e}")
                break

        # Failure path
        self._convert_embedding_errors(Exception("embedding failed after retries"))
        return [(idx, None) for idx in indices]

    # ------------------------------------------------------------------
    # Deduplication & Skip Logic
    # ------------------------------------------------------------------
    def _get_record_model_key(self, record_id: str, embedding_model: str) -> str:
        """Stable cache key to remember processed record/model combos this run."""
        return f"{record_id}::{embedding_model}"

    def _extract_raw_text(self, record: Record) -> str:
        """Return the raw text used for hashing (pre-chunk)."""
        if hasattr(record, "content") and record.content:
            return record.content if isinstance(record.content, str) else str(record.content)
        if hasattr(record, "text_content") and record.text_content:
            return record.text_content if isinstance(record.text_content, str) else str(record.text_content)
        return ""

    def _get_content_hash(self, record: Record) -> str:
        """Compute a stable content hash for deduplication (content + minimal metadata)."""
        import hashlib
        import json
        raw_text = self._extract_raw_text(record)
        # Include a shallow, deterministic subset of metadata that might affect semantics
        meta = {}
        if hasattr(record, "metadata") and isinstance(record.metadata, dict):
            # Pick only stable scalar fields to avoid hash churn
            for k, v in record.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
        payload = json.dumps({"text": raw_text, "meta": meta}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _query_collection_single(
        self,
        where: dict[str, Any],
        limit: int = 5,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """Synchronous helper to query collection safely."""
        try:
            return self.collection.get(where=where, limit=limit, include=include or ["metadatas", "ids"])
        except Exception as e:
            logger.warning(f"Collection query failed (where={where}): {e}")
            return {"ids": [], "metadatas": []}

    async def _should_skip_record(
        self,
        record: Record,
        force_reprocess: bool = False,
        embedding_model: str | None = None,
    ) -> tuple[bool, str]:
        """Determine whether to skip processing a record.

        Returns:
            (should_skip, reason)

        """
        if force_reprocess:
            return False, "force reprocess requested"

        embedding_model = embedding_model or getattr(self, "_embedding_model", self.embedding_model)
        cache_key = self._get_record_model_key(record.record_id, embedding_model)

        # Fast in-run cache: if we've processed this combo already in this run, skip.
        if cache_key in self._processed_combinations_cache:
            return True, "already processed this run"

        strategy = self.deduplication_strategy
        content_hash = self._get_content_hash(record)

        # Helper closures
        def has_matching_content(metadatas: list[dict[str, Any]]) -> bool:
            for md in metadatas:
                if md and md.get("content_hash") == content_hash and md.get("embedding_model") == embedding_model:
                    return True
            return False

        if strategy in ("record_id", "both"):
            # Query by record_id first
            res = await asyncio.to_thread(
                self._query_collection_single,
                {"document_id": record.record_id, "embedding_model": embedding_model},
            )
            ids = res.get("ids", []) or []
            metas = res.get("metadatas", []) or []

            if not ids:
                # No existing chunks for this record id -> cannot skip yet
                if strategy == "record_id":
                    return False, "record_id not present"
            else:
                if strategy == "record_id":
                    return True, "record_id already embedded"
                # strategy == both: need to verify content hash
                if has_matching_content(metas):
                    return True, "record_id & content hash match"
                return False, "record_id exists but content changed"

        if strategy == "content_hash":
            # Query by content_hash only
            res = await asyncio.to_thread(
                self._query_collection_single,
                {"content_hash": content_hash, "embedding_model": embedding_model},
            )
            ids = res.get("ids", []) or []
            if ids:
                return True, "content hash already embedded"
            return False, "content hash not present"

        if strategy == "both":
            # If we reach here, record_id was absent; fall back to content_hash check.
            res = await asyncio.to_thread(
                self._query_collection_single,
                {"content_hash": content_hash, "embedding_model": embedding_model},
            )
            ids = res.get("ids", []) or []
            if ids:
                # This means same content under a different record_id; treat as duplicate.
                return True, "identical content (hash) already embedded under different record_id"
            return False, "new record_id & content hash"

        # Fallback (should not occur)
        return False, "no matching deduplication strategy"

    # ------------------------------------------------------------------
    # (Rest of class continues...)
    # ------------------------------------------------------------------
