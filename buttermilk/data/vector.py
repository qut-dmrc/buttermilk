import asyncio
import json
import signal
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self, TypeVar  # Corrected import for Tuple

import chromadb
import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import pydantic
import semchunk
from chromadb import Collection, Documents, EmbeddingFunction, Embeddings
from chromadb.api import ClientAPI
from google import genai
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, PrivateAttr
from tqdm.asyncio import tqdm
from vertexai.language_models import (
    TextEmbeddingInput,
)

from buttermilk import (
    buttermilk as bm,  # Global Buttermilk instance
)
from buttermilk import (
    logger,
)
from buttermilk._core.exceptions import RateLimit  # Import RateLimit exception
from buttermilk._core.log import logger  # noqa # Import logger from Buttermilk core
from buttermilk._core.retry import RetryWrapper  # Add retry functionality
from buttermilk._core.storage_config import VectorStorageConfig
from buttermilk._core.types import Record

ProcessingStatus = Literal["processed", "skipped", "failed"]
from buttermilk.utils.utils import convert_numpy_to_list, ensure_chromadb_cache

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


class ProcessingResult(BaseModel):
    """Comprehensive result from record processing."""

    record: Record | None
    status: Literal["processed", "skipped", "failed"]
    reason: str = Field(default="", description="Reason for skipping or failure")
    chunks_created: int = Field(default=0, description="Number of chunks created during processing")
    embedding_model: str = Field(default="n/a", description="Embedding model used for processing")
    processing_time_ms: float = Field(default=-1, description="Time taken to process the record in milliseconds")
    metadata: dict[str, Any] = Field(default={}, description="Additional metadata about the processing result")


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
        self.client: genai.Client = bm.genai
        self._embedding_model = embedding_model

    def __call__(self, input: Documents) -> Embeddings:
        response = self.client.models.embed_content(
            model=self._embedding_model,
            contents=input,
            config={
                "output_dimensionality": self.dimensionality,
            })

        # Extract embeddings from response
        embeddings = []
        for embedding in response.embeddings:
            # Convert to list if it's a numpy array
            embeddings.append(convert_numpy_to_list(embedding.values))

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
    embeddings_cache_dir: str = Field(default=".cache/embeddings", description="Directory to cache embeddings")

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
        if self.arrow_save_dir:
            Path(self.arrow_save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.embeddings_cache_dir).mkdir(parents=True, exist_ok=True)
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
                raise ValueError(
                    f"Record {record.record_id} has no chunks to process. Ensure it was chunked before processing.",
                )

            # --- Try to load embeddings from cache first ---
            cache_loaded = await self._load_embeddings_from_cache(record)

            if cache_loaded:
                # Embeddings loaded from cache, skip API call
                embedding_ok = True
                logger.info(f"📋 [VECTORIZER-{record.record_id}] Using cached embeddings, skipping API call")
            else:
                # --- Embeddings (now with robust retry) ---
                logger.debug(
                    f"🧬 [VECTORIZER-{record.record_id}] Generating embeddings for {len(record.chunks)} chunks..."
                )
                embedding_ok = await self._embed_chunks(record.chunks)

                # Save embeddings to cache if successful
                if embedding_ok:
                    await self._save_embeddings_to_cache(record)

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
                # Convert numpy array or list of numpy floats to regular Python floats
                if hasattr(chunk.embedding, "tolist"):
                    # It's a numpy array
                    embeddings_list.append(chunk.embedding.tolist())
                else:
                    # Convert any numpy float32/float64 to regular Python floats
                    embeddings_list.append([float(x) for x in chunk.embedding])  # type: ignore

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
            logger.error(f"Failed to store chunks for record {record.record_id}: {str(e)[:500]}")
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

    def _get_embeddings_cache_path(self, record: Record) -> Path:
        """Get the path to the embeddings cache file for a record.
        
        Returns the cache file path using the configured embeddings cache directory.
        """
        cache_dir = Path(self.embeddings_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{record.record_id}_embeddings.json"

    async def _save_embeddings_to_cache(self, record: Record) -> bool:
        """Save embeddings to cache file.
        
        Returns True if successfully saved, False otherwise.
        """
        cache_path = self._get_embeddings_cache_path(record)

        try:
            # Prepare embeddings data
            embeddings_data = {
                "record_id": record.record_id,
                "embedding_model": self._embedding_model,
                "dimensionality": self.dimensionality,
                "timestamp": datetime.now(UTC).isoformat(),
                "chunks": [],
            }

            for chunk in record.chunks:
                if chunk.embedding is not None:
                    chunk_data = {
                        "chunk_id": chunk.chunk_id,
                        "chunk_index": chunk.chunk_index,
                        "embedding": convert_numpy_to_list(chunk.embedding)  # Ensure it's regular Python list
                    }
                    embeddings_data["chunks"].append(chunk_data)

            # Save to file
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(embeddings_data, f, ensure_ascii=False, indent=2)

            logger.info(f"💾 Saved embeddings to cache: {cache_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save embeddings cache for {record.record_id}: {e}")
            return False

    async def _load_embeddings_from_cache(self, record: Record) -> bool:
        """Load embeddings from cache file if available and valid.

        Returns True if embeddings were loaded from cache, False otherwise.
        """
        cache_path = self._get_embeddings_cache_path(record)
        if not cache_path.exists():
            return False

        try:
            with cache_path.open("r", encoding="utf-8") as f:
                embeddings_data = json.load(f)

            # Validate cache is for correct model and record
            if (
                embeddings_data.get("record_id") != record.record_id
                or embeddings_data.get("embedding_model") != self._embedding_model
            ):
                logger.debug(f"Cache mismatch for {record.record_id}")
                return False

            # Check if we have the right number of chunks
            cached_chunks = embeddings_data.get("chunks", [])
            if len(cached_chunks) != len(record.chunks):
                logger.debug(f"Chunk count mismatch for {record.record_id}: cached={len(cached_chunks)}, current={len(record.chunks)}")
                return False

            # Load embeddings into chunks
            chunk_map = {chunk.chunk_id: chunk for chunk in record.chunks}
            loaded_count = 0

            for cached_chunk in cached_chunks:
                chunk_id = cached_chunk.get("chunk_id")
                if chunk_id in chunk_map:
                    chunk_map[chunk_id].embedding = cached_chunk.get("embedding")
                    loaded_count += 1

            if loaded_count == len(record.chunks):
                logger.info(f"✅ Loaded {loaded_count} embeddings from cache for record {record.record_id}")
                return True
            else:
                logger.warning(f"Only loaded {loaded_count}/{len(record.chunks)} embeddings from cache")
                # Clear partial embeddings
                for chunk in record.chunks:
                    chunk.embedding = None
                return False

        except Exception as e:
            logger.error(f"Failed to load embeddings cache for {record.record_id}: {e}")
            return False

    async def _embed_chunks(self, chunks: list[ChunkedDocument]) -> bool:
        """Generate embeddings for a list of chunks in place.
        
        Returns:
            bool: True if at least one embedding succeeded AND no hard failure.

        """
        if not chunks:
            return False

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
            texts.append(text_input)

        # Lazily initialize a retry wrapper tuned for embeddings if not already set.
        # We intentionally use higher wait times than the default because the
        # embedding API is prone to rate limiting in our workloads.
        if not hasattr(self, "_retry_wrapper") or self._retry_wrapper is None:
            try:
                self._retry_wrapper = RetryWrapper(
                    client=None,  # not used directly; we pass the callable
                    cooldown_seconds=self.embedding_cooldown_seconds,
                    max_retries=self.embedding_max_retries,
                    # Ensure minimum waits are elevated (at least 5s) and allow a higher ceiling.
                    min_wait_seconds=max(5.0, getattr(self, "embedding_min_wait_seconds", 5.0)),
                    max_wait_seconds=max(180.0, getattr(self, "embedding_max_wait_seconds", 120.0)),
                    jitter_seconds=10.0,
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"Failed to init embedding retry wrapper, falling back to single attempt: {e}")
                self._retry_wrapper = None

        async def _run_embed() -> list[Any]:
            return await asyncio.to_thread(self._embedding_function, texts)

        try:
            if self._retry_wrapper:
                embeddings = await self._retry_wrapper._execute_with_retry(_run_embed)
            else:
                embeddings = await _run_embed()
        except Exception as e:  # All retries exhausted or non-retryable error surfaced
            logger.error(f"Embedding failed after retries: {e}")
            self._convert_embedding_errors(e)
            return [(idx, None) for idx in indices]

        # Successful path
        results: list[tuple[int, list[float] | None]] = []
        for i, idx in enumerate(indices):
            if i < len(embeddings):
                results.append((idx, embeddings[i]))
            else:
                results.append((idx, None))
        return results

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

    def _write_record_to_parquet(self, record: Record, file_path: Path):
        """Synchronous helper to write Record chunks to a Parquet file."""
        if not record.chunks:
            logger.warning(
                f"Attempted to write empty chunks for record {record.record_id} to {file_path}. Skipping.",
            )
            return

        data = {
            "chunk_id": [c.chunk_id for c in record.chunks],
            "document_id": [c.document_id for c in record.chunks],
            "document_title": [c.document_title for c in record.chunks],
            "chunk_index": [c.chunk_index for c in record.chunks],
            "chunk_text": [c.chunk_text for c in record.chunks],
            "embedding": [list(c.embedding) if c.embedding is not None else None for c in record.chunks],
            "chunk_metadata": [json.dumps(c.metadata) if c.metadata else None for c in record.chunks],
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

        record_meta_serializable = {
            "record_id": record.record_id,
            "title": record.metadata.get("title", ""),
            "file_path": record.metadata.get("file_path", ""),
            "record_path": record.metadata.get("record_path", ""),
            "metadata": json.dumps(record.metadata),
        }
        arrow_metadata = {k.encode("utf-8"): str(v).encode("utf-8") for k, v in record_meta_serializable.items()}

        final_schema = table.schema.with_metadata(arrow_metadata)
        table = table.cast(final_schema)

        pq.write_table(table, file_path, compression="snappy")

    def _convert_embedding_errors(self, exc: Exception) -> None:
        """Convert embedding-specific errors to RateLimit exceptions for retry handling."""
        error_str = str(exc).lower()
        if any(keyword in error_str for keyword in ["quota", "rate limit", "429", "too many requests"]):
            raise RateLimit(str(exc)) from exc

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
        doc_iterator: AsyncIterator[Record],
    ) -> tuple[int, int]:
        """Upserts all chunks for each Record from the iterator into ChromaDB."""
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
                # Convert numpy array or list of numpy floats to regular Python floats
                if hasattr(rec.embedding, "tolist"):
                    # It's a numpy array
                    embeddings_list.append(rec.embedding.tolist())
                else:
                    # Convert any numpy float32/float64 to regular Python floats
                    embeddings_list.append([float(x) for x in rec.embedding])  # type: ignore
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
                    failed_doc_filename = Path(bm.save_dir) / Path(FAILED_BATCH_DIR) / f"failed_upsert_doc_{doc.record_id}_{uuid.uuid4()}.pkl"
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


# --- Async Pipeline Stages ---
class DocProcessor(BaseModel):
    """Callable class for processing documents from an iterator."""

    concurrency: int = Field(default=20)
    max_docs: int | None = Field(default=None, description="Maximum documents to process")
    count_only_yielded: bool = Field(
        default=True,
        description="If True, max_docs applies only to successfully yielded (processed) records; "
        "otherwise it applies to all attempts (including skipped/failed).",
    )
    return_results: bool = Field(
        default=False,
        description="If True, yield ProcessingResult objects (including skipped/failed) instead of filtering them out.",
    )
    _semaphore: asyncio.Semaphore = PrivateAttr()
    _record_cache: Any = PrivateAttr(default=None)
    doc_iterator: AsyncIterator[Record] | None = Field(default=None, exclude=True)
    processor: Callable[[Record], Awaitable[ProcessingResult | Record | None]] | None = Field(
        default=None, exclude=True
    )
    stage_name: str | None = Field(default=None, description="Explicit stage name to disambiguate caching/logging")
    _name: str = PrivateAttr(default="")
    _original_name: str = PrivateAttr(default="")
    _used_stage_names: dict[str, int] = {}
    enable_record_cache: bool = Field(default=True, description="Enable generic per-stage Record caching")
    force_reprocess: bool = Field(default=False, description="Ignore existing cache and re-run processor")

    # Internal counters (for diagnostics)
    _attempted: int = PrivateAttr(default=0)
    _yielded: int = PrivateAttr(default=0)
    _skipped: int = PrivateAttr(default=0)
    _failed: int = PrivateAttr(default=0)

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @pydantic.model_validator(mode="after")
    def _init(self) -> Self:
        # Initialize semaphore
        self._semaphore = asyncio.Semaphore(self.concurrency)
        # Resolve stage name
        base = self.stage_name or getattr(self.processor, "__name__", "stage")
        norm = base.lower().replace(" ", "_")
        # Ensure uniqueness inside a single process run
        count = self._used_stage_names.get(norm, 0)
        if count:
            unique = f"{norm}_{count + 1}"
            self._used_stage_names[norm] = count + 1
            self._name = unique
        else:
            self._used_stage_names[norm] = 1
            self._name = norm
        self._original_name = norm
        # Lazy import of record cache to avoid cycles
        try:
            from buttermilk._core.record_cache import RecordCache  # noqa

            self._record_cache = RecordCache()
        except Exception:  # pragma: no cover
            self._record_cache = None
        return self

    def _validate_cached_record(self, cached_record: Record) -> bool:
        if not cached_record:
            return False
        lowered = self._name.lower()
        if any(k in lowered for k in ("chunk", "split")):
            return bool(getattr(cached_record, "chunks", None))
        if any(k in lowered for k in ("vectoriz", "process_record")):
            # Embedding stage validation could be richer; assume presence of embedding_ids meta
            return True
        return True

    async def _process(self, doc: Record) -> ProcessingResult | Record | None:
        async with self._semaphore:
            try:
                # Cache shortcut
                if self.enable_record_cache and self._record_cache and not self.force_reprocess:
                    cached = self._record_cache.load(doc.record_id, self._name)
                    if cached and self._validate_cached_record(cached):
                        logger.debug(
                            f"⚡ Cache hit for record {doc.record_id} at stage '{self._name}' – skipping processing"
                        )
                        # Wrap cached as pseudo ProcessingResult (processed) if returning results
                        if self.return_results:
                            return ProcessingResult(
                                status="processed",
                                record=cached,
                                reason="cache hit",
                                chunks_created=len(cached.chunks) if hasattr(cached, "chunks") else 0,
                                embedding_model="unknown",
                                processing_time_ms=0,
                                metadata={"skip_validation": False, "cache_hit": True, "stage": self._name},
                            )
                        return cached
                if self.processor is None:
                    logger.error(f"[{self._name}] No processor callable configured")
                    if self.return_results:
                        return ProcessingResult(
                            status="failed",
                            record=doc,
                            reason="no_processor",
                            chunks_created=0,
                            embedding_model="unknown",
                            processing_time_ms=0,
                            metadata={"skip_validation": False, "cache_hit": False, "stage": self._name},
                        )
                    return None

                logger.info(
                    f"🔷 [{self._name}-{doc.record_id}] Processing record '{doc.title[:50] if doc.title else 'Unknown'}'"
                )
                result = await self.processor(doc)

                # Normalize to ProcessingResult for unified accounting
                if isinstance(result, ProcessingResult):
                    if result.status == "processed" and result.record:
                        # Cache only processed
                        if self.enable_record_cache and self._record_cache:
                            try:
                                include_chunks = bool(getattr(result.record, "chunks", None))
                                self._record_cache.save(result.record, self._name, include_chunks=include_chunks)
                            except Exception as ce:  # pragma: no cover
                                logger.debug(f"Cache save failed {doc.record_id} @ {self._name}: {ce}")
                    return result

                if result is None:
                    if self.return_results:
                        return ProcessingResult(status="skipped", record=doc, reason="processor_returned_none")
                    return None

                # result is a Record
                if self.enable_record_cache and self._record_cache:
                    try:
                        include_chunks = bool(getattr(result, "chunks", None))
                        self._record_cache.save(result, self._name, include_chunks=include_chunks)
                    except Exception as ce:  # pragma: no cover
                        logger.debug(f"Cache save failed {doc.record_id} @ {self._name}: {ce}")
                if self.return_results:
                    return ProcessingResult(status="processed", record=result)
                return result

            except Exception as e:
                logger.error(f"Error processing document {doc.record_id} in stage {self._name}: {e}")
                if self.return_results:
                    return ProcessingResult(status="failed", record=doc, reason=str(e))
                return None

    async def __call__(self) -> AsyncIterator[Record | ProcessingResult]:  # noqa: C901
        pending: set[asyncio.Task] = set()
        log_interval = 15.0
        last_log = time.monotonic()

        async def schedule(doc: Record):
            return await self._process(doc)

        upstream = self.doc_iterator
        if upstream is None:
            logger.error(f"[{self._name}] No upstream iterator configured")
            return

        async def maybe_log_status():
            nonlocal last_log
            now = time.monotonic()
            if now - last_log >= log_interval:
                logger.debug(
                    f"📊 Stage '{self._name}': attempted={self._attempted} yielded={self._yielded} "
                    f"skipped={self._skipped} failed={self._failed} pending={len(pending)}"
                )
                last_log = now

        try:
            upstream_aiter = upstream if hasattr(upstream, "__anext__") else upstream.__aiter__()
            async for doc in upstream_aiter:
                # Respect max_docs (attempt vs yielded semantics)
                if self.max_docs is not None:
                    if self.count_only_yielded:
                        if self._yielded >= self.max_docs:
                            logger.info(
                                f"🔚 Stage '{self._name}' reached max_docs (yielded={self._yielded}) – stopping intake"
                            )
                            break
                    elif self._attempted >= self.max_docs:
                        logger.info(
                            f"🔚 Stage '{self._name}' reached max_docs (attempted={self._attempted}) – stopping intake"
                        )
                        break

                # Maintain in-flight up to concurrency
                while len(pending) >= self.concurrency:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for t in done:
                        out = t.result()
                        await maybe_log_status()
                        if isinstance(out, ProcessingResult):
                            self._attempted += 1
                            if out.status == "processed":
                                self._yielded += 1
                                yield out if self.return_results else out.record
                            elif out.status == "skipped":
                                self._skipped += 1
                                if self.return_results:
                                    yield out
                            else:
                                self._failed += 1
                                if self.return_results:
                                    yield out
                        elif out:
                            self._attempted += 1
                            self._yielded += 1
                            yield out

                pending.add(asyncio.create_task(schedule(doc)))

            # Drain remaining
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    out = t.result()
                    await maybe_log_status()
                    if isinstance(out, ProcessingResult):
                        self._attempted += 1
                        if out.status == "processed":
                            self._yielded += 1
                            yield out if self.return_results else out.record
                        elif out.status == "skipped":
                            self._skipped += 1
                            if self.return_results:
                                yield out
                        else:
                            self._failed += 1
                            if self.return_results:
                                yield out
                    elif out:
                        self._attempted += 1
                        self._yielded += 1
                        yield out

            logger.info(
                f"✅ Stage '{self._name}' complete: attempted={self._attempted} yielded={self._yielded} "
                f"skipped={self._skipped} failed={self._failed}"
            )
        except Exception as e:
            logger.error(f"Stage '{self._name}' aborted: {e}")

# --- Main Execution ---


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg) -> None:
    # Track start time for statistics
    start_time = time.time()
    interrupted = False

    OmegaConf.resolve(cfg)

    bm = hydra.utils.instantiate(cfg.bm)

    from buttermilk._core.dmrc import set_bm

    set_bm(bm)  # Set the Buttermilk instance using the singleton pattern

    objs = hydra.utils.instantiate(cfg)
    vectoriser: ChromaDBEmbeddings = objs.vectoriser
    input_docs_source = objs.input_docs
    preprocessor_instance = objs.preprocessor
    processor_instance = objs.processor
    text_splitter_instance = objs.chunker

    # Set up signal handlers for graceful shutdown
    def handle_interrupt(signum, frame):
        nonlocal interrupted
        logger.warning("🛑 Interrupt received, finishing current batch...")
        interrupted = True

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    # Print startup banner
    logger.info("🚀 Vector Database Builder")
    logger.info("=" * 50)
    if hasattr(cfg, "name"):
        logger.info(f"Job: {cfg.name}")
    if hasattr(cfg, "storage"):
        logger.info(f"Storage: {cfg.storage.collection_name} at {cfg.storage.persist_directory}")
    logger.info("=" * 50)

    logger.info("Setting vector store instance on input document source.")
    input_docs_source.set_vector_store(vectoriser)

    loop = asyncio.get_event_loop()
    loop.slow_callback_duration = 35.0

    async def run_pipeline():
        from buttermilk._core.standalone_trace import create_standalone_trace

        # Create a standalone trace context for the entire pipeline
        trace_attributes = {
            "job_name": getattr(cfg, "name", "vector_batch"),
            "collection_name": cfg.storage.collection_name if hasattr(cfg, "storage") else None,
            "start_from": getattr(cfg, "start_from", 0),
            "max_docs": getattr(cfg, "max_docs", MAX_TOTAL_TASKS_PER_RUN),
        }

        async with create_standalone_trace("vector_pipeline", **trace_attributes):
            logger.info("Starting data processing pipeline...")

            # Initialize cache for remote storage
            await vectoriser.ensure_cache_initialized()

            # Get existing stats
            existing_count = vectoriser.collection.count()
            logger.info(f"📊 Existing embeddings in collection: {existing_count}")

            # 1. Source Documents
            start_from = getattr(cfg, "start_from", 0)
            max_docs = getattr(cfg, "max_docs", MAX_TOTAL_TASKS_PER_RUN)
            doc_iterator = input_docs_source.get_all_records(start=start_from, max_docs=max_docs)

            # 2. Pre-process (Extract Text if needed)
            pre_processed_iterator = DocProcessor(
                doc_iterator=doc_iterator,
                processor=preprocessor_instance.process,
                max_docs=max_docs,
                stage_name="preprocess",
            )

            # 3. Process Documents (e.g., add citations)
            processed_doc_iterator = DocProcessor(
                doc_iterator=pre_processed_iterator(),
                processor=processor_instance.process,
                max_docs=max_docs,
                stage_name="enrich",
            )

            # 4. Chunk Documents (Adds chunks to Record)
            chunked_doc_iterator = DocProcessor(
                doc_iterator=processed_doc_iterator(),
                processor=text_splitter_instance.process,
                max_docs=max_docs,
                stage_name="chunk",
            )

            # 5. Vectorize and Upsert
            vectorizer_processor = DocProcessor(
                doc_iterator=chunked_doc_iterator(),
                processor=vectoriser.process_record,
                concurrency=vectoriser.concurrency,
                max_docs=max_docs,
                stage_name="vectorize",
                return_results=True,
            )

            # Process documents through the complete pipeline with a limit
            quiet = getattr(cfg, "quiet", False)

            pbar = tqdm(total=max_docs, desc="Processing documents", disable=quiet)
            stats = {
                "total": 0,
                "embedded": 0,
                "skipped": 0,
                "failed": 0,
            }

            # Use the pipeline to process documents
            async for doc in vectorizer_processor():
                if interrupted:
                    logger.warning("🛑 Processing interrupted by user")
                    break

                stats["total"] += 1

                if doc is not None:
                    stats["embedded"] += 1
                else:
                    stats["failed"] += 1

                pbar.update(1)
                pbar.set_postfix({
                    "processed": stats["embedded"],
                    "failed": stats["failed"],
                })

                if stats["embedded"] >= max_docs:
                    logger.info(f"Reached document limit: {max_docs}")
                    break

            pbar.close()

            # Final sync for remote storage
            if not interrupted:
                logger.info("🔄 Performing final sync...")
                await vectoriser.finalize_processing()

            # Print summary statistics
            duration = time.time() - start_time
            final_count = vectoriser.collection.count()

            logger.info("\n" + "=" * 50)
            logger.info("📊 PROCESSING SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Total documents found: {stats['total']}")
            logger.info(f"Successfully processed: {stats['embedded']}")
            logger.info(f"Failed: {stats['failed']}")
            logger.info(f"Time elapsed: {duration:.1f} seconds")
            logger.info(f"Processing rate: {stats['total'] / duration:.1f} docs/second")
            logger.info(f"Total embeddings in collection: {final_count} (added {final_count - existing_count})")
            logger.info("=" * 50)

            if interrupted:
                logger.warning("⚠️  Processing was interrupted. Run again to resume.")
            else:
                logger.info("✅ Processing completed successfully!")

    loop.run_until_complete(run_pipeline())


if __name__ == "__main__":
    main()
