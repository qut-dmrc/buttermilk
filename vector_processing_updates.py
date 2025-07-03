"""Updated vector processing methods to work with the enhanced Record class.

This module shows how to update the existing vector processing code in 
buttermilk/data/vector.py to work with the new EnhancedRecord while 
maintaining backward compatibility with InputDocument.
"""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from vertexai.language_models import TextEmbedding, TextEmbeddingInput, TextEmbeddingModel

from enhanced_record_design import EnhancedRecord, ChunkData
from migration_strategy import CompatibilityLayer, vector_processor_adapter
from buttermilk._core.storage_config import MultiFieldEmbeddingConfig
from buttermilk import logger

# ========== UPDATED CHROMADB EMBEDDINGS CLASS ==========

class EnhancedChromaDBEmbeddings(BaseModel):
    """Enhanced ChromaDB embeddings class that works with EnhancedRecord.
    
    This replaces the existing ChromaDBEmbeddings class with full support for
    the EnhancedRecord format while maintaining backward compatibility.
    """
    
    # Configuration fields (same as before)
    type: Literal["chromadb"] = "chromadb"
    embedding_model: str = Field(default="gemini-embedding-001")
    task: str = "RETRIEVAL_DOCUMENT"
    collection_name: str = Field(default=...)
    dimensionality: int = Field(default=3072)
    persist_directory: str = Field(default=...)
    concurrency: int = Field(default=20)
    upsert_batch_size: int = Field(default=10)
    embedding_batch_size: int = Field(default=1)
    arrow_save_dir: str = Field(default="")
    multi_field_config: MultiFieldEmbeddingConfig | None = Field(default=None)

    # Private attributes
    _embedding_semaphore: asyncio.Semaphore = PrivateAttr()
    _collection: Any = PrivateAttr()  # ChromaDB Collection
    _embedding_model: TextEmbeddingModel = PrivateAttr()
    _embedding_function: Any = PrivateAttr()
    _client: Any = PrivateAttr()  # ChromaDB ClientAPI

    model_config = {"extra": "ignore", "arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def load_models(self):
        """Initialize models and connections."""
        # Same initialization as before
        logger.info(f"Loading embedding model: {self.embedding_model}")
        self._embedding_model = TextEmbeddingModel.from_pretrained(self.embedding_model)
        self._embedding_semaphore = asyncio.Semaphore(self.concurrency)
        
        # Initialize ChromaDB client (implementation same as before)
        # ... (ChromaDB initialization code)
        
        return self

    # ========== ENHANCED RECORD PROCESSING METHODS ==========

    def process_enhanced_record(self, record: EnhancedRecord) -> EnhancedRecord:
        """Process an EnhancedRecord directly with multi-field chunking."""
        return asyncio.run(self.process_enhanced_record_async(record))

    async def process_enhanced_record_async(self, record: EnhancedRecord) -> EnhancedRecord | None:
        """Async processing of EnhancedRecord with multi-field chunking."""
        # Create chunks using configuration
        chunks = self.create_multi_field_chunks_enhanced(record)
        
        if not chunks:
            logger.warning(f"No chunks created for record {record.record_id}")
            return None
        
        # Update record with chunks
        record.chunks = chunks
        
        # Log chunk breakdown if multi-field config is used
        if self.multi_field_config:
            chunk_types = {}
            for chunk in record.chunks:
                chunk_type = chunk.chunk_type
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            breakdown = ", ".join([f"{count} {ctype}" for ctype, count in chunk_types.items()])
            logger.info(f"Multi-field chunks for {record.record_id}: {breakdown}")
        else:
            logger.info(f"Single-field chunks for {record.record_id}: {len(record.chunks)} content")
        
        # Generate embeddings
        record_with_embeddings = await self.embed_record_enhanced(record)
        if not record_with_embeddings:
            return None
            
        # Store in ChromaDB 
        return await self.store_chunks_enhanced(record_with_embeddings)

    def create_multi_field_chunks_enhanced(self, record: EnhancedRecord) -> list[ChunkData]:
        """Create chunks for multiple content types using EnhancedRecord."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        chunks = []
        
        # If no multi-field config, use traditional single-field chunking
        if not self.multi_field_config:
            content_text = record.text_content
            if content_text:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
                content_chunks = text_splitter.split_text(content_text)
                
                for i, chunk_text in enumerate(content_chunks):
                    if chunk_text.strip():
                        chunk = ChunkData(
                            chunk_index=len(chunks),
                            chunk_text=chunk_text.strip(),
                            chunk_type="content",
                            source_field="content",
                            metadata=record.metadata.copy()
                        )
                        chunks.append(chunk)
            return chunks
        
        # Multi-field chunking based on configuration
        config = self.multi_field_config
        
        # 1. Main content field (chunked)
        content_text = record.text_content
        if content_text:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
            content_chunks = text_splitter.split_text(content_text)
            
            for i, chunk_text in enumerate(content_chunks):
                if chunk_text.strip():
                    chunk = ChunkData(
                        chunk_index=len(chunks),
                        chunk_text=chunk_text.strip(),
                        chunk_type="content",
                        source_field=config.content_field,
                        metadata={
                            **record.metadata,
                            "content_type": config.content_field,
                            "chunk_type": "content",
                            "chunk_sequence": i
                        }
                    )
                    chunks.append(chunk)
        
        # 2. Additional fields (single chunks each)
        for field_config in config.additional_fields:
            field_value = record.metadata.get(field_config.source_field, '')
            
            # Only embed if field exists and meets minimum length requirement
            if field_value and len(str(field_value).strip()) >= field_config.min_length:
                chunk = ChunkData(
                    chunk_index=len(chunks),
                    chunk_text=str(field_value).strip(),
                    chunk_type=field_config.chunk_type,
                    source_field=field_config.source_field,
                    metadata={
                        **record.metadata,
                        "content_type": field_config.source_field,
                        "chunk_type": field_config.chunk_type
                    }
                )
                chunks.append(chunk)
                
        return chunks

    async def embed_record_enhanced(self, record: EnhancedRecord) -> EnhancedRecord | None:
        """Generate embeddings for chunks within an EnhancedRecord."""
        if not record.chunks:
            logger.warning(f"No chunks found for record {record.record_id}, cannot embed.")
            return None

        logger.debug(f"Generating embeddings for record {record.record_id} with {len(record.chunks)} chunks.")

        # Prepare embedding inputs
        embeddings_input: list[tuple[int, TextEmbeddingInput]] = []
        for chunk in record.chunks:
            embeddings_input.append((
                chunk.chunk_index,
                TextEmbeddingInput(
                    text=chunk.chunk_text,
                    task_type=self.task,
                    title=chunk.chunk_title,
                )
            ))

        # Get embeddings
        embedding_results = await self._embed_batch(embeddings_input)

        # Assign embeddings to chunks
        successful_embeddings = 0
        for chunk_index, embedding in embedding_results:
            # Find the chunk with this index
            chunk = next((c for c in record.chunks if c.chunk_index == chunk_index), None)
            if chunk and embedding is not None:
                chunk.embedding = embedding
                successful_embeddings += 1
            elif chunk:
                logger.warning(f"Embedding failed for chunk index {chunk_index} in record {record.record_id}")

        if successful_embeddings == 0:
            logger.error(f"All embeddings failed for record {record.record_id}")
            return None

        # Save to Parquet if configured
        if self.arrow_save_dir:
            arrow_file_path = Path(self.arrow_save_dir) / f"{record.record_id}.parquet"
            record.chunks_path = str(arrow_file_path)
            
            try:
                await asyncio.to_thread(self._write_record_to_parquet, record, arrow_file_path)
                logger.info(f"Successfully saved record chunks to {arrow_file_path}")
            except Exception as e:
                logger.error(f"Failed to save record {record.record_id} to Parquet: {e}")
                record.chunks_path = None

        return record

    async def store_chunks_enhanced(self, record: EnhancedRecord) -> EnhancedRecord | None:
        """Store record chunks with metadata in ChromaDB."""
        try:
            ids = []
            documents = []
            embeddings_list = []
            metadatas = []

            chunks_to_upsert = [c for c in record.chunks if c.embedding is not None]

            for chunk in chunks_to_upsert:
                ids.append(chunk.chunk_id)
                documents.append(chunk.chunk_text)
                embeddings_list.append(list(chunk.embedding))

                # Enhanced metadata with content type tagging
                enhanced_metadata = {
                    "document_title": record.title or "Untitled",
                    "chunk_index": chunk.chunk_index,
                    "document_id": record.record_id,
                    "content_type": chunk.metadata.get("content_type", chunk.source_field),
                    "chunk_type": chunk.chunk_type,
                    "source_field": chunk.source_field,
                    **{k: v for k, v in chunk.metadata.items() 
                       if k not in ["content_type", "chunk_type", "source_field"]}
                }
                metadatas.append(self._sanitize_metadata_for_chroma(enhanced_metadata))

            logger.info(f"Upserting {len(ids)} enhanced chunks for record {record.record_id}...")

            # Execute the upsert operation
            await asyncio.to_thread(
                self.collection.upsert,
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents,
            )

            logger.info(f"Successfully stored enhanced chunks for record {record.record_id}")
            
            # Update vector metadata
            record.update_vector_metadata("stored_at", asyncio.get_event_loop().time())
            record.update_vector_metadata("chunk_count", len(chunks_to_upsert))
            record.update_vector_metadata("embedding_model", self.embedding_model)
            
            return record

        except Exception as e:
            logger.error(f"Failed to store enhanced chunks for record {record.record_id}: {e}")
            return None

    # ========== BACKWARD COMPATIBILITY METHODS ==========

    @vector_processor_adapter
    def process_record(self, record: Any) -> EnhancedRecord | None:
        """Process any record type (legacy compatibility)."""
        enhanced_record = CompatibilityLayer.normalize_record(record)
        return asyncio.run(self.process_enhanced_record_async(enhanced_record))

    def record_to_enhanced_record(self, record: Any) -> EnhancedRecord:
        """Convert any record format to EnhancedRecord."""
        return CompatibilityLayer.normalize_record(record)

    # Legacy method for backward compatibility
    def record_to_input_document(self, record: Any) -> "InputDocument":
        """Convert a Record to InputDocument format (legacy compatibility)."""
        enhanced_record = CompatibilityLayer.normalize_record(record)
        return enhanced_record.as_input_document()

    # ========== UTILITY METHODS ==========

    def _write_record_to_parquet(self, record: EnhancedRecord, file_path: Path):
        """Write EnhancedRecord chunks to a Parquet file."""
        if not record.chunks:
            logger.warning(f"Attempted to write empty chunks for record {record.record_id}")
            return

        data = {
            "chunk_id": [c.chunk_id for c in record.chunks],
            "document_id": [record.record_id for c in record.chunks],
            "document_title": [record.title or "Untitled" for c in record.chunks],
            "chunk_index": [c.chunk_index for c in record.chunks],
            "chunk_text": [c.chunk_text for c in record.chunks],
            "chunk_type": [c.chunk_type for c in record.chunks],
            "source_field": [c.source_field for c in record.chunks],
            "embedding": [list(c.embedding) if c.embedding is not None else None for c in record.chunks],
            "chunk_metadata": [json.dumps(c.metadata) if c.metadata else None for c in record.chunks],
        }

        # Create schema with enhanced fields
        embedding_type = pa.list_(pa.float32())
        if self.dimensionality:
            embedding_type = pa.list_(pa.float32(), self.dimensionality)

        schema = pa.schema([
            pa.field("chunk_id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("document_title", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("chunk_text", pa.string()),
            pa.field("chunk_type", pa.string()),
            pa.field("source_field", pa.string()),
            pa.field("embedding", embedding_type),
            pa.field("chunk_metadata", pa.string()),
        ])

        table = pa.Table.from_pydict(data, schema=schema)

        # Enhanced record metadata
        record_meta_serializable = {
            "record_id": record.record_id,
            "title": record.title or "Untitled",
            "file_path": record.file_path or "",
            "record_path": record.record_path or "",
            "uri": record.uri or "",
            "mime": record.mime or "text/plain",
            "metadata": json.dumps(record.metadata),
            "vector_metadata": json.dumps(record.vector_metadata),
            "format_version": "enhanced_record_v1"
        }
        
        arrow_metadata = {k.encode("utf-8"): str(v).encode("utf-8") 
                         for k, v in record_meta_serializable.items()}

        final_schema = table.schema.with_metadata(arrow_metadata)
        table = table.cast(final_schema)

        pq.write_table(table, file_path, compression="snappy")

    async def _embed_batch(self, inputs: Sequence[tuple[int, TextEmbeddingInput]]) -> list[tuple[int, list[float] | None]]:
        """Internal async method to call the Vertex AI embedding model concurrently."""
        if not inputs:
            return []
        
        tasks = [self._run_embedding_task(chunk_input, idx) for idx, chunk_input in inputs]
        results = await asyncio.gather(*tasks)
        return results

    async def _run_embedding_task(self, chunk_input: TextEmbeddingInput, index: int) -> tuple[int, list[float] | None]:
        """Helper coroutine to run a single embedding task."""
        async with self._embedding_semaphore:
            kwargs = dict(
                output_dimensionality=self.dimensionality,
                auto_truncate=False,
            )
            try:
                embeddings_result: list[TextEmbedding] = await self._embedding_model.get_embeddings_async(
                    [chunk_input], **kwargs
                )
                if embeddings_result:
                    return index, embeddings_result[0].values
                logger.warning(f"No embedding result returned for input {index}.")
                return index, None
            except Exception as exc:
                logger.error(f"Error getting embedding for input {index}: {exc}")
                return index, None

    def _sanitize_metadata_for_chroma(self, metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        """Convert metadata values to types supported by ChromaDB."""
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
                    logger.warning(f"Could not JSON serialize metadata value for key '{k}': {e}")
            else:
                try:
                    sanitized[k] = str(v)
                except Exception as e:
                    logger.warning(f"Could not convert metadata value for key '{k}' to string: {e}")
        
        return sanitized

    @property
    def collection(self):
        """Get ChromaDB collection (implementation same as before)."""
        # Implementation same as existing ChromaDBEmbeddings.collection
        pass


# ========== PIPELINE PROCESSING FUNCTIONS ==========

async def process_records_pipeline(
    records: AsyncIterator[Any],
    embedder: EnhancedChromaDBEmbeddings,
    batch_size: int = 10
) -> AsyncIterator[EnhancedRecord]:
    """Process records through the enhanced embedding pipeline."""
    
    async def process_single_record(record: Any) -> EnhancedRecord | None:
        """Process a single record."""
        try:
            enhanced_record = CompatibilityLayer.normalize_record(record)
            result = await embedder.process_enhanced_record_async(enhanced_record)
            return result
        except Exception as e:
            logger.error(f"Failed to process record: {e}")
            return None
    
    # Process records in batches
    batch = []
    async for record in records:
        batch.append(record)
        
        if len(batch) >= batch_size:
            # Process batch concurrently
            tasks = [process_single_record(r) for r in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, EnhancedRecord):
                    yield result
                elif isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
            
            batch = []
    
    # Process remaining records
    if batch:
        tasks = [process_single_record(r) for r in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, EnhancedRecord):
                yield result


# ========== SEARCH AND RETRIEVAL ENHANCEMENTS ==========

class EnhancedVectorSearch:
    """Enhanced vector search that leverages multi-field chunking."""
    
    def __init__(self, embedder: EnhancedChromaDBEmbeddings):
        self.embedder = embedder
        self.collection = embedder.collection
    
    async def search_by_content_type(
        self,
        query: str,
        content_types: list[str] | None = None,
        chunk_types: list[str] | None = None,
        limit: int = 10
    ) -> list[dict[str, Any]]:
        """Search with filtering by content type and chunk type."""
        
        # Generate query embedding
        query_embedding = await self._embed_query(query)
        
        # Build filter conditions
        where_conditions = {}
        if content_types:
            where_conditions["content_type"] = {"$in": content_types}
        if chunk_types:
            where_conditions["chunk_type"] = {"$in": chunk_types}
        
        # Perform search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            where=where_conditions if where_conditions else None,
            n_results=limit,
            include=["documents", "metadatas", "distances"]
        )
        
        return self._format_search_results(results)
    
    async def search_by_record_field(
        self,
        query: str,
        source_field: str,
        limit: int = 10
    ) -> list[dict[str, Any]]:
        """Search within chunks from a specific record field."""
        return await self.search_by_content_type(
            query=query,
            content_types=[source_field],
            limit=limit
        )
    
    async def _embed_query(self, query: str) -> list[float]:
        """Generate embedding for search query."""
        query_input = TextEmbeddingInput(
            text=query,
            task_type="RETRIEVAL_QUERY"
        )
        
        embeddings_result = await self.embedder._embedding_model.get_embeddings_async([query_input])
        return embeddings_result[0].values if embeddings_result else []
    
    def _format_search_results(self, raw_results: dict) -> list[dict[str, Any]]:
        """Format ChromaDB search results."""
        formatted_results = []
        
        if not raw_results.get("documents"):
            return formatted_results
        
        for i, doc in enumerate(raw_results["documents"][0]):
            result = {
                "document": doc,
                "metadata": raw_results["metadatas"][0][i] if raw_results.get("metadatas") else {},
                "distance": raw_results["distances"][0][i] if raw_results.get("distances") else None,
                "id": raw_results["ids"][0][i] if raw_results.get("ids") else None
            }
            formatted_results.append(result)
        
        return formatted_results


# ========== EXAMPLE USAGE ==========

async def example_enhanced_processing():
    """Example of how to use the enhanced vector processing."""
    
    # Create an enhanced record
    record = EnhancedRecord(
        content="This is a research paper about machine learning techniques.",
        metadata={
            "title": "ML Research Paper",
            "abstract": "A comprehensive study of modern ML approaches",
            "authors": "Smith, J. and Doe, A.",
            "doi": "10.1234/example.2024"
        }
    )
    
    # Configure multi-field embedding
    from buttermilk._core.storage_config import MultiFieldEmbeddingConfig, AdditionalFieldConfig
    
    multi_field_config = MultiFieldEmbeddingConfig(
        content_field="content",
        additional_fields=[
            AdditionalFieldConfig(
                source_field="title",
                chunk_type="title",
                min_length=5
            ),
            AdditionalFieldConfig(
                source_field="abstract", 
                chunk_type="abstract",
                min_length=20
            )
        ],
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Create embedder with multi-field config
    embedder = EnhancedChromaDBEmbeddings(
        collection_name="enhanced_test",
        persist_directory="./test_chromadb",
        multi_field_config=multi_field_config
    )
    
    # Process the record
    processed_record = await embedder.process_enhanced_record_async(record)
    
    if processed_record:
        print(f"Processed record with {len(processed_record.chunks)} chunks")
        print(f"Chunk types: {processed_record.chunk_types}")
        print(f"Has embeddings: {processed_record.has_embeddings}")
        
        # Search example
        search = EnhancedVectorSearch(embedder)
        results = await search.search_by_content_type(
            query="machine learning",
            content_types=["content", "abstract"],
            limit=5
        )
        print(f"Found {len(results)} search results")


if __name__ == "__main__":
    asyncio.run(example_enhanced_processing())