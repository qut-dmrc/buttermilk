"""Enhanced Record class design that unifies Record and InputDocument functionality.

This is a design document/implementation blueprint for the unified Record class.
It shows how to extend the current Record class to support vector operations
while maintaining full backward compatibility.
"""

import datetime
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self

import shortuuid
from autogen_core.models import AssistantMessage, UserMessage
from cloudpathlib import CloudPath
from PIL.Image import Image
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


class ChunkData(BaseModel):
    """Represents a chunk of text derived from a Record's content.
    
    This replaces the separate ChunkedDocument class, integrating chunking
    directly into the Record ecosystem.
    """
    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chunk_index: int = Field(description="Sequential index of this chunk within the document")
    chunk_text: str = Field(description="The actual text content of this chunk")
    chunk_type: str = Field(default="content", description="Type of chunk (content, summary, title, etc.)")
    source_field: str = Field(default="content", description="Which Record field this chunk came from")
    embedding: Sequence[float] | None = Field(default=None, description="Vector embedding for this chunk")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional chunk-specific metadata")

    @property
    def chunk_title(self) -> str:
        """Generate a title hint for embedding models."""
        return f"chunk_{self.chunk_index}_{self.chunk_type}"


class EnhancedRecord(BaseModel):
    """Enhanced Record class that unifies Record and InputDocument functionality.
    
    This class extends the current Record to support vector operations while maintaining
    full backward compatibility. It incorporates all InputDocument fields and methods
    as optional/computed properties.
    
    Key Enhancements:
    - Chunking support for vector processing
    - Multi-field embedding capabilities
    - Vector storage metadata
    - Lazy loading for performance
    - Full InputDocument compatibility
    """

    # ========== CORE RECORD FIELDS (unchanged) ==========
    record_id: str = Field(
        default_factory=lambda: str(shortuuid.ShortUUID().uuid()),
        description="Unique identifier for the record.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata associated with the record.",
    )
    alt_text: str | None = Field(
        default=None,
        description="Textual description or transcript of media content within this record.",
    )
    ground_truth: dict[str, Any] | None = Field(
        default=None,
        description="Optional ground truth data associated with this record for evaluation.",
    )
    uri: str | None = Field(
        default=None,
        description="Optional URI pointing to the original source of the record's content.",
    )
    content: str | Sequence[str | Image] = Field(
        description="Main content of the record: a string, or a sequence of strings and Pillow Images.",
    )
    mime: str | None = Field(
        default="text/plain",
        description="Primary MIME type of the content.",
    )

    # ========== VECTOR/CHUNKING EXTENSIONS ==========
    # These fields support vector operations but are optional for backward compatibility
    
    full_text: str | None = Field(
        default=None,
        description="Complete text content for vector processing (auto-populated from content if string)",
    )
    chunks: list[ChunkData] = Field(
        default_factory=list,
        description="Text chunks derived from this record for vector processing",
    )
    
    # Vector storage metadata
    chunks_path: str | None = Field(
        default=None,
        description="Path to stored chunk data (Parquet/Arrow file)",
    )
    vector_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata specific to vector processing (embeddings, chunking config, etc.)",
    )
    
    # InputDocument compatibility fields
    file_path: str | None = Field(
        default=None,
        description="Original file path if record was created from a file",
    )
    record_path: str | None = Field(
        default=None,
        description="Path to the stored record data",
    )

    # ========== COMPUTED PROPERTIES (unchanged from Record) ==========
    @computed_field
    @property
    def images(self) -> list[Image] | None:
        """Extracts all Pillow `Image` objects from the `content` attribute."""
        images_found: list[Image] = []
        if isinstance(self.content, Sequence) and not isinstance(self.content, str):
            for item in self.content:
                if isinstance(item, Image):
                    images_found.append(item)
        return images_found or None

    @property
    def title(self) -> str | None:
        """Convenience property to access the 'title' from the `metadata` dictionary."""
        return self.metadata.get("title")

    # ========== VECTOR PROCESSING COMPUTED PROPERTIES ==========
    @computed_field
    @property
    def text_content(self) -> str:
        """Get textual content for vector processing.
        
        Priority: full_text > string content > alt_text > empty string
        """
        if self.full_text:
            return self.full_text
        if isinstance(self.content, str):
            return self.content
        if self.alt_text:
            return self.alt_text
        return ""

    @computed_field
    @property
    def has_embeddings(self) -> bool:
        """Check if any chunks have embeddings."""
        return any(chunk.embedding is not None for chunk in self.chunks)

    @computed_field 
    @property
    def embedding_count(self) -> int:
        """Count of chunks that have embeddings."""
        return sum(1 for chunk in self.chunks if chunk.embedding is not None)

    @computed_field
    @property
    def chunk_types(self) -> set[str]:
        """Get all chunk types present in this record."""
        return {chunk.chunk_type for chunk in self.chunks}

    # ========== COMPATIBILITY METHODS ==========
    def as_input_document(self) -> "InputDocument":
        """Convert to InputDocument for legacy compatibility.
        
        This allows gradual migration from InputDocument to EnhancedRecord.
        """
        # Import here to avoid circular imports
        from buttermilk.data.vector import InputDocument, ChunkedDocument
        
        # Convert chunks to ChunkedDocument format
        legacy_chunks = []
        for chunk in self.chunks:
            legacy_chunks.append(ChunkedDocument(
                chunk_id=chunk.chunk_id,
                document_title=self.title or "Untitled",
                chunk_index=chunk.chunk_index,
                chunk_text=chunk.chunk_text,
                document_id=self.record_id,
                embedding=chunk.embedding,
                metadata=chunk.metadata
            ))
        
        return InputDocument(
            record_id=self.record_id,
            file_path=self.file_path or "",
            record_path=self.record_path or "",
            chunks_path=self.chunks_path or "",
            full_text=self.text_content,
            chunks=legacy_chunks,
            title=self.title or "Untitled",
            metadata=self.metadata
        )

    @classmethod
    def from_input_document(cls, input_doc: "InputDocument") -> "EnhancedRecord":
        """Create EnhancedRecord from InputDocument for migration."""
        # Convert ChunkedDocuments to ChunkData
        chunks = []
        for chunk in input_doc.chunks:
            chunks.append(ChunkData(
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                chunk_text=chunk.chunk_text,
                chunk_type="content",  # Default type
                source_field="content",  # Default source
                embedding=chunk.embedding,
                metadata=chunk.metadata
            ))
        
        # Build metadata with title
        metadata = input_doc.metadata.copy()
        if input_doc.title and input_doc.title != "Untitled":
            metadata["title"] = input_doc.title
            
        return cls(
            record_id=input_doc.record_id,
            content=input_doc.full_text,
            metadata=metadata,
            full_text=input_doc.full_text,
            chunks=chunks,
            chunks_path=input_doc.chunks_path,
            file_path=input_doc.file_path,
            record_path=input_doc.record_path
        )

    # ========== VECTOR PROCESSING METHODS ==========
    def add_chunk(
        self, 
        text: str, 
        chunk_type: str = "content",
        source_field: str = "content",
        metadata: dict[str, Any] | None = None
    ) -> ChunkData:
        """Add a new chunk to this record."""
        chunk = ChunkData(
            chunk_index=len(self.chunks),
            chunk_text=text,
            chunk_type=chunk_type,
            source_field=source_field,
            metadata=metadata or {}
        )
        self.chunks.append(chunk)
        return chunk

    def get_chunks_by_type(self, chunk_type: str) -> list[ChunkData]:
        """Get all chunks of a specific type."""
        return [chunk for chunk in self.chunks if chunk.chunk_type == chunk_type]

    def get_chunks_by_source(self, source_field: str) -> list[ChunkData]:
        """Get all chunks from a specific source field."""
        return [chunk for chunk in self.chunks if chunk.source_field == source_field]

    def clear_chunks(self) -> None:
        """Remove all chunks from this record."""
        self.chunks.clear()

    def update_vector_metadata(self, key: str, value: Any) -> None:
        """Update vector processing metadata."""
        self.vector_metadata[key] = value

    # ========== EXISTING RECORD METHODS (unchanged) ==========
    def as_markdown(self) -> str:
        """Combines metadata and text content into a single string."""
        parts: list[str] = []

        if self.metadata:
            if self.title:
                parts.append(f"### {self.title}")
            parts.append(f"**Record ID**: {self.record_id}")

            for key, value in self.metadata.items():
                if key not in [
                    "title",
                    "fetch_timestamp_utc", 
                    "fetch_source_id",
                    "components",
                ]:
                    parts.append(f"**{key}**: {value!s}")
            if parts:
                parts.append("---")

        # Handle content based on its type
        if isinstance(self.content, str):
            parts.append(self.content)
        elif isinstance(self.content, Sequence):
            text_parts_from_content: list[str] = []
            has_non_text = False
            for item in self.content:
                if isinstance(item, str):
                    text_parts_from_content.append(item)
                elif isinstance(item, Image):
                    has_non_text = True

            if text_parts_from_content:
                parts.append("\n".join(text_parts_from_content))

            if has_non_text and self.alt_text:
                parts.append(f"\n**Alternative Text for Media**: {self.alt_text}")
            elif has_non_text and not self.alt_text:
                parts.append("\n[Non-text content present, no alternative text provided]")

        elif self.alt_text:
            parts.append(self.alt_text)

        return "\n\n".join(p for p in parts if p)

    def as_message(self, role: Literal["user", "assistant"] = "user") -> UserMessage | AssistantMessage:
        """Converts the Record into an Autogen UserMessage or AssistantMessage."""
        if role == "assistant":
            return AssistantMessage(content=self.as_markdown(), source=self.record_id)

        message_content: str | list[Any]

        if isinstance(self.content, str):
            message_content = self.content
        elif isinstance(self.content, Sequence):
            processed_parts: list[Any] = []
            for item in self.content:
                if isinstance(item, (str, Image)):
                    processed_parts.append(item)
            message_content = processed_parts
        else:
            message_content = str(self.content)

        return UserMessage(content=message_content, source=self.record_id)

    # ========== MODEL CONFIGURATION AND VALIDATORS ==========
    model_config = ConfigDict(
        extra="ignore",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        exclude_unset=True,
        exclude_none=True,
        exclude={"title", "images", "text_content", "has_embeddings", "embedding_count", "chunk_types"},
    )

    @model_validator(mode="after")
    def vld_input(self) -> Self:
        """Process model_extra and auto-populate full_text."""
        # Handle model_extra (same as original Record)
        if self.model_extra:
            extra_keys = list(self.model_extra.keys())
            for key in extra_keys:
                value = self.model_extra.pop(key)
                if key == "components":
                    pass
                elif key in self.metadata:
                    raise ValueError(f"Extra field '{key}' conflicts with existing metadata key in Record.")
                elif key in self.model_computed_fields:
                    raise ValueError(f"Extra field '{key}' conflicts with a computed field name in Record.")
                elif value is not None:
                    self.metadata[key] = value

        # Auto-populate full_text from content if not provided
        if not self.full_text and isinstance(self.content, str):
            self.full_text = self.content

        return self

    @field_validator("uri")
    @classmethod
    def vld_path(cls, path: Any) -> str | None:
        """Validates and normalizes the `uri` attribute."""
        if path is None:
            return None
        if isinstance(path, CloudPath):
            return path.as_uri()
        if isinstance(path, Path):
            return path.as_posix()
        return str(path)


# Legacy InputDocument class for backward compatibility
# This will be imported from vector.py but we show the interface here
class InputDocument(BaseModel):
    """Legacy InputDocument class - kept for backward compatibility.
    
    New code should use EnhancedRecord instead.
    """
    model_config = ConfigDict(extra="ignore")

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str = Field(...)
    record_path: str = Field(default="")
    chunks_path: str = Field(default="")
    full_text: str = Field(default="")
    chunks: list = Field(default_factory=list)  # ChunkedDocument list
    title: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_enhanced_record(self) -> EnhancedRecord:
        """Convert to EnhancedRecord."""
        return EnhancedRecord.from_input_document(self)


# ========== MIGRATION STRATEGY ==========

class RecordMigration:
    """Utility class for migrating between Record formats."""
    
    @staticmethod
    def is_enhanced_record(record: Any) -> bool:
        """Check if a record is an EnhancedRecord."""
        return isinstance(record, EnhancedRecord)
    
    @staticmethod
    def ensure_enhanced_record(record: Any) -> EnhancedRecord:
        """Ensure we have an EnhancedRecord, converting if necessary.""" 
        if isinstance(record, EnhancedRecord):
            return record
        elif hasattr(record, 'to_enhanced_record'):
            return record.to_enhanced_record()
        elif hasattr(record, 'record_id'):  # Looks like a Record-like object
            # Try to convert from dict-like object
            if hasattr(record, 'model_dump'):
                data = record.model_dump()
            else:
                data = record.__dict__
            return EnhancedRecord(**data)
        else:
            raise ValueError(f"Cannot convert {type(record)} to EnhancedRecord")

    @staticmethod
    def batch_migrate_records(records: list[Any]) -> list[EnhancedRecord]:
        """Convert a batch of records to EnhancedRecord format."""
        return [RecordMigration.ensure_enhanced_record(record) for record in records]


# ========== CONFIGURATION INTEGRATION ==========

def create_multi_field_chunks(
    record: EnhancedRecord, 
    config: "MultiFieldEmbeddingConfig"
) -> list[ChunkData]:
    """Create chunks for multiple content types using configuration.
    
    This replaces the method in ChromaDBEmbeddings.create_multi_field_chunks
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    chunks = []
    
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
                        "content_type": config.content_field,
                        "chunk_type": "content",
                        "chunk_sequence": i
                    }
                )
                chunks.append(chunk)
    
    # 2. Additional fields (single chunks each)
    for field_config in config.additional_fields:
        field_value = record.metadata.get(field_config.source_field, '')
        
        if field_value and len(str(field_value).strip()) >= field_config.min_length:
            chunk = ChunkData(
                chunk_index=len(chunks),
                chunk_text=str(field_value).strip(),
                chunk_type=field_config.chunk_type,
                source_field=field_config.source_field,
                metadata={
                    "content_type": field_config.source_field,
                    "chunk_type": field_config.chunk_type
                }
            )
            chunks.append(chunk)
    
    return chunks


# ========== PERFORMANCE CONSIDERATIONS ==========

class LazyChunkLoader:
    """Lazy loader for chunk data to optimize memory usage."""
    
    def __init__(self, chunks_path: str):
        self.chunks_path = chunks_path
        self._chunks: list[ChunkData] | None = None
    
    def load_chunks(self) -> list[ChunkData]:
        """Load chunks from storage on demand."""
        if self._chunks is None:
            self._chunks = self._load_from_file()
        return self._chunks
    
    def _load_from_file(self) -> list[ChunkData]:
        """Load chunks from Parquet file."""
        import pyarrow.parquet as pq
        import json
        
        if not Path(self.chunks_path).exists():
            return []
        
        table = pq.read_table(self.chunks_path)
        chunks = []
        
        for i in range(table.num_rows):
            row = table.slice(i, 1).to_pydict()
            chunk = ChunkData(
                chunk_id=row['chunk_id'][0],
                chunk_index=row['chunk_index'][0],
                chunk_text=row['chunk_text'][0],
                chunk_type=row.get('chunk_type', ['content'])[0],
                source_field=row.get('source_field', ['content'])[0],
                embedding=row['embedding'][0] if row['embedding'][0] else None,
                metadata=json.loads(row['chunk_metadata'][0]) if row['chunk_metadata'][0] else {}
            )
            chunks.append(chunk)
        
        return chunks


# ========== SERIALIZATION EFFICIENCY ==========

def serialize_enhanced_record(record: EnhancedRecord, include_chunks: bool = True) -> dict[str, Any]:
    """Efficiently serialize EnhancedRecord with options."""
    data = record.model_dump(
        exclude_none=True,
        exclude_unset=True,
        exclude={"images", "text_content", "has_embeddings", "embedding_count", "chunk_types"}
    )
    
    if not include_chunks:
        data.pop("chunks", None)
    
    return data


def deserialize_enhanced_record(data: dict[str, Any]) -> EnhancedRecord:
    """Deserialize EnhancedRecord from dict."""
    return EnhancedRecord(**data)


# Example usage and testing
if __name__ == "__main__":
    # Example: Create an enhanced record
    record = EnhancedRecord(
        content="This is a test document with some content for chunking.",
        metadata={"title": "Test Document", "source": "example"}
    )
    
    # Add some chunks
    record.add_chunk("This is a test document", chunk_type="content")
    record.add_chunk("Test Document", chunk_type="title", source_field="title")
    
    print(f"Record has {len(record.chunks)} chunks")
    print(f"Chunk types: {record.chunk_types}")
    print(f"Text content: {record.text_content}")
    
    # Test serialization
    serialized = serialize_enhanced_record(record)
    deserialized = deserialize_enhanced_record(serialized)
    
    print(f"Serialization successful: {deserialized.record_id == record.record_id}")