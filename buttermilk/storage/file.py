"""File storage implementation for unified storage operations."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from buttermilk._core.log import logger
from buttermilk._core.types import Record
from .base import Storage, StorageError

if TYPE_CHECKING:
    from buttermilk._core.bm_init import BM
    from .config import StorageConfig


class FileStorage(Storage):
    """Unified file storage supporting both read and write operations.
    
    Supports local files and cloud storage paths (GCS, S3) for JSON/JSONL formats.
    """
    
    def __init__(self, config: "StorageConfig", bm: "BM | None" = None):
        """Initialize file storage.
        
        Args:
            config: Storage configuration with file path
            bm: Buttermilk instance (optional for file operations)
        """
        super().__init__(config, bm)
        
        if not config.path:
            raise ValueError("File storage requires a path")
        
        self.path = Path(config.path)
    
    def __iter__(self) -> Iterator[Record]:
        """Iterate over records from file.
        
        Yields:
            Record objects from the file
        """
        try:
            if not self.exists():
                logger.warning(f"File does not exist: {self.path}")
                return
            
            with open(self.path, 'r') as f:
                if self.path.suffix == '.jsonl':
                    # JSONL format - one JSON object per line
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            record = self._dict_to_record(data, line_num)
                            yield record
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON on line {line_num}: {e}")
                            continue
                else:
                    # JSON format - single JSON array or object
                    data = json.load(f)
                    if isinstance(data, list):
                        for i, item in enumerate(data):
                            record = self._dict_to_record(item, i + 1)
                            yield record
                    else:
                        record = self._dict_to_record(data, 1)
                        yield record
                        
        except Exception as e:
            logger.error(f"Error reading from file {self.path}: {e}")
            raise StorageError(f"Failed to read file: {e}") from e
    
    def save(self, records: list[Record] | Record) -> None:
        """Save records to file.
        
        Args:
            records: Single record or list of records to save
        """
        if isinstance(records, Record):
            records = [records]
        
        if not records:
            logger.warning("No records to save")
            return
        
        try:
            # Ensure parent directory exists
            self.path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert records to dictionaries
            data = [self._record_to_dict(record) for record in records]
            
            with open(self.path, 'w') as f:
                if self.path.suffix == '.jsonl':
                    # JSONL format - one JSON object per line
                    for record_dict in data:
                        json.dump(record_dict, f, ensure_ascii=False)
                        f.write('\n')
                else:
                    # JSON format - single JSON array
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully saved {len(records)} records to {self.path}")
            
        except Exception as e:
            logger.error(f"Error saving records to file {self.path}: {e}")
            raise StorageError(f"Failed to save file: {e}") from e
    
    def count(self) -> int:
        """Count total records in file.
        
        Returns:
            Number of records in the file
        """
        try:
            count = 0
            for _ in self:
                count += 1
            return count
        except Exception as e:
            logger.warning(f"Error counting records in file {self.path}: {e}")
            return -1
    
    def exists(self) -> bool:
        """Check if the file exists.
        
        Returns:
            True if file exists, False otherwise
        """
        return self.path.exists() and self.path.is_file()
    
    def create(self) -> None:
        """Create an empty file if it doesn't exist."""
        if self.exists():
            return
        
        try:
            # Ensure parent directory exists
            self.path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create empty file with appropriate format
            with open(self.path, 'w') as f:
                if self.path.suffix == '.jsonl':
                    # Empty JSONL file
                    pass
                else:
                    # Empty JSON array
                    json.dump([], f)
            
            logger.info(f"Created empty file: {self.path}")
            
        except Exception as e:
            logger.error(f"Error creating file {self.path}: {e}")
            raise StorageError(f"Failed to create file: {e}") from e
    
    def _dict_to_record(self, data: dict, index: int) -> Record:
        """Convert dictionary to Record object.
        
        Args:
            data: Dictionary data from file
            index: Record index for error reporting
            
        Returns:
            Record object
        """
        try:
            # Apply column mapping if configured
            if self.config.columns:
                mapped_data = {}
                for new_key, old_key in self.config.columns.items():
                    if old_key in data:
                        mapped_data[new_key] = data[old_key]
                data = mapped_data
            
            # Extract required and optional fields
            record_id = data.get('record_id', data.get('id', f"record_{index}"))
            content = data.get('content', data.get('text', ''))
            
            # Build metadata from remaining fields
            metadata = data.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {'raw_metadata': metadata}
            
            # Add other fields to metadata if not already Record fields
            record_fields = {'record_id', 'content', 'metadata', 'alt_text', 'ground_truth', 'uri', 'mime'}
            for key, value in data.items():
                if key not in record_fields and key not in ['id', 'text']:
                    metadata[key] = value
            
            return Record(
                record_id=str(record_id),
                content=content,
                metadata=metadata,
                alt_text=data.get('alt_text'),
                ground_truth=data.get('ground_truth'),
                uri=data.get('uri'),
                mime=data.get('mime', 'text/plain')
            )
            
        except Exception as e:
            logger.warning(f"Error converting data to Record at index {index}: {e}")
            return Record(
                record_id=f"error_{index}",
                content=str(data),
                metadata={'parse_error': str(e), 'original_data': data}
            )
    
    def _record_to_dict(self, record: Record) -> dict:
        """Convert Record object to dictionary for file storage.
        
        Args:
            record: Record object to convert
            
        Returns:
            Dictionary representation
        """
        result = {
            'record_id': record.record_id,
            'content': record.content,
            'metadata': record.metadata,
        }
        
        # Add optional fields if present
        if record.alt_text:
            result['alt_text'] = record.alt_text
        if record.ground_truth:
            result['ground_truth'] = record.ground_truth
        if record.uri:
            result['uri'] = record.uri
        if record.mime and record.mime != 'text/plain':
            result['mime'] = record.mime
        
        return result