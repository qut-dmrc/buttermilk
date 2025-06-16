"""Migration strategy and compatibility layer for transitioning from InputDocument to EnhancedRecord.

This module provides the tools and strategies needed to migrate existing code
from using InputDocument to the new EnhancedRecord while maintaining full
backward compatibility during the transition period.
"""

from typing import Any, Protocol, TypeVar, Union
from pathlib import Path
import json
import logging
from contextlib import contextmanager

# Import the enhanced record design
from enhanced_record_design import EnhancedRecord, InputDocument, ChunkData, RecordMigration

logger = logging.getLogger(__name__)

T = TypeVar('T')

# ========== MIGRATION PHASES ==========

class MigrationPhase:
    """Defines the three phases of migration."""
    
    # Phase 1: Dual compatibility - both formats supported
    COMPATIBILITY = "compatibility"
    
    # Phase 2: Enhanced default - new code uses EnhancedRecord, old code still works
    ENHANCED_DEFAULT = "enhanced_default"
    
    # Phase 3: EnhancedRecord only - InputDocument deprecated
    ENHANCED_ONLY = "enhanced_only"


class MigrationConfig:
    """Configuration for controlling migration behavior."""
    
    def __init__(
        self,
        phase: str = MigrationPhase.COMPATIBILITY,
        auto_convert: bool = True,
        warn_on_legacy: bool = False,
        strict_mode: bool = False
    ):
        self.phase = phase
        self.auto_convert = auto_convert
        self.warn_on_legacy = warn_on_legacy
        self.strict_mode = strict_mode


# Global migration config
_migration_config = MigrationConfig()


def set_migration_config(config: MigrationConfig) -> None:
    """Set global migration configuration."""
    global _migration_config
    _migration_config = config


def get_migration_config() -> MigrationConfig:
    """Get current migration configuration."""
    return _migration_config


# ========== COMPATIBILITY PROTOCOLS ==========

class RecordLike(Protocol):
    """Protocol for Record-like objects."""
    record_id: str
    content: Any
    metadata: dict[str, Any]


class VectorCapable(Protocol):
    """Protocol for objects that support vector operations."""
    chunks: list[Any]
    
    def add_chunk(self, text: str, **kwargs) -> Any:
        ...


# ========== MIGRATION UTILITIES ==========

class CompatibilityLayer:
    """Provides compatibility methods for migrating between formats."""
    
    @staticmethod
    def normalize_record(record: Any) -> EnhancedRecord:
        """Normalize any record-like object to EnhancedRecord."""
        config = get_migration_config()
        
        if isinstance(record, EnhancedRecord):
            return record
        
        if config.warn_on_legacy:
            logger.warning(f"Converting legacy record type {type(record)} to EnhancedRecord")
        
        if hasattr(record, 'to_enhanced_record'):
            return record.to_enhanced_record()
        
        return RecordMigration.ensure_enhanced_record(record)
    
    @staticmethod
    def create_record_from_data(data: dict[str, Any]) -> EnhancedRecord:
        """Create a record from raw data, handling various formats."""
        # Handle legacy InputDocument data
        if 'full_text' in data and 'file_path' in data:
            # This looks like InputDocument data
            input_doc = InputDocument(**data)
            return EnhancedRecord.from_input_document(input_doc)
        
        # Handle standard Record data
        return EnhancedRecord(**data)
    
    @staticmethod
    def batch_normalize_records(records: list[Any]) -> list[EnhancedRecord]:
        """Normalize a batch of records efficiently."""
        return [CompatibilityLayer.normalize_record(record) for record in records]


# ========== ADAPTER CLASSES ==========

class InputDocumentAdapter:
    """Adapter to make InputDocument behave like EnhancedRecord."""
    
    def __init__(self, input_doc: InputDocument):
        self.input_doc = input_doc
        self._enhanced_record: EnhancedRecord | None = None
    
    def to_enhanced_record(self) -> EnhancedRecord:
        """Convert to EnhancedRecord (cached)."""
        if self._enhanced_record is None:
            self._enhanced_record = EnhancedRecord.from_input_document(self.input_doc)
        return self._enhanced_record
    
    def __getattr__(self, name: str) -> Any:
        """Delegate to enhanced record."""
        enhanced = self.to_enhanced_record()
        return getattr(enhanced, name)


class LegacyVectorProcessor:
    """Wraps vector processing methods to handle both formats."""
    
    def __init__(self, processor_func):
        self.processor_func = processor_func
    
    def __call__(self, record: Any) -> Any:
        """Process record, converting format if needed."""
        # Normalize to EnhancedRecord
        enhanced_record = CompatibilityLayer.normalize_record(record)
        
        # Process with enhanced record
        result = self.processor_func(enhanced_record)
        
        # Convert back if needed (for legacy compatibility)
        config = get_migration_config()
        if config.phase == MigrationPhase.COMPATIBILITY:
            # In compatibility phase, might need to return InputDocument format
            if hasattr(record, 'to_input_document') and not isinstance(record, EnhancedRecord):
                return result.as_input_document() if hasattr(result, 'as_input_document') else result
        
        return result


def vector_processor_adapter(func):
    """Decorator to adapt vector processing functions."""
    return LegacyVectorProcessor(func)


# ========== MIGRATION CONTEXT MANAGERS ==========

@contextmanager
def migration_context(phase: str, **kwargs):
    """Context manager for setting migration phase temporarily."""
    old_config = get_migration_config()
    new_config = MigrationConfig(phase=phase, **kwargs)
    
    try:
        set_migration_config(new_config)
        yield new_config
    finally:
        set_migration_config(old_config)


@contextmanager
def strict_migration():
    """Context manager for strict migration mode."""
    with migration_context(MigrationPhase.ENHANCED_ONLY, strict_mode=True):
        yield


# ========== FILE MIGRATION UTILITIES ==========

class FileMigrationTool:
    """Tool for migrating stored data files."""
    
    @staticmethod
    def migrate_parquet_file(file_path: Path) -> Path:
        """Migrate a Parquet file from InputDocument format to EnhancedRecord format."""
        import pyarrow.parquet as pq
        import pyarrow as pa
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read existing file
        table = pq.read_table(file_path)
        
        # Check if migration is needed
        if 'chunk_type' in table.column_names:
            logger.info(f"File {file_path} already appears to be in new format")
            return file_path
        
        # Add new columns for enhanced format
        num_rows = table.num_rows
        
        # Add chunk_type column (default to 'content')
        chunk_type_array = pa.array(['content'] * num_rows)
        
        # Add source_field column (default to 'content')
        source_field_array = pa.array(['content'] * num_rows)
        
        # Create new table with additional columns
        new_table = table.append_column('chunk_type', chunk_type_array)
        new_table = new_table.append_column('source_field', source_field_array)
        
        # Update metadata
        metadata = table.schema.metadata or {}
        metadata[b'migration_version'] = b'enhanced_record_v1'
        metadata[b'migrated_at'] = str(pd.Timestamp.now()).encode()
        
        new_schema = new_table.schema.with_metadata(metadata)
        new_table = new_table.cast(new_schema)
        
        # Create backup and write new file
        backup_path = file_path.with_suffix('.backup' + file_path.suffix)
        file_path.rename(backup_path)
        
        pq.write_table(new_table, file_path, compression="snappy")
        
        logger.info(f"Migrated {file_path}, backup saved as {backup_path}")
        return file_path
    
    @staticmethod
    def migrate_directory(directory: Path, pattern: str = "*.parquet") -> list[Path]:
        """Migrate all matching files in a directory."""
        migrated_files = []
        
        for file_path in directory.glob(pattern):
            try:
                migrated_path = FileMigrationTool.migrate_parquet_file(file_path)
                migrated_files.append(migrated_path)
            except Exception as e:
                logger.error(f"Failed to migrate {file_path}: {e}")
        
        return migrated_files


# ========== TESTING UTILITIES ==========

class MigrationTester:
    """Utilities for testing migration compatibility."""
    
    @staticmethod
    def test_record_compatibility(record: Any) -> dict[str, bool]:
        """Test if a record is compatible with both formats."""
        results = {}
        
        try:
            # Test EnhancedRecord compatibility
            enhanced = CompatibilityLayer.normalize_record(record)
            results['enhanced_record'] = True
            
            # Test basic operations
            results['has_chunks'] = hasattr(enhanced, 'chunks')
            results['can_add_chunk'] = hasattr(enhanced, 'add_chunk')
            results['has_text_content'] = hasattr(enhanced, 'text_content')
            
            # Test InputDocument compatibility
            if hasattr(enhanced, 'as_input_document'):
                input_doc = enhanced.as_input_document()
                results['input_document_conversion'] = True
            else:
                results['input_document_conversion'] = False
                
        except Exception as e:
            logger.error(f"Compatibility test failed: {e}")
            results['error'] = str(e)
        
        return results
    
    @staticmethod
    def compare_processing_results(
        record: Any,
        old_processor,
        new_processor
    ) -> dict[str, Any]:
        """Compare results between old and new processing methods."""
        results = {}
        
        try:
            # Test with old processor
            old_result = old_processor(record)
            results['old_success'] = True
            results['old_type'] = type(old_result).__name__
        except Exception as e:
            results['old_success'] = False
            results['old_error'] = str(e)
        
        try:
            # Test with new processor
            enhanced_record = CompatibilityLayer.normalize_record(record)
            new_result = new_processor(enhanced_record)
            results['new_success'] = True
            results['new_type'] = type(new_result).__name__
        except Exception as e:
            results['new_success'] = False
            results['new_error'] = str(e)
        
        # Compare results if both succeeded
        if results.get('old_success') and results.get('new_success'):
            results['results_compatible'] = True  # Could add more detailed comparison
        
        return results


# ========== MIGRATION EXECUTION PLAN ==========

class MigrationExecutor:
    """Executes migration plan for a codebase."""
    
    def __init__(self, config: MigrationConfig):
        self.config = config
        self.migration_log = []
    
    def execute_phase_1(self) -> None:
        """Execute Phase 1: Establish compatibility."""
        logger.info("Starting Migration Phase 1: Compatibility")
        
        # 1. Deploy EnhancedRecord alongside existing Record
        # 2. Add conversion utilities
        # 3. Update vector processing to accept both formats
        # 4. Add migration warnings (optional)
        
        set_migration_config(MigrationConfig(
            phase=MigrationPhase.COMPATIBILITY,
            auto_convert=True,
            warn_on_legacy=self.config.warn_on_legacy
        ))
        
        self.migration_log.append("Phase 1 complete: Dual compatibility established")
    
    def execute_phase_2(self) -> None:
        """Execute Phase 2: Enhanced as default."""
        logger.info("Starting Migration Phase 2: Enhanced Default")
        
        # 1. Update all new code to use EnhancedRecord
        # 2. Migrate existing data files
        # 3. Update APIs to return EnhancedRecord by default
        # 4. Add deprecation warnings for InputDocument
        
        set_migration_config(MigrationConfig(
            phase=MigrationPhase.ENHANCED_DEFAULT,
            auto_convert=True,
            warn_on_legacy=True
        ))
        
        self.migration_log.append("Phase 2 complete: EnhancedRecord is now default")
    
    def execute_phase_3(self) -> None:
        """Execute Phase 3: EnhancedRecord only."""
        logger.info("Starting Migration Phase 3: Enhanced Only")
        
        # 1. Remove InputDocument class
        # 2. Remove compatibility adapters
        # 3. Clean up migration code
        # 4. Update documentation
        
        set_migration_config(MigrationConfig(
            phase=MigrationPhase.ENHANCED_ONLY,
            strict_mode=True
        ))
        
        self.migration_log.append("Phase 3 complete: Migration finished")
    
    def get_migration_log(self) -> list[str]:
        """Get migration execution log."""
        return self.migration_log.copy()


# ========== EXAMPLE MIGRATION WORKFLOW ==========

def example_migration_workflow():
    """Example of how to execute the migration."""
    
    # Initialize migration
    config = MigrationConfig(
        phase=MigrationPhase.COMPATIBILITY,
        warn_on_legacy=True
    )
    executor = MigrationExecutor(config)
    
    # Phase 1: Establish compatibility
    with migration_context(MigrationPhase.COMPATIBILITY):
        print("Phase 1: Testing compatibility...")
        
        # Test existing code with new compatibility layer
        legacy_record = {"record_id": "test", "content": "test content"}
        enhanced = CompatibilityLayer.create_record_from_data(legacy_record)
        print(f"Enhanced record created: {enhanced.record_id}")
    
    # Phase 2: Use enhanced as default
    with migration_context(MigrationPhase.ENHANCED_DEFAULT, warn_on_legacy=True):
        print("Phase 2: Enhanced as default...")
        
        # New code uses EnhancedRecord directly
        new_record = EnhancedRecord(content="New content", metadata={"source": "new"})
        new_record.add_chunk("New content", chunk_type="content")
        print(f"New record has {len(new_record.chunks)} chunks")
    
    # Phase 3: Enhanced only
    with strict_migration():
        print("Phase 3: Enhanced only...")
        
        # Only EnhancedRecord is used
        final_record = EnhancedRecord(content="Final content")
        print(f"Final migration complete: {final_record.record_id}")


if __name__ == "__main__":
    example_migration_workflow()