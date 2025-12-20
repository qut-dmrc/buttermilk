"""Standard Unified Processors implementations.

This module contains concrete implementations of standard processors:
- GroupchatProcessor: Runs a group chat session.
- ExpanderProcessor: Expands a record into multiple records (1:N).

Processors are automatically registered on import.
"""

from typing import AsyncGenerator

from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.processor_config import ExpanderProcessorConfig, GroupchatProcessorConfig
from buttermilk._core.processor_registry import register_processor
from buttermilk._core.types import BaseRecord
from buttermilk._core.unified_processor import UnifiedProcessor
from buttermilk import logger


class GroupchatProcessor(UnifiedProcessor):
    """Executes a group chat session for each record."""
    
    def __init__(self, config: GroupchatProcessorConfig):
        super().__init__(config)
        self.config: GroupchatProcessorConfig = config

    async def _process_record(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[BaseRecord, None]:
        logger.info(
            f"Starting groupchat with {self.config.participants}",
            record_id=context.record.record_id
        )
        # TODO: Implement actual groupchat execution logic here
        # For now, just pass through the record
        yield context.record


class ExpanderProcessor(UnifiedProcessor):
    """Expands a single record into multiple records based on a list field."""
    
    def __init__(self, config: ExpanderProcessorConfig):
        super().__init__(config)
        self.config: ExpanderProcessorConfig = config

    async def _process_record(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[BaseRecord, None]:
        field_name = self.config.field_to_expand
        
        # Access the field from the record (attribute or metadata)
        values = getattr(context.record, field_name, None)
        if values is None:
            values = context.record.metadata.get(field_name)
            
        if not isinstance(values, list):
            logger.warning(
                f"Field '{field_name}' is not a list, cannot expand. Skipping expansion.",
                record_id=context.record.record_id,
                value_type=type(values)
            )
            yield context.record
            return

        # Expand logic
        for i, value in enumerate(values):
            # Create updates dictionary
            updates = {"record_id": f"{context.record.record_id}_{i}"}
            
            # Prepare metadata update
            # We copy existing metadata to avoid mutating the original record's metadata if it's shared
            new_metadata = context.record.metadata.copy()
            new_metadata["expansion_source_id"] = context.record.record_id
            new_metadata["expansion_index"] = i
            
            if hasattr(context.record, field_name):
                 updates[field_name] = value
            else:
                 new_metadata[field_name] = value
            
            updates["metadata"] = new_metadata

            # Create new record with updates
            expanded_record = context.record.model_copy(update=updates)

            yield expanded_record


# Register processors on module import
register_processor("groupchat", GroupchatProcessor)
register_processor("expander", ExpanderProcessor)
