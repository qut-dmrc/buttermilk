"""Standard Unified Processors implementations.

This module contains concrete implementations of standard processors:
- GroupchatProcessor: Runs a group chat session.
- ExpanderProcessor: Expands a record into multiple records (1:N).
- TransformProcessor: Applies JMESPath transformations to records.

Processors are automatically registered on import.
"""

from typing import AsyncGenerator

import jmespath
from jmespath.exceptions import JMESPathError

from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.processor_config import ExpanderProcessorConfig, GroupchatProcessorConfig, TransformProcessorConfig
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


class TransformProcessor(UnifiedProcessor):
    """Applies JMESPath expressions to transform records.

    Evaluates a JMESPath expression against the record and stores the result
    in context.metadata. This enables declarative data extraction and transformation
    without custom Python code.

    Features:
    - Compiles JMESPath expression once during initialization for performance
    - Stores transformation result in context metadata
    - Fail-fast on invalid JMESPath expressions
    - Graceful handling when expression returns None
    """

    def __init__(self, config: TransformProcessorConfig):
        super().__init__(config)
        self.config: TransformProcessorConfig = config

        # Compile JMESPath expression once during initialization
        try:
            self._compiled_expression = jmespath.compile(config.expression)
        except JMESPathError as e:
            logger.error(
                "Invalid JMESPath expression",
                expression=config.expression,
                error=str(e),
            )
            raise ValueError(
                f"Invalid JMESPath expression: {config.expression}"
            ) from e

    async def _process_record(
        self,
        context: ProcessingContext,
    ) -> AsyncGenerator[BaseRecord, None]:
        """Apply JMESPath transformation and store result in context metadata.

        Args:
            context: Processing context containing the record to transform

        Yields:
            The original record (transformation result is stored in context.metadata)

        Raises:
            ValueError: If JMESPath expression evaluation fails
        """
        logger.debug(
            "Applying JMESPath transformation",
            record_id=context.record.record_id,
            expression=self.config.expression,
            output_field=self.config.output_field,
        )

        # Convert record to dict for JMESPath processing
        record_dict = context.record.model_dump()

        try:
            # Apply JMESPath expression
            result = self._compiled_expression.search(record_dict)

            if result is not None:
                # Store result in context metadata
                context.update_metadata(self.config.output_field, result)

                logger.debug(
                    "JMESPath transformation complete",
                    record_id=context.record.record_id,
                    output_field=self.config.output_field,
                    result_type=type(result).__name__,
                )
            else:
                logger.debug(
                    "JMESPath expression returned None, no metadata stored",
                    record_id=context.record.record_id,
                    expression=self.config.expression,
                )

        except Exception as e:
            logger.error(
                "Error applying JMESPath expression",
                record_id=context.record.record_id,
                expression=self.config.expression,
                error=str(e),
            )
            raise ValueError(
                f"Error applying JMESPath expression '{self.config.expression}': {str(e)}"
            ) from e

        # Yield the original record (metadata is stored in context)
        yield context.record


# Register processors on module import
register_processor("groupchat", GroupchatProcessor)
register_processor("expander", ExpanderProcessor)
register_processor("transform", TransformProcessor)
