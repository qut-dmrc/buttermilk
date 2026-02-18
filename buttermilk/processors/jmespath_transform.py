"""JMESPath-based transformation processor for Buttermilk pipelines.

This processor applies JMESPath expressions to BaseRecord objects to extract and
transform data declaratively without writing custom Python code.
"""

from typing import AsyncGenerator

import jmespath
from jmespath.exceptions import JMESPathError
from pydantic import BaseModel, Field

from buttermilk import logger
from buttermilk._core.types import BaseRecord


class JMESPathTransform(BaseModel):
    """Transform BaseRecord objects using JMESPath expressions.

    This processor accepts a BaseRecord and applies JMESPath mappings to extract
    and transform fields. Each mapping defines a new field name and the JMESPath
    expression to compute its value from the record.

    Features:
    - Declarative field extraction and transformation
    - Nested object construction from JMESPath expressions
    - Graceful handling of missing fields (no field added if expression returns None)
    - Fail-fast on invalid JMESPath expressions
    - Preserves all original record fields

    Example:
        >>> processor = JMESPathTransform(
        ...     mappings={"answer": "metadata.outputs.result"}
        ... )
        >>> async for record in processor.process(record, processor_stage="transform"):
        ...     print(record.answer)  # Extracted field
    """

    mappings: dict[str, str] = Field(
        ...,
        description="Mapping of field names to JMESPath expressions. "
        "Each key becomes a new field on the record, with value computed by the expression.",
    )

    # Cache compiled JMESPath expressions for performance
    _compiled_expressions: dict[str, jmespath.parser.ParsedResult] | None = None

    def model_post_init(self, __context) -> None:
        """Compile JMESPath expressions once during initialization."""
        self._compiled_expressions = {}
        for field_name, expression in self.mappings.items():
            try:
                self._compiled_expressions[field_name] = jmespath.compile(expression)
            except JMESPathError as e:
                logger.error(
                    "Invalid JMESPath expression",
                    field_name=field_name,
                    expression=expression,
                    error=str(e),
                )
                raise ValueError(f"Invalid JMESPath expression for field '{field_name}': {expression}") from e

    async def process(self, record: BaseRecord, *, processor_stage: str, **kwargs) -> AsyncGenerator[BaseRecord, None]:
        """Process a record by applying JMESPath transformations.

        Args:
            record: BaseRecord to transform
            processor_stage: Stage name for metadata tracking
            **kwargs: Additional keyword arguments (ignored)

        Yields:
            BaseRecord with additional fields from JMESPath transformations

        Raises:
            ValueError: If JMESPath expression is invalid (fail-fast)
        """
        logger.debug(
            "JMESPathTransform processing record",
            record_id=record.record_id,
            mappings_count=len(self.mappings),
            processor_stage=processor_stage,
        )

        # Convert record to dict for JMESPath processing
        record_dict = record.model_dump()

        # Apply each JMESPath expression
        transformed_fields = {}
        for field_name, compiled_expr in self._compiled_expressions.items():
            try:
                # Apply JMESPath expression
                result = compiled_expr.search(record_dict)

                # Only add field if result is not None (graceful handling of missing fields)
                if result is not None:
                    transformed_fields[field_name] = result
                    logger.debug(
                        "Applied JMESPath mapping",
                        field_name=field_name,
                        result_type=type(result).__name__,
                        processor_stage=processor_stage,
                    )
                else:
                    logger.debug(
                        "JMESPath expression returned None, field not added",
                        field_name=field_name,
                        processor_stage=processor_stage,
                    )

            except Exception as e:
                logger.error(
                    "Error applying JMESPath expression",
                    field_name=field_name,
                    error=str(e),
                    processor_stage=processor_stage,
                )
                raise ValueError(f"Error applying JMESPath for field '{field_name}': {str(e)}") from e

        # Create new record with additional fields
        # BaseRecord has extra="allow" and frozen=True, so we need to reconstruct it
        if transformed_fields:
            # Get all existing fields from the record
            record_data = record.model_dump()

            # Add the transformed fields as extra fields
            record_data.update(transformed_fields)

            # Create a new record instance with all fields
            # This works because BaseRecord has extra="allow"
            updated_record = record.__class__(**record_data)

            logger.debug(
                "JMESPath transformation complete",
                record_id=record.record_id,
                fields_added=list(transformed_fields.keys()),
                processor_stage=processor_stage,
            )

            yield updated_record
        else:
            logger.info(
                "No fields transformed (all expressions returned None)",
                record_id=record.record_id,
                processor_stage=processor_stage,
            )
            yield record
