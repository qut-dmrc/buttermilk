"""Utilities for converting Pydantic models to BigQuery schemas."""

from typing import Any, Dict, List, Union
from google.cloud import bigquery
from pydantic import BaseModel
from pydantic.fields import FieldInfo
import datetime
from PIL.Image import Image


def pydantic_to_bigquery_schema(model_class: type[BaseModel], extra_fields: List[Dict[str, str]] = None) -> List[bigquery.SchemaField]:
    """Convert a Pydantic model to BigQuery schema fields.

    Args:
        model_class: The Pydantic model class
        extra_fields: Additional fields to add (e.g., dataset_name, split_type)

    Returns:
        List of BigQuery SchemaField objects
    """
    schema_fields = []

    # Add extra fields first (these are typically required metadata)
    if extra_fields:
        for field in extra_fields:
            schema_fields.append(bigquery.SchemaField(name=field["name"], field_type=field["type"], mode=field.get("mode", "NULLABLE")))

    # Process Pydantic model fields
    for field_name, field_info in model_class.model_fields.items():
        # Skip computed fields - they're not stored
        if field_name in getattr(model_class, "model_computed_fields", {}):
            continue
            
        # Skip alt_text field - it's legacy and redundant with metadata
        if field_name == "alt_text":
            continue

        bq_field = _convert_pydantic_field_to_bq(field_name, field_info)
        if bq_field:
            schema_fields.append(bq_field)

    return schema_fields


def _convert_pydantic_field_to_bq(field_name: str, field_info: FieldInfo) -> bigquery.SchemaField:
    """Convert a single Pydantic field to BigQuery SchemaField."""

    # Get the field type
    field_type = field_info.annotation

    # Handle Optional types (Union[T, None] or T | None)
    is_optional = False
    
    # Handle both Union[str, None] and str | None syntax
    if hasattr(field_type, "__args__") and field_type.__args__:
        args = field_type.__args__
        if len(args) == 2 and type(None) in args:
            is_optional = True
            field_type = args[0] if args[1] is type(None) else args[1]
    elif hasattr(field_type, "__origin__") and field_type.__origin__ is type(Union):
        args = field_type.__args__
        if len(args) == 2 and type(None) in args:
            is_optional = True
            field_type = args[0] if args[1] is type(None) else args[1]

    # Determine BigQuery type
    bq_type = "STRING"  # default

    if field_type == str:
        bq_type = "STRING"
    elif field_type == int:
        bq_type = "INTEGER"
    elif field_type == float:
        bq_type = "FLOAT"
    elif field_type == bool:
        bq_type = "BOOLEAN"
    elif field_type == datetime.datetime:
        bq_type = "TIMESTAMP"
    elif field_type == dict or str(field_type).startswith("dict"):
        bq_type = "JSON"
    elif hasattr(field_type, "__origin__") and field_type.__origin__ is list:
        bq_type = "JSON"  # Store lists as JSON
    elif field_type == Image:
        # Images aren't stored directly in BigQuery
        return None
    else:
        # For complex types like Sequence[str | Image], store as JSON if it contains strings
        # Special case for content field - always store even if it contains images
        if field_name == "content":
            bq_type = "JSON"  # Store content as JSON regardless of image presence
        elif "Image" in str(field_type):
            return None  # Skip other image-heavy fields
        else:
            bq_type = "JSON"

    # Determine mode
    # Special case: record_id should always be REQUIRED even though it has a default_factory
    if field_name == "record_id":
        mode = "REQUIRED"
    else:
        mode = "NULLABLE" if is_optional or field_info.default is not None else "REQUIRED"

    return bigquery.SchemaField(name=field_name, field_type=bq_type, mode=mode, description=field_info.description)


def get_record_bigquery_schema() -> List[bigquery.SchemaField]:
    """Get the complete BigQuery schema for Record objects including metadata fields."""
    from buttermilk._core.types import Record

    # Define the extra fields needed for BigQuery storage
    extra_fields = [
        {"name": "dataset_name", "type": "STRING", "mode": "REQUIRED"},
        {"name": "split_type", "type": "STRING", "mode": "NULLABLE"},
        {"name": "created_at", "type": "TIMESTAMP", "mode": "NULLABLE"},
        {"name": "updated_at", "type": "TIMESTAMP", "mode": "NULLABLE"},
    ]

    return pydantic_to_bigquery_schema(Record, extra_fields)
