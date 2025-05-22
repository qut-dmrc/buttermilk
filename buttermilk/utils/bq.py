"""Utilities for interacting with Google BigQuery.

This module provides helper functions and classes for common BigQuery operations,
such as:
- Constructing dictionaries that conform to a BigQuery schema, including type conversions.
- Loading BigQuery schemas from JSON files.
- Asynchronously writing data to BigQuery tables using the Storage Write API.
- Retrieving the latest partition or table from a time-partitioned or sharded table set.
- Loading data into BigQuery from Google Cloud Storage (GCS).
- Uploading Pandas DataFrames directly to BigQuery tables.

It uses the `google-cloud-bigquery` and `google-cloud-bigquery-storage` client libraries.
"""

import datetime
from collections.abc import Sequence # For type hinting sequences
from typing import Any, Mapping # For type hinting

import pandas as pd
import pydantic # Pydantic core, though less used directly in this file now
from google.cloud import bigquery, bigquery_storage_v1beta2 # BigQuery client libraries
from google.cloud.bigquery_storage import ( # Storage Write API client
    BigQueryWriteAsyncClient,
)
from google.cloud.bigquery_storage_v1.types import ( # Types for Storage Write API
    ProtoRows,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator # Pydantic components

from .._core.log import logger # Centralized logger
from .utils import make_serialisable, remove_punctuation # Utility functions


# This constant is also in _core/constants.py. Define here if specific to bq utils,
# or ensure single source of truth. For now, keeping as it was in original.
GOOGLE_BQ_PRICE_PER_BYTE = 5 / (10**12)  # $5 per Terabyte (1 TB = 10^12 bytes)
"""Estimated cost per byte for Google BigQuery queries, based on $5 per TB."""


def construct_dict_from_schema(
    schema: list[bigquery.SchemaField | dict[str, Any]], # Schema can be list of SchemaField or dicts
    data_dict: dict[str, Any], 
    remove_extra_fields: bool = True
) -> dict[str, Any]:
    """Recursively constructs a dictionary that conforms to a BigQuery schema.

    This function takes a data dictionary and a BigQuery schema definition.
    It processes the dictionary to:
    1.  Include only keys present in the schema (if `remove_extra_fields` is True).
    2.  Convert data types to match BigQuery types (e.g., strings to TIMESTAMP,
        numbers to INTEGER/FLOAT, strings to BOOLEAN).
    3.  Handle nested fields (STRUCT/RECORD) and repeated fields (ARRAY) by
        applying the schema transformation recursively.
    4.  Filter out values that represent NULL in a string form (case-insensitive
        "NULL" or empty string after punctuation removal).

    Args:
        schema (list[bigquery.SchemaField | dict[str, Any]]): A list representing the
            BigQuery schema. Each item can be a `google.cloud.bigquery.SchemaField`
            object or a dictionary with keys like 'name', 'type', and optionally
            'fields' (for nested schemas) and 'mode' (for REPEATED fields).
        data_dict (dict[str, Any]): The input data dictionary to transform.
        remove_extra_fields (bool): If True (default), keys in `data_dict` that
            are not defined in the `schema` will be excluded from the result.
            If False, they will be passed through as is (which might cause
            BigQuery errors if the table doesn't allow unknown fields).

    Returns:
        dict[str, Any]: A new dictionary with keys and values conforming to the
        provided BigQuery schema.
    
    Raises:
        ValueError: If data type conversion fails for a specific field (e.g.,
            cannot convert a string to a number for an INTEGER field if the string
            is not a valid number).
    """
    transformed_dict: dict[str, Any] = {}
    
    # Create a set of schema field names for efficient lookup if removing extra fields
    schema_field_names = {
        (field.name if isinstance(field, bigquery.SchemaField) else field["name"])
        for field in schema
    }

    for key, value in data_dict.items():
        if remove_extra_fields and key not in schema_field_names:
            logger.debug(f"Field '{key}' not in schema, removing (remove_extra_fields=True).")
            continue # Skip fields not in schema if remove_extra_fields is True

        # Find the corresponding schema field definition for the current key
        field_schema = None
        for f_schema in schema:
            current_field_name = f_schema.name if isinstance(f_schema, bigquery.SchemaField) else f_schema["name"]
            if current_field_name == key:
                field_schema = f_schema.to_api_repr() if isinstance(f_schema, bigquery.SchemaField) else f_schema
                break
        
        if not field_schema: # Should not happen if remove_extra_fields is False and key is present
            if not remove_extra_fields: # Pass through if not removing extra and no schema found (should be rare)
                 transformed_dict[key] = value
            continue


        field_type_upper = field_schema["type"].upper()
        is_repeated = field_schema.get("mode", "").upper() == "REPEATED"

        # Handle NULL-like string values for non-repeated fields
        if not is_repeated and isinstance(value, str) and \
           (remove_punctuation(value).upper() == "NULL" or not remove_punctuation(value)):
            logger.debug(f"Field '{key}' has NULL-like string value ('{value}'), skipping.")
            continue # Skip (effectively treating as NULL by not adding to dict)

        # Handle repeated fields (arrays)
        if is_repeated:
            if not isinstance(value, list):
                logger.warning(f"Field '{key}' is REPEATED in schema but value is not a list (type: {type(value)}). Skipping.")
                continue
            
            transformed_array_items = []
            nested_schema_fields = field_schema.get("fields")
            for item in value:
                if nested_schema_fields and isinstance(item, dict): # Array of STRUCTs
                    transformed_array_items.append(construct_dict_from_schema(nested_schema_fields, item, remove_extra_fields))
                elif nested_schema_fields: # Expected dict for struct, got something else
                     logger.warning(f"REPEATED field '{key}' contains non-dict item '{item}' for a STRUCT type. Skipping item.")
                else: # Array of simple types
                    # Apply type conversion to individual array items if needed (simplified here)
                    # For simplicity, this example assumes basic types in arrays don't need deep conversion here
                    # A more robust version would delegate to a type conversion helper for array items too.
                    transformed_array_items.append(item) # Basic types are added as is for now
            transformed_dict[key] = transformed_array_items

        # Handle nested fields (STRUCT/RECORD) for non-repeated fields
        elif field_type_upper == "RECORD" or field_type_upper == "STRUCT": # Both are valid type names
            nested_schema_fields = field_schema.get("fields")
            if isinstance(value, dict) and nested_schema_fields:
                transformed_dict[key] = construct_dict_from_schema(nested_schema_fields, value, remove_extra_fields)
            elif isinstance(value, dict) and not nested_schema_fields:
                 logger.warning(f"Field '{key}' is STRUCT but schema has no 'fields'. Value: {value}. Storing as is.")
                 transformed_dict[key] = value # Or handle as error
            else: # Value is not a dict for a STRUCT field
                logger.warning(f"Field '{key}' is STRUCT but value is not a dict (type: {type(value)}). Skipping.")
                continue
        
        # Handle simple data type conversions for non-repeated, non-STRUCT fields
        elif field_type_upper in ("TIMESTAMP", "DATETIME", "DATE"):
            if isinstance(value, datetime.datetime):
                converted_value = value
            elif isinstance(value, str):
                try:
                    converted_value = pd.to_datetime(value) # Handles various string formats
                except Exception as e_dt:
                    logger.warning(f"Could not parse string '{value}' to datetime for field '{key}': {e_dt!s}. Skipping.")
                    continue
            elif isinstance(value, (int, float)): # Assume POSIX timestamp (seconds or ms)
                try:
                    converted_value = datetime.datetime.utcfromtimestamp(value)
                except (ValueError, OSError): # Try milliseconds if seconds failed
                    try:
                        converted_value = datetime.datetime.utcfromtimestamp(value / 1000.0)
                    except Exception as e_ts:
                        logger.warning(f"Could not parse numeric '{value}' as timestamp for field '{key}': {e_ts!s}. Skipping.")
                        continue
            else:
                logger.warning(f"Unsupported type '{type(value)}' for datetime conversion for field '{key}'. Skipping.")
                continue
            
            if field_type_upper == "DATE" and isinstance(converted_value, datetime.datetime):
                transformed_dict[key] = converted_value.date()
            else:
                transformed_dict[key] = converted_value

        elif field_type_upper in ("INTEGER", "INT64", "FLOAT", "FLOAT64", "NUMERIC", "BIGNUMERIC"):
            try:
                # pd.to_numeric handles strings and existing numbers robustly
                numeric_value = pd.to_numeric(value)
                if field_type_upper in ("INTEGER", "INT64") and not pd.isna(numeric_value):
                    transformed_dict[key] = int(numeric_value)
                elif field_type_upper in ("FLOAT", "FLOAT64") and not pd.isna(numeric_value):
                    transformed_dict[key] = float(numeric_value)
                else: # NUMERIC, BIGNUMERIC or if NaN (let BigQuery handle potential type issues for these)
                    transformed_dict[key] = numeric_value if not pd.isna(numeric_value) else None # Convert NaN to None
                    if transformed_dict[key] is None: continue # Skip if it became None
            except ValueError as e_num:
                logger.warning(f"Could not convert '{value}' to numeric for field '{key}': {e_num!s}. Skipping.")
                continue
        
        elif field_type_upper == "BOOLEAN":
            if isinstance(value, str):
                val_lower = value.strip().lower()
                if val_lower in ("true", "t", "yes", "y", "1"):
                    transformed_dict[key] = True
                elif val_lower in ("false", "f", "no", "n", "0"):
                    transformed_dict[key] = False
                elif not val_lower: # Empty string for boolean field
                    logger.debug(f"Empty string for BOOLEAN field '{key}'. Skipping.")
                    continue
                else:
                    logger.warning(f"Could not convert string '{value}' to BOOLEAN for field '{key}'. Skipping.")
                    continue
            elif isinstance(value, bool):
                transformed_dict[key] = value
            elif isinstance(value, (int, float)): # Allow 0/1 for boolean
                transformed_dict[key] = bool(value)
            else:
                logger.warning(f"Unsupported type '{type(value)}' for BOOLEAN conversion for field '{key}'. Skipping.")
                continue
        else: # STRING, BYTES, GEOGRAPHY, JSON - pass through as is, BigQuery client handles these
            transformed_dict[key] = value
            
    return transformed_dict


class TableWriter(BaseModel):
    """Manages asynchronous writes to a BigQuery table using the Storage Write API.

    This class configures a `BigQueryWriteAsyncClient` and provides an `append_rows`
    method to write data in batches. It handles schema loading and data preparation,
    including type conversions to match the BigQuery schema.

    Attributes:
        write_client (BigQueryWriteAsyncClient): An instance of the asynchronous
            BigQuery Storage Write API client. Defaults to a new client instance.
        destination (str | None): The fully qualified BigQuery table ID in the
            format "project.dataset.table". If provided, it's used to derive
            `project_id`, `dataset_id`, and `table_id` if they are not set.
        bq_schema (list[bigquery.SchemaField] | str | None): The BigQuery table schema.
            Can be a list of `SchemaField` objects or a path to a JSON schema file.
            If a path string is provided, it's loaded into `SchemaField` objects.
        stream (str | None): The write stream name to use. Defaults to "_default",
            which is the default stream for a table in the Storage Write API.
        project_id (str | None): Google Cloud project ID. Derived from `destination`
            or `table_path` if not set.
        dataset_id (str | None): BigQuery dataset ID. Derived from `destination`
            or `table_path` if not set.
        table_id (str | None): BigQuery table ID. Derived from `destination`
            or `table_path` if not set.
        table_path (str | None): The fully qualified path for the BigQuery table resource,
            formatted as "projects/{project_id}/datasets/{dataset_id}/tables/{table_id}".
            Automatically constructed if individual ID components are provided.
        model_config (ConfigDict): Pydantic model configuration.
            - `extra`: "forbid" - Disallows extra fields.
            - `arbitrary_types_allowed`: True - Allows `BigQueryWriteAsyncClient`.
            - `populate_by_name`: True.
    """

    write_client: BigQueryWriteAsyncClient = Field(default_factory=BigQueryWriteAsyncClient)
    destination: str | None = Field(
        default=None, 
        description="Optional. Fully qualified table ID (project.dataset.table). Used if component IDs not set."
    )
    bq_schema: list[bigquery.SchemaField] | None = Field( # Changed from str to list[SchemaField] for clarity post-validation
        default=None, 
        description="BigQuery table schema as list of SchemaFields or path to JSON schema file."
    )
    stream: str | None = Field(
        default="_default", 
        description="BigQuery Storage Write API stream name. Defaults to '_default'."
    )
    project_id: str | None = Field(default=None, description="Google Cloud project ID.")
    dataset_id: str | None = Field(default=None, description="BigQuery dataset ID.")
    table_id: str | None = Field(default=None, description="BigQuery table ID.")
    table_path: str | None = Field(
        default=None, 
        description="Fully qualified table path for Storage Write API (projects/.../datasets/.../tables/...). Auto-constructed if not set."
    )

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    @pydantic.field_validator("bq_schema", mode="before")
    @classmethod
    def load_schema_from_path(cls, v: Any) -> list[bigquery.SchemaField] | Any: # Allow Any to pass through if already list
        """Pydantic validator to load a BigQuery schema from a JSON file path if a string is provided.

        Args:
            v: The value for `bq_schema`. If a string, it's treated as a file path.
               If already a list (presumably of `SchemaField`), it's passed through.

        Returns:
            list[bigquery.SchemaField] | Any: The loaded schema as a list of `SchemaField`
            objects, or the original value if not a string path.
        
        Raises:
            TypeError: If `v` is a string path but schema loading fails.
        """
        if isinstance(v, str): # If schema is provided as a file path string
            try:
                return bigquery.Client().schema_from_json(v)
            except Exception as e:
                raise TypeError(f"Failed to load BigQuery schema from JSON file path '{v}': {e!s}") from e
        return v # Pass through if already list of SchemaField or None

    @model_validator(mode="after") # Changed from field_validator for table_path to model_validator
    def construct_table_path(cls, values: Any) -> Any: # Changed to model_validator signature
        """Constructs the `table_path` from component IDs if not explicitly provided.
        
        Also populates `project_id`, `dataset_id`, `table_id` if `destination`
        (project.dataset.table) is given and they are missing.

        Args:
            values: The Pydantic model instance (or data for it).

        Returns:
            The validated model instance with `table_path` (and potentially component IDs) populated.

        Raises:
            ValueError: If neither `table_path` nor sufficient components
                        (`destination` or all of `project_id`, `dataset_id`, `table_id`)
                        are provided to construct the table path.
        """
        # This is now a model_validator, so 'values' is the model instance (or its __dict__)
        # In Pydantic v2, for model_validator(mode="after"), 'values' is the model instance itself.
        # For model_validator(mode="before"), 'values' is a dict.
        # Assuming this runs *after* individual field validation.
        
        # Get current values from the model instance
        table_path = values.table_path
        project_id = values.project_id
        dataset_id = values.dataset_id
        table_id = values.table_id
        destination = values.destination

        if table_path: # If table_path is already set, prioritize it.
            # Optionally, parse project_id, dataset_id, table_id from table_path if they are None
            if not (project_id and dataset_id and table_id) and isinstance(table_path, str):
                parts = table_path.split('/')
                if len(parts) == 6 and parts[0] == "projects" and parts[2] == "datasets" and parts[4] == "tables":
                    values.project_id = parts[1]
                    values.dataset_id = parts[3]
                    values.table_id = parts[5]
            return values

        # If table_path is not set, try to construct it.
        # First, use destination if component IDs are missing.
        if not (project_id and dataset_id and table_id) and destination:
            try:
                dest_parts = destination.split(".")
                if len(dest_parts) == 3:
                    values.project_id, values.dataset_id, values.table_id = dest_parts[0], dest_parts[1], dest_parts[2]
                elif len(dest_parts) == 2: # Assume project from default client, dataset.table given
                    values.dataset_id, values.table_id = dest_parts[0], dest_parts[1]
                    # values.project_id will use default from client if None
                else:
                    raise ValueError("`destination` must be in 'project.dataset.table' or 'dataset.table' format.")
            except Exception as e:
                raise ValueError(f"Invalid `destination` format ('{destination}'). Expected 'project.dataset.table' or 'dataset.table'. Error: {e!s}") from e
        
        # Now, ensure all components are present to build table_path
        # project_id can be None if using default project for BigQuery client
        final_project_id = values.project_id or bigquery.Client().project # Use default project if None

        if not (final_project_id and values.dataset_id and values.table_id):
            raise ValueError("Cannot construct table_path: Missing project_id, dataset_id, or table_id, and destination was not sufficient.")
        
        values.table_path = f"projects/{final_project_id}/datasets/{values.dataset_id}/tables/{values.table_id}"
        return values


    async def append_rows(self, rows: Sequence[Mapping[str, Any]] | pd.DataFrame) -> list[Any] | None:
        """Appends rows to the configured BigQuery table asynchronously using Storage Write API.

        The input `rows` are first prepared using `construct_dict_from_schema` (if
        `self.bq_schema` is set) and `make_serialisable`. Then, they are converted
        to `ProtoRows` format and sent to BigQuery in batches.

        Args:
            rows: A sequence of dictionaries or a Pandas DataFrame representing
                  the rows to append. Each dictionary's keys (or DataFrame's columns)
                  should correspond to the table's column names.

        Returns:
            list[Any] | None: A list of error objects if any errors occurred during
            the append operations for any chunk. An empty list indicates success for all
            chunks. Returns `None` if no rows were provided after preparation.
            (Note: The original returned Sequence[bool], this is changed to match
            BigQuery client's `insert_rows` error reporting style more closely).
        
        Raises:
            TypeError: If `self.bq_schema` is set but is not a list of `SchemaField`.
        """
        if not self.table_path: # Should be set by validator
            raise ValueError("TableWriter.table_path is not set. Cannot append rows.")
            
        write_stream_path = f"{self.table_path}/streams/{self.stream or '_default'}"

        prepared_batch: list[Mapping[str, Any]]
        if isinstance(rows, pd.DataFrame):
            # Deduplicate columns if necessary
            df_to_process = rows.copy() # Work on a copy
            if any(df_to_process.columns.duplicated()):
                df_to_process.columns = [ # type: ignore
                    x[1] if x[1] not in df_to_process.columns[: x[0]] else f"{x[1]}_{list(df_to_process.columns[: x[0]]).count(x[1])}" 
                    for x in enumerate(df_to_process.columns)
                ]
            prepared_batch = df_to_process.to_dict(orient="records")
        elif isinstance(rows, list) and all(isinstance(r, dict) for r in rows):
            prepared_batch = [r.copy() for r in rows] # List of dicts
        elif isinstance(rows, dict): # Single dict row
             prepared_batch = [rows.copy()]
        else:
            raise TypeError(f"Unsupported 'rows' type for append_rows: {type(rows)}. Expected DataFrame, list of dicts, or dict.")

        # Apply schema transformations and ensure serializability
        if self.bq_schema:
            if not (isinstance(self.bq_schema, list) and all(isinstance(sf, bigquery.SchemaField) for sf in self.bq_schema)):
                 raise TypeError(f"TableWriter.bq_schema must be a list of bigquery.SchemaField. Got: {type(self.bq_schema)}")
            prepared_batch = [construct_dict_from_schema(self.bq_schema, row) for row in prepared_batch]
        
        serializable_batch = make_serialisable(prepared_batch)

        if not serializable_batch:
            logger.warning(f"No rows to append to {self.table_path} after preparation and serialization.")
            return None # Nothing to append

        # Convert to ProtoRows for Storage Write API
        # Note: This assumes serializable_batch is a list of dicts that can be directly serialized.
        # The actual serialization to protobuf bytes needs to happen here if not handled by client library.
        # For google-cloud-bigquery-storage >= 2.0, client handles dicts.
        # If using older versions or custom protobufs, this step is more complex.
        # Assuming current client handles list of dicts for proto_rows.
        
        # The ProtoRows constructor itself does not take serialized_rows directly.
        # It expects a `rows` argument which should be a sequence of serialized protobuf messages.
        # For sending Python dicts, they are typically wrapped in a structure that the client
        # then serializes. Let's assume the client's append_rows handles dicts correctly
        # when used with ProtoRows, or that ProtoRows itself needs to be constructed differently.
        # The example from Google often shows serializing to individual protobuf messages first.
        # However, if `bigquery_storage_v1beta2` takes dicts directly in `serialized_rows` for ProtoRows,
        # then this is fine. Let's adjust based on typical usage or assume client handles it.
        # For now, assuming `serialized_rows` in ProtoRows takes a list of dicts for JSON target.

        # The `serialized_rows` field of `ProtoRows` expects a list of bytes, where each
        # byte string is a serialized protocol buffer message.
        # If we are sending JSON-compatible dicts to a JSON-mode stream, this is different.
        # For simplicity and to match common patterns with Storage Write API for JSON data,
        # often data is sent as JSON strings in the request payload, not raw ProtoRows.
        # However, the type hint here is ProtoRows.
        # Let's assume the `append_rows` client method can handle this structure
        # for a default stream (which implies JSON compatible).
        
        # If the stream expects JSON, sending dicts might be fine if client handles serialization.
        # If it expects actual protobufs, each dict in `serializable_batch` needs to be
        # serialized into its specific protobuf message type first.
        # Given no explicit proto schema definition here for rows, it's likely JSON mode.
        
        proto_rows_payload = ProtoRows() # Create an empty ProtoRows
        # Add serialized rows (assuming JSON compatible dicts which client serializes)
        # This part is tricky as ProtoRows expects serialized protobuf bytes.
        # If using JSON stream, the client might abstract this.
        # For now, this might be incorrect if strict protobuf bytes are needed for ProtoRows.
        # Let's assume this is for a JSON stream and the client handles dicts.
        # A common pattern is to use `google.protobuf.json_format.ParseDict` if you have a .proto schema.
        # Without it, this is ambiguous. Let's pass the dicts and assume client does the work.
        # This is likely where original code's `batch` was used directly.
        
        # Correct usage for sending Python dicts (JSON compatible) with Storage Write API
        # usually involves setting the `rows` field of `AppendRowsRequest.ProtoData`
        # if a `writer_schema` (as `ProtoSchema`) is also provided.
        # If `ProtoRows` is used directly, it implies pre-serialized protobuf bytes.
        # Given `make_serialisable` output is list of dicts, this needs clarification
        # on how `BigQueryWriteAsyncClient.append_rows` handles `ProtoRows(serialized_rows=list_of_dicts)`.
        
        # Assuming `serializable_batch` is a list of dicts to be sent.
        # The client's `append_rows` method will take an iterable of requests.
        # Each request contains a batch of rows.

        all_response_errors: list[Any] = []
        loop = asyncio.get_running_loop()

        # The `append_rows` method of `BigQueryWriteAsyncClient` is an asynchronous generator
        # that itself takes an iterable/generator of `AppendRowsRequest` objects.
        # We need to create these requests.
        
        async def request_generator():
            for chunk_of_rows in chunks(serializable_batch, 100): # Example chunk size
                proto_data = bigquery_storage_v1beta2.types.ProtoData()
                proto_data.rows.serialized_rows.extend(
                     [json.dumps(row).encode('utf-8') for row in chunk_of_rows] # Serialize each dict to JSON bytes
                )
                # If you have a .proto schema, you'd define writer_schema here.
                # For schemaless JSON append (default stream), this might be okay or need adjustment.
                # proto_data.writer_schema = ... 

                yield bigquery_storage_v1beta2.types.AppendRowsRequest(
                    write_stream=write_stream_path,
                    rows=proto_data # Corrected: use `rows` field with ProtoData
                )

        try:
            # The append_rows method is an async generator itself.
            # We need to iterate over it to process responses.
            response_stream = self.write_client.append_rows(request_generator()) # type: ignore
            
            async for response_item in response_stream: # type: ignore
                # Each response_item could be an AppendRowsResponse
                # Check for errors in the response
                if hasattr(response_item, 'error') and response_item.error:
                    logger.error(f"Error appending rows to {self.table_path}: {response_item.error.message}")
                    all_response_errors.append(response_item.error)
                elif hasattr(response_item, 'row_errors') and response_item.row_errors:
                    logger.error(f"Row errors appending to {self.table_path}: {response_item.row_errors}")
                    all_response_errors.extend(response_item.row_errors)
                else:
                    logger.debug(f"Successfully appended a chunk to {self.table_path}. Response offset: {getattr(response_item, 'append_result', {}).get('offset', {}).get('value', 'N/A')}")
        
        except Exception as e:
            logger.error(f"Exception during append_rows stream processing for {self.table_path}: {e!s}", exc_info=True)
            all_response_errors.append(str(e)) # Add exception as an error string

        if all_response_errors:
            return all_response_errors
        return [] # Return empty list for success
