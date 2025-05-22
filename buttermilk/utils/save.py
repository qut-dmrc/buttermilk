"""Utilities for saving various data types to local disk or cloud storage.

This module provides a flexible `save` function that attempts to serialize and
upload data using different methods, prioritizing cloud storage (like Google
Cloud Storage and BigQuery) and falling back to local disk storage if necessary.
It includes helper functions for specific upload tasks like handling Pandas
DataFrames, JSON data, binary data, and pickled objects. Retry mechanisms are
implemented for cloud operations using the `tenacity` library.
"""

import asyncio
import io # For in-memory binary streams (BytesIO)
import json
import pickle # For serializing Python objects
import tempfile # For creating temporary files/directories
from collections.abc import Hashable, Mapping # For type hinting
from pathlib import Path # For local path manipulation
from typing import Any

import google.cloud.storage # For GCS interactions (though client often from bm)
import pandas as pd
import shortuuid # For generating short unique IDs
from cloudpathlib import AnyPath, CloudPath, GSPath # For handling local and cloud paths
from cloudpathlib.exceptions import InvalidPrefixError # Specific CloudPathlib exception
from google.api_core.exceptions import ClientError, GoogleAPICallError # Google Cloud exceptions
from google.cloud import bigquery, storage # Google Cloud clients
from pydantic import BaseModel # For type checking if data is a Pydantic model
from tenacity import ( # Retry library components
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)


# Use deferred import to avoid circular references if this module is imported early
def get_bm() -> Any: # Return type should be 'BM' from bm_init.py if type hint is resolvable
    """Gets the Buttermilk global singleton instance (`bm`) with a delayed import.
    
    This helps avoid circular dependencies that can occur if `bm` is imported
    at the module level by many utility files.

    Returns:
        Any: The Buttermilk global instance (`bm`).
    """
    from buttermilk._core.bm_init import get_bm as _get_bm # Actual import of get_bm
    return _get_bm()


from buttermilk._core.config import SaveInfo # Configuration model for save operations

from .._core.log import logger # Centralized logger
from .utils import ( # Other utility functions from the same package
    chunks,
    make_serialisable,
    reset_index_and_dedup_columns,
    scrub_serializable,
)


def save(
    data: Any,
    save_dir: AnyPath | str = "",
    uri: CloudPath | str = "",
    basename: str = "",
    extension: str = "",
    **parameters: Any,
) -> str | None:
    """Saves various data types to a specified location (cloud or local).

    This function acts as a versatile saver, attempting different serialization
    and upload methods based on the data type and provided parameters.
    It prioritizes saving to Google BigQuery if schema and dataset parameters are
    given. If a `uri` (especially a cloud URI like "gs://") is provided or
    constructed, it attempts to upload there (e.g., DataFrames as JSON).
    As fallbacks, it tries generic JSON upload, binary upload, and finally,
    saving to local disk (as JSON or pickle).

    Args:
        data (Any): The data to be saved. Can be a Pandas DataFrame, list of dicts,
            Pydantic model, string, bytes, or other pickleable Python object.
        save_dir (AnyPath | str): The base directory for saving. If empty,
            it attempts to use `bm.save_dir` from the global Buttermilk instance.
            Can be a local path string or a `CloudPath` object.
        uri (CloudPath | str): An optional full URI (e.g., "gs://bucket/file.json",
            "s3://...", or a local file path) where the data should be saved.
            If provided, it often takes precedence over constructing a path from
            `save_dir`, `basename`, and `extension`.
        basename (str): The base name for the file (without extension). If not
            provided, a name might be generated (e.g., using a UUID).
        extension (str): The file extension (e.g., ".json", ".csv", ".pkl").
            If provided, it's appended to `uri` or the constructed filename.
        **parameters (Any): Additional parameters that can control saving behavior:
            - `schema` (list | str): BigQuery schema (list of SchemaField or path to schema JSON).
              If provided along with `dataset`, triggers BigQuery upload attempt.
            - `dataset` (str): BigQuery dataset ID (e.g., "project.dataset.table").
              If provided along with `schema`, triggers BigQuery upload attempt.
            - `uuid` (str): An optional UUID to include in the generated filename if
              `basename` is also used.

    Returns:
        str | None: The URI or path of the saved file as a string if successful,
        or `None` if all save attempts fail.

    Raises:
        OSError: If all attempted save methods fail.
        TypeError: If data types are incompatible with chosen save methods.
    """
    # from .utils import reset_index_and_dedup_columns # Already imported at module level

    final_save_dir_str: str | None = None
    if isinstance(save_dir, (CloudPath, Path)):
        final_save_dir_str = str(save_dir)
    elif isinstance(save_dir, str) and save_dir:
        final_save_dir_str = save_dir
    else: # save_dir is empty or not a path type
        try:
            bm_instance = get_bm()
            final_save_dir_str = bm_instance.save_dir
            if not final_save_dir_str: # If bm.save_dir is also None or empty
                logger.warning("`save_dir` not provided and `bm.save_dir` is not set. Saving might default to temporary or current dir.")
        except Exception as e:
            logger.warning(f"Could not find default save_dir from BM object (bm.save_dir). Error: {e!s}. Saving might default to temporary or current dir.")

    # Prepare data: Ensure DataFrame index is serializable if it's a DataFrame
    if isinstance(data, pd.DataFrame):
        if not (len(data.index.names) == 1 and data.index.name is None): # Not a simple RangeIndex
            data = reset_index_and_dedup_columns(data)

    # Attempt 1: Upload to BigQuery if schema and dataset are provided
    if "schema" in parameters and "dataset" in parameters:
        try:
            destination_table = upload_rows(rows=data, **parameters) # Pass all params
            logger.info(f"Successfully uploaded data to BigQuery table: {destination_table}.")
            return destination_table
        except Exception as e:
            logger.error(f"Failed to upload data to BigQuery (schema/dataset provided). Error: {e!s}", exc_info=True)
            # Continue to other save methods if BigQuery upload fails

    # Determine final URI for file-based saving
    final_uri_str: str | None = None
    if uri: # If a full URI is explicitly provided
        final_uri_str = str(uri)
    elif final_save_dir_str: # Construct URI from save_dir, basename, extension
        try:
            # Convert string path to AnyPath for consistent handling
            base_path = AnyPath(final_save_dir_str)
            file_id = parameters.get("uuid", shortuuid.uuid())
            effective_basename = "_".join(filter(None, [basename, file_id])) or f"data_{file_id}"
            
            target_path = base_path / effective_basename
            if extension:
                # Ensure extension starts with a dot
                dot_extension = extension if extension.startswith(".") else "." + extension
                target_path = target_path.with_suffix(dot_extension)
            final_uri_str = str(target_path)
        except InvalidPrefixError: # If save_dir was a local path string not convertible to CloudPath directly
             logger.warning(f"Interpreting save_dir '{final_save_dir_str}' as local path due to CloudPath InvalidPrefixError.")
             # Fallback to local Path logic if CloudPath parsing fails for local-like strings
             base_path_local = Path(final_save_dir_str)
             file_id_local = parameters.get("uuid", shortuuid.uuid())
             effective_basename_local = "_".join(filter(None, [basename, file_id_local])) or f"data_{file_id_local}"
             target_path_local = base_path_local / effective_basename_local
             if extension:
                dot_extension_local = extension if extension.startswith(".") else "." + extension
                target_path_local = target_path_local.with_suffix(dot_extension_local)
             final_uri_str = str(target_path_local)

        except Exception as e: # Catch other path construction errors
            logger.warning(
                f"Error constructing save URI from save_dir='{final_save_dir_str}', basename='{basename}'. Error: {e!s}. Parameters: {parameters}",
                exc_info=True
            )
            # final_uri_str remains None, will try fallback saving to temp dir

    # List of upload methods to try for file-based saving
    # These are ordered from more specific/cloud-preferred to general/local fallbacks.
    file_upload_methods: list[Callable[..., str | None]] = []
    
    if final_uri_str and AnyPath(final_uri_str).is_cloud: # Prioritize cloud methods if URI is cloud
        if isinstance(data, pd.DataFrame):
            # Add DataFrame-specific cloud upload first if applicable
            file_upload_methods.append(lambda **kwargs_lambda: upload_dataframe_json(data=data, uri=final_uri_str, **kwargs_lambda.get("kwargs_for_upload",{}))) # type: ignore
        file_upload_methods.extend([
            upload_json,  # Tries to serialize to JSON and upload as text
            upload_binary, # Tries to upload as binary (e.g., for pickled data if serialized to bytes)
        ])

    # Fallback to local disk saving methods
    file_upload_methods.extend([
        dump_to_disk, # Saves as JSON locally
        dump_pickle,  # Saves as pickle locally
    ])

    # Try each method until one succeeds
    for method_func in file_upload_methods:
        try:
            logger.debug(f"Attempting to save data using method: {method_func.__name__} to path/URI: '{final_uri_str if final_uri_str else 'temporary_path'}'.")
            # Prepare arguments for the save method
            method_args = {
                "data": data,
                "uri": final_uri_str, # Pass URI if available (for cloud methods)
                "save_dir": final_save_dir_str or tempfile.gettempdir(), # save_dir for local methods
                "extension": extension or ( ".json" if method_func in [upload_json, dump_to_disk] else ".pkl" if method_func == dump_pickle else ".bin"), # Sensible default extension
                **parameters.get("kwargs_for_upload", {}) # Pass through extra kwargs if any
            }
            # Filter out None args that the method might not expect (like uri for dump_to_disk)
            # This needs to be more nuanced based on each method's signature.
            # For simplicity, assuming methods can handle None for uri/save_dir if not applicable.

            destination_path = method_func(**method_args) # type: ignore # Call with prepared args
            if destination_path: # If method returns a path (success)
                logger.info(f"Successfully saved data using {method_func.__name__} to: {destination_path}.")
                return str(destination_path)
        except (GoogleAPICallError, ClientError) as e_cloud: # Specific cloud errors
            logger.warning(f"Cloud save error using {method_func.__name__} for '{final_uri_str}': {e_cloud!s}. Trying next method.")
        except TypeError as e_type: # Catch TypeErrors from methods not expecting certain data types
            logger.warning(f"Type error using {method_func.__name__} (data type: {type(data)}): {e_type!s}. Trying next method.")
        except Exception as e_general: # Catch other errors
            logger.warning(f"Failed to save data using {method_func.__name__}: {e_general!s}", exc_info=True) # Log with stack trace for debugging

    # If all methods fail
    err_msg = f"Critical failure: Unable to save data using any available method. Tried: {[m.__name__ for m in file_upload_methods]}."
    logger.error(err_msg)
    raise OSError(err_msg)


@retry(
    retry=retry_if_exception_type((GoogleAPICallError, ClientError)),
    wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
    stop=stop_after_attempt(5),
)
def upload_dataframe_json(data: pd.DataFrame, uri: str, **kwargs: Any) -> str:
    """Uploads a Pandas DataFrame to a specified URI as newline-delimited JSON.

    Primarily targets Google Cloud Storage (GCS). It first attempts to serialize
    the DataFrame to NDJSON in memory and upload it as a binary blob. If that
    fails, it falls back to using Pandas' `to_json` method with GCS path support.
    Ensures DataFrame columns are unique before serialization.

    Args:
        data (pd.DataFrame): The Pandas DataFrame to upload.
        uri (str): The destination URI, typically a GCS path (e.g., "gs://bucket/file.jsonl").
        **kwargs: Additional keyword arguments (not directly used by this function but
                  available for future extensions or if wrapped).

    Returns:
        str: The URI where the DataFrame was saved.

    Raises:
        TypeError: If `data` is not a Pandas DataFrame.
        google.api_core.exceptions.GoogleAPICallError: For GCS API errors after retries.
        google.api_core.exceptions.ClientError: For GCS client errors after retries.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input `data` must be a Pandas DataFrame for upload_dataframe_json.")

    if data.empty:
        logger.info(f"DataFrame is empty. No data uploaded to {uri}.")
        # Depending on desired behavior, could create an empty file or just return.
        # For now, let's try to write an empty file to signify an empty DataFrame.
        try:
            gcs_client = storage.Client()
            blob = google.cloud.storage.blob.Blob.from_string(uri=uri, client=gcs_client)
            blob.upload_from_string("".encode("utf-8"), content_type="application/jsonl") # Empty NDJSON
            logger.info(f"Uploaded empty DataFrame placeholder to {uri}.")
            return uri
        except Exception as e:
            logger.warning(f"Failed to upload empty DataFrame placeholder to {uri}: {e!s}")
            return uri # Return URI even if empty placeholder fails, as data was empty

    # Ensure unique column names; reset index if it's complex
    if any(data.columns.duplicated()):
        data = reset_index_and_dedup_columns(data) # Assumes this handles multi-index too
    
    try:
        gcs_client = storage.Client()
        # Serialize DataFrame to newline-delimited JSON string
        rows_dict = data.to_dict(orient="records")
        scrubbed_rows = scrub_serializable(rows_dict) # Ensure all data is JSON serializable
        
        json_data_str = "\n".join([json.dumps(row) for row in scrubbed_rows])
        json_data_bytes = json_data_str.encode("utf-8")
        
        blob = google.cloud.storage.blob.Blob.from_string(uri=uri, client=gcs_client)
        
        # Upload from an in-memory bytes buffer
        with io.BytesIO(json_data_bytes) as buffer:
            blob.upload_from_file(file_obj=buffer, content_type="application/jsonl")
        logger.info(f"Successfully uploaded DataFrame as NDJSON to GCS: {uri}")
        return uri
    except Exception as e_bytesio: # Fallback to pandas direct GCS upload if BytesIO fails
        logger.warning(
            f"Error saving DataFrame to {uri} using in-memory BytesIO method: {e_bytesio!s}. "
            "Falling back to pandas `to_json` with GCSFS."
        )
        try:
            # Pandas to_json can write directly to GCS if gcsfs is installed
            data.to_json(uri, orient="records", lines=True, compression="gzip" if uri.endswith(".gz") else None)
            logger.info(f"Successfully uploaded DataFrame via pandas.to_json to GCS: {uri}")
            return uri
        except Exception as e_pandas:
            logger.error(f"Pandas to_json fallback also failed for {uri}: {e_pandas!s}", exc_info=True)
            raise # Re-raise the pandas error if fallback also fails


def data_to_export_rows(
    data: pd.DataFrame | dict[str, Any] | list[Mapping[str, Any]] | BaseModel,
    schema: list[bigquery.SchemaField], # Expecting BigQuery SchemaField objects
) -> list[Mapping[Hashable, Any]]:
    """Converts various data types into a list of dictionaries suitable for BigQuery row insertion.

    This function handles:
    - Pandas DataFrames: Converts to list of dicts. Deduplicates columns if needed.
    - Pydantic Models: Dumps the model to a JSON-compatible dictionary.
    - Single Dictionaries or Lists of Dictionaries: Copies the data.

    After initial conversion, it applies schema transformations using
    `construct_dict_from_schema` (from `.bq` module) and ensures all values
    are JSON serializable using `make_serialisable`.

    Args:
        data: The input data. Can be a Pandas DataFrame, a single dictionary,
            a list of dictionaries, or a Pydantic BaseModel instance.
        schema (list[bigquery.SchemaField]): The BigQuery schema definition (list of
            `SchemaField` objects) to which the rows should conform.

    Returns:
        list[Mapping[Hashable, Any]]: A list of dictionaries, where each dictionary
        represents a row ready for BigQuery insertion.
    """
    from .bq import construct_dict_from_schema # Deferred import

    bq_rows: list[Mapping[str, Any]] | Mapping[str,Any] # Adjusted type hint

    if isinstance(data, pd.DataFrame):
        # Deduplicate columns if necessary
        if any(data.columns.duplicated()):
             data.columns = [ # type: ignore
                x[1] if x[1] not in data.columns[: x[0]] else f"{x[1]}_{list(data.columns[: x[0]]).count(x[1])}" 
                for x in enumerate(data.columns)
            ]
        bq_rows = data.to_dict(orient="records")
    elif isinstance(data, BaseModel): # Check for Pydantic BaseModel
        bq_rows = [data.model_dump(mode="json")] # Use model_dump for Pydantic v2
    elif isinstance(data, dict): # Single dictionary row
        bq_rows = [data.copy()]
    elif isinstance(data, list) and all(isinstance(i, dict) for i in data): # List of dictionaries
        bq_rows = [d.copy() for d in data] # Create copies of dicts
    else:
        raise TypeError(f"Unsupported data type for data_to_export_rows: {type(data)}. "
                        "Expected DataFrame, Pydantic BaseModel, dict, or list of dicts.")

    # Apply schema transformations and ensure serializability
    # Ensure bq_rows is a list before list comprehension
    rows_as_list = bq_rows if isinstance(bq_rows, list) else [bq_rows]
    
    # Ensure schema is a list of SchemaField objects for construct_dict_from_schema
    if not (isinstance(schema, list) and all(isinstance(sf, bigquery.SchemaField) for sf in schema)):
        raise TypeError(f"Schema must be a list of bigquery.SchemaField objects. Got: {type(schema)}")

    transformed_rows = [construct_dict_from_schema(schema, row) for row in rows_as_list]
    serializable_rows = make_serialisable(transformed_rows) # make_serialisable should return list

    return serializable_rows # type: ignore # Final type should be list[Mapping[Hashable, Any]]


async def upload_rows_async(
    rows: Any, 
    *, 
    schema: list[bigquery.SchemaField] | str | None = None, 
    dataset: str | None = None, 
    save_dest: SaveInfo | None = None
) -> str | None:
    """Uploads rows to a Google BigQuery table asynchronously.

    The target table and schema can be specified directly or via a `SaveInfo` object.
    Data is prepared using `data_to_export_rows` and uploaded in chunks.
    This function uses `asyncio.get_running_loop().run_in_executor` to run
    blocking BigQuery client operations in a separate thread pool, making the
    overall operation non-blocking for an async application.

    Args:
        rows (Any): The data to upload. Can be a Pandas DataFrame, Pydantic model,
            dict, or list of dicts compatible with `data_to_export_rows`.
        schema (list[bigquery.SchemaField] | str | None): BigQuery table schema.
            Can be a list of `SchemaField` objects or a path to a JSON schema file.
            If None, attempts to get from `save_dest.bq_schema`.
        dataset (str | None): The BigQuery table ID (format: "project.dataset.table").
            If None, attempts to get from `save_dest.dataset`.
        save_dest (SaveInfo | None): A `SaveInfo` object containing schema and
            dataset information. Used if `schema` or `dataset` are not directly provided.

    Returns:
        str | None: The BigQuery table ID if upload was successful, `None` otherwise.

    Raises:
        AssertionError: If both `schema` (or `save_dest.bq_schema`) and `dataset`
            (or `save_dest.dataset`) are not resolved.
        OSError: If there are errors during BigQuery table operations (getting table,
            inserting rows).
        TypeError: If schema format is incorrect.
    """
    final_schema = schema or (save_dest.bq_schema if save_dest else None)
    final_dataset = dataset or (save_dest.dataset if save_dest else None)
    assert final_schema is not None, "Schema must be provided either directly or via save_dest."
    assert final_dataset is not None, "Dataset (table ID) must be provided either directly or via save_dest."

    loop = asyncio.get_running_loop()
    # Instantiate BigQuery client in executor as it might do I/O or be blocking
    bq_client: bigquery.Client = await loop.run_in_executor(None, bigquery.Client)

    # Resolve schema if it's a path string
    if isinstance(final_schema, str):
        try:
            final_schema = bq_client.schema_from_json(final_schema) # This is a sync call
        except Exception as e:
            raise TypeError(f"Failed to load schema from JSON path '{final_schema}': {e!s}") from e
    
    if not (isinstance(final_schema, list) and all(isinstance(sf, bigquery.SchemaField) for sf in final_schema)):
        raise TypeError(f"Resolved schema is not a list of bigquery.SchemaField. Type: {type(final_schema)}")


    # data_to_export_rows can be CPU-bound, consider executor if it's very heavy for large data.
    # For now, assuming it's acceptable to run in the current thread before dispatching BQ I/O.
    bq_prepared_rows = data_to_export_rows(rows, schema=final_schema)

    if not bq_prepared_rows:
        logger.warning(f"No rows to upload to BigQuery table {final_dataset} after preparation.")
        return None

    try:
        table_ref = await loop.run_in_executor(None, bq_client.get_table, final_dataset)
    except Exception as e:
        err_msg = (f"Unable to get BigQuery table '{final_dataset}' for async upload. "
                   f"It may not exist or there are permission issues. Error: {e!s}")
        logger.error(err_msg)
        raise OSError(err_msg) from e # Re-raise as OSError for consistency with sync version

    logger.debug(f"Inserting {len(bq_prepared_rows)} rows asynchronously to BigQuery table {final_dataset}.")

    insertion_tasks = []
    # bq_client.insert_rows is a blocking I/O call, run each chunk insertion in an executor thread
    for row_chunk in chunks(bq_prepared_rows, 100): # Default chunk size of 100
        insertion_tasks.append(
            loop.run_in_executor(None, bq_client.insert_rows, table_ref, row_chunk, final_schema)
        )

    insertion_results = await asyncio.gather(*insertion_tasks) # Gather results from all chunk insertions
    all_errors = [error for sublist_errors in insertion_results for error in sublist_errors if sublist_errors] # Flatten and filter

    if not all_errors:
        logger.info(f"Successfully pushed {len(bq_prepared_rows)} rows asynchronously to BigQuery table {final_dataset}.")
    else:
        # Log detailed errors if possible, summarize for exception
        error_summary = str(all_errors)[:1000] # Limit error string length
        logger.error(f"Errors during async BigQuery upload to {final_dataset}: {all_errors}")
        raise OSError(f"Google BigQuery returned errors during async upload to {final_dataset}: {error_summary}")

    return final_dataset


def upload_rows(
    rows: Any, 
    *, 
    schema: list[bigquery.SchemaField] | str | None = None, 
    dataset: str | None = None, 
    save_dest: SaveInfo | None = None, 
    create_if_not_exists: bool = False, # Parameter not used in current implementation
    **parameters: Any # Catch-all for other params, not used directly here
) -> str | None:
    """Uploads rows to a Google BigQuery table synchronously.

    Similar to `upload_rows_async` but performs operations synchronously.
    It prepares data using `data_to_export_rows` and uploads it in chunks.
    The `create_if_not_exists` parameter is defined but not currently implemented;
    the table must exist.

    Args:
        rows (Any): Data to upload (Pandas DataFrame, Pydantic model, dict, list of dicts).
        schema (list[bigquery.SchemaField] | str | None): BigQuery table schema or path to schema JSON.
            Uses `save_dest.bq_schema` if None.
        dataset (str | None): BigQuery table ID ("project.dataset.table").
            Uses `save_dest.dataset` if None.
        save_dest (SaveInfo | None): `SaveInfo` object with schema and dataset.
        create_if_not_exists (bool): If True, would attempt to create the table
            if it doesn't exist. **Currently not implemented.** Defaults to False.
        **parameters: Additional parameters (currently ignored by this function).

    Returns:
        str | None: The BigQuery table ID if upload successful, `None` otherwise.

    Raises:
        AssertionError: If schema or dataset cannot be resolved.
        OSError: If BigQuery operations fail (e.g., table not found, insertion errors).
        TypeError: If schema format is incorrect.
    """
    # Resolve schema and dataset, preferring direct args, then from save_dest
    final_schema = schema or (save_dest.bq_schema if save_dest else None)
    final_dataset = dataset or (save_dest.dataset if save_dest else None)
    assert final_schema is not None, "Schema must be provided either directly or via save_dest."
    assert final_dataset is not None, "Dataset (table ID) must be provided either directly or via save_dest."

    bq_client = bigquery.Client()  # Uses application default credentials

    if isinstance(final_schema, str): # If schema is a path, load it
        try:
            final_schema = bq_client.schema_from_json(final_schema)
        except Exception as e:
            raise TypeError(f"Failed to load schema from JSON path '{final_schema}': {e!s}") from e
            
    if not (isinstance(final_schema, list) and all(isinstance(sf, bigquery.SchemaField) for sf in final_schema)):
        raise TypeError(f"Resolved schema is not a list of bigquery.SchemaField. Type: {type(final_schema)}")

    bq_prepared_rows = data_to_export_rows(rows, schema=final_schema)

    if not bq_prepared_rows:
        logger.warning(f"No rows to upload to BigQuery table {final_dataset} after preparation.")
        return None # Nothing to upload

    try:
        table_ref = bq_client.get_table(final_dataset) # Check if table exists
    except Exception as e:
        # TODO: Implement create_if_not_exists if flag is True
        # if create_if_not_exists:
        #     logger.info(f"Table {final_dataset} not found. `create_if_not_exists` is True, attempting to create.")
        #     table = bigquery.Table(final_dataset, schema=final_schema)
        #     table = bq_client.create_table(table)
        #     logger.info(f"Created table {table.path}")
        # else:
        err_msg = (f"Unable to get BigQuery table '{final_dataset}'. "
                   f"It may not exist or there are permission issues. Error: {e!s}")
        logger.error(err_msg)
        raise OSError(err_msg) from e

    logger.debug(f"Inserting {len(bq_prepared_rows)} rows into BigQuery table {final_dataset}.")

    all_errors = []
    for row_chunk in chunks(bq_prepared_rows, 100): # Process in chunks of 100 rows
        chunk_errors = bq_client.insert_rows(table_ref, row_chunk, selected_fields=final_schema)
        if chunk_errors:
            all_errors.extend(chunk_errors)

    if not all_errors:
        logger.info(f"Successfully pushed {len(bq_prepared_rows)} rows to BigQuery table {final_dataset}.")
    else:
        error_summary = str(all_errors)[:1000] # Limit error string length for logging
        logger.error(f"Errors during BigQuery upload to {final_dataset}: {all_errors}")
        raise OSError(f"Google BigQuery returned errors during upload to {final_dataset}: {error_summary}")

    return final_dataset


@retry(
    retry=retry_if_exception_type((GoogleAPICallError, ClientError)),
    wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
    stop=stop_after_attempt(5),
)
def upload_binary(data: bytes | io.BufferedIOBase, *, uri: str) -> str:
    """Uploads binary data to a Google Cloud Storage (GCS) URI.

    The data can be provided as raw bytes or an in-memory binary stream
    (e.g., `io.BytesIO`).

    Args:
        data (bytes | io.BufferedIOBase): The binary data to upload.
        uri (str): The GCS destination URI (e.g., "gs://bucket/object.bin").

    Returns:
        str: The GCS URI where the data was uploaded.

    Raises:
        AssertionError: If `data` is None.
        google.api_core.exceptions.GoogleAPICallError: For GCS API errors after retries.
        google.api_core.exceptions.ClientError: For GCS client errors after retries.
    """
    assert data is not None, "Data for upload_binary cannot be None."
    gcs_client = storage.Client()

    logger.debug(f"Uploading binary data to GCS URI: {uri}.")
    # Create a Blob object from the GCS URI
    blob = google.cloud.storage.blob.Blob.from_string(uri=uri, client=gcs_client)

    if isinstance(data, io.BufferedIOBase): # If data is already a file-like object
        data.seek(0) # Ensure reading from the beginning
        blob.upload_from_file(file_obj=data)
    elif isinstance(data, bytes): # If data is raw bytes
        with io.BytesIO(data) as buffer:
            blob.upload_from_file(file_obj=buffer)
    else:
        raise TypeError(f"Unsupported data type for upload_binary: {type(data)}. Expected bytes or BufferedIOBase.")
    
    logger.info(f"Successfully uploaded binary data to {uri}.")
    return uri


def dump_to_disk(data: Any, *, save_dir: str, extension: str = ".json", **kwargs: Any) -> str:
    """Saves data to a temporary file on the local disk, typically as JSON.

    Creates the `save_dir` if it doesn't exist. If `data` is a Pandas DataFrame,
    it's saved as newline-delimited JSON. Otherwise, `json.dumps` is used.

    Args:
        data (Any): The data to save.
        save_dir (str): The local directory where the temporary file will be created.
        extension (str): The file extension (including the dot) for the temporary file.
                         Defaults to ".json".
        **kwargs: Additional keyword arguments (currently unused).

    Returns:
        str: The full path to the saved temporary file.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True) # Ensure directory exists
    # Use NamedTemporaryFile to get a unique filename and manage it safely
    with tempfile.NamedTemporaryFile(
        delete=False, # Keep the file after closing for the caller to use
        dir=save_dir,
        mode="w", # Write in text mode
        suffix=extension,
        encoding="utf-8" # Specify encoding
    ) as out_file:
        if isinstance(data, pd.DataFrame):
            data.to_json(out_file, orient="records", lines=True, date_format="iso")
        else:
            # Ensure data is JSON serializable before dumping
            try:
                json.dump(scrub_serializable(data), out_file, indent=kwargs.get("indent")) # Use scrub_serializable
            except TypeError as e:
                logger.error(f"Data is not JSON serializable for dump_to_disk: {e!s}. Data type: {type(data)}", exc_info=True)
                # Fallback or re-raise, depending on desired behavior. For now, log and continue.
                # This might leave an empty temp file.
        
        saved_filepath = out_file.name
    logger.info(f"Successfully dumped data to local disk (JSON): {saved_filepath}.")
    return saved_filepath


def dump_pickle(data: Any, *, save_dir: str, extension: str = ".pickle", **kwargs: Any) -> str:
    """Saves Python objects to a temporary file on local disk using `pickle`.

    Creates the `save_dir` if it doesn't exist.

    Args:
        data (Any): The Python object to pickle.
        save_dir (str): The local directory for the temporary pickled file.
        extension (str): File extension for the pickled file (including the dot).
                         Defaults to ".pickle".
        **kwargs: Additional keyword arguments (currently unused).

    Returns:
        str: The full path to the saved pickled file.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        delete=False, # Keep the file
        dir=save_dir,
        mode="wb", # Write in binary mode for pickle
        suffix=extension,
    ) as out_file:
        pickle.dump(data, out_file)
        saved_filepath = out_file.name
    logger.info(f"Successfully dumped data to local disk (pickle): {saved_filepath}.")
    return saved_filepath


def read_pickle(filename: str | GSPath) -> Any:
    """Reads a pickled object from a local file or a GCS path.

    Args:
        filename (str | GSPath): The local file path (as a string) or a
            `cloudpathlib.GSPath` object pointing to the pickled file in GCS.

    Returns:
        Any: The unpickled Python object.
    """
    path_to_read: AnyPath
    if isinstance(filename, str):
        # Determine if it's a GCS path string or local path string
        path_to_read = AnyPath(filename) # AnyPath handles gs:// or local
    elif isinstance(filename, GSPath): # Already a GSPath
        path_to_read = filename
    else:
        raise TypeError(f"Unsupported filename type for read_pickle: {type(filename)}. Expected str or GSPath.")

    logger.info(f"Reading pickled object from: {path_to_read}")
    try:
        pickle_bytes = path_to_read.read_bytes()
        with io.BytesIO(pickle_bytes) as buffer:
            unpickled_object = pickle.load(buffer)
        return unpickled_object
    except Exception as e:
        logger.error(f"Failed to read or unpickle from {path_to_read}: {e!s}", exc_info=True)
        raise


@retry(
    retry=retry_if_exception_type((GoogleAPICallError, ClientError)),
    wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
    stop=stop_after_attempt(5),
)
def upload_text(data: str, *, uri: str, **kwargs: Any) -> str:
    """Uploads text data to a Google Cloud Storage (GCS) URI.

    Args:
        data (str): The string data to upload.
        uri (str): The GCS destination URI (e.g., "gs://bucket/object.txt").
        **kwargs: Additional keyword arguments (not directly used by this function).

    Returns:
        str: The GCS URI where the data was uploaded.

    Raises:
        google.api_core.exceptions.GoogleAPICallError: For GCS API errors after retries.
        google.api_core.exceptions.ClientError: For GCS client errors after retries.
    """
    gcs_client = storage.Client()
    logger.debug(f"Uploading text data to GCS URI: {uri}.")
    blob = google.cloud.storage.blob.Blob.from_string(uri, client=gcs_client)
    blob.upload_from_string(data, content_type=kwargs.get("content_type", "text/plain"))
    
    # blob.size might not be populated immediately after upload_from_string without a reload.
    # For logging, len(data) is more direct for uncompressed text.
    logger.info(
        f"Successfully uploaded text file to {uri} with {len(data)} characters written.",
    )
    return uri


def upload_json(data: Any, *, uri: str, **kwargs: Any) -> str:
    """Serializes Python data to a newline-delimited JSON string and uploads it as text to GCS.

    Before serialization, it processes the data using `scrub_serializable` to
    handle common non-JSON-serializable types (like datetime objects).
    If `data` is a Pandas DataFrame, it's first converted to a list of dictionaries.

    Args:
        data (Any): The data to serialize and upload. Can be a Pandas DataFrame,
            list of dicts, or other JSON-serializable Python objects.
        uri (str): The GCS destination URI (e.g., "gs://bucket/data.jsonl").
        **kwargs: Additional keyword arguments passed to `upload_text` (e.g., `content_type`).

    Returns:
        str: The GCS URI where the JSON data was uploaded.
    """
    rows_to_serialize: list[Any]
    if isinstance(data, pd.DataFrame):
        rows_to_serialize = data.to_dict(orient="records")
    elif isinstance(data, list) and all(isinstance(i, dict) for i in data):
        rows_to_serialize = data
    elif isinstance(data, dict): # Single dictionary
        rows_to_serialize = [data]
    else: # Attempt to serialize other types if possible, might fail if not list/dict like
        logger.warning(f"Data for upload_json is not a DataFrame, list of dicts, or dict (type: {type(data)}). Attempting direct serialization.")
        rows_to_serialize = data # Will be scrubbed next

    scrubbed_rows = scrub_serializable(rows_to_serialize) # Ensure serializability

    # Convert to newline-delimited JSON string
    # If scrubbed_rows is not a list (e.g., a single dict was passed and scrubbed), wrap it for join
    if not isinstance(scrubbed_rows, list):
        scrubbed_rows_list = [scrubbed_rows]
    else:
        scrubbed_rows_list = scrubbed_rows
        
    json_string_payload = "\n".join([json.dumps(row_item) for row_item in scrubbed_rows_list])

    # Set content_type to application/jsonl for newline-delimited JSON
    kwargs.setdefault("content_type", "application/jsonl")
    return upload_text(json_string_payload, uri=uri, **kwargs)
