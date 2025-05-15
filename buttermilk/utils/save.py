import asyncio
import io
import json
import pickle
import tempfile
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

import google.cloud.storage
import pandas as pd
import shortuuid
from cloudpathlib import AnyPath, CloudPath, GSPath
from cloudpathlib.exceptions import InvalidPrefixError
from google.api_core.exceptions import ClientError, GoogleAPICallError
from google.cloud import bigquery, storage
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from buttermilk._core.config import SaveInfo
from buttermilk._core.job import Job

from .._core.log import logger
from .bq import construct_dict_from_schema
from .utils import (
    chunks,
    make_serialisable,
    reset_index_and_dedup_columns,
    scrub_serializable,
)


def save(
    data,
    save_dir: AnyPath | str = "",
    uri: CloudPath | str = "",
    basename: str = "",
    extension: str = "",
    **parameters,
):
    from .utils import reset_index_and_dedup_columns

    if not save_dir:
        try:

            save_dir = BM().save_dir
        except Exception as e:
            logger.warning(f"Could not find save dir from BM object (maybe not configured or initialised?) Error: {e}, {e.args=}")

    # Failsafe save routine. We should be able to find some way of dumping the data.
    # Try multiple methods in order until we get a result.
    if isinstance(data, pd.DataFrame):
        # Check if dataframe has a single range index
        if len(data.index.names) == 1 and data.index.name is None:
            pass
        else:
            # if not, reset the index to make sure we can save it
            data = reset_index_and_dedup_columns(data)

    try:
        if "schema" in parameters and "dataset" in parameters:
            destination = upload_rows(rows=data, **parameters)
            logger.debug(f"Uploaded data to BigQuery: {destination}.")
            return destination
    except Exception as e:
        logger.error(
            msg=f"Critical failure. Unable to upload to data BigQuery: {e!s}",
        )

    if not uri:
        try:
            save_dir = CloudPath(save_dir)
            id = parameters.get("uuid", shortuuid.uuid())
            basename = "_".join([x for x in [basename, id] if x])
            uri = save_dir / basename
            if extension:
                uri = uri.with_suffix(extension)
        except InvalidPrefixError:
            pass
        except Exception as e:
            logger.warning(
                f"Error saving data to cloud uri: {e} {e.args=}, {parameters}",
            )

    if isinstance(uri, CloudPath):
        uri = uri.as_uri()

    upload_methods = []
    if uri:
        try:
            # Try to upload to GCS
            if isinstance(data, pd.DataFrame):
                return upload_dataframe_json(data=data, uri=uri)
        except Exception as e:
            logger.warning(
                f"Error saving data to {uri} using upload_dataframe_json: {e} {e.args=}",
            )

        upload_methods = [
            upload_json,
            upload_binary,
        ]

    # save to disk as a last resort
    upload_methods.extend(
        [
            dump_to_disk,
            dump_pickle,
        ],
    )

    for method in upload_methods:
        try:
            logger.debug(f"Trying to save data using {method.__name__}.")
            destination = method(
                data=data,
                uri=uri,
                save_dir=save_dir,
                extension=extension,
            )
            logger.info(
                f"Saved data using {method.__name__} to: {destination}.",
            )
            return destination
        except (GoogleAPICallError, ClientError) as e:
            logger.warning(
                f"Error saving data to {uri} using {method.__name__}: {e} {e.args=}",
            )
        except Exception as e:
            logger.warning(
                f"Could not save data using {method.__name__}: {e} {e.args=}",
            )

    raise OSError(
        f"Critical failure. Unable to save using any method in {upload_methods}",
    )


@retry(
    retry=retry_if_exception_type((GoogleAPICallError, ClientError)),
    # Wait interval:  10 seconds first, increasing exponentially up to a max of two minutes between retries
    wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
    # Retry up to five times before giving up
    stop=stop_after_attempt(5),
)
def upload_dataframe_json(data: pd.DataFrame, uri, **kwargs) -> str:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")

    if any(data.columns.duplicated()):
        data = reset_index_and_dedup_columns(data)
    if not data.empty:
        try:
            gcs = storage.Client()
            rows = data.to_dict(orient="records")
            rows = scrub_serializable(rows)
            # Try to upload as newline delimited json
            json_data = "\n".join([json.dumps(row) for row in rows])
            json_data = json_data.encode("utf-8")
            blob = google.cloud.storage.blob.Blob.from_string(uri=uri, client=gcs)

            # Try to upload as binary from a file like object
            with io.BytesIO(json_data) as b:
                blob.upload_from_file(file_obj=b)
                return uri
        except Exception as e:
            logger.warning(
                f"Error saving data to {uri} in upload_dataframe_json using BytesIO: {e} {e.args=}",
            )
            # use pandas to upload to GCS
            data.to_json(uri, orient="records", lines=True)
            return uri

    return uri


def data_to_export_rows(
    data: pd.DataFrame | Job | dict | list[Mapping[str, Any]],
    schema: list,
) -> list[Mapping[Hashable, Any]]:
    if isinstance(data, pd.DataFrame):
        # deduplicate columns
        data.columns = [
            x[1] if x[1] not in data.columns[: x[0]] else f"{x[1]}_{list(data.columns[: x[0]]).count(x[1])}" for x in enumerate(data.columns)
        ]

        bq_rows = data.to_dict(orient="records")

    elif isinstance(data, BaseModel):
        bq_rows = [data.model_dump(mode="json")]
    else:
        bq_rows = data.copy()

    # Handle other conversions required for bigquery
    bq_rows = [construct_dict_from_schema(schema, row) for row in bq_rows]

    bq_rows = make_serialisable(bq_rows)

    return bq_rows


async def upload_rows_async(rows, *, schema=None, dataset=None, save_dest: SaveInfo = None):
    """Upload results to Google Bigquery asynchronously"""
    schema = schema or save_dest.bq_schema
    dataset = dataset or save_dest.dataset
    assert schema is not None and dataset is not None

    loop = asyncio.get_running_loop()
    bq = await loop.run_in_executor(None, bigquery.Client)  # Run sync client instantiation in executor

    if isinstance(schema, str):
        schema = bq.schema_from_json(schema)

    # data_to_export_rows is CPU-bound, can run directly or in executor if very heavy
    bq_rows = data_to_export_rows(rows, schema=schema)

    if not bq_rows:
        logger.warning("No rows found in async save function.")
        return None

    try:
        # get_table is a network call, run in executor
        table = await loop.run_in_executor(None, bq.get_table, dataset)
    except Exception as e:
        msg = f"Unable to save rows asynchronously. Table {dataset} does not exist or there was some other problem getting the table: {e} {e.args=}"
        # Consider using a specific exception type if needed
        raise OSError(msg)

    logger.debug(f"Inserting {len(bq_rows)} rows asynchronously to BigQuery table {dataset}.")

    tasks = []
    # insert_rows is a blocking I/O call, run each chunk insertion in the executor
    for chunk in chunks(bq_rows, 100):  # Adjust chunk size as needed for BQ limits/performance
        tasks.append(loop.run_in_executor(None, bq.insert_rows, table, chunk, schema))

    results = await asyncio.gather(*tasks)
    errors = [error for sublist in results for error in sublist]  # Flatten list of lists

    if not errors:
        logger.info(
            f"Successfully pushed {len(bq_rows)} rows asynchronously to BigQuery table {dataset}.",
        )
    else:
        # Consider more specific error handling or logging details
        raise OSError(f"Google BigQuery returned an error result during async upload: {str(errors)[:1000]}")

    return dataset


def upload_rows(rows, *, schema=None, dataset=None, save_dest: SaveInfo = None, create_if_not_exists=False, **parameters) -> str:
    """Upload results to Google Bigquery asynchronously"""
    schema = schema or save_dest.bq_schema
    dataset = dataset or save_dest.dataset
    assert schema is not None and dataset is not None

    """Upload results to Google Bigquery"""
    bq = bigquery.Client()  # use application default credentials

    if isinstance(schema, str):
        schema = bq.schema_from_json(schema)

    bq_rows = data_to_export_rows(rows, schema=schema)

    if not bq_rows:
        logger.warning("No rows found in save function.")
        return None

    try:
        table = bq.get_table(dataset)
    except Exception as e:
        msg = f"Unable to save rows. Table {dataset} does not exist or there was some other problem getting the table: {e} {e.args=}"
        raise OSError(msg)

    logger.debug(f"Inserting {len(bq_rows)} rows to BigQuery table {dataset}.")

    errors = []
    for chunk in chunks(bq_rows, 100):
        errors.extend(bq.insert_rows(table, chunk, selected_fields=schema))

    if not errors:
        logger.debug(
            f"Successfully pushed {len(bq_rows)} rows to BigQuery table {dataset}.",
        )
    else:
        raise OSError(f"Google BigQuery returned an error result: {str(errors)[:1000]}")

    return dataset


@retry(
    retry=retry_if_exception_type((GoogleAPICallError, ClientError)),
    wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
    # Retry up to five times before giving up
    stop=stop_after_attempt(5),
)
def upload_binary(data=None, *, uri=None) -> str:
    assert data is not None
    gcs = storage.Client()

    logger.debug(f"Uploading file {uri}.")
    blob = google.cloud.storage.blob.Blob.from_string(uri=uri, client=gcs)

    if isinstance(data, io.BufferedIOBase):
        blob.upload_from_file(file_obj=data)
        return uri
    # Try to upload as binary from a file like object
    with io.BytesIO(data) as b:
        blob.upload_from_file(file_obj=b)
        return uri


def dump_to_disk(data, *, save_dir, extension=".json", **kwargs) -> str:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=save_dir,
        mode="w",
        suffix=extension,
    ) as out:
        if isinstance(data, pd.DataFrame):
            data.to_json(out, orient="records", lines=True)
        else:
            out.write(json.dumps(data))
        logger.warning(f"Successfully dumped to json on disk: {out.name}.")
        return out.name


def dump_pickle(data, *, save_dir, extension=".pickle", **kwargs) -> str:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=save_dir,
        mode="wb",
        suffix=extension,
    ) as out:
        pickle.dump(data, out)
        filename = out.name

    logger.warning(f"Successfully dumped to pickle on disk: {filename}.")
    return filename


def read_pickle(filename):
    with io.BytesIO() as incoming:
        incoming.write(GSPath(filename).read_bytes())
        incoming.seek(0)
        return pickle.load(incoming)


@retry(
    retry=retry_if_exception_type((GoogleAPICallError, ClientError)),
    # Wait interval:  10 seconds first, increasing exponentially up to a max of two minutes between retries
    wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
    # Retry up to five times before giving up
    stop=stop_after_attempt(5),
)
def upload_text(data, *, uri, **kwargs) -> str:
    gcs = storage.Client()

    logger.debug(f"Uploading file {uri}.")
    blob = google.cloud.storage.blob.Blob.from_string(uri, client=gcs)
    blob.upload_from_string(data)
    # TODO @nicsuzor: Do we have size info here to report?
    logger.debug(
        f"Successfully uploaded file {uri} with {len(data)} characters written (blob size {blob.size}).",
    )

    return uri


def upload_json(data, *, uri, **kwargs) -> str:
    # make sure the data is serializable first
    if isinstance(data, pd.DataFrame):
        # First, convert to serialisable formats
        rows = data.to_dict(orient="records")
    rows = scrub_serializable(data)

    # Try to upload as newline delimited json
    rows = "\n".join([json.dumps(row) for row in rows])

    return upload_text(rows, uri=uri)
