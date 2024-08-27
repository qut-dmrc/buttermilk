import pandas as pd
import shortuuid
import pandas as pd
from .utils import scrub_serializable
import json
from cloudpathlib import GSPath
import pickle
import io
from tenacity import retry, wait_exponential_jitter,  retry_if_exception_type, stop_after_attempt
import google.cloud.storage
import tempfile
from google.api_core.exceptions import AlreadyExists, ClientError, GoogleAPICallError

from google.cloud import bigquery, storage
from .utils import reset_index_and_dedup_columns
from .utils import construct_dict_from_schema, make_serialisable
from .log import logger

def save(data, **params):
    from .utils import reset_index_and_dedup_columns

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
        if "schema" in params and "dataset" in params:
            destination = upload_rows(rows=data, **params)
            logger.debug(f"Uploaded data to BigQuery: {destination}.")
            return destination
    except Exception as e:
        logger.error(
            msg=f"Critical failure. Unable to upload to data BigQuery: {str(e)}"
        )
        pass

    upload_methods = []

    if filename := params.get("filename"):
        if id := params.get("uuid"):
            filename = f"{filename}_{id}"
    else:
        filename = params.get("uuid", shortuuid.uuid())

    if save_dir :=  params.get("save_dir"):
        # Try to upload to GCS

        uri = params.get("uri", f"{save_dir}/{filename}")
        upload_methods = []

        if isinstance(data, pd.DataFrame):
            upload_methods.append(upload_dataframe_json)

        upload_methods.extend([
            upload_json,
            upload_binary,
        ])


    else:
        uri = filename

    # save to disk as a last resort
    upload_methods.extend(
        [
            dump_to_disk,
            dump_pickle,
        ]
    )

    for method in upload_methods:
        try:
            logger.debug(f"Trying to save data using {method.__name__}.")
            destination = method(data=data, uri=uri)
            logger.info(
                f"Saved data using {method.__name__} to: {destination}."
            )
            return destination
        except (GoogleAPICallError, ClientError) as e:
            logger.warning(
                f"Error saving data to {uri} using {method.__name__}: {e} {e.args=}"
            )
        except Exception as e:
            logger.warning(
                f"Could not save data using {method.__name__}: {e} {e.args=}"
            )

    raise IOError(
        f"Critical failure. Unable to save using any method in {upload_methods}"
    )
@retry(
    retry=retry_if_exception_type((GoogleAPICallError, ClientError)),
    # Wait interval:  10 seconds first, increasing exponentially up to a max of two minutes between retries
    wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
    # Retry up to five times before giving up
    stop=stop_after_attempt(5),
)
def upload_dataframe_json(data: pd.DataFrame, uri):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")


    # use pandas to upload to GCS
    if not uri[-5:] == ".json":
        uri = uri + ".json"

    if any(data.columns.duplicated()):
        data = reset_index_and_dedup_columns(data)
    if not data.empty:
        data.to_json(uri, orient="records", lines=True)

    return uri

def upload_rows(schema, rows, dataset, create_if_not_exists=False, **params):
    """Upload results to Google Bigquery"""
    bq = bigquery.Client()  # use application default credentials

    if isinstance(rows, pd.DataFrame):
        bq_rows = rows.to_dict(orient="records")
    else:
        bq_rows = rows.copy()


    # Handle other conversions required for bigquery
    bq_rows = [construct_dict_from_schema(schema, row) for row in bq_rows]

    bq_rows = make_serialisable(bq_rows)
    if not bq_rows:
        logger.warning("No rows found in save function.")
        return None

    try:
        table = bq.get_table(dataset)
    except Exception as e:
        msg=f"Unable to save rows. Table {dataset} does not exist or there was some other problem getting the table: {e} {e.args=}"
        raise IOError(msg)

    logger.debug(f"Inserting {len(bq_rows)} rows to BigQuery table {dataset}.")
    try:
        errors = bq.insert_rows(table, bq_rows)
    except:
        errors = bq.insert_rows(table, bq_rows, selected_fields=schema)

    if not errors:
        inserted = True

        logger.debug(
            f"Successfully pushed {len(bq_rows)} rows to BigQuery table {dataset}."
        )
    else:
        raise IOError(f"Google BigQuery returned an error result: {str(errors[:3])}")

    return dataset


@retry(
    retry=retry_if_exception_type((GoogleAPICallError, ClientError)),
    wait=wait_exponential_jitter(initial=1, max=60, jitter=5),
    # Retry up to five times before giving up
    stop=stop_after_attempt(5),
)
def upload_binary(*, save_dir, data=None, uri=None, extension=None):
    assert data is not None

    if uri is None:
        uri = f"{save_dir}/{shortuuid.uuid()}"
    if extension:
        uri = f"{uri}.{extension}"

    logger.debug(f"Uploading file {uri}.")
    blob = google.cloud.storage.blob.Blob.from_string(uri=uri, client=gcs)

    try:
        blob.upload_from_string(data)
        return uri
    except (Exception, TypeError) as e:
        logger.warning(
            f"Error uploading file {uri} as string: {e}. Trying with BytesIO instead."
        )
        if isinstance(data, io.BufferedIOBase):
            blob.upload_from_file(file_obj=data)
            return uri
        else:
            # Try to upload as binary from a file like object
            with io.BytesIO(data) as b:
                blob.upload_from_file(file_obj=b)
                return uri

def dump_to_disk(data, **kwargs):
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as out:
        if isinstance(data, pd.DataFrame):
            data.to_json(out)
        else:
            out.write(json.dumps(data))
        logger.warning(f"Successfully dumped to pickle on disk: {out.name}.")
        return out.name

def dump_pickle(data, **kwargs):
    filename = f"data-dumped-{shortuuid.uuid()}.pickle"
    with open(filename, "wb") as f:
        pickle.dump(data, f)

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
def upload_text(data, save_dir, uri=None, base_name=None, extension="html"):
    if not uri:
        # Create a random URI in our GCS directory
        if base_name:
            base_name = f"{shortuuid.uuid()[:6]}_{base_name}"
        else:
            base_name = f"{shortuuid.uuid()}"
        uri = f"{save_dir}/{base_name}.{extension}"

    logger.debug(f"Uploading file {uri}.")
    blob = google.cloud.storage.blob.Blob.from_string(uri, client=gcs)
    blob.upload_from_string(data)
    logger.debug(
        f"Successfully uploaded file {uri} with {len(data)} lines written."
    )

    return uri

def upload_json(data, save_dir, uri=None, id=None):
    id = id or shortuuid.uuid()
    uri = uri or f"{save_dir}/{id}.json"

    # make sure the data is serializable first
    if isinstance(data, pd.DataFrame):
        # First, convert to serialisable formats
        rows = data.to_dict(orient="records")
    rows = scrub_serializable(data)

    # Try to upload as newline delimited json
    rows = "\n".join([json.dumps(row) for row in rows])

    if uri[-5:] != ".json":
        uri = uri + ".json"

    return upload_text(rows, uri, extension="json")
