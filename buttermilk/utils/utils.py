import datetime
import json
from cloudpathlib import AnyPath, CloudPath
import pathlib
import fsspec
import requests
import yaml
import regex as re
import pandas as pd
import numpy as np
import pydantic
import uuid
import validators
import math

from google.cloud.bigquery import SchemaField

from typing import Any, List, Optional, TypeVar, Union

T = TypeVar("T")

def download_limited(url, *, allow_arbitrarily_large_downloads=False,
                     max_size: int=1024 * 1024 * 10, token=None) -> bytes:
    if token:
        # add Authorization: Bearer to the request
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(url, headers=headers, stream=True)
    else:
        r = requests.get(url, stream=True)
    if not allow_arbitrarily_large_downloads and int(r.headers.get("Content-Length", 0)) > max_size:
        return b""

    data = []
    length = 0

    for chunk in r.iter_content(1024):
        data.append(chunk)
        length += len(chunk)
        if not allow_arbitrarily_large_downloads and length > max_size:
            return b""

    return b"".join(data)

def read_file(path, auth_token: Optional[str] = None, allow_arbitrarily_large_downloads: bool = False) -> bytes:
    try:
        uri = AnyPath(path)
    except Exception as e:
        uri = path

    if isinstance(uri, CloudPath):
        return uri.read_bytes()
    elif isinstance(uri, pathlib.Path):
        with fsspec.open(path, "rb") as f:
            # For local files
            return f.read()
    elif validators.url(uri):
        return download_limited(path, token=auth_token,allow_arbitrarily_large_downloads=allow_arbitrarily_large_downloads)
    else:
        raise ValueError(f"Did not recognise {path} as valid path type or url.")



def read_yaml(filename: Union[pathlib.Path, str]) -> dict:
    file = read_file(filename)
    return yaml.load(file, Loader=yaml.FullLoader)

def read_json(filename: Union[pathlib.Path, str]) -> dict:
    file = read_file(filename)
    return json.loads(file)

def read_text(filename: Union[pathlib.Path, str]) -> str:
    file = read_file(filename)
    return file.decode()


def make_serialisable( rows):
    """Prepare dataframe for export"""
    if isinstance(rows, pd.DataFrame):
        bq_rows = rows.to_dict(orient="records")
    if isinstance(rows, pydantic.BaseModel):
        bq_rows = rows.model_dump()
    else:
        bq_rows = rows

    # Make sure objects are serializable.
    bq_rows = scrub_serializable(bq_rows)

    return bq_rows

def scrub_serializable(d) -> T:
    if isinstance(d, list):
        return [scrub_serializable(x) for x in d]

    elif isinstance(d, np.ndarray):
        # Numpy arrays: convert to list and apply same processing to each element
        return [scrub_serializable(x) for x in d.tolist()]

    elif isinstance(d, dict):
        new_val = {key: scrub_serializable(value) for key, value in d.items()}
        # remove empty values
        new_val = {k: v for k, v in new_val.items() if v is not None}
        return new_val

    elif isinstance(d, pd.DataFrame):
        return scrub_serializable(d.to_dict(orient="records"))

    else:
        # from here we know it's a single value
        if d is None or pd.isna(d):
            return None
        elif isinstance(d, np.generic):
            # This should catch all other numpy objects
            return d.item()
        elif isinstance(d, datetime.date) or isinstance(d, datetime.datetime):
            # ensure dates and datetimes are stored as strings in ISO format for uploading
            d = d.isoformat()
        elif isinstance(d, uuid.UUID):
            # if the obj is uuid, we simply return the value of uuid as a string
            d = str(d)
        elif isinstance(d, (CloudPath, pathlib.Path)):
            # convert path objects to strings
            d = str(d)

    return d


def construct_dict_from_schema(schema: list, d: dict, remove_extra=True):
    """Recursively construct a new dictionary, changing data types to match BigQuery.
    Only uses fields from d that are in schema.

    INPUT: schema - list of dictionaries, each with keys 'name', 'type', and optionally 'fields'
    """

    new_dict = {}
    keys_deleted = []

    for row in schema:
        if isinstance(row, SchemaField):
            row = row.to_api_repr()
        key_name = row["name"]
        if key_name not in d:
            keys_deleted.append(key_name)
            continue

        # Handle nested fields
        if isinstance(d[key_name], dict) and "fields" in row:
            new_dict[key_name] = construct_dict_from_schema(row["fields"], d[key_name])

        # Handle repeated fields - use the same schema as we were passed
        elif isinstance(d[key_name], list) and "fields" in row:
            new_dict[key_name] = [
                construct_dict_from_schema(row["fields"], item) for item in d[key_name]
            ]

        elif isinstance(d[key_name], str) and (
            str.upper(remove_punctuation(d[key_name])) == "NULL"
            or remove_punctuation(d[key_name]) == ""
        ):
            # don't add null values
            keys_deleted.append(key_name)
            continue

        elif str.upper(row["type"]) in ["TIMESTAMP", "DATETIME", "DATE"]:
            # convert string dates to datetimes
            if not isinstance(d[key_name], datetime.datetime):
                _ts = None
                if type(d[key_name]) == str:
                    if d[key_name].isnumeric():
                        _ts = float(d[key_name])
                    else:
                        new_dict[key_name] = pd.to_datetime(d[key_name])

                if type(d[key_name]) == int or type(d[key_name]) == float or _ts:
                    if not _ts:
                        _ts = d[key_name]

                    try:
                        new_dict[key_name] = datetime.datetime.utcfromtimestamp(_ts)
                    except (ValueError, OSError):
                        # time is likely in milliseconds
                        new_dict[key_name] = datetime.datetime.utcfromtimestamp(
                            _ts / 1000
                        )

                elif not isinstance(d[key_name], datetime.datetime):
                    new_dict[key_name] = pd.to_datetime(d[key_name])
            else:
                # Already a datetime, move it over
                new_dict[key_name] = d[key_name]

            # if it's a date only field, remove time
            if str.upper(row["type"]) == "DATE":
                new_dict[key_name] = new_dict[key_name].date()

        elif str.upper(row["type"]) in ["INTEGER", "FLOAT"]:
            # convert string numbers to integers
            if isinstance(d[key_name], str):
                new_dict[key_name] = pd.to_numeric(d[key_name])
            else:
                new_dict[key_name] = d[key_name]

        elif str.upper(row["type"]) == "BOOLEAN":
            if isinstance(d[key_name], str):
                try:
                    new_dict[key_name] = pydantic.TypeAdapter(bool).validate_python(d[key_name])
                except ValueError as e:
                    if new_dict[key_name] == "":
                        pass  # no value
                    raise e
            else:
                new_dict[key_name] = pydantic.TypeAdapter(bool).validate_python(d[key_name])

        else:
            new_dict[key_name] = d[key_name]

    return new_dict

def remove_punctuation(text):
    return re.sub(r"\p{P}+", "", text)

def dedup_columns(df):
    if any(df.columns.duplicated()):
        cols = pd.Series(df.columns)
        dup_count = cols.value_counts()
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [dup] + [
                dup + str(i) for i in range(2, dup_count[dup] + 1)
            ]

        df.columns = cols

    return df

def reset_index_and_dedup_columns(df: pd.DataFrame):
    """Reset index and rename any duplicate columns in a dataframe"""

    df = df.reset_index(allow_duplicates=True)

    return dedup_columns(df)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


def split(l, num_chunks):
    """Split a list into a number of chunks"""
    # get number of full chunks
    chunk_size = math.ceil(len(l) / num_chunks)

    # yield them successively
    yield from (l[i : i + chunk_size] for i in range(0, len(l), chunk_size))
