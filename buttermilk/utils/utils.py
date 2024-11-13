import asyncio
import datetime
import itertools
import json
import math
import pathlib
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

import fsspec
import httpx
import numpy as np
import pandas as pd
import pydantic
import regex as re
import requests
import validators
import yaml
from cloudpathlib import AnyPath, CloudPath, exceptions

from .._core.log import logger

T = TypeVar("T")


async def run_async_newthread(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


async def download_limited_async(
    url,
    *,
    allow_arbitrarily_large_downloads=False,
    max_size: int = 1024 * 1024 * 10,
    token: str | None = None,
) -> tuple[bytes, str]:
    headers = {"Authorization": f"Bearer {token}"} if token else None

    try:
        url = CloudPath(url)
        data = await run_async_newthread(url.read_bytes)
        return data, "application/octet-stream"  # Default for CloudPath
    except exceptions.InvalidPrefixError:
        # not a cloudpath url
        pass

    async with httpx.AsyncClient() as client:
        r = await client.get(url, headers=headers)

        if (
            not allow_arbitrarily_large_downloads
            and int(r.headers.get("Content-Length", 0)) > max_size
        ):
            return b""

        data = []
        length = 0

        async for chunk in r.aiter_bytes(1024):
            data.append(chunk)
            length += len(chunk)
            if not allow_arbitrarily_large_downloads and length > max_size:
                raise OSError("File too large, download aborted")

        mimetype = r.headers.get("Content-Type", "application/octet-stream")

        return b"".join(data), mimetype


def download_limited(
    url,
    *,
    allow_arbitrarily_large_downloads=False,
    max_size: int = 1024 * 1024 * 10,
    token=None,
) -> bytes:
    if token:
        # add Authorization: Bearer to the request
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(url, headers=headers, stream=True)
    else:
        r = requests.get(url, stream=True)
    if (
        not allow_arbitrarily_large_downloads
        and int(r.headers.get("Content-Length", 0)) > max_size
    ):
        return b""

    data = []
    length = 0

    for chunk in r.iter_content(1024):
        data.append(chunk)
        length += len(chunk)
        if not allow_arbitrarily_large_downloads and length > max_size:
            return b""

    return b"".join(data)


def read_file(
    path,
    auth_token: str | None = None,
    allow_arbitrarily_large_downloads: bool = False,
) -> bytes:
    try:
        uri = AnyPath(path)
    except Exception:
        uri = path

    if isinstance(uri, CloudPath):
        return uri.read_bytes()
    if isinstance(uri, pathlib.Path):
        with fsspec.open(path, "rb") as f:
            # For local files
            return f.read()
    elif validators.url(uri):
        return download_limited(
            path,
            token=auth_token,
            allow_arbitrarily_large_downloads=allow_arbitrarily_large_downloads,
        )
    else:
        raise ValueError(f"Did not recognise {path} as valid path type or url.")


def read_yaml(filename: pathlib.Path | str) -> dict:
    file = read_file(filename)
    return yaml.load(file, Loader=yaml.FullLoader)


def read_json(filename: pathlib.Path | str) -> dict:
    file = read_file(filename)
    return json.loads(file)


def read_text(filename: pathlib.Path | str) -> str:
    file = read_file(filename)
    return file.decode()


def scrub_keys(data: Sequence | Mapping) -> T:
    # Remove anything that looks like an key or credential from a hierarchical dict or list
    if isinstance(data, Sequence) and not isinstance(data, str):
        return [scrub_keys(v) for v in data]
    if isinstance(data, Mapping):
        return {
            k: scrub_keys(v)
            for k, v in data.items()
            if not any(
                x in str(k).lower()
                for x in ["key", "token", "password", "secret", "credential"]
            )
        }
    return data


def make_serialisable(rows):
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

    if isinstance(d, np.ndarray):
        # Numpy arrays: convert to list and apply same processing to each element
        return [scrub_serializable(x) for x in d.tolist()]

    if isinstance(d, dict):
        new_val = {key: scrub_serializable(value) for key, value in d.items()}
        # remove empty values
        new_val = {k: v for k, v in new_val.items() if v is not None}
        return new_val

    if isinstance(d, pd.DataFrame):
        return scrub_serializable(d.to_dict(orient="records"))

    # from here we know it's a single value
    if d is None or pd.isna(d):
        return None
    if isinstance(d, np.generic):
        # This should catch all other numpy objects
        return d.item()
    if (
        isinstance(d, datetime.date)
        or isinstance(d, datetime.datetime)
        or isinstance(d, pd.Timestamp)
    ):
        # ensure dates and datetimes are stored as strings in ISO format for uploading
        d = d.isoformat()
    elif isinstance(d, uuid.UUID):
        # if the obj is uuid, we simply return the value of uuid as a string
        d = str(d)
    elif isinstance(d, (CloudPath, pathlib.Path)):
        # convert path objects to strings
        d = str(d)

    return d


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


def list_files(
    directory: str | list[pathlib.Path],
    filename: str = "",
    extension: str = "",
    parent: str = "",
) -> list[tuple[str, str]]:
    """Return a list of matching files in the given directory.

    Args:
    directory (str): The path to the directory to search.
    pattern (str): The regex pattern to match filenames against.

    Raises:
    ValueError: If the provided directory does not exist.

    """
    if isinstance(directory, str):
        dir_path = [pathlib.Path(directory)]
    else:
        dir_path = [pathlib.Path(d) if isinstance(d, str) else d for d in directory]

    for dir in dir_path:
        if not dir.is_dir():
            raise ValueError(
                f"The provided path '{directory}' is not a valid directory.",
            )

        extension = extension or ".*"
        if filename:
            pattern = f"*{parent}*/*{filename}*{extension}"
        else:
            pattern = f"*{parent}*/*{extension}"

        for file_path in dir.rglob(pattern):
            if file_path.is_file():  # and re.search(pattern, file_path.name):
                yield file_path


def list_files_with_content(
    directory: str | list[pathlib.Path],
    filename: str = "",
    extension: str = "",
    parent: str = "",
) -> list[tuple[str, str]]:
    """Return a list of tuples containing (filenames matching a pattern, file content)
    for all files in the given directory.

    Returns:
    List[Tuple[str, str]]: A list of tuples, each containing a matching filename and its content.

    """
    result = []
    for file_path in list_files(directory, filename, extension, parent):
        try:
            with open(file_path, encoding="utf-8") as file:
                content = file.read()
            result.append((file_path.name, content))
        except OSError as e:
            logger.warning(f"Could not read file {file_path}: {e!s}")

    return result


def split(l, num_chunks):
    """Split a list into a number of chunks"""
    # get number of full chunks
    chunk_size = math.ceil(len(l) / num_chunks)

    # yield them successively
    yield from (l[i : i + chunk_size] for i in range(0, len(l), chunk_size))


def get_ip() -> str:
    # Get external IP address
    try:
        ip_addr = requests.get("https://api.ipify.org").content.decode("utf8")
        return ip_addr
    except Exception:
        logger.error("Unable to get host IP address from external source.")
        return None


def find_key_string_pairs(data):
    if isinstance(data, str):
        return
    if isinstance(data, Mapping):
        for key, value in data.items():
            if isinstance(value, str):
                yield (key, value)
            elif isinstance(value, (Mapping, Sequence)):
                yield from find_key_string_pairs(value)
    elif isinstance(data, Sequence):
        for item in data:
            yield from find_key_string_pairs(item)


def find_in_nested_dict(result: dict, value: str) -> object:
    nested = value.split(".")
    for n in nested:
        result = result.get(n, {})
    if result:
        return result
    raise ValueError(f"Unable to find key {value} in dict.")


def find_all_keys_in_dict(x, search_key):
    results = []

    if isinstance(x, dict):
        if x.get(search_key):
            if isinstance(x[search_key], list):
                results.extend(x[search_key])
            else:
                results.append(x[search_key])
        else:
            for k in x:
                result = find_all_keys_in_dict(x[k], search_key)
                if result:
                    results.extend(result)
    elif isinstance(x, list):
        for y in x:
            result = find_all_keys_in_dict(y, search_key)
            if result:
                results.extend(result)

    return results


def expand_dict(d: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not d:
        return [{}]

    # Separate keys with list values and keys with single values
    list_keys = {
        k: v
        for k, v in d.items()
        if v and isinstance(v, Sequence) and not isinstance(v, str)
    }
    single_keys = {
        k: v for k, v in d.items() if not isinstance(v, Sequence) or isinstance(v, str)
    }

    # Generate all combinations of list values
    combinations = list(itertools.product(*list_keys.values()))

    # Create a list of dictionaries with all combinations
    expanded_dicts = [
        {**single_keys, **dict(zip(list_keys.keys(), combo, strict=False))}
        for combo in combinations
    ]

    # Guarantee at least a list with an empty dict
    if len(expanded_dicts) == 0:
        expanded_dicts = [d]

    return expanded_dicts
