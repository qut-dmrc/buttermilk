import asyncio
import base64
import datetime
import itertools
import math
import mimetypes
import pathlib
import shutil
import tempfile
import threading
import uuid
from collections.abc import Mapping, Sequence
from io import IOBase
from typing import Any, TypeVar
from urllib.parse import urlparse

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
from fake_useragent import UserAgent
from omegaconf import DictConfig, ListConfig, OmegaConf
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

from .._core.log import logger

T = TypeVar("T")


def extract_url(text: str) -> str | None:
    """Find first valid URL in  string."""
    words = text.split()  # Split into words to avoid partial matches
    for word in words:
        try:
            parsed_url = urlparse(word)
            if all(
                [
                    parsed_url.scheme,
                    parsed_url.netloc,
                ],
            ):  # Check for valid scheme and netloc
                return word
        except ValueError:  # Invalid url
            pass
    return None


def is_filepath(value: Any, check_exists=True) -> bool:
    # Check if the string is a valid filepath
    try:
        x = pathlib.Path(value)
        if check_exists:
            return x.exists()
    except:
        return False
    return True


def is_uri(value: Any) -> bool:
    # Check if the string is a valid URI
    try:
        x = pydantic.AnyUrl(value)
    except:
        return False
    else:
        return not (not x.scheme or not x.host)


def is_b64(value: Any) -> bool:
    if not value:
        return False
    # Check if the string is a valid base64-encoded string
    try:
        return base64.b64encode(base64.b64decode(value)) == value.encode("utf-8")
    except:
        return False


async def run_async_newthread(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


# Initialise here to keep these the same for the whole session.
ua = UserAgent()

session_headers = {
    "User-Agent": ua.random,
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "http://www.google.com/",
}


async def download_limited_async(
    url: str | httpx.URL | pydantic.AnyUrl | AnyPath,
    *,
    allow_arbitrarily_large_downloads: bool = False,
    max_size: int = 1024 * 1024 * 10,
    token: str | None = None,
) -> tuple[bytes, str]:
    try:
        url = CloudPath(url)
        data = await run_async_newthread(url.read_bytes)
        return data, "application/octet-stream"  # Default for CloudPath
    except exceptions.InvalidPrefixError:
        # not a cloudpath url
        pass

    if not isinstance(url, httpx.URL):
        url = httpx.URL(str(url))

    async with httpx.AsyncClient() as client:
        headers = session_headers.copy()
        if token:
            headers.update({"Authorization": f"Bearer {token}"})

        r = await client.get(url, headers=headers, follow_redirects=True)

        if not allow_arbitrarily_large_downloads and int(r.headers.get("Content-Length", 0)) > max_size:
            raise OSError("File too large, download aborted")

        data = []
        length = 0

        async for chunk in r.aiter_bytes():
            data.append(chunk)
            length += len(chunk)
            if not allow_arbitrarily_large_downloads and length > max_size:
                raise OSError("File too large, download aborted")

        mimetype = r.headers.get("Content-Type")
        if not mimetype or mimetype == "application/octet-stream":
            # Try to guess the mimetype from the content
            mimetype = mimetypes.guess_type(url.path)[0] or "application/octet-stream"

        return b"".join(data), mimetype


def download_limited(
    url,
    *,
    allow_arbitrarily_large_downloads=False,
    max_size: int = 1024 * 1024 * 10,
    token=None,
    timeout=300,
) -> tuple[bytes, str]:
    if token:
        # add Authorization: Bearer to the request
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(url, headers=headers, stream=True, timeout=timeout)
    else:
        r = requests.get(url, stream=True, timeout=timeout)
    if not allow_arbitrarily_large_downloads and int(r.headers.get("Content-Length", 0)) > max_size:
        raise OSError("File too large, download aborted")

    data = []
    length = 0

    mimetype = r.headers.get("Content-Type")
    if not mimetype or mimetype == "application/octet-stream":
        # Try to guess the mimetype from the content
        mimetype = mimetypes.guess_type(url.path)[0] or "application/octet-stream"

    for chunk in r.iter_content(1024):
        data.append(chunk)
        length += len(chunk)
        if not allow_arbitrarily_large_downloads and length > max_size:
            return b""

    return b"".join(data), mimetype


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
    if isinstance(uri, pathlib.Path) and is_filepath(uri, check_exists=True):
        with fsspec.open(uri, "rb") as f:
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
    return load_json_flexi(file)


def read_text(filename: pathlib.Path | str) -> str:
    file = read_file(filename)
    return file.decode()


def scrub_keys(data: Sequence | Mapping) -> T:
    # Remove anything that looks like an key or credential from a hierarchical dict or list
    if isinstance(data, Sequence) and not isinstance(data, str):
        return [scrub_keys(v) for v in data]
    if isinstance(data, Mapping):
        return {
            k: scrub_keys(v) for k, v in data.items() if not any(x in str(k).lower() for x in ["key", "token", "password", "secret", "credential"])
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
    if isinstance(d, datetime.date) or isinstance(d, datetime.datetime) or isinstance(d, pd.Timestamp):
        # ensure dates and datetimes are stored as strings in ISO format for uploading
        d = d.isoformat()
    elif isinstance(d, uuid.UUID):
        # if the obj is uuid, we simply return the value of uuid as a string
        d = str(d)
    elif isinstance(d, (CloudPath, pathlib.Path)):
        # convert path objects to strings
        d = str(d)
    
    # Handle PIL Image objects - convert to base64 for JSON storage
    try:
        from PIL.Image import Image
        if isinstance(d, Image):
            d = image_to_base64(d)
    except ImportError:
        # PIL not available, skip image handling
        pass

    return d


def remove_punctuation(text):
    return re.sub(r"\p{P}+", "", text)


def dedup_columns(df):
    if any(df.columns.duplicated()):
        cols = pd.Series(df.columns)
        dup_count = cols.value_counts()
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [dup] + [dup + str(i) for i in range(2, dup_count[dup] + 1)]

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
    directory: str | pathlib.Path | list[pathlib.Path],
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
    elif isinstance(directory, pathlib.Path):
        dir_path = [directory]
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
    directory: str | pathlib.Path | list[pathlib.Path],
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


async def get_ip() -> str:
    # Get external IP address
    try:
        url = "https://api.ipify.org"
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                timeout=10,
            )
        ip_addr = response.content.decode("utf8")
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
    # Loop through supplied object, looking for keys that match
    # 'value'. Note that 'value' might be in dotted notation,
    # in which case we will descend each level of the 'result'
    # dictionary, consuming one level of 'value.split(".")'
    if isinstance(result, Mapping) and value in result:
        found = result[value]
        if isinstance(found, (DictConfig, ListConfig)):
            return OmegaConf.to_object(found)
        return found

    if isinstance(result, Sequence) and not isinstance(result, str):
        # Handle lists of dicts too
        found = [find_in_nested_dict(item, value=value) for item in result]
        return [x for x in found if x]

    if "." in value:
        nested = value.split(".", maxsplit=1)
        return find_in_nested_dict(result.get(nested[0], {}), value=nested[1])

    return None


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
    list_keys = {k: v for k, v in d.items() if v and isinstance(v, Sequence) and not isinstance(v, str)}
    single_keys = {k: v for k, v in d.items() if not isinstance(v, Sequence) or isinstance(v, str)}

    # Generate all combinations of list values
    combinations = list(itertools.product(*list_keys.values()))

    # Create a list of dictionaries with all combinations
    expanded_dicts = [{**single_keys, **dict(zip(list_keys.keys(), combo, strict=False))} for combo in combinations]

    # Guarantee at least a list with an empty dict
    if len(expanded_dicts) == 0:
        expanded_dicts = [d]

    return expanded_dicts


def load_json_flexi(contents):
    """Be a bit more flexible with JSON syntax."""
    try:
        import json5
    except:
        logger.warning("Install json5 for more lenient JSON parsing")

    return json5.loads(contents)


URL_PATTERN = r"https?://[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)"


def extract_url_regex(text):
    # Simple regex pattern that matches common URL formats
    match = re.search(URL_PATTERN, text)
    return match.group(0) if match else None


def get_pdf_text(file: str | IOBase) -> str | None:
    try:
        return extract_text(file, laparams=LAParams())
    except Exception as e:
        logger.error(
            f"Error extracting text from PDF {file}: {e} {e.args=}",
        )
        return None


def pydantic_to_dict(obj):  # -> dict[str, Any] | dict[Any, dict[str, Any] | dict[Any, Any...:#
    """Recursively converts Pydantic models within a structure to dictionaries."""
    if isinstance(obj, pydantic.BaseModel):
        # Convert Pydantic model to dict, handles nested models automatically
        return obj.model_dump()
    if isinstance(obj, dict):
        # Recursively process dictionary values
        return {k: pydantic_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        # Recursively process list/tuple elements
        return [pydantic_to_dict(item) for item in obj]
    # Return other types as is
    return obj


def clean_empty_values(data):
    """Recursively removes keys with empty values from a nested dictionary structure.
    Empty values are: None, empty strings, empty lists, and empty dictionaries.

    Args:
        data: Dictionary, list, or other value to clean

    Returns:
        The cleaned data structure (modifies dictionaries in-place)

    """
    if isinstance(data, dict):
        # Process each key in the dictionary
        keys_to_delete = []
        for key, value in list(data.items()):
            # Recursively clean the value
            cleaned_value = clean_empty_values(value)

            # Check if the cleaned value is empty
            if cleaned_value is None or cleaned_value == "" or (isinstance(cleaned_value, (dict, list)) and not cleaned_value):
                keys_to_delete.append(key)
            else:
                data[key] = cleaned_value

        # Remove empty keys
        for key in keys_to_delete:
            del data[key]
        return data

    if isinstance(data, list):
        # Clean each element in the list
        cleaned_list = [clean_empty_values(item) for item in data]
        # Filter out empty values
        return [item for item in cleaned_list
                if not (item is None or
                       item == "" or
                       (isinstance(item, (dict, list)) and not item))]
    # Return primitive values unchanged
    return data


# Global state for managing concurrent downloads
_download_locks: dict[str, threading.Lock] = {}
_download_locks_lock = threading.Lock()


def _get_cache_dir() -> pathlib.Path:
    """Get the cache directory for ChromaDB databases."""
    cache_dir = pathlib.Path.home() / ".cache" / "buttermilk" / "chromadb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_download_lock(persist_directory: str) -> threading.Lock:
    """Get or create a download lock for a specific persist_directory to prevent concurrent downloads."""
    with _download_locks_lock:
        if persist_directory not in _download_locks:
            _download_locks[persist_directory] = threading.Lock()
        return _download_locks[persist_directory]


async def ensure_chromadb_cache(persist_directory: str) -> pathlib.Path:
    """Ensure ChromaDB database files are available locally, downloading from remote if needed.
    
    This function handles:
    1. Checking if the ChromaDB files already exist in cache
    2. Thread-safe downloading to prevent multiple concurrent downloads
    3. Downloading the complete ChromaDB directory structure from GCS/remote storage
    
    Args:
        persist_directory: The remote path (e.g., "gs://bucket/path") or local path to ChromaDB data
        
    Returns:
        pathlib.Path: Local path to the cached ChromaDB directory
        
    Raises:
        ValueError: If the persist_directory format is invalid
        OSError: If download fails or files are corrupted
    """
    # If it's already a local path, return as-is
    try:
        local_path = pathlib.Path(persist_directory)
        if local_path.exists() and (local_path / "chroma.sqlite3").exists():
            logger.debug(f"Using existing local ChromaDB at {persist_directory}")
            return local_path
    except (OSError, ValueError):
        pass  # Not a valid local path, treat as remote
    
    # Generate cache key from persist_directory
    cache_key = persist_directory.replace("/", "_").replace(":", "_").replace(".", "_")
    cache_dir = _get_cache_dir()
    local_cache_path = cache_dir / cache_key
    
    # Check if we already have cached data
    chroma_db_path = local_cache_path / "chroma.sqlite3"
    if chroma_db_path.exists():
        logger.debug(f"Found cached ChromaDB at {local_cache_path}")
        return local_cache_path
    
    # Use thread-safe download to prevent multiple parallel downloads of the same DB
    download_lock = _get_download_lock(persist_directory)
    
    def _download_chromadb_sync():
        """Synchronous download function to be run in thread."""
        with download_lock:
            # Double-check after acquiring lock - another thread might have completed the download
            if chroma_db_path.exists():
                logger.debug(f"ChromaDB cache created by another thread at {local_cache_path}")
                return
                
            logger.info(f"Downloading ChromaDB from {persist_directory} to {local_cache_path}")
            
            try:
                # Create temporary directory for atomic download
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = pathlib.Path(temp_dir) / "chromadb_download"
                    temp_path.mkdir(parents=True, exist_ok=True)
                    
                    # Download the entire ChromaDB directory
                    remote_path = CloudPath(persist_directory)
                    
                    if not remote_path.exists():
                        # For new vector stores, the remote directory won't exist yet
                        # Create empty local directory that ChromaDB can initialize
                        logger.info(f"Remote ChromaDB directory does not exist: {persist_directory}")
                        logger.info(f"Creating new empty ChromaDB directory for initialization")
                        # temp_path already exists, so we just leave it empty for ChromaDB to initialize
                    else:
                        # Download all files recursively
                        _download_chromadb_recursive(remote_path, temp_path)
                    
                    # For new vector stores, we skip verification since ChromaDB will create the files
                    if remote_path.exists():
                        # Only verify for existing remote stores
                        temp_chroma_db = temp_path / "chroma.sqlite3"
                        if not temp_chroma_db.exists():
                            raise OSError(f"Required file chroma.sqlite3 not found in downloaded ChromaDB from {persist_directory}")
                    
                    # Atomic move to final location
                    local_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    if local_cache_path.exists():
                        shutil.rmtree(local_cache_path)
                    shutil.move(str(temp_path), str(local_cache_path))
                    
                    logger.info(f"Successfully cached ChromaDB at {local_cache_path}")
                    
            except Exception as e:
                logger.error(f"Failed to download ChromaDB from {persist_directory}: {e}")
                # Clean up any partial download
                if local_cache_path.exists():
                    shutil.rmtree(local_cache_path, ignore_errors=True)
                raise OSError(f"ChromaDB download failed: {e}") from e
    
    # Run download in thread to avoid blocking async operations
    await asyncio.to_thread(_download_chromadb_sync)
    
    return local_cache_path


def _download_chromadb_recursive(remote_path: CloudPath, local_path: pathlib.Path) -> None:
    """Recursively download ChromaDB directory structure.
    
    Args:
        remote_path: CloudPath to remote ChromaDB directory
        local_path: Local path to download to
    """
    try:
        # List all items in the remote directory
        for item in remote_path.iterdir():
            local_item_path = local_path / item.name
            
            if item.is_dir():
                # Create local directory and recurse
                local_item_path.mkdir(parents=True, exist_ok=True)
                _download_chromadb_recursive(item, local_item_path)
            else:
                # Download file
                logger.debug(f"Downloading {item} to {local_item_path}")
                local_item_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Use CloudPath's read_bytes/write_bytes for efficient transfer
                file_data = item.read_bytes()
                local_item_path.write_bytes(file_data)
                
    except Exception as e:
        logger.error(f"Error downloading from {remote_path}: {e}")
        raise


async def get_chromadb_cache_size(persist_directory: str) -> int:
    """Get the size of cached ChromaDB files in bytes.
    
    Args:
        persist_directory: The persist_directory identifier
        
    Returns:
        int: Size in bytes, or 0 if cache doesn't exist
    """
    cache_key = persist_directory.replace("/", "_").replace(":", "_").replace(".", "_")
    cache_dir = _get_cache_dir()
    local_cache_path = cache_dir / cache_key
    
    if not local_cache_path.exists():
        return 0
    
    def _calculate_size():
        total_size = 0
        for file_path in local_cache_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    return await asyncio.to_thread(_calculate_size)


async def clear_chromadb_cache(persist_directory: str | None = None) -> int:
    """Clear ChromaDB cache files.
    
    Args:
        persist_directory: Specific cache to clear, or None to clear all caches
        
    Returns:
        int: Number of bytes freed
    """
    cache_dir = _get_cache_dir()
    
    if persist_directory is None:
        # Clear all caches
        if not cache_dir.exists():
            return 0
            
        def _clear_all():
            total_freed = 0
            for cache_path in cache_dir.iterdir():
                if cache_path.is_dir():
                    for file_path in cache_path.rglob("*"):
                        if file_path.is_file():
                            total_freed += file_path.stat().st_size
                    shutil.rmtree(cache_path, ignore_errors=True)
            return total_freed
            
        return await asyncio.to_thread(_clear_all)
    else:
        # Clear specific cache
        cache_key = persist_directory.replace("/", "_").replace(":", "_").replace(".", "_")
        local_cache_path = cache_dir / cache_key
        
        if not local_cache_path.exists():
            return 0
            
        def _clear_specific():
            total_freed = 0
            for file_path in local_cache_path.rglob("*"):
                if file_path.is_file():
                    total_freed += file_path.stat().st_size
            shutil.rmtree(local_cache_path, ignore_errors=True)
            return total_freed
            
        return await asyncio.to_thread(_clear_specific)


async def upload_chromadb_cache(local_cache_path: str, persist_directory: str) -> None:
    """Upload local ChromaDB cache to remote storage.
    
    This function uploads a local ChromaDB directory to remote storage (GCS, S3, etc.)
    to persist local changes back to the shared remote storage.
    
    Args:
        local_cache_path: Path to local ChromaDB directory to upload
        persist_directory: Remote destination path (e.g., "gs://bucket/path")
        
    Raises:
        ValueError: If paths are invalid
        OSError: If upload fails
    """
    local_path = pathlib.Path(local_cache_path)
    
    if not local_path.exists() or not local_path.is_dir():
        raise ValueError(f"Local cache path does not exist or is not a directory: {local_cache_path}")
    
    # Check if it's actually a ChromaDB directory
    if not (local_path / "chroma.sqlite3").exists():
        raise ValueError(f"Local path does not appear to be a ChromaDB directory (missing chroma.sqlite3): {local_cache_path}")
    
    try:
        remote_path = CloudPath(persist_directory)
        
        def _upload_chromadb_sync():
            """Synchronous upload function to be run in thread."""
            logger.info(f"Uploading ChromaDB from {local_cache_path} to {persist_directory}")
            
            # Ensure remote directory exists
            if not remote_path.exists():
                remote_path.mkdir(parents=True, exist_ok=True)
            
            # Upload all files recursively
            _upload_chromadb_recursive(local_path, remote_path)
            
            logger.info(f"Successfully uploaded ChromaDB to {persist_directory}")
        
        # Run upload in thread to avoid blocking async operations
        await asyncio.to_thread(_upload_chromadb_sync)
        
    except Exception as e:
        logger.error(f"Failed to upload ChromaDB to {persist_directory}: {e}")
        raise OSError(f"ChromaDB upload failed: {e}") from e


def _upload_chromadb_recursive(local_path: pathlib.Path, remote_path: CloudPath) -> None:
    """Recursively upload ChromaDB directory structure.
    
    Args:
        local_path: Local path to ChromaDB directory
        remote_path: CloudPath to remote destination
    """
    try:
        # Upload all items in the local directory
        for item in local_path.iterdir():
            remote_item_path = remote_path / item.name
            
            if item.is_dir():
                # Create remote directory and recurse
                if not remote_item_path.exists():
                    remote_item_path.mkdir(parents=True, exist_ok=True)
                _upload_chromadb_recursive(item, remote_item_path)
            else:
                # Upload file
                logger.debug(f"Uploading {item} to {remote_item_path}")
                
                # Use CloudPath's read_bytes/write_bytes for efficient transfer
                file_data = item.read_bytes()
                remote_item_path.write_bytes(file_data)
                
    except Exception as e:
        logger.error(f"Error uploading to {remote_path}: {e}")
        raise


# Image utility functions (replaces MediaObj functionality)
def image_to_base64(image, format: str = "PNG", longest_edge: int = -1, shortest_edge: int = -1) -> str:
    """Convert PIL Image to base64 string with optional resizing.
    
    Args:
        image: PIL Image object
        format: Image format for encoding (PNG, JPEG, etc.)
        longest_edge: Resize to this max size on longest edge
        shortest_edge: Resize to this max size on shortest edge
        
    Returns:
        Base64 encoded string
    """
    from io import BytesIO
    from PIL import Image
    
    # Resize if requested
    if longest_edge > 0:
        if image.width > longest_edge or image.height > longest_edge:
            if image.width > image.height:
                new_width = longest_edge
                new_height = int(longest_edge * image.height / image.width)
            else:
                new_height = longest_edge
                new_width = int(longest_edge * image.width / image.height)
            image = image.resize((new_width, new_height))
    elif shortest_edge > 0:
        if image.width > shortest_edge or image.height > shortest_edge:
            if image.width > image.height:
                new_height = shortest_edge
                new_width = int(shortest_edge * image.width / image.height)
            else:
                new_width = shortest_edge
                new_height = int(shortest_edge * image.height / image.width)
            image = image.resize((new_width, new_height))
    
    # Convert to base64
    buffered = BytesIO()
    image.save(buffered, format=format)
    image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return image_b64


def image_to_content_part(image, model_type: str = "openai", mime_type: str = "image/png") -> dict[str, Any]:
    """Convert PIL Image to LLM-specific content part format.
    
    Args:
        image: PIL Image object
        model_type: Target LLM provider ("openai" or "anthropic")
        mime_type: MIME type for the image
        
    Returns:
        Dictionary in the appropriate format for the LLM provider
    """
    b64_data = image_to_base64(image)
    
    if model_type == "openai":
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
        }
    elif model_type == "anthropic":
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": b64_data,
            },
        }
    else:
        # Fallback format
        return {"type": "image", "data": b64_data, "mime_type": mime_type}


def base64_to_image(b64_string: str):
    """Convert base64 string to PIL Image.
    
    Args:
        b64_string: Base64 encoded image data
        
    Returns:
        PIL Image object
    """
    from io import BytesIO
    from PIL import Image
    
    image_data = base64.b64decode(b64_string)
    return Image.open(BytesIO(image_data))
