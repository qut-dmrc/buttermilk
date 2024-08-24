import datetime
import json
import platform
import psutil
import shortuuid
from cloudpathlib import AnyPath, CloudPath
from pathlib import Path
import fsspec
import requests
import yaml

from typing import Any, List, Optional, TypeVar, Union

def make_run_id() -> str:
    # Create a unique identifier for this run
    node_name = platform.uname().node
    username = psutil.Process().username()
    username = str.split(username, "\\")[
        -1
    ]  # get rid of windows domain if present

    # Format the current datetime as an ISO 8601 string with time zone
    #run_time = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds') + 'Z'

    # The ISO format has too many special characters for a filename, so we'll use a simpler format
    run_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%MZ")

    run_id = f"{run_time}-{shortuuid.uuid()[:4]}-{platform.uname().node}"

    return run_id

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
        with fsspec.open(path, "rb") as f:
            # For local files
            return f.read()
    except Exception:
        pass
    try:
        # this will work for local and cloud storage paths
        uri = AnyPath(path)
        return uri.read_bytes()
    except Exception:
        pass
    return download_limited(path, token=auth_token,allow_arbitrarily_large_downloads=allow_arbitrarily_large_downloads)

def read_yaml(filename: Union[Path, str]) -> dict:
    file = read_file(filename)
    return yaml.load(file, Loader=yaml.FullLoader)

def read_json(filename: Union[Path, str]) -> dict:
    file = read_file(filename)
    return json.loads(file)
