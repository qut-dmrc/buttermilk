
import datetime
from typing import Self
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    MutableMapping,
    Optional,
    Self,
    Type,
    TypeVar,
    Union,
)
import fsspec
import requests
import psutil

import os
import platform
import pydantic
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
    root_validator,)
from cloudpathlib import AnyPath, CloudPath
import shortuuid
from tempfile import mkdtemp
from ..utils import save, get_ip

_global_run_id = ''

def _make_run_id() -> str:
    global _global_run_id
    if _global_run_id:
        return _global_run_id
    # Create a unique identifier for this run
    node_name = platform.uname().node
    username = psutil.Process().username()
    # get rid of windows domain if present
    username = str.split(username, "\\")[-1]

    # The ISO 8601 format has too many special characters for a filename, so we'll use a simpler format
    run_time = datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y%m%dT%H%MZ")

    run_id = f"{run_time}-{shortuuid.uuid()[:4]}-{node_name}-{username}"
    return run_id
    
_global_run_id = _make_run_id()


class SessionInfo(pydantic.BaseModel):
    run_id: str = Field(default_factory=lambda: _make_run_id())
    ip: str = Field(default_factory=get_ip)
    node_name: str = Field(default_factory=lambda: platform.uname().node)
    username: str = Field(default_factory=lambda: psutil.Process().username().split("\\")[-1])
    save_dir: str = Field(default=None, validate_default=True)
    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, populate_by_name=True
    )

    def __str__(self):
        return _global_run_id
    
    @pydantic.field_validator("save_dir", mode="before")
    def get_save_dir(cls, save_dir) -> str:
        if not save_dir:
            save_dir = mkdtemp()
        elif isinstance(save_dir, str):
            pass
        elif isinstance(save_dir, Path):
            save_dir = save_dir.as_posix()
        elif isinstance(save_dir, (CloudPath)):
            save_dir = save_dir.as_uri()
        else:
            raise ValueError(
                f"save_path must be a string, Path, or CloudPath, got {type(save_dir)}"
            )
        return save_dir
