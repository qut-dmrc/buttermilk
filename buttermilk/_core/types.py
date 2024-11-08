
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

from ..utils import save, get_ip

_global_run_id = None

class SessionInfo(pydantic.BaseModel):
    project: str
    job: str
    run_id: str = Field(default_factory=lambda: SessionInfo.make_run_id())
    save_bucket: Optional[str] = ''
    save_dir: Optional[str] = ''

    ip: str = Field(default_factory=get_ip)
    node_name: str = Field(default_factory=lambda: platform.uname().node)
    username: str = Field(default_factory=lambda: psutil.Process().username().split("\\")[-1])
    save_dir: str

    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, populate_by_name=True
    )

    @property
    def __str__(self) -> str:
        return self.run_id

    @pydantic.model_validator(mode="after")
    def get_save_dir(self) -> Self:
        if self.save_dir:
            if isinstance(self.save_dir, str):
                save_dir = self.save_dir
            elif isinstance(self.save_dir, Path):
                save_dir = self.save_dir.as_posix()
            elif isinstance(self.save_dir, (CloudPath)):
                save_dir = self.save_dir.as_uri()
            else:
                raise ValueError(
                    f"save_path must be a string, Path, or CloudPath, got {type(self.save_dir)}"
                )
        else:
            # Get the save directory from the configuration or use a default
            save_dir = (
                f"gs://{self.save_bucket}/runs/{self.project}/{self.job}/{self.run_id}"
            )
            del self.save_bucket
        # # Make sure the save directory is a valid path
        try:
            _ = AnyPath(save_dir)
        except Exception as e:
            raise ValueError(f"Invalid cloud save directory: {save_dir}. Error: {e}")
        self.save_dir = save_dir

        return self

    @classmethod
    def make_run_id(cls) -> str:
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
        _global_run_id = run_id
        return run_id