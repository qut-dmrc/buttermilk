
import datetime
import platform
from pathlib import Path
from tempfile import mkdtemp
from typing import Self

import psutil
import pydantic
import shortuuid
from cloudpathlib import AnyPath, CloudPath
from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
)

from ..utils import get_ip

_global_run_id = ""


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
    run_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%MZ")

    run_id = f"{run_time}-{shortuuid.uuid()[:4]}-{node_name}-{username}"
    return run_id


_global_run_id = _make_run_id()


class SessionInfo(pydantic.BaseModel):
    name: str
    job: str
    run_id: str = Field(default=_global_run_id)
    ip: str = Field(default_factory=get_ip)
    node_name: str = Field(default_factory=lambda: platform.uname().node)
    username: str = Field(
        default_factory=lambda: psutil.Process().username().split("\\")[-1],
    )
    save_dir_base: str 
    _save_dir: str = PrivateAttr()
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    def __str__(self):
        return _global_run_id

    @property
    def save_dir(self):
        return self._save_dir

    @pydantic.field_validator("save_dir_base", mode="before")
    def get_save_dir(cls, save_dir_base, values) -> str:
        if isinstance(save_dir_base, str):
            pass
        elif isinstance(save_dir_base, Path):
            save_dir_base = save_dir_base.as_posix()
        elif isinstance(save_dir_base, CloudPath):
            save_dir_base = save_dir_base.as_uri()
        else:
            raise ValueError(
                f"save_dir_base must be a string, Path, or CloudPath, got {type(save_dir_base)}",
            )
        return save_dir_base
    

    @pydantic.model_validator(mode="after")
    def set_full_save_dir(self) -> Self:
        save_dir = AnyPath(self.save_dir_base) / self.name / self.job / self.run_id
        self._save_dir = save_dir.as_uri()
        return self