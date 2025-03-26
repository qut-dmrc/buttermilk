
import asyncio
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
    run_time = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%MZ")

    run_id = f"{run_time}-{shortuuid.uuid()[:4]}-{node_name}-{username}"
    return run_id


_global_run_id = _make_run_id()


class SessionInfo(pydantic.BaseModel):
    platform: str = "local"
    name: str
    job: str
    run_id: str = Field(default=_global_run_id)
    max_concurrency: int = -1
    ip: str = Field(default="")
    node_name: str = Field(default_factory=lambda: platform.uname().node)
    username: str = Field(
        default_factory=lambda: psutil.Process().username().split("\\")[-1],
    )
    save_dir: str | None = None
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )
    save_dir_base: str = Field(
        default_factory=mkdtemp,
        validate_default=True,
    )  # Default to temp dir

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=False)

    def __str__(self):
        return _global_run_id

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
        self.save_dir = str(save_dir)
        return self

    @pydantic.model_validator(mode="after")
    def schedule_get_ip(self) -> Self:
        asyncio.get_event_loop().create_task(self.get_ip())
        return self

    async def get_ip(self):
        if not self.ip:
            self.ip = await get_ip()
