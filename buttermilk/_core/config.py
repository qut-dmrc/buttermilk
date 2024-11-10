import os
from pathlib import Path
from typing import Any, AsyncGenerator, Generator, Literal, Optional, Self, Sequence, Tuple, Type, TypeVar, Union,Mapping
from cloudpathlib import CloudPath, GSPath
import cloudpathlib
from pydantic import (
    AliasChoices,
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    TypeAdapter,
    field_validator,
    model_validator,
)

from hydra_zen import instantiate, builds

from buttermilk.defaults import BQ_SCHEMA_DIR

from abc import abstractmethod
import asyncio
import datetime
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import cloudpathlib
from fastapi import BackgroundTasks
import hydra
import pandas as pd
import regex as re
import shortuuid
from humanfriendly import format_timespan
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator
from promptflow.tracing import trace

from buttermilk.defaults import BQ_SCHEMA_DIR
from buttermilk.utils.errors import extract_error_info
from buttermilk.utils.save import upload_rows, save
from .log import logger

BASE_DIR = Path(__file__).absolute().parent
import datetime
import itertools
from itertools import cycle
from tempfile import NamedTemporaryFile
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Literal,
    Mapping,
    Optional,
    Self,
    Sequence,
    Type,
    TypeVar,
)

import cloudpathlib


# from functools import wraps
# from typing import Any, Callable
# from pydantic import BaseModel

# async def run_flow(job: Job, flow: callable):
#     response = await flow.call_async(**job.inputs)
#     job.outputs = Result(**response)
#     return job

# def flow():
#     def inner(func):
#         @wraps(func)
#         def _impl(job: Job) -> Job:
#             return run_flow(job, func)
#         return _impl


CloudProvider = Literal["gcp", "bq", "aws", "azure", "hashicorp", "env", "vault", "local"]

class CloudProviderCfg(BaseModel):
    type: CloudProvider

    model_config = ConfigDict(
        extra="allow", 
        arbitrary_types_allowed=True, 
        populate_by_name=True, 
        exclude_none=True, 
        exclude_unset=True, 
        include_extra=True
    )

class SaveInfo(CloudProviderCfg):
    destination: str|cloudpathlib.AnyPath
    db_schema: Optional[str] = Field(..., validation_alias='schema')

    @field_validator("db_schema")
    def file_must_exist(cls, v):
        if not os.path.exists(v):
            f = Path(BQ_SCHEMA_DIR) / v
            if f.exists():
                return f.as_posix()
            
            raise ValueError(f"File '{v}' does not exist.")
        return v

class AgentInfo(BaseModel):
    name: str
    save_params: Optional[SaveInfo] = None

    model_config = ConfigDict(
        extra="allow", arbitrary_types_allowed=True, populate_by_name=True, exclude_none=True, exclude_unset=True,
    )


class DataSource(BaseModel):
    name: str
    max_records_per_group: int = -1
    type: Literal["job", "file", "bq", "generator", "plaintext"]
    path: str = Field(..., validation_alias=AliasChoices("path", "dataset", "uri", "func"))
    filter: Optional[Mapping[str, str]] = Field(default_factory=dict)
    join: Optional[Mapping[str, str]] = Field(default_factory=dict)
    agg: Optional[bool] = Field(default=False)
    group: Optional[Mapping[str, str]] = Field(default_factory=dict)
    columns: Optional[Mapping[str, str]] = Field(default_factory=dict)

class Flow(BaseModel):
    name: str
    num_runs: int = 1
    concurrency: int = 1
    agent: AgentInfo
    data: Optional[Sequence[Any]] = Field(default_factory=list)
    parameters: Optional[Mapping] = Field(default_factory=dict)

class Tracing(BaseModel):
    enabled: bool = False
    endpoint: Optional[str] = None
    otlp_headers: Optional[Mapping] = Field(default_factory=dict)

class RunCfg(BaseModel):
    platform: str = 'local'
    parameters: Mapping[str, Any] = Field(default_factory=dict)

class Project(BaseModel):
    name: str
    job: str
    connections: Sequence[str] = Field(default_factory=list)
    secret_provider: CloudProviderCfg
    save_dest: CloudProviderCfg
    logger: CloudProviderCfg
    flows: list[Flow] = Field(default_factory=list)
    tracing: Optional[Tracing] = Field(default_factory=Tracing)
    verbose: bool = True
    cloud: list[CloudProviderCfg] = Field(default_factory=list)
    run: RunCfg

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, populate_by_name=True,          exclude_none=True, exclude_unset=True,)



# @hydra.main(version_base="1.3", config_path="../conf", config_name="config")
# def main(cfg: DictConfig) -> None:
#     validated_config = instantiate(Config, cfg)
#     print(validated_config)

if __name__ == '__main__':
    Config = builds(Project, populate_full_signature=True)
    Config
    pass