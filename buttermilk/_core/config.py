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

    class Config:
        # Exclude fields with None values when serializing
        exclude_none = True
        arbitrary_types_allowed=True
        populate_by_name=True
        # Ignore extra fields not defined in the model
        extra = "allow"
        exclude_none=True
        exclude_unset=True
        include_extra=True
    

class SaveInfo(CloudProviderCfg):
    destination: Optional[str|cloudpathlib.AnyPath] = None
    db_schema: Optional[str] = Field(..., validation_alias='schema')
    dataset: Optional[str] = Field(default=None)

    @field_validator("db_schema")
    def file_must_exist(cls, v):
        if not os.path.exists(v):
            f = Path(BQ_SCHEMA_DIR) / v
            if f.exists():
                return f.as_posix()
            
            raise ValueError(f"File '{v}' does not exist.")
        return v
    
    @model_validator(mode='after')
    def check_destination(self) -> Self:
        if not self.destination and not self.dataset:
            raise ValueError("Nowhere to save to! Either destination or dataset must be provided.")
        return self

class AgentInfo(BaseModel):
    type: str

    # Optional because it will be added when the agent is instantiated.
    flow: Optional[str] = Field(default=None, validation_alias=AliasChoices("flow", "name"))

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True
        populate_by_name = True
        exclude_none = True
        exclude_unset = True
        

class DataSource(BaseModel):
    name: str
    max_records_per_group: int = -1
    type: Literal["job", "file", "bq", "generator", "plaintext"]
    path: str = Field(..., validation_alias=AliasChoices("path", "dataset", "uri", "func"))
    glob: str = Field(default="**/*")
    filter: Optional[Mapping[str, str|Sequence[str]|None]] = Field(default_factory=dict)
    join: Optional[Mapping[str, str]] = Field(default_factory=dict)
    agg: Optional[bool] = Field(default=False)
    group: Optional[Mapping[str, str]] = Field(default_factory=dict)
    columns: Optional[Mapping[str, str|Mapping]] = Field(default_factory=dict)
    last_n_days: int = Field(default=7)

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = False
        populate_by_name = True
        exclude_none = True
        exclude_unset = True
        
class Flow(BaseModel):
    name: str
    num_runs: int = 1
    concurrency: int = 1
    agent: AgentInfo
    save: SaveInfo
    data: Optional[Sequence[DataSource]] = Field(default_factory=list)
    parameters: Optional[Mapping] = Field(default_factory=dict)

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = False
        populate_by_name = True
        exclude_none = True
        exclude_unset = True

    @field_validator("data", mode="before")
    def convert_data(cls, value):
        datasources = []
        for source in value:
            if not isinstance(source, DataSource):
                source = DataSource(**source)
            datasources.append(source)
        return datasources
    

class Tracing(BaseModel):
    enabled: bool = False
    endpoint: Optional[str] = None
    otlp_headers: Optional[Mapping] = Field(default_factory=dict)

class RunCfg(BaseModel):
    platform: str = 'local'
    max_concurrency: int = -1
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
    run: RunCfg

    class Config:
        extra="forbid"
        arbitrary_types_allowed=True
        populate_by_name=True
        exclude_none=True
        exclude_unset=True



# @hydra.main(version_base="1.3", config_path="../conf", config_name="config")
# def main(cfg: DictConfig) -> None:
#     validated_config = instantiate(Config, cfg)
#     print(validated_config)

if __name__ == '__main__':
    Config = builds(Project, populate_full_signature=True)
    pass
