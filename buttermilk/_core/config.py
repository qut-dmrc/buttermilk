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

from .runner_types import AgentInfo, Job, RecordInfo, Result, AgentInfo
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


CloudProvider = TypeVar(Literal["gcp", "bq", "aws", "azure", "hashicorp", "env", "vault", "local"])

class CloudProviderCfg(BaseModel):
    type: CloudProvider
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True, populate_by_name=True, exclude_none=True, exclude_unset=True, include_extra=True)


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


#########
# Agent
#
# A simple class with a function that process a job.
# 
# It takes a Job with Inputs and returns a Job with Inputs and Outputs.
# The completed Job is stored in a database (BigQuery) for tracing and analysis.
#
# The primary type of Job is a "flow" which is a sequence of steps that process data
# using a model or client of some sort. In the standard implementation, this is a 
# langchain based template processed by an interchangeable LLM Chat model.
#
##########


class Agent(BaseModel):
    """
    Receive data, processes it, save the results to BQ, and acknowledge completion.
    """
    name: str
    concurrency: Optional[int] = 4            # Max number of async tasks to run
    _agent_info: Optional[AgentInfo] = None  # The metadata for this run
    

    _sem: asyncio.Semaphore = PrivateAttr()  # Semaphore for limiting concurrent tasks

    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)
    
    @model_validator(mode='after')
    def init(self) -> Self:
        # configure agent info
        params = self.model_dump(exclude_unset=True, mode='json',exclude_none=True)
        if self.model_extra:
            params.update(self.model_extra)
        self._agent_info = AgentInfo(**params)
        return self
    
    @field_validator("save_params", mode="before")
    def validate_save_params(cls, value: Optional[SaveInfo|Mapping]) -> SaveInfo:
        if not isinstance(value, SaveInfo):
            return SaveInfo(**value)
        return value
    
    @field_validator("name", mode="before")
    def validate_agent(cls, value: Optional[str|int]) -> str:
        # Make a unique worker name for identification and logging
        value =  f"{value}_{shortuuid.uuid()[:6]}"
        return value
    
    @model_validator(mode="after")
    def validate_concurrency(self) -> Self:
        if self.concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self._sem = asyncio.Semaphore(value=self.concurrency)
        return self

    @trace
    async def run(self, job: Job) -> Job:
        async with self._sem:
            try:
                job.agent_info = self._agent_info
                job = await self.process_job(job=job)
            except Exception as e:
                job.error = extract_error_info(e=e)
                if job.record:
                    logger.error(
                        f"Error processing task {self.name} by {self.name} with job {job.job_id} and record {job.record.record_id}. Error: {e or type(e)} {e.args=}"
                    )
            finally: 
                if self.save_params:
                    rows = [job.model_dump()]
                    if self.save_params.type == 'bq':
                        upload_rows(rows=rows, dataset=self.save_params.destination, schema=self.save_params.db_schema)
                    else:
                        save(save_dir=self.save_params.destination)
            return job
    
    async def process_job(self, *, job: Job) -> Job:
        """ Take a Job with Inputs, process it, and 
        return a Job with Inputs and Outputs."""
        raise NotImplementedError()
    


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
    agent: "Agent"
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
    logger: CloudProvider
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