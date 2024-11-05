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
from buttermilk._core.runner_types import AgentInfo, Job, RecordInfo, Result, AgentInfo
from buttermilk.defaults import BQ_SCHEMA_DIR
from buttermilk.utils.errors import extract_error_info
from buttermilk.utils.save import upload_rows, save
from buttermilk._core.log import logger

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

class SaveInfo(BaseModel):
    destination: str|cloudpathlib.AnyPath
    db_schema: Optional[str] = Field(..., validation_alias='schema')
    type: Literal['bq', 'gcs']

    @field_validator("db_schema")
    def file_must_exist(cls, v):
        if not os.path.exists(v):
            f = Path(BQ_SCHEMA_DIR) / v
            if f.exists():
                return f.as_posix()
            
            raise ValueError(f"File '{v}' does not exist.")
        return v

class Agent(BaseModel):
    """
    Receive data, processes it, save the results to BQ, and acknowledge completion.
    """
    flow: str
    name: str
    concurrent: int = 4            # Max number of async tasks to run
    agent_info: Optional[AgentInfo] = None  # The metadata for this run
    
    save_params: Optional[SaveInfo] = None

    _sem: asyncio.Semaphore = PrivateAttr()  # Semaphore for limiting concurrent tasks

    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)
    
    @model_validator(mode='after')
    def init(self) -> Self:
        # configure agent info
        self.agent_info = AgentInfo(agent=self.name, **self.model_extra, **self.model_dump(exclude_unset=True, mode='json',exclude_none=True, exclude=["name"]))
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
    def validate_concurrent(self) -> Self:
        if self.concurrent < 1:
            raise ValueError("concurrent must be at least 1")
        self._sem = asyncio.Semaphore(value=self.concurrent)
        return self

    @trace
    async def run(self, job: Job) -> Job:
        async with self._sem:
            try:
                job.agent_info = self.agent_info
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
    
