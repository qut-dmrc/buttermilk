from abc import abstractmethod
import asyncio
import datetime
import json
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

from buttermilk.runner._runner_types import AgentInfo, Job, RecordInfo, Result, AgentInfo
from buttermilk.utils.save import upload_rows

BASE_DIR = Path(__file__).absolute().parent
import datetime
import itertools
from itertools import cycle
from tempfile import NamedTemporaryFile
from typing import (
    Any,
    AsyncGenerator,
    Callable,
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
class Agent(BaseModel):
    """
    Receive data, processes it, save the results to BQ, and acknowledge completion.
    """
    flow_obj: Type = None
    client: object = None
    agent: str      # The name of this process that is used to get the result
    concurrent: int = 10     # Max number of async tasks to run
    agent_info: Optional[AgentInfo] = None  # The metadata for this run

    save_dest: Optional[str] = None
    save_schema: Optional[str] = None

    _sem: asyncio.Semaphore = PrivateAttr()  # Semaphore for limiting concurrent tasks
    init_vars: dict = Field(default_factory=dict)  # Store the original kwargs
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)


    @model_validator(mode='before')
    def preprocess_data(cls, values):
        # Store the original kwargs in a private attribute
        values['init_vars'] = { k: values.pop(k) for k in values.copy().keys() if k not in cls.model_fields.keys() }

        values['agent_info'] = AgentInfo(agent=values['agent'], **values['init_vars'])

        # get save info if it's there
        if save_cfg := values.get('save', None):
            values['_save_dest'] = cloudpathlib.CloudPath(save_cfg.destination)
        return values
    
    @model_validator(mode='after')
    def init(self) -> Self:
        if self.client is None:
            self.client = self.flow_obj(**self.init_vars)
            del self.flow_obj  # Don't keep the class around
        return self
    
    @field_validator("agent", mode="before")
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

    
    def enqueue(self, job: Job, background_tasks: BackgroundTasks)->bool:
        background_tasks.add_task(self._process_job_with_semaphore, job)
        return True

    async def _process_job_with_semaphore(self, job: Job) -> Job:
        async with self._sem:
            result = await self.process_job(job)
            if self.save_dest:
                rows = [result.model_dump()]
                upload_rows(rows=rows, dataset=self.save_dest, schema=self.save_schema)
            return result
        
    @abstractmethod
    async def _process(self, *, job: Job) -> Job:
        """ Take a Job with Inputs, process it, and 
        return a Job with Inputs and Outputs."""
        raise NotImplementedError()
    