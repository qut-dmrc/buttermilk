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


from buttermilk._core.config import AgentInfo, SaveInfo
from buttermilk._core.runner_types import Job
from buttermilk.defaults import BQ_SCHEMA_DIR

from abc import abstractmethod
import asyncio
import os
from pathlib import Path

import cloudpathlib
import shortuuid
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator
from promptflow.tracing import trace

from buttermilk.defaults import BQ_SCHEMA_DIR
from buttermilk.utils.errors import extract_error_info
from buttermilk.utils.save import upload_rows, save
from .log import logger

import datetime

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
    name: str
    type: str
    concurrency: int = Field(default=4)            # Max number of async tasks to run
    save: SaveInfo                                 # Where to save the results
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
    
    @field_validator("save", mode="before")
    def validate_save_params(cls, value: Optional[SaveInfo|Mapping]) -> Optional[SaveInfo]:
        if value is None or isinstance(value, SaveInfo):
            return value
        return SaveInfo(**value)
    
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
                if self.save:
                    rows = [job.model_dump()]
                    if self.save.type == 'bq':
                        save(data=rows, dataset=self.save.dataset, schema=self.save.db_schema, save_dir=self.save.destination)
                    else:
                        save(data=rows, save_dir=self.save.destination)
            return job
    
    async def process_job(self, *, job: Job) -> Job:
        """ Take a Job with Inputs, process it, and 
        return a Job with Inputs and Outputs."""
        raise NotImplementedError()
    