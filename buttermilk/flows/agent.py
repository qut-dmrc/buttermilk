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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from buttermilk import BM
from buttermilk.libs import (
    HFInferenceClient,
    Llama2ChatMod,
    hf_pipeline,
    replicatellama2,
    replicatellama3,
)
from buttermilk.exceptions import FatalError
from buttermilk.lc import LC
from buttermilk.runner._runner_types import Job, RecordInfo, Result, RunInfo, StepInfo
from buttermilk.runner.flow import ResultsSaver, run_flow, LC
from buttermilk.runner.helpers import group_and_filter_jobs, load_data
from buttermilk.runner.runner import Consumer, ResultsCollector, TaskDistributor
from buttermilk.tools.metrics import Metriciser, Scorer
from buttermilk.utils import (
    col_mapping_hydra_to_local,
    find_key_string_pairs,
    make_serialisable,
)

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
import datasets
import torch
import tqdm

from buttermilk.utils.flows import col_mapping_hydra_to_pf
from buttermilk.utils.log import logger

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

    agent: str      # The name of this process that is used to get the result
    run_info: RunInfo
    init_vars: dict = {}    # Vars to use when initialising the client
    concurrent: int = 1     # Max number of async tasks to run
    cfg: DictConfig         # The configuration for this agent

    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)

    @field_validator("agent", mode="before")
    def validate_agent(cls, value: Optional[str|int]) -> str:
        return str(value)
    
    @model_validator(mode="after")
    def validate_concurrent(self) -> Self:
        if self.concurrent < 1:
            raise ValueError("concurrent must be at least 1")

        # Make a unique worker name for identification and logging
        self.agent =  "_".join([x for x in [self.step_name, self.agent, shortuuid.uuid()[:6]] if x])

        self._sem = asyncio.Semaphore(value=self.concurrent)
        return self

    @property
    def step_info(self) -> StepInfo:
        step_info = StepInfo(agent=self.agent,
                      step=self.step_name, **self.init_vars)

        return step_info

    @model_validator(mode='after')
    def init(self) -> Self:
        self._client = self.flow_obj(**self.init_vars)
        del self.flow_obj  # Don't keep the class around

        return self

    @abstractmethod
    async def process(self, *, job: Job) -> Job:
        """ Take a Job with Inputs, process it, and 
        return a Job with Inputs and Outputs."""
        raise NotImplementedError()

    async def _process_job_with_semaphore(self, job: Job) -> Job:
        async with self._sem:
            return await self.process_job(job)

    def submit_job(self, job: Job, background_tasks: BackgroundTasks):
        background_tasks.add_task(self._process_job_with_semaphore, job)
