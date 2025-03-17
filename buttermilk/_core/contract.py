import pydantic

import os
from pathlib import Path
from typing import Any, AsyncGenerator, Generator, Literal, Optional, Protocol, Sequence, Tuple, Type, TypeVar, Union,Mapping
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

from buttermilk._core.agent import Agent
from buttermilk._core.config import DataSource, SaveInfo
from buttermilk._core.job import Job
from buttermilk._core.ui import IOInterface
from buttermilk.bm import BM
from buttermilk.defaults import BQ_SCHEMA_DIR

import os
from pathlib import Path

import cloudpathlib
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from .log import logger

BASE_DIR = Path(__file__).absolute().parent
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
)


class Orchestrator(Protocol):
    bm: BM
    save: SaveInfo = Field(default=None)
    data: Sequence[DataSource] = Field(default_factory=list)
    flow: Sequence[Agent] = Field(default_factory=list)
    interface: IOInterface 

    model_config = ConfigDict(
        extra = "forbid",
        arbitrary_types_allowed = False,
        populate_by_name = True,
        exclude_none = True,
        exclude_unset = True,
    ) # type: ignore

    async def run(self, job: Job) -> Job:
        results = job.inputs.model_copy()
        
        for step in self.flow:
            results[step.name] = await step(results)

        job.outputs = results
        return results
    
    async def __call__(self, job: Job) -> Job:
        return await self.run(job)
        
