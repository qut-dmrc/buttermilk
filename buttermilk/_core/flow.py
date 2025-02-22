import os
from pathlib import Path
from typing import Any, AsyncGenerator, Generator, Literal, Optional, Sequence, Tuple, Type, TypeVar, Union,Mapping
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
      
class Flow(BaseModel):
    name: str
    num_runs: int = 1
    concurrency: int = 4
    agent: Agent
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
    