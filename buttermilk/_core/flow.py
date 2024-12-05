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

from hydra_zen import instantiate, builds

from buttermilk._core.agent import Agent
from buttermilk._core.config import DataSource, SaveInfo
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
    Sequence,
    Type,
    TypeVar,
)
      
class Flow(BaseModel):
    name: str
    num_runs: int = 1
    concurrency: int = 1
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
    