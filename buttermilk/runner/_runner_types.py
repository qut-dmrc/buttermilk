import datetime
import platform
from typing import Any, AsyncGenerator, Generator, Optional, Self, Type, Union

import numpy as np
import psutil
import pydantic
import shortuuid
from cloudpathlib import CloudPath, GSPath
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    TypeAdapter,
    field_validator,
    model_validator,
)

from buttermilk.utils.utils import get_ip


class StepInfo(BaseModel):
    step: str
    agent: str

    model_config = ConfigDict(
        extra="allow", arbitrary_types_allowed=True, populate_by_name=True
    )

class RunInfo(BaseModel):
    run_id: str
    project: str
    job: str

    ip: str = Field(default_factory=get_ip)
    node_name: str = Field(default_factory=lambda: platform.uname().node)
    username: str = Field(default_factory=lambda: psutil.Process().username().split("\\")[-1])

    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, populate_by_name=True
    )

class Result(BaseModel):
    category: Optional[str|int] = None
    result: Optional[float] = None
    labels: Optional[list[str]] = None
    confidence: Optional[float|str] = None
    reasons: Optional[list[str]] = None
    scores: Optional[dict|list] = None

    @field_validator("labels", "reasons", mode="before")
    def vld_list(labels):
        # ensure labels is a list
        if isinstance(labels, str):
            return [labels]


class RecordInfo(BaseModel):
    record_id: str = Field(default_factory=lambda: str(shortuuid.ShortUUID().uuid()))
    source: str
    content: Optional[str] = ""
    image: Optional[object] = None
    alt_text: Optional[str] = ""
    ground_truth: Optional[Result] = None
    path:  Optional[str] = ""

    model_config = ConfigDict(
        extra="allow", arbitrary_types_allowed=True, populate_by_name=True
    )

    @model_validator(mode="after")
    def vld_input(self) -> Self:
        if (
            self.content is None
            and self.image is None
            and self.alt_text is None
        ):
            raise ValueError("InputRecord must have text or image or alt_text.")

        return self

    # @field_validator("labels")
    # def vld_labels(labels):
    #     # ensure labels is a list
    #     if isinstance(labels, str):
    #         return [labels]

    @field_validator("path")
    def vld_path(path):
        if isinstance(path, CloudPath):
            return str(path.as_uri())



##################################
# A single unit of work, including
# a result once we get it.
##################################
class Job(BaseModel):
    # A unique identifier for this particular unit of work
    job_id: str = pydantic.Field(default_factory=lambda: shortuuid.uuid())
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(tz=datetime.timezone.utc))

    run_info: RunInfo
    record_id: str
    parameters: Optional[dict[str, Any]] = {}     # Additional options for the worker
    source: str|list[str]
    inputs: dict =  {}              # The data to be processed by the worker

    step_info: Optional[StepInfo] = None  # These fields will be added once
    outputs: Optional[Result] = None      # the record is processed

    error: Optional[dict[str, Any]] = None
    metadata: Optional[dict] = {}

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        use_enum_values=True,
        json_encoders={np.bool_: lambda v: bool(v)},
        validate_assignment=True
    )

    @field_validator("source", mode="before")
    def vld_list(v):
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("outputs", mode="before")
    def convert_result(v):
        if v and isinstance(v, Mapping):
            return Result(**v)
        elif isinstance(v, Result):
            return v
        else:
            raise ValueError(f'Job constructor expected outputs as type Result, got {type(v)}.')