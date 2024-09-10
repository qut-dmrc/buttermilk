import datetime
from typing import Any, AsyncGenerator, Generator, Optional, Self, Type, Union

import numpy as np
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



class InputRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: str(shortuuid.ShortUUID().uuid()))
    source: str
    text: Optional[str] = ""
    image: Optional[object] = None
    alt_text: Optional[str] = ""
    expected: Union[bool, None] = False
    labels: list[str] = Field(default=[], validation_alias="label")
    path:  Optional[str] = ""

    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, populate_by_name=True
    )

    @model_validator(mode="after")
    def vld_input(self) -> Self:
        if (
            self.text is None
            and self.image is None
            and self.alt_text is None
        ):
            raise ValueError("InputRecord must have text or image or alt_text.")

        return self

    @field_validator("labels")
    def vld_labels(labels):
        # ensure labels is a list
        if isinstance(labels, str):
            return [labels]

    @field_validator("path")
    def vld_path(path):
        if isinstance(path, CloudPath):
            return str(path.as_uri())


################################
# The result of a job
################################
class JobResultBase(BaseModel):
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

    source: Optional[str] = None
    error: Optional[str] = None
    worker: Optional[str] = None
    raw: Optional[dict|str] = (
        None  # when we receive an invalid response, log it in this field
    )
    metadata: Optional[dict] = {}

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        use_enum_values=True,
        json_encoders={np.bool_: lambda v: bool(v)},
    )

################################
# A unit of work
################################
class Job(BaseModel):
    # A unique identifier for this particular unit of work
    job_id: str = pydantic.Field(default_factory=lambda: shortuuid.uuid())
    name: str  # The name of the evaluation run
    job: str  # The job of the evaluator
    record: InputRecord                                   # The data to be processed by the worker
    result: Optional[Any] = None      # The result of the operation
    options: dict[str, Any] = {}                # Additional options for the worker
    error: Optional[dict[str, Any]] = None
    metadata: dict = {}



####################################
# A response from a single LLM call
####################################
class LLMOutput(JobResultBase):
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    query_id: str = Field(default_factory=lambda: str(shortuuid.ShortUUID().uuid()))

    result: Optional[str] = None
    reasons: Optional[list] = None
    labels: Optional[list] = None
    scores: Optional[dict] = None
    predicted: Optional[bool] = None

    source: Optional[str] = None
    error: Optional[str] = None
    worker: Optional[str] = None
    raw: Optional[dict|str] = (
        None  # when we receive an invalid response, log it in this field
    )
    metadata: Optional[dict] = {}

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        use_enum_values=True,
        json_encoders={np.bool_: lambda v: bool(v)},
    )
