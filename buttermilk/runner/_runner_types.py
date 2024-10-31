import asyncio
import datetime
import platform
from typing import Any, AsyncGenerator, Generator, Literal, Optional, Self, Type, Union,Mapping

from langchain_core.messages.base import BaseMessage, BaseMessageChunk
import numpy as np
import psutil
import pydantic
from pydantic.functional_validators import AfterValidator
import shortuuid
import base64
from cloudpathlib import CloudPath, GSPath
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

from buttermilk import BM

from buttermilk.buttermilk import SessionInfo
from buttermilk.utils.utils import download_limited_async

class AgentInfo(BaseModel):
    # job: str
    # step: str
    agent: str
    model: Optional[str] = None

    model_config = ConfigDict(
        extra="allow", arbitrary_types_allowed=True, populate_by_name=True
    )

class Result(BaseModel):
    category: Optional[str|int] = None
    prediction: Optional[bool|int] = Field(default=None, validation_alias=AliasChoices("prediction", "prediction", "pred"))
    result: Optional[float|str] = None
    labels: Optional[list[str]] = Field(default=[], validation_alias=AliasChoices("labels", "label"))
    confidence: Optional[float|str] = None
    severity: Optional[float|str] = None
    reasons: Optional[list] = Field(default=[], validation_alias=AliasChoices("reasoning", "reason"))
    scores: Optional[dict|list] = None
    metadata: Optional[dict] = {}

    @field_validator("labels", "reasons", mode="before")
    def vld_list(labels):
        # ensure labels is a list
        if isinstance(labels, str):
            return [labels]
        return labels

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        use_enum_values=True,
        json_encoders={np.bool_: lambda v: bool(v)},
        validate_assignment=True
    )

class Base64Str(str):
    @classmethod
    def validate(cls, value: str) -> str:
        try:
            # Check if the string is a valid base64-encoded string
            base64.b64decode(value, validate=True)
            return value
        except Exception:
            raise ValueError("Invalid base64-encoded string")
        
class RecordInfo(BaseModel):
    record_id: str = Field(default_factory=lambda: str(shortuuid.ShortUUID().uuid()))
    name: Optional[str] = None
    source: str
    content: Optional[str] = Field(default=None, validation_alias=AliasChoices("content", "text", "body"))
    uri: Optional[AnyUrl] = None
    image: Optional[Union[AnyUrl, Base64Str]] = None  # Allow AnyUrl or base64 string
    video: Optional[Union[AnyUrl, Base64Str]] = None  # Allow AnyUrl or base64 string
    alt_text: Optional[str] = Field(default=None, validation_alias=AliasChoices("alt_text", "alt", "caption"))
    ground_truth: Optional[Result] = None
    path:  Optional[str] = ""

    model_config = ConfigDict(
        extra="allow", arbitrary_types_allowed=True, populate_by_name=True
    )

    @AfterValidator
    async def vld_input(self) -> Self:
        if (
            self.content is None
            and self.image is None
            and self.alt_text is None
            and self.video is None
            and self.uri is None
        ):
            raise ValueError("InputRecord must have text or image or video or alt_text or uri.")
        return self
    
    @field_validator("image", "video","uri", mode="before")
    async def validate_image(cls, value: Optional[Union[AnyUrl, Base64Str]]) -> Optional[str]:
        if value and isinstance(value, AnyUrl):
            return await download_limited_async(value)
        return value

    # @field_validator("labels")
    # def vld_labels(labels):
    #     # ensure labels is a list
    #     if isinstance(labels, str):
    #         return [labels]

    @field_validator("path")
    def vld_path(path):
        if isinstance(path, CloudPath):
            return str(path.as_uri())

    def as_langchain_message(self, type: Literal["human","system"]='human') -> BaseMessage:
        # Return the fields as a langchain message
        parts = [x for x in [self.text, self.alt_text] if x is not None ]
    
        if self.imaeg_b64:
            parts.append(f"![Image](data:image/png;base64,{self.image_b64})")
        if self.video_b64:
            parts.append(f"![Video](data:video/mp4;base64,{self.video_b64})")

        content = '\n'.join(parts)

        message = BaseMessage(type=type, content=content, id=self.record_id, name=self.name)


##################################
# A single unit of work, including
# a result once we get it.
##################################
class Job(BaseModel):
    # A unique identifier for this particular unit of work
    job_id: str = pydantic.Field(default_factory=lambda: shortuuid.uuid())
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(tz=datetime.timezone.utc))
    run_info: SessionInfo = pydantic.Field(default_factory=lambda: BM()._run_metadata)

    parameters: Optional[dict[str, Any]] = Field(default_factory=dict)     # Additional options for the worker
    source: str|list[str]
    
    record: Optional[RecordInfo] = None
    inputs: dict = Field(default_factory=dict)             # Other data to be processed by the worker

    # These fields will be fully filled once the record is processed
    agent_info: Optional[AgentInfo] = None
    outputs: Optional[Result] = None     
    error: Optional[dict[str, Any]] = None
    metadata: Optional[dict] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        use_enum_values=True,
        json_encoders={np.bool_: lambda v: bool(v)},
        validate_assignment=True,
        exclude_unset=True
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
    
        
    @model_validator(mode="after")
    def move_metadata(self) -> Self:
        if self.outputs and hasattr(self.outputs, 'metadata') and self.outputs.metadata:
            if self.metadata is None:
                self.metadata = {}
            self.metadata['outputs'] = self.outputs.metadata
            self.outputs.metadata = None
            
        return self
