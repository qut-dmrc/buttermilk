import asyncio
import datetime
import platform
from typing import Any, AsyncGenerator, Generator, Literal, Optional, Self, Sequence, Tuple, Type, Union,Mapping

from langchain_core.messages import BaseMessage, BaseMessageChunk, HumanMessage, AIMessage
from langchain_core.prompts import (
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
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
from pydantic import PositiveInt, ValidationError, validate_call


from .types import SessionInfo
from buttermilk.utils.utils import download_limited_async
import httpx
from bs4 import BeautifulSoup

from buttermilk.utils.validators import make_list_validator


@validate_call
async def validate_uri_extract_text(value: Optional[Union[AnyUrl, str]]) -> Optional[str]:
    if value:
        try:
            _ = AnyUrl(value)
        except:
            return value
        
        # It's a URL, go fetch
        obj, mimetype = await download_limited_async(value)

        # try to extract text from object
        if mimetype.startswith('text/html'):
            soup = BeautifulSoup(obj, 'html.parser')
            value = soup.get_text()
        else:
            value = obj.decode()
    return value

@validate_call
async def validate_uri_or_b64(value: Optional[Union[AnyUrl, str]]) -> Optional[str]:
    if value:
        try:
            # Check if the string is a valid base64-encoded string
            base64.b64decode(value, validate=True)
            return value
        except:
            pass
    
        try:
            if isinstance(value, AnyUrl) or AnyUrl(value):
                # It's a URL, go fetch and encode it
                obj = await download_limited_async(value)
                value = base64.b64encode(obj).decode("utf-8")
                return value
        except Exception as e:
            raise ValueError("Invalid URI or base64-encoded string")
    return None
        
class AgentInfo(BaseModel):
    name: str

    model_config = ConfigDict(
        extra="allow", arbitrary_types_allowed=True, populate_by_name=True, exclude_none=True, exclude_unset=True,
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

    _ensure_list = field_validator("labels", "reasons", mode="before")(make_list_validator())

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        use_enum_values=True,
        json_encoders={np.bool_: lambda v: bool(v)},
        validate_assignment=True,
        exclude_unset=True,
        exclude_none=True
    )

        
class RecordInfo(BaseModel):
    record_id: str = Field(default_factory=lambda: str(shortuuid.ShortUUID().uuid()))
    name: Optional[str] = Field(default=None, validation_alias=AliasChoices("name", "title", "heading"))
    content: Optional[str] = Field(default=None, validation_alias=AliasChoices("content", "text", "body"))
    image: Optional[str] = None  # Allow URL or base64 string
    video: Optional[str] = None  # Allow URL or base64 string
    alt_text: Optional[str] = Field(default=None, validation_alias=AliasChoices("alt_text", "alt", "caption"))
    ground_truth: Optional[Result] = None
    path:  Optional[str] = ""

    model_config = ConfigDict(
        extra="allow", arbitrary_types_allowed=True, populate_by_name=True, exclude_unset=True, exclude_none=True
    )

    @model_validator(mode="after")
    def vld_input(self) -> Self:
        if (
            self.content is None
            and self.image is None
            and self.alt_text is None
            and self.video is None
        ):
            raise ValueError("InputRecord must have text or image or video or alt_text.")
        
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

        
    def as_langchain_message(self, type: Literal["human","system"]='human', components: list[str] = ['content']) -> BaseMessage:
        # Return the fields as a langchain message
        parts = [x for x in [self.content, self.alt_text] if x is not None ]
        # TODO: make multimodal
        # Example from Claude:
        message = """"content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "<base64_encoded_image>"
                    }
                }
            ]"""
        # if self.image:
        #     parts.append(f"![Image](data:image/png;base64,{self.image})")
        # if self.video:
        #     parts.append(f"![Video](data:video/mp4;base64,{self.video})")

        content = '\n'.join(parts)
        if type == "human":
            message = HumanMessage(content=content)
        else:
            # No idea why, but this one doesn't seem to work with type='human'
            message = BaseMessage(content=str(content), type=type)#, id=self.record_id, name=self.name)

        return message


##################################
# A single unit of work, including
# a result once we get it.
##################################
class Job(BaseModel):
    # A unique identifier for this particular unit of work
    job_id: str = pydantic.Field(default_factory=lambda: shortuuid.uuid())
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(tz=datetime.timezone.utc))
    run_info: Optional[SessionInfo] = None

    source: str|list[str]
    
    record: Optional[RecordInfo] = None
    prompt: Optional[Sequence[str]] = Field(default_factory=list)
    parameters: Optional[dict[str, Any]] = Field(default_factory=dict)     # Additional options for the worker

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
        exclude_unset=True,
        exclude_none=True,
    )
    _ensure_list = field_validator("source","prompt", mode="before")(make_list_validator())

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
        
        # Store a copy of the run info in this model's metadata
        if self.run_info is None:
            from ..buttermilk import BM
            run_info = BM()._run_metadata
            self.run_info = run_info
        return self
