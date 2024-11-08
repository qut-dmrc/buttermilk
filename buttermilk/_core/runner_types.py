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
from buttermilk.utils.utils import download_limited, download_limited_async
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

def is_b64(value: str) -> bool:
    # Check if the string is a valid base64-encoded string
    try:
        base64.b64decode(value, validate=True)
        return True
    except:
        return False
    
@validate_call
async def validate_uri_or_b64(value: Optional[Union[AnyUrl, str]]) -> Optional[str]:
    if value:
        if is_b64(value):
            return value
    
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

class Media(BaseModel):
    base_64: Optional[str] = Field(default=None, validation_alias=AliasChoices("base_64", "base64", "b64"))
    uri: Optional[str] = Field(default=None, validation_alias=AliasChoices("uri", "path", "url"))
    binary: Optional[bytes] = None
    mimetype: Optional[str] = None
    download: Optional[bool] = False
    

    @field_validator("base64", mode="after")
    def ensure_b64(cls, values):
        if values:
            for n, v in enumerate(values):
                if not is_b64(v):
                    raise ValueError(f"Invalid base64 string passed (idx: {n})")
        
        return values
    
    @model_validator(mode="after")
    def convert(self) -> Self:
        if self.download:
            if self.uri and (isinstance(self.uri, AnyUrl) or AnyUrl(self.uri)):
                # It's a URL, go fetch and encode it
                obj = download_limited(self.uri)
                self.base_64 = base64.b64encode(obj).decode("utf-8")
        elif isinstance(self.uri, CloudPath):
            self.uri = str(self.uri.as_uri())
        elif isinstance(self.uri, GSPath):
            self.uri = str(self.uri.as_uri())

        if self.binary:
            self.base_64 = base64.b64encode(self.binary).decode("utf-8")
            del self.binary

        return self

class RecordInfo(BaseModel):
    record_id: str = Field(default_factory=lambda: str(shortuuid.ShortUUID().uuid()))
    name: Optional[str] = Field(default=None, validation_alias=AliasChoices("name", "title", "heading"))
    text: Optional[str] = Field(default=None, validation_alias=AliasChoices("text", "content", "body", "alt_text", "alt", "caption"))
    media: Optional[Media|Sequence[Media]] = Field(default_factory=list, validation_alias=AliasChoices("media", "image", "video", "audio"))
    ground_truth: Optional[Result] = None
    uri:  Optional[str] =  Field(default=None, validation_alias=AliasChoices("uri","path","url"))

    _ensure_list = field_validator("media", mode="before")(make_list_validator())
    
    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, populate_by_name=True, exclude_unset=True, exclude_none=True
    )


    @model_validator(mode="after")
    def vld_input(self) -> Self:
        if (
            self.text is None and self.media is None
        ):
            raise ValueError("InputRecord must have text or image or video or alt_text.")
        
        return self

    @field_validator("path")
    def vld_path(path):
        if isinstance(path, CloudPath):
            return str(path.as_uri())

    def as_langchain_message(self, type: Literal["human","system"]='human') -> BaseMessage:
        # Return the fields as a langchain message

        if not self.media:
            BaseMessage(content=self.text, type=type)

        components = []
        # Prepare input for model consumption
        if self.text:
            components.append(
                {
                    "type": "text",
                    "text": self.text,
                }
            )
        for obj in self.media:
            if obj.base_64:
                mime_type = mime_type or "image/png"
                media_message = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{obj.base_64}",
                        },
                }
            elif obj.uri:
                if mime_type:
                    media_message = {"type": "media", 'mime_type': obj.mimetype,
                            'file_uri': obj.uri}
                else:
                    media_message = {
                        "type": "image_url",
                        "image_url": {
                            "url": obj.uri,
                        },
                    }
            else:
                raise ValueError("Unknown media attachment provided.")
            
            components.append(media_message)

        return HumanMessage(content=components)
        


##################################
# A single unit of work, including
# a result once we get it.
#
# A Job contains all the information that we need to log
# to know about an agent's operation on a single datum.
#
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
