import base64
import datetime
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import (
    Any,
    Literal,
    Self,
)

import numpy as np
import pydantic
import shortuuid
from bs4 import BeautifulSoup
from cloudpathlib import CloudPath
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import (
    AliasChoices,
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
    validate_call,
)

from buttermilk.utils.utils import download_limited_async
from buttermilk.utils.validators import make_list_validator

from .types import SessionInfo


@validate_call
async def validate_uri_extract_text(value: AnyUrl | str | None) -> str | None:
    if value:
        try:
            _ = AnyUrl(value)
        except:
            return value

        # It's a URL, go fetch
        obj, mimetype = await download_limited_async(value)

        # try to extract text from object
        if mimetype.startswith("text/html"):
            soup = BeautifulSoup(obj, "html.parser")
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
async def validate_uri_or_b64(value: AnyUrl | str | None) -> str | None:
    if value:
        if is_b64(value):
            return value

        try:
            if isinstance(value, AnyUrl) or AnyUrl(value):
                # It's a URL, go fetch and encode it
                obj = await download_limited_async(value)
                value = base64.b64encode(obj).decode("utf-8")
                return value
        except Exception:
            raise ValueError("Invalid URI or base64-encoded string")
    return None


class Result(BaseModel):
    category: str | int | None = None
    prediction: bool | int | None = Field(
        default=None,
        validation_alias=AliasChoices("prediction", "prediction", "pred"),
    )
    result: float | str | None = None
    labels: list[str] | None = Field(
        default=[],
        validation_alias=AliasChoices("labels", "label"),
    )
    confidence: float | str | None = None
    severity: float | str | None = None
    reasons: list | None = Field(
        default=[],
        validation_alias=AliasChoices("reasoning", "reason"),
    )
    scores: dict | list | None = None
    metadata: dict | None = {}

    _ensure_list = field_validator("labels", "reasons", mode="before")(
        make_list_validator(),
    )

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        use_enum_values=True,
        json_encoders={np.bool_: lambda v: bool(v)},
        validate_assignment=True,
        exclude_unset=True,
        exclude_none=True,
    )


class MediaObj(BaseModel):
    mime: str = Field(
        default="image/png",
        alias=AliasChoices("mime", "mimetype", "type", "mime_type"),
    )
    data: bytes | None = Field(default=None, alias=AliasChoices("data", "content"))
    uri: str | None = Field(default=None, alias=AliasChoices("uri", "path", "url"))
    base_64: str | None = Field(
        default=None,
        alias=AliasChoices("base_64", "b64", "b64_data"),
    )

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        exclude_unset=True,
        exclude_none=True,
        exclude=["data", "base64"],
    )

    @computed_field
    @property
    def len(self) -> int:
        """We exclude the potentially large fields from the export. Instead just indicate the length
        and the mimetype+uri so the user knows data was passed in.
        """
        if self.base_64:
            return len(self.base_64)
        return -1

    @model_validator(mode="after")
    def vld_input(self) -> Self:
        if not (self.data or self.uri or self.base_64):
            raise ValueError("MediaObj must have data, a uri, or a base64 string.")
        if self.data and not self.base_64:
            self.base_64 = base64.b64encode(self.data).decode("utf-8")
        return self

    @field_validator("uri")
    def vld_path(path):
        if isinstance(path, CloudPath):
            return path.as_uri()
        if isinstance(path, Path):
            return path.as_posix()
        return path

    def as_content_part(self):
        if self.base_64:
            part = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{self.mime};base64,{self.base_64}",
                },
            }
        else:
            part = {
                "type": "image_url",
                "image_url": {
                    "url": self.uri,
                },
            }
        return part


class RecordInfo(BaseModel):
    record_id: str = Field(default_factory=lambda: str(shortuuid.ShortUUID().uuid()))
    name: str | None = Field(
        default=None,
        alias=AliasChoices("name", "title", "heading"),
    )
    text: str | None = Field(
        default=None,
        alias=AliasChoices("text", "content", "body", "alt_text", "alt", "caption"),
    )
    media: MediaObj | Sequence[MediaObj] | None = Field(
        default_factory=list,
        alias=AliasChoices("media", "image", "video", "audio"),
    )
    ground_truth: Result | None = None
    uri: str | None = Field(default=None, alias=AliasChoices("uri", "path", "url"))

    _ensure_list = field_validator("media", mode="before")(make_list_validator())

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        exclude_unset=True,
        exclude_none=True,
    )

    @model_validator(mode="after")
    def vld_input(self) -> Self:
        if self.text is None and self.media is None:
            raise ValueError(
                "InputRecord must have text or image or video or alt_text.",
            )

        return self

    @field_validator("uri")
    def vld_path(path):
        if isinstance(path, CloudPath):
            return path.as_uri()
        if isinstance(path, Path):
            return path.as_posix()
        return path

    def as_langchain_message(
        self,
        type: Literal["human", "system"] = "human",
    ) -> BaseMessage:
        # Return the fields as a langchain message

        if not self.media:
            BaseMessage(content=self.text, type=type)

        components = []
        # Prepare input for model consumption
        if text := self.text or "see attached":
            components.append(
                {
                    "type": "text",
                    "text": text,
                },
            )

        for obj in self.media:
            components.append(obj.as_content_part())

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
    job_id: str = pydantic.Field(default_factory=shortuuid.uuid)
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(tz=datetime.UTC),
        description="The date and time a job was created.",
    )
    source: str = Field(..., description="Where this particular job came from")

    run_info: SessionInfo | None = Field(
        default=None,
        description="Information about the context in which this job runs",
    )

    record: RecordInfo | None = Field(
        default=None,
        description="The data the job will process.",
    )
    prompt: Sequence[str] | None = Field(default_factory=list)
    parameters: Mapping | None = Field(
        default_factory=dict,
        description="Additional options for the worker",
    )
    inputs: Any | Mapping | None = None

    # These fields will be fully filled once the record is processed
    agent_info: Any | None = None
    outputs: Any | Result | None = None
    error: dict[str, Any] | None = None
    metadata: dict | None = Field(default_factory=dict)

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        use_enum_values=True,
        json_encoders={np.bool_: bool},
        validate_assignment=True,
        exclude_unset=True,
        exclude_none=True,
    )
    _ensure_list = field_validator("prompt", mode="before")(make_list_validator())

    @field_validator("outputs", mode="before")
    def convert_result(v):
        if v and isinstance(v, Mapping):
            return Result(**v)
        return v

    @model_validator(mode="after")
    def move_metadata(self) -> Self:
        if self.outputs and hasattr(self.outputs, "metadata") and self.outputs.metadata:
            if self.metadata is None:
                self.metadata = {}
            self.metadata["outputs"] = self.outputs.metadata
            self.outputs.metadata = None

        return self
