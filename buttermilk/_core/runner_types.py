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
from cloudpathlib import AnyPath, CloudPath
from langchain_core.messages import BaseMessage, HumanMessage
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from buttermilk.utils.utils import is_uri, remove_punctuation
from buttermilk.utils.validators import make_list_validator

from .types import SessionInfo


class Result(BaseModel):
    category: str | int | None = Field(default=None)
    prediction: bool | int | None = Field(
        default=None,
        validation_alias=AliasChoices("prediction", "prediction", "pred"),
    )
    result: float | str | None = Field(default=None)
    labels: list[str] | None = Field(
        default=[],
        validation_alias=AliasChoices("labels", "label"),
    )
    confidence: float | str | None = Field(default=None)
    severity: float | str | None = Field(default=None)
    reasons: list | None = Field(
        default=[],
        validation_alias=AliasChoices("reasoning", "reason", "analysis", "answer"),
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
        json_encoders={
            np.bool_: bool,
            datetime.datetime: lambda v: v.isoformat(),
            ListConfig: lambda v: OmegaConf.to_container(v, resolve=True),
            DictConfig: lambda v: OmegaConf.to_container(v, resolve=True),
        },
        validate_assignment=True,
        exclude_unset=True,
        exclude_none=True,
    )

    def model_dump(self, **kwargs):
        # Use the default model_dump method with exclude_none and exclude_unset
        data = super().model_dump(**kwargs, exclude_none=True, exclude_unset=True)
        # Remove keys with empty values
        return {k: v for k, v in data.items() if v}


class MediaObj(BaseModel):
    mime: str = Field(
        default="image/png",
        validation_alias=AliasChoices("mime", "mimetype", "type", "mime_type"),
    )

    text: str = Field(
        default="",
        validation_alias=AliasChoices("text", "alt", "caption", "alt_text"),
    )

    data: bytes | None = Field(
        default=None,
        validation_alias=AliasChoices("data", "content"),
    )

    uri: str | None = Field(
        default=None,
        validation_alias=AliasChoices("uri", "path", "url"),
    )

    base_64: str | None = Field(
        default=None,
        validation_alias=AliasChoices("base_64", "b64", "b64_data"),
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
        # We exclude the potentially large fields from the export. Instead
        # just indicate the length and the mimetype+uri so the user knows
        # data was passed in.
        if self.base_64:
            return len(self.base_64)
        return -1

    @model_validator(mode="after")
    def vld_input(self) -> Self:
        if not (self.data or self.uri or self.base_64):
            raise ValueError("MediaObj must have data, a uri, or a base64 string.")
        if self.data and not self.base_64:
            self.base_64 = base64.b64encode(self.data).decode("utf-8")

            # Strip binary data away once it's converted to b64
            self.data = None

        return self

    @field_validator("uri")
    def vld_path(path):
        if isinstance(path, CloudPath):
            return path.as_uri()
        if isinstance(path, Path):
            return path.as_posix()
        return path

    def as_image_url(self) -> dict:
        return {
            "type": "image_url",
            "image_url": {"url": self.uri},
        }

    def as_text(self) -> dict:
        return {"type": "text", "text": self.text}

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
        validation_alias=AliasChoices("name", "title", "heading"),
    )
    text: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "text",
            "content",
            "body",
            "alt_text",
            "alt",
            "caption",
        ),
    )
    media: list[MediaObj] = Field(
        default_factory=list,
        validation_alias=AliasChoices("media", "image", "video", "audio"),
    )
    ground_truth: Result | None = None
    uri: str | None = Field(
        default=None,
        validation_alias=AliasChoices("uri", "path", "url"),
    )

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

        if not self.record_id:
            if self.name:
                self.record_id = remove_punctuation(self.name)
            elif self.uri:
                self.record_id = AnyPath(self.uri).stem

        return self

    @field_validator("uri")
    @classmethod
    def vld_path(cls, path: object) -> str:
        if not is_uri(path):
            raise ValueError(f"Invalid URI: {path}")
        if isinstance(path, CloudPath):
            return path.as_uri()
        if isinstance(path, Path):
            return path.as_posix()
        return str(path)

    def as_langchain_message(
        self,
        role: Literal["user", "human", "system"] = "user",
    ) -> BaseMessage | None:
        components = self.as_openai_message(role=role)["content"]
        if role in {"user", "human"}:
            return HumanMessage(content=components)
        return BaseMessage(content=components, type=role)

    def as_openai_message(
        self,
        role: Literal["user", "human", "system"] = "user",
    ) -> dict:
        # Prepare input for model consumption
        components = [obj.as_content_part() for obj in self.media]

        if not self.media and not self.text:
            raise OSError("No text or media provided for {self.record_id}")
        if self.text:
            components.append({"type": "text", "text": self.text})

        message = {
            "role": role,
            "content": components,
        }
        return message


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
    flow_id: str
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(tz=datetime.UTC),
        description="The date and time a job was created.",
    )
    source: Sequence[str] = Field(
        ...,
        description="Where this particular job came from",
    )

    run_info: SessionInfo | None = Field(
        default=None,
        description="Information about the context in which this job runs",
    )

    record: RecordInfo | None = Field(
        default=None,
        description="The data the job will process.",
    )
    prompt: str = Field(default_factory=str)
    parameters: dict | None = Field(
        default_factory=dict,
        description="Additional options for the worker",
    )
    inputs: dict = Field(default_factory=dict)

    # These fields will be fully filled once the record is processed
    agent_info: dict | None = Field(default_factory=dict)
    outputs: Result | None = Field(
        default=None,
        description="The results of the job",
    )
    error: dict[str, Any] | None = Field(default_factory=dict)
    metadata: dict | None = Field(default_factory=dict)

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=False,
        populate_by_name=True,
        use_enum_values=True,
        json_encoders={
            np.bool_: bool,
            datetime.datetime: lambda v: v.isoformat(),
            ListConfig: lambda v: OmegaConf.to_container(v, resolve=True),
            DictConfig: lambda v: OmegaConf.to_container(v, resolve=True),
        },
        validate_assignment=True,
        exclude_unset=True,
        exclude_none=True,
    )
    _ensure_list = field_validator("source", mode="before")(
        make_list_validator(),
    )

    # @field_serializer('flow')
    # def serialize_omegaconf(cls, value):
    #     return OmegaConf.to_container(value, resolve=True)

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
