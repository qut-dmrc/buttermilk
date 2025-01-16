import base64
import datetime
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pydantic
import shortuuid
from cloudpathlib import CloudPath
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

from buttermilk import logger
from buttermilk.llms import LLMCapabilities
from buttermilk.utils.utils import download_limited_async, is_uri
from buttermilk.utils.validators import convert_omegaconf_objects, make_list_validator

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
    metadata: dict = {}

    mime: str | None = Field(
        default="text/plain",
        validation_alias=AliasChoices("mime", "mimetype", "type", "mime_type"),
    )

    text: list[str] | str = Field(
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
    def vld_input(self) -> "MediaObj":
        if not (self.data or self.text or self.uri or self.base_64):
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

    def as_content_part(self, model_type="openai"):
        part = None
        if self.base_64:
            if model_type == "openai":
                part = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{self.mime};base64,{self.base_64}",
                    },
                }
            elif model_type == "anthropic":
                part = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": self.mime,
                        "data": self.base_64,
                    },
                }
        elif self.mime.startswith("text/"):
            part = {"type": "text", "text": self.text}
        elif self.uri:  # try as imageurl?
            part = {
                "type": "image_url",
                "image_url": {
                    "url": self.uri,
                },
            }
        elif self.text:  # not explicitly text, but we have text?
            part = {"type": "text", "text": self.text}
        return part

        """ Anthropic expects: 
        messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image1_media_type,
                        "data": image1_data,
                    },
                },
                {
                    "type": "text",
                    "text": "Describe this image."
                }
            ],
        }
    ],"""


class RecordInfo(BaseModel):
    record_id: str = Field(default_factory=lambda: str(shortuuid.ShortUUID().uuid()))
    metadata: dict[str, str] = {}
    components: list[MediaObj] = []

    @property
    def title(self):
        return self.metadata.get("title")

    @property
    def text(self):
        all_text = [f"{k}: {v}" for k, v in self.metadata.items()]
        all_text.extend([
            f"{part.section}: {part.text}" for part in self.components if part.text
        ])
        return "\n".join(all_text)

    ground_truth: Result | None = None
    uri: str | None = Field(
        default=None,
        validation_alias=AliasChoices("uri", "path", "url"),
    )

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        exclude_unset=True,
        exclude_none=True,
    )

    @model_validator(mode="after")
    def vld_input(self) -> "RecordInfo":
        if not self.components:
            raise ValueError(
                "InputRecord must have at least one component.",
            )
        for key, value in self.model_extra.popitem():
            if key not in self.metadata:
                self.metadata[key] = value
            else:
                raise ValueError(f"Received multiple values for {key} in RecordInfo")

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

    @classmethod
    async def from_uri(cls, uri: str, mimetype: str = None, **metadata):
        if not is_uri(uri):
            raise ValueError(f"Invalid URI provided: {uri}")
        obj, detected_mimetype = await download_limited_async(uri)
        if not mimetype or mimetype == "application/octet-stream":
            mimetype = detected_mimetype

        return await cls.from_object(obj=obj, mimetype=mimetype, **metadata)

    @classmethod
    async def from_object(cls, obj, uri: str = None, mimetype: str = None, **metadata):
        from buttermilk.utils.media import convert_media

        components = [convert_media(obj=obj, mimetype=mimetype)]
        return RecordInfo(uri=uri, components=components, metadata=metadata)

    def update_from(self, result: Result, fields: list | str | None = None):
        update_dict = {}
        if result:
            update_dict = result.model_dump()
        if fields and fields != "record":
            update_dict = {f: update_dict[f] for f in fields}

        # exclude null values
        update_dict = {k: v for k, v in update_dict.items() if v}

        self.__dict__.update(**update_dict)

    def as_langchain_message(
        self,
        model_capabilities: LLMCapabilities,
        role: Literal["user", "human", "system"] = "user",
        include_text: bool = True,
    ) -> BaseMessage | None:
        components = self.as_openai_message(
            role=role, model_capabilities=model_capabilities, include_text=include_text
        )
        if components and (components := components.get("content")):
            if role in {"user", "human"}:
                return HumanMessage(content=components)
            return BaseMessage(content=components, type=role)
        return None

    def as_openai_message(
        self,
        model_capabilities: LLMCapabilities,
        role: Literal["user", "human", "system"] = "user",
        include_text: bool = False,
    ) -> dict | None:
        # Prepare input for model consumption
        components = []
        for obj in self.components:
            # attach media objects if the model supports them
            if (
                (obj.mime.startswith("image") and model_capabilities.image)
                or (obj.mime.startswith("video") and model_capabilities.video)
                or (obj.mime.startswith("audio") and model_capabilities.audio)
            ):
                components.append(obj.as_content_part())

        if not components and not (self.text and include_text):
            logger.warning(
                f"No text or model compatible media provided for {self.record_id}"
            )
            return None

        if include_text:
            text = self.text or "see attached media"
            components.append({"type": "text", "text": text})

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
    prompt: str | None = Field(default=None)

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
    error: dict[str, Any] = Field(default_factory=dict)
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
    _convert = field_validator("outputs", "inputs", "parameters", mode="before")(
        convert_omegaconf_objects(),
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
    def move_metadata(self) -> "Job":
        if self.outputs and hasattr(self.outputs, "metadata") and self.outputs.metadata:
            if self.metadata is None:
                self.metadata = {}
            self.metadata["outputs"] = self.outputs.metadata
            self.outputs.metadata = None

        return self
