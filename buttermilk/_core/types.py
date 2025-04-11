import asyncio
import datetime
import platform
from pathlib import Path
from tempfile import mkdtemp
from typing import Self

import psutil
import pydantic
import shortuuid
from cloudpathlib import AnyPath, CloudPath
from pydantic import (
    ConfigDict,
    Field,
)

import base64
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self
from PIL.Image import Image
import shortuuid
from cloudpathlib import CloudPath
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    field_validator,
    model_validator,
)

from buttermilk.utils.utils import is_b64

from .log import logger
from .contract import AssistantMessage, UserMessage


_global_run_id = ""


def _make_run_id() -> str:
    global _global_run_id
    if _global_run_id:
        return _global_run_id
    # Create a unique identifier for this run
    node_name = platform.uname().node
    username = psutil.Process().username()
    # get rid of windows domain if present
    username = str.split(username, "\\")[-1]

    # The ISO 8601 format has too many special characters for a filename, so we'll use a simpler format
    run_time = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%MZ")

    run_id = f"{run_time}-{shortuuid.uuid()[:4]}-{node_name}-{username}"
    return run_id


_global_run_id = _make_run_id()


class SessionInfo(pydantic.BaseModel):
    platform: str = "local"
    name: str
    job: str
    run_id: str = Field(default=_global_run_id)
    max_concurrency: int = -1
    ip: str = Field(default="")
    node_name: str = Field(default_factory=lambda: platform.uname().node)
    username: str = Field(
        default_factory=lambda: psutil.Process().username().split("\\")[-1],
    )
    save_dir: str | None = None
    flow_api: str | None = None

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )
    _get_ip_task: asyncio.Task

    save_dir_base: str = Field(
        default_factory=mkdtemp,
        validate_default=True,
    )  # Default to temp dir

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=False)

    def __str__(self):
        return _global_run_id

    @pydantic.field_validator("save_dir_base", mode="before")
    def get_save_dir(cls, save_dir_base, values) -> str:
        if isinstance(save_dir_base, str):
            pass
        elif isinstance(save_dir_base, Path):
            save_dir_base = save_dir_base.as_posix()
        elif isinstance(save_dir_base, CloudPath):
            save_dir_base = save_dir_base.as_uri()
        else:
            raise ValueError(
                f"save_dir_base must be a string, Path, or CloudPath, got {type(save_dir_base)}",
            )
        return save_dir_base

    @pydantic.model_validator(mode="after")
    def set_full_save_dir(self) -> Self:
        save_dir = AnyPath(self.save_dir_base) / self.name / self.job / self.run_id
        self.save_dir = str(save_dir)
        return self

    @pydantic.model_validator(mode="after")
    def schedule_get_ip(self) -> Self:
        self._get_ip_task = asyncio.get_event_loop().create_task(self.get_ip())
        return self

    async def get_ip(self):
        if not self.ip:
            from ..utils import get_ip
            self.ip = await get_ip()


class MediaObj(BaseModel):
    label: str | None = Field(
        default=None,
        description="Section or type of content part (e.g. heading, body paragraph, caption, image, etc)",
        validation_alias=AliasChoices("arg0"),
    )

    metadata: dict = {}

    uri: str | None = Field(
        default=None,
        description="Media objects that can be passed as URIs to cloud storage",
    )

    mime: str | None = Field(
        default="text/plain",
        validation_alias=AliasChoices("mime", "mimetype", "type", "mime_type"),
    )

    content: str | bytes | None = Field(default=None, description="Temporary field used in loading data only.")

    _text: str | None = PrivateAttr(
        default=None,
    )
    _image: Image | None = PrivateAttr(
        default=None,
    )
    _base_64: str | None = PrivateAttr(
        default=None,
    )

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        exclude_unset=True,
        exclude_none=True,
        exclude=["data", "base_64"],
    )

    def __str__(self):
        return f"{self.label or 'unknown'} object ({self.mime}) {self._text[:50]}..."

    @model_validator(mode="after")
    def interpret_data(self) -> Self:
        if isinstance(self.content, str):
            if self._base_64:
                raise ValueError("MediaObj received two string fields.")

            if not self.mime:
                self.mime = "text/plain"

        if isinstance(self.content, bytes):
            if self._base_64:
                raise ValueError(
                    "MediaObj can have either bytes or base64 data, but not both.",
                )
            self._base_64 = base64.b64encode(self.content).decode("utf-8")
            if not self.mime:
                self.mime = "application/octet-stream"

            # Strip binary data away once it's converted to b64
            self.content = None
        elif isinstance(self.content, Image):
            self._image = self.content
            self.content = None
        else:
            # String content, move
            if is_b64(self.content):
                self._base_64 = str(self.content)
            else:
                self._text = str(self.content)
            self.content = None

        # del self.content

        return self

    def as_url(self):
        return (f"data:{self.mime};base64,{self._base_64}",)

    def as_image_url_message(self) -> dict:
        return {
            "type": "image_url",
            "image_url": {"url": self.uri},
        }

    def as_text(self) -> str:
        return self._text

    def as_content_part(self, model_type="openai") -> dict:
        part = {}
        if self._base_64:
            if model_type == "openai":
                part = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{self.mime};base64,{self._base_64}",
                    },
                }
            elif model_type == "anthropic":
                part = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": self.mime,
                        "data": self._base_64,
                    },
                }
        elif self.uri:
            part = self.as_image_url_message()
        else:
            part = self.as_text()

        return part

    #     """ Anthropic expects:
    #     messages=[
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "image",
    #                 "source": {
    #                     "type": "base64",
    #                     "media_type": image1_media_type,
    #                     "data": image1_data,
    #                 },
    #             },
    #             {
    #                 "type": "text",
    #                 "text": "Describe this image."
    #             }
    #         ],
    #     }
    # ],"""


class Record(BaseModel):
    record_id: str = Field(default_factory=lambda: str(shortuuid.ShortUUID().uuid()))
    metadata: dict[str, Any] = Field(default={})
    alt_text: str | None = Field(
        default=None,
        description="Text description or transcript of media objects contained in this record.",
    )
    ground_truth: dict | None = Field(
        default=None,
        validation_alias=AliasChoices("ground_truth", "golden"),
    )

    uri: str | None = Field(
        default=None,
    )

    components: list[MediaObj] = Field(default=[])

    def __str__(self) -> str:
        return self._fulltext

    @computed_field
    @property
    def _fulltext(self) -> str:
        """Combines metadata and text content into a single string.

        Excludes ground truth and component labels.
        """
        parts = []

        if self.metadata:
            parts.append("### Metadata")
            for k, v in self.metadata.items():
                parts.append(f"**{k}**: {v}")
            parts.append("### Content")  # Separator

        component_texts = [part._text for part in self.components if part._text]
        if component_texts:
            parts.extend(component_texts)

        return "\n".join(parts)

    @computed_field
    @property
    def _text(self) -> str:
        # Only text. Metadata and Ground truth not included
        return "\n".join(self._paragraphs)

    @computed_field
    @property
    def _paragraphs(self) -> list[str]:
        # Only text. Metadata and Ground truth not included
        all_text = []
        for part in self.components:
            if part._text:
                all_text.append(part._text)

        return all_text

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        exclude_unset=True,
        exclude_none=True,
        exclude=["_fulltext", "_text", "_paragraphs"],
        positional_args=True,
    )  # type: ignore

    @model_validator(mode="after")
    def vld_input(self) -> Self:
        # Place extra arguments in the metadata field
        if self.model_extra:
            while len(self.model_extra.keys()) > 0:
                key, value = self.model_extra.popitem()
                if key not in self.metadata and key not in self.model_computed_fields:
                    self.metadata[key] = value
                else:
                    raise ValueError(
                        f"Received multiple values for {key} in Record",
                    )
        self.data = None
        return self

    @field_validator("uri")
    @classmethod
    def vld_path(cls, path: object) -> str | None:
        if not path:
            return None
        if isinstance(path, CloudPath):
            return path.as_uri()
        if isinstance(path, Path):
            return path.as_posix()
        return str(path)

    @property
    def title(self) -> str | None:
        return self.metadata.get("title")

    @property
    def all_text(self) -> str:
        # Also with paragraph labels etc.
        all_text = [f"**{k}**: {v}" for k, v in self.metadata.items()]
        for part in self.components:
            if part._text:
                if part.label:
                    all_text.append(f"{part.label}: {part._text}")
                else:
                    all_text.append(part._text)
        return "\n".join(all_text)

    def update_from(self, update_dict) -> Self:
        # exclude null values
        update_dict = {k: v for k, v in update_dict.items() if v}

        self.metadata.update(**update_dict)

        return self

    def as_message(self, role: Literal["user", "human", "system", "assistant"] = "user", source: str = "record") -> UserMessage | AssistantMessage:
        # Prepare input for model consumption
        leading_components = []
        trailing_components = []

        for obj in self.components:
            # attach media objects if the model supports them
            if obj.mime.startswith("image") or obj.mime.startswith("video") or obj.mime.startswith("audio"):
                leading_components.append(obj.as_content_part())
            elif obj.mime.startswith("text"):
                trailing_components.append(obj.as_content_part())
            elif "uri" in obj.model_fields_set:
                trailing_components.append(obj.as_image_url_message())

        for k, v in self.metadata.items():
            # add in metadata (title, byline, date, exif, etc.)
            # llama3.2 at least expects images first
            trailing_components.append({"type": "text", "text": f"{k}: {v}"})

        components = leading_components + trailing_components
        if not components:
            logger.warning(
                f"No text or model compatible media provided for {self.record_id}",
            )
            return None

        if role == "user":
            message = UserMessage(content=self._fulltext, source=source)
        else:
            message = AssistantMessage(content=self._fulltext, source=source)
        return message
