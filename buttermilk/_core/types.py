import asyncio
import base64
import datetime
import platform
from collections.abc import Sequence
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Literal, Self

import psutil
import pydantic
import shortuuid
from autogen_core.models import AssistantMessage, UserMessage
from cloudpathlib import AnyPath, CloudPath
from PIL.Image import Image
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

    # @pydantic.model_validator(mode="after")
    # def schedule_get_ip(self) -> Self:
    #     self._get_ip_task = asyncio.create_task(self.get_ip())
    #     return self

    async def get_ip(self):
        if not self.ip:
            from ..utils import get_ip

            self.ip = await get_ip()


class MediaObj(BaseModel):
    label: str | None = Field(
        default=None,
        description="Section or type of content part (e.g. heading, body paragraph, caption, image, etc)",
    )

    metadata: dict = {}

    uri: str | None = Field(
        default=None,
        description="Media objects that can be passed as URIs to cloud storage",
    )

    mime: str | None = Field(
        default="text/plain",
    )

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
        extra="ignore",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        exclude_unset=True,
        exclude_none=True,
        exclude={"data", "base_64"},  # Use a set for exclude
    )

    def __str__(self) -> str:
        return f"{self.label or 'unknown'} object ({self.mime}) {self.as_text()[:50]}..."

    @model_validator(mode="after")
    def interpret_data(self) -> Self:
        if self.model_extra and (content := self.model_extra.get("content")):
            if isinstance(content, str):
                if self._base_64:
                    raise ValueError("MediaObj received two string fields.")

                if not self.mime:
                    self.mime = "text/plain"

                # String content, move
                if is_b64(content):
                    self._base_64 = str(content)
                else:
                    self._text = str(content)

            if isinstance(content, bytes):
                if self._base_64:
                    raise ValueError(
                        "MediaObj can have either bytes or base64 data, but not both.",
                    )
                self._base_64 = base64.b64encode(content).decode("utf-8")
                if not self.mime:
                    self.mime = "application/octet-stream"

            elif isinstance(content, Image):
                self._image = content

        return self

    def as_url(self):
        return (f"data:{self.mime};base64,{self._base_64}",)

    def as_image_url_message(self) -> dict:
        return {
            "type": "image_url",
            "image_url": {"url": self.uri},
        }

    def as_text(self) -> str:
        return str(self._text)

    def as_content_part(self, model_type="openai") -> dict | str:  # Fix return type hint
        part: dict | str = {}
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
    )

    uri: str | None = Field(
        default=None,
    )

    content: str | Sequence[str | Image] = Field()

    mime: str | None = Field(
        default="text/plain",
    )

    def __str__(self) -> str:
        return self.text

    @computed_field
    @property
    def images(self) -> list[Image] | None:
        images = []
        for x in self.content:
            if x and isinstance(x, Image):
                images.append(x)
        if images:
            return images
        return None

    @computed_field
    @property
    def text(self) -> str:
        """Combines metadata and text content into a single string.

        Excludes ground truth and component labels.
        """
        parts = []

        if self.metadata:
            if self.title:
                parts.append(f"### {self.title}")
            parts.append(f"**{self.record_id}**")
            for k, v in self.metadata.items():
                if k not in ["title", "fetch_timestamp_utc", "fetch_source_id"]:
                    parts.append(f"**{k}**: {v}")
            parts.append("")  # Separator

        if isinstance(self.content, str):
            parts.append(self.content)
        else:
            parts.append(self.alt_text)

        return "\n\n".join(parts)

    model_config = ConfigDict(
        extra="ignore",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        exclude_unset=True,
        exclude_none=True,
        exclude=["title", "text", "images"],
        positional_args=True,
    )  # type: ignore

    @model_validator(mode="after")
    def vld_input(self) -> Self:
        # Retrieve content if passed and place extra arguments in the metadata field
        if self.model_extra:
            while len(self.model_extra.keys()) > 0:
                key, value = self.model_extra.popitem()
                if key == "components":
                    pass

                if value and key not in self.metadata and key not in self.model_computed_fields:
                    self.metadata[key] = value
                else:
                    raise ValueError(
                        f"Received multiple values for {key} in Record",
                    )

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

    def as_message(self, role: Literal["user", "assistant"] = "user") -> UserMessage | AssistantMessage:
        if role == "assistant":
            # Assistant message content should likely be string representation
            return AssistantMessage(content=self.text, source=self.record_id)

        # Ensure content matches UserMessage expected type (str | List[str | Image])
        message_content: str | list[str | Image]
        if isinstance(self.content, str):
            message_content = self.content
        else:
            # Convert Sequence to List
            message_content = list(self.content)

        return UserMessage(content=message_content, source=self.record_id)


# --- Flow Protocol Start signal ---
class RunRequest(BaseModel):
    """Input object to initiate an orchestrator run.
    
    This is the unified request object for all entry points (CLI, API, batch jobs),
    containing fields for both interactive and batch processing.
    """

    # Common flow execution fields
    flow: str = Field(..., description="The name of the flow to execute")
    prompt: str | None = Field(default="", description="The main prompt or question for the run.", validation_alias=AliasChoices("prompt", "q"))
    record_id: str | None = Field(default="", description="Record to lookup")
    session_id: Any = Field(default=None, exclude=False)
    uri: str | None = Field(default="", description="URI to fetch")
    records: list[Record] = Field(default_factory=list, description="Input records, potentially including ground truth.")
    parameters: dict = Field(default_factory=dict, description="Additional parameters for flow execution")

    # Exclude; these fields are needed for some clients, but we don't need to keep them after initialisation
    callback_to_ui: Any = Field(..., exclude=True)
    ui_type: str = Field(..., description="Type of UI to use for this run", exclude=True)

    # Batch processing fields
    batch_id: str | None = Field(default=None, description="ID of the parent batch, if this is a batch job")
    status: str = Field(default="pending", description="Current status of this job (pending, running, completed, failed, cancelled)")
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat(), description="Creation timestamp")
    started_at: str | None = Field(default=None, description="Execution start timestamp")
    completed_at: str | None = Field(default=None, description="Execution completion timestamp")
    error: str | None = Field(default=None, description="Error message if job failed")
    result_uri: str | None = Field(default=None, description="URI to job results if available")

    # API-specific fields
    source: list[str] = Field(default_factory=list, description="Source identifiers")
    mime_type: str | None = Field(default=None, description="MIME type for input data")

    # Data could be passed in different formats
    data: bytes | None = Field(default=None, description="Binary data input")

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for extensibility
        populate_by_name=True,
    )

    @field_validator("prompt", mode="before")
    def sanitize_prompt(cls, v):
        if v and isinstance(v, str):
            v = v.strip()
            # Could add additional sanitization here
        return v

    @property
    def is_batch_job(self) -> bool:
        """Check if this is a batch job request."""
        return self.batch_id is not None

    @property
    def job_id(self) -> str:
        """Generate a unique job ID."""
        if self.batch_id and self.record_id:
            return f"{self.batch_id}:{self.record_id}"
        return shortuuid.uuid()

    # --- Tracing ---
    @computed_field
    def tracing_attributes(self) -> dict:

        metadata = {
            **self.parameters,
            "session_id": self.session_id,
            "flow_name": self.flow,
            "name": self.name,
            "record_id": self.record_id,
            "uri": self.uri,
            "batch_id": self.batch_id,
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}
        return metadata

    @computed_field
    def name(self) -> str:
        request_parts = [self.flow, self.parameters.get("record_id"), self.parameters.get("criteria"), self.session_id]
        request_parts = [str(part) for part in request_parts if part]
        display_name = " ".join(request_parts).strip()
        return display_name
