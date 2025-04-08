import base64
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

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

from .log import logger
from .contract import UserMessage


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

    content: str | bytes | None = Field(
        default=None,
        validation_alias=AliasChoices("content", "data", "bytes"),
    )

    base_64: str | None = Field(
        default=None,
        validation_alias=AliasChoices("base_64", "base64", "b64", "b64_data"),
    )

    # move base64 representation out of the fields into a private attribute
    _b64: str = PrivateAttr(default=None)

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        exclude_unset=True,
        exclude_none=True,
        exclude=["data", "base_64"],
    )

    def __str__(self):
        return f"{self.label or 'unknown'} object ({self.mime}) {self.content[:30]} {self._b64[:30]}..."

    @model_validator(mode="after")
    def interpret_data(self) -> Self:
        if isinstance(self.content, str):
            if self.base_64:
                raise ValueError("MediaObj received two string fields.")

            if not self.mime:
                self.mime = "text/plain"

        if isinstance(self.content, bytes):
            if self.base_64:
                raise ValueError(
                    "MediaObj can have either bytes or base64 data, but not both.",
                )
            self.base_64 = base64.b64encode(self.content).decode("utf-8")
            if not self.mime:
                self.mime = "application/octet-stream"

            # Strip binary data away once it's converted to b64
            self.content = None
        else:
            # String content, leave as is
            pass

        if self.base_64:
            # move base64 representation out of the fields into a private attribute
            self._b64 = str(self.base_64)
            # del self.base_64

        return self

    def as_url(self):
        return (f"data:{self.mime};base64,{self._b64}",)

    def as_image_url_message(self) -> dict:
        return {
            "type": "image_url",
            "image_url": {"url": self.uri},
        }

    def as_text(self) -> dict:
        return {"type": "text", "text": self.content}

    def as_content_part(self, model_type="openai") -> dict:
        part = {}
        if self._b64:
            if model_type == "openai":
                part = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{self.mime};base64,{self._b64}",
                    },
                }
            elif model_type == "anthropic":
                part = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": self.mime,
                        "data": self._b64,
                    },
                }
        elif self.content:
            part = self.as_text()
        elif self.uri:
            part = self.as_image_url_message()

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
    data: Any | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "arg0",
            # for backwards compatibility
            "content",
            "image",
            "video",
            "media",
        ),
    )

    record_id: str = Field(default_factory=lambda: str(shortuuid.ShortUUID().uuid()))
    metadata: dict[str, Any] = Field(default={})
    alt_text: str | None = Field(
        default=None,
        description="Text description of media objects contained in this record.",
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
        return self.fulltext
    
 
    @computed_field
    @property
    def fulltext(self) -> str:
        """Combines metadata and text content into a single string.

        Excludes ground truth and component labels.
        """
        parts = []

        if self.metadata:
            parts.append("--- Metadata ---")
            for k, v in self.metadata.items():
                parts.append(f"{k}: {v}")
            parts.append("---")  # Separator

        component_texts = [part.content for part in self.components if part.content]
        if component_texts:
             parts.extend(component_texts)
             
        return "\n".join(parts)

    @computed_field
    @property
    def text(self) -> str:
        # Only text. Metadata and Ground truth not included
        return "\n".join(self.paragraphs)

    @computed_field
    @property
    def paragraphs(self) -> list[str]:
        # Only text. Metadata and Ground truth not included
        all_text = []
        for part in self.components:
            if part.content:
                all_text.append(part.content)

        return all_text

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        exclude_unset=True,
        exclude_none=True,
        exclude=["components", "fulltext"],
        positional_args=True,
    )  # type: ignore

    @model_validator(mode="after")
    def vld_input(self) -> Self:
        # Take data arguments and turn them into MediaObj components
        if isinstance(self.data, MediaObj):
            self.components.append(self.data)
        elif isinstance(self.data, str | bytes | Mapping):
            self.data = [self.data]
            for element in self.data:
                if isinstance(element, Mapping):
                    self.components.append(MediaObj(**element))
                elif isinstance(element, str):
                    self.components.append(MediaObj(content=element))
                else:
                    self.components.append(MediaObj(content=element))
        elif isinstance(self.data, list):
            for x in self.data:
                if isinstance(x, MediaObj):
                    self.components.append(x)
                else:
                    self.components.append(MediaObj(content=x))
        elif self.data:
            raise ValueError(f"Unknown component type: {type(self.data)}")

        # Place extra arguments in the metadata field
        if self.model_extra:
            while len(self.model_extra.keys()) > 0:
                key, value = self.model_extra.popitem()
                if key not in self.metadata and key not in self.model_computed_fields:
                    self.metadata[key] = value
                else:
                    raise ValueError(
                        f"Received multiple values for {key} in RecordInfo",
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
        all_text = [f"{k}: {v}" for k, v in self.metadata.items()]
        for part in self.components:
            if part.content:
                if part.label:
                    all_text.append(f"{part.label}: {part.content}")
                else:
                    all_text.append(part.content)
        return "\n".join(all_text)

    def update_from(self, update_dict) -> Self:
        # exclude null values
        update_dict = {k: v for k, v in update_dict.items() if v}

        self.metadata.update(**update_dict)

        return self

    def as_message(
        self,
        role: Literal["user", "human", "system", "assistant"] = "user",
    ) -> UserMessage:
        # Prepare input for model consumption
        leading_components = []
        trailing_components = []

        for obj in self.components:
            # attach media objects if the model supports them
            if (
                obj.mime.startswith("image")
                or obj.mime.startswith("video")
                or obj.mime.startswith("audio")
            ):
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
        
        message = UserMessage(content=self.fulltext, source=role)

        return message
