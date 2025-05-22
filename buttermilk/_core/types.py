"""Core data types and Pydantic models used throughout the Buttermilk framework.

This module defines fundamental data structures such as `Record` for representing
individual data items, `MediaObj` for handling multimedia content, and `RunRequest`
for encapsulating parameters for initiating orchestrator runs. These types ensure
consistent data handling and interfaces across different components of Buttermilk.
"""

import base64
import datetime
from collections.abc import Sequence  # For type hinting sequences
from pathlib import Path  # For path manipulation
from typing import Any, Literal, Self  # Standard typing utilities

import shortuuid  # For generating short unique IDs
from autogen_core.models import AssistantMessage, UserMessage  # Autogen message types
from cloudpathlib import CloudPath  # For handling cloud storage paths
from PIL.Image import Image  # For image manipulation with Pillow
from pydantic import (
    AliasChoices,  # For field aliasing
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,  # For private attributes not part of the model schema
    computed_field,  # For fields computed from other fields
    field_validator,  # For custom field validation
    model_validator,  # For model-level validation
)

from buttermilk.utils.utils import is_b64  # Utility to check for base64 encoding


class MediaObj(BaseModel):
    """Represents a media object, which can be text, an image, or other binary data.

    This model is used to encapsulate different types of media content, providing
    methods to access the content in various formats (e.g., as text, data URL,
    or as a content part for specific LLM providers like OpenAI or Anthropic).

    Attributes:
        label (str | None): An optional label describing the section or type of
            this media content (e.g., "heading", "body_paragraph", "caption", "image").
        metadata (dict): A dictionary for storing arbitrary metadata associated
            with the media object. Defaults to an empty dict.
        uri (str | None): An optional URI (Uniform Resource Identifier) that points
            to the media content, especially if it's stored in cloud storage or
            accessible via a URL.
        mime (str | None): The MIME type of the media content (e.g., "text/plain",
            "image/jpeg", "application/pdf"). Defaults to "text/plain".
        _text (str | None): Private attribute storing the textual content if the
            media is text-based.
        _image (Image | None): Private attribute storing a Pillow `Image` object
            if the media is an image.
        _base_64 (str | None): Private attribute storing the base64 encoded string
            if the media is binary data or an image represented in base64.
        model_config (ConfigDict): Pydantic model configuration.
            - `extra`: "ignore" - Ignores extra fields during model parsing.
            - `arbitrary_types_allowed`: True.
            - `populate_by_name`: True.
            - `exclude_unset`: True.
            - `exclude_none`: True.
            - `exclude`: {"data", "base_64"} - Excludes these from serialization.

    """

    label: str | None = Field(
        default=None,
        description="Optional label for the type or section of content (e.g., 'heading', 'image').",
    )
    metadata: dict[str, Any] = Field(  # Added type hint for dict value
        default_factory=dict,  # Use factory for mutable default
        description="Arbitrary metadata associated with the media object.",
    )
    uri: str | None = Field(
        default=None,
        description="Optional URI for media content, especially if cloud-stored or URL-accessible.",
    )
    mime: str | None = Field(
        default="text/plain",  # Default MIME type if not specified
        description="MIME type of the media content (e.g., 'text/plain', 'image/jpeg').",
    )

    _text: str | None = PrivateAttr(default=None)
    _image: Image | None = PrivateAttr(default=None)
    _base_64: str | None = PrivateAttr(default=None)

    model_config = ConfigDict(
        extra="ignore",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        exclude_unset=True,
        exclude_none=True,
        exclude={"data", "base_64"},  # Exclude fields not directly part of the model
    )

    def __str__(self) -> str:
        """Returns a string representation of the MediaObj, showing its label, MIME type, and a snippet of its text content."""
        text_snippet = self.as_text()[:50] + "..." if self.as_text() and len(self.as_text()) > 50 else self.as_text()
        return f"{self.label or 'MediaObj'} ({self.mime or 'unknown/mime'}) - Content: '{text_snippet}'"

    @model_validator(mode="after")
    def interpret_data(self) -> Self:
        """Interprets raw content passed via `model_extra` (e.g., a 'content' field).

        This validator attempts to determine if the raw content is text, base64
        encoded data, raw bytes, or a Pillow `Image` object, and populates the
        appropriate private attributes (`_text`, `_base_64`, `_image`) and `mime` type.

        Returns:
            Self: The updated `MediaObj` instance.

        Raises:
            ValueError: If conflicting data types are provided (e.g., both text
                and base64 for the primary content).

        """
        if self.model_extra and (content := self.model_extra.get("content")):
            if isinstance(content, str):
                if self._base_64 and self._text:  # Check if both are already set
                    raise ValueError("MediaObj received conflicting string and base64 content.")

                if not self.mime or self.mime == "application/octet-stream":  # Default or if bytes were processed first
                    self.mime = "text/plain"  # Assume text if not specified or was generic bytes

                if is_b64(content):
                    if self._text:  # If text was already set, this is ambiguous
                        raise ValueError(
                            "MediaObj received string content that is also valid base64, and text was already set.",
                        )
                    self._base_64 = content
                    # Mime type for base64 should ideally be more specific if known, e.g., image/jpeg
                    if self.mime == "text/plain":  # If it was defaulted to text/plain but is b64
                        self.mime = "application/octet-stream"  # A generic default for b64
                else:
                    self._text = content

            elif isinstance(content, bytes):
                if self._base_64 or self._text or self._image:  # Check for conflicts
                    raise ValueError("MediaObj received bytes content but other content forms already exist.")
                self._base_64 = base64.b64encode(content).decode("utf-8")
                if not self.mime or self.mime == "text/plain":  # If no specific mime or defaulted to text
                    self.mime = "application/octet-stream"  # Default for raw bytes

            elif isinstance(content, Image):
                if self._image or self._text or self._base_64:  # Check for conflicts
                    raise ValueError("MediaObj received Image content but other content forms already exist.")
                self._image = content
                # Attempt to infer MIME type from image format if not already set or is generic
                if (not self.mime or self.mime in ["text/plain", "application/octet-stream"]) and self._image.format:
                    self.mime = Image.MIME.get(self._image.format.upper()) or self.mime

        return self

    def as_url(self) -> tuple[str] | None:  # Changed to return None if no base64
        """Returns the media content as a data URL (if base64 encoded).

        Returns:
            tuple[str] | None: A tuple containing the data URL string
            (e.g., "data:image/jpeg;base64,...") if `_base_64` is set.
            Returns `None` otherwise.

        """
        if self._base_64:
            return (f"data:{self.mime};base64,{self._base_64}",)
        return None

    def as_image_url_message(self) -> dict[str, Any] | None:
        """Formats the media as an image URL message, typically for LLM APIs, using its URI.

        Returns:
            dict[str, Any] | None: A dictionary structured for image URL messages
            (e.g., `{"type": "image_url", "image_url": {"url": self.uri}}`)
            if `self.uri` is set. Returns `None` otherwise.

        """
        if self.uri:
            return {
                "type": "image_url",
                "image_url": {"url": self.uri},
            }
        return None

    def as_text(self) -> str:
        """Returns the textual representation of the media content.

        Returns:
            str: The text content from `self._text`, or an empty string if None.

        """
        return str(self._text) if self._text is not None else ""

    def as_content_part(self, model_type: str = "openai") -> dict[str, Any] | str:
        """Formats the media object as a content part for multimodal LLM requests.

        Adapts the output format based on the specified `model_type` (e.g.,
        "openai" or "anthropic") to match their expected API structures for
        images or text.

        Args:
            model_type (str): The target LLM provider type. Currently supports
                "openai" and "anthropic". Defaults to "openai".

        Returns:
            dict[str, Any] | str: A dictionary representing the structured content part
            (typically for images) or a plain string (for text content).

        """
        part: dict[str, Any] | str
        if self._base_64 and self.mime and not self.mime.startswith("text"):  # Prioritize base64 for non-text
            if model_type == "openai":
                part = {
                    "type": "image_url",
                    "image_url": {"url": f"data:{self.mime};base64,{self._base_64}"},
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
            else:  # Fallback for unknown model_type with base64 data
                part = self.as_text() or f"[Unsupported base64 media for {model_type}]"
        elif self.uri and self.mime and not self.mime.startswith("text"):  # Then URI for non-text
            # OpenAI format is common for URI based images too
            part = self.as_image_url_message() or self.as_text()  # Fallback to text if URI message fails
        else:  # Default to text
            part = self.as_text()

        return part


class Record(BaseModel):
    """Represents a single data record within the Buttermilk framework.

    A `Record` encapsulates a piece of data to be processed, along with its
    metadata, alternative text representations, ground truth information (if any),
    and a unique identifier. It can handle both simple text content and more
    complex, potentially multimodal content (e.g., sequences of text and images).

    Attributes:
        record_id (str): A unique identifier for the record. Defaults to a
            new short UUID.
        metadata (dict[str, Any]): A dictionary for storing arbitrary metadata
            associated with the record (e.g., source, creation date, tags).
        alt_text (str | None): A textual description or transcript of the media
            objects contained in this record, especially useful for non-text content.
        ground_truth (dict | None): Optional dictionary containing ground truth
            data associated with this record, for evaluation or reference.
        uri (str | None): An optional URI pointing to the original source or
            location of the record's content.
        content (str | Sequence[str | Image]): The main content of the record.
            Can be a simple string (for text-only records) or a sequence of
            strings and Pillow `Image` objects for multimodal content.
        mime (str | None): The primary MIME type of the `content`. Defaults to
            "text/plain".
        images (list[Image] | None): A computed property that extracts all Pillow
            `Image` objects from `content` if it's a sequence.
        text (str): A computed property that generates a combined textual
            representation of the record, including its ID, selected metadata,
            and either the string `content` or `alt_text`.
        title (str | None): A computed property that retrieves the 'title' from
            `metadata`, if present.
        model_config (ConfigDict): Pydantic model configuration.

    """

    record_id: str = Field(
        default_factory=lambda: str(shortuuid.ShortUUID().uuid()),
        description="Unique identifier for the record.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,  # Use factory for mutable default
        description="Arbitrary metadata associated with the record.",
    )
    alt_text: str | None = Field(
        default=None,
        description="Textual description or transcript of media content within this record.",
    )
    ground_truth: dict[str, Any] | None = Field(  # Added type hint for dict value
        default=None,
        description="Optional ground truth data associated with this record for evaluation.",
    )
    uri: str | None = Field(
        default=None,
        description="Optional URI pointing to the original source of the record's content.",
    )
    content: str | Sequence[str | Image | MediaObj] = Field(  # Allow MediaObj in content
        description="Main content of the record: a string, or a sequence of strings, Pillow Images, or MediaObj.",
    )
    mime: str | None = Field(
        default="text/plain",
        description="Primary MIME type of the content.",
    )

    @computed_field
    @property
    def images(self) -> list[Image] | None:
        """Extracts all Pillow `Image` objects from the `content` attribute.

        Returns:
            list[Image] | None: A list of `Image` objects if any are found in
            `content` (when `content` is a sequence). Returns `None` if no images
            are found or if `content` is a simple string.

        """
        images_found: list[Image] = []
        if isinstance(self.content, Sequence) and not isinstance(self.content, str):
            for item in self.content:
                if isinstance(item, Image):
                    images_found.append(item)
                elif isinstance(item, MediaObj) and item._image:
                    images_found.append(item._image)
        return images_found or None

    def as_markdown(self) -> str:
        """Combines metadata and text content into a single string.

        Includes the record ID, selected metadata fields (excluding title,
        fetch timestamps/sources by default here), and the textual content
        (either direct string content or `alt_text` if content is complex).

        Returns:
            str: A string combining key information from the record.

        """
        parts: list[str] = []

        if self.metadata:
            if self.title:  # Add title first if it exists
                parts.append(f"### {self.title}")
            parts.append(f"**Record ID**: {self.record_id}")  # Make ID clear

            # Add other metadata items, perhaps filtering some common/internal ones
            for key, value in self.metadata.items():
                if key not in [
                    "title",
                    "fetch_timestamp_utc",
                    "fetch_source_id",
                    "components",
                ]:  # Exclude some common internal/structural keys
                    parts.append(f"**{key}**: {value!s}")
            if parts:  # Add a separator only if metadata was added
                parts.append("---")  # Separator after metadata block

        # Handle content based on its type
        if isinstance(self.content, str):
            parts.append(self.content)
        elif isinstance(self.content, Sequence):  # It's a list of parts
            text_parts_from_content: list[str] = []
            has_non_text = False
            for item in self.content:
                if isinstance(item, str):
                    text_parts_from_content.append(item)
                elif isinstance(item, MediaObj):
                    text_parts_from_content.append(item.as_text())
                    if item.mime and not item.mime.startswith("text/"):
                        has_non_text = True
                elif isinstance(item, Image):
                    has_non_text = True  # Mark that there's image content

            if text_parts_from_content:
                parts.append("\n".join(text_parts_from_content))

            if has_non_text and self.alt_text:  # If there was image content and alt_text is available
                parts.append(f"\n**Alternative Text for Media**: {self.alt_text}")
            elif has_non_text and not self.alt_text:
                parts.append("\n[Non-text content present, no alternative text provided]")

        elif self.alt_text:  # Fallback to alt_text if content is not a simple string and not handled above
            parts.append(self.alt_text)

        return "\n\n".join(p for p in parts if p)  # Join non-empty parts

    model_config = ConfigDict(
        extra="ignore",  # Ignore extra fields not defined in the model
        arbitrary_types_allowed=True,  # Allow types like PIL.Image
        populate_by_name=True,  # Allow population by field name or alias
        exclude_unset=True,  # Exclude fields not explicitly set during serialization
        exclude_none=True,  # Exclude fields with None values during serialization
        exclude={"title", "images"},  # Exclude computed properties from model_dump
        # positional_args=True, # Removed as it's less common and can be ambiguous
    )

    @model_validator(mode="after")
    def vld_input(self) -> Self:
        """Processes `model_extra` (fields not defined in the model schema).

        Any extra fields passed during instantiation (e.g., from a dictionary
        with more keys than the model defines) are moved into the `metadata`
        dictionary, unless the key is "components" (which might be handled
        elsewhere or ignored) or if it would overwrite an existing metadata key
        or a computed field.

        Returns:
            Self: The `Record` instance with `model_extra` processed.

        Raises:
            ValueError: If an extra field would overwrite an existing metadata
                key or conflicts with a computed field name.

        """
        if self.model_extra:
            extra_keys = list(self.model_extra.keys())  # Iterate over a copy of keys
            for key in extra_keys:
                value = self.model_extra.pop(key)  # Process and remove
                if key == "components":  # Specific handling for "components" if needed
                    # If 'components' should also go into metadata or be processed:
                    # self.metadata[key] = value
                    pass  # Currently, just passes over 'components'

                # Check for conflicts with existing metadata or computed fields
                elif key in self.metadata:
                    raise ValueError(f"Extra field '{key}' conflicts with existing metadata key in Record.")
                elif key in self.model_computed_fields:
                    raise ValueError(f"Extra field '{key}' conflicts with a computed field name in Record.")
                elif value is not None:  # Add to metadata if value is not None
                    self.metadata[key] = value
        return self

    @field_validator("uri")
    @classmethod
    def vld_path(cls, path: Any) -> str | None:  # path can be various types initially
        """Validates and normalizes the `uri` attribute.

        Converts `CloudPath` instances to their URI string representation and
        `Path` (local path) instances to their POSIX string representation.
        Other types are converted to string.

        Args:
            path: The input value for the `uri`.

        Returns:
            str | None: The normalized string representation of the URI/path,
            or `None` if the input `path` is None.

        """
        if path is None:
            return None
        if isinstance(path, CloudPath):
            return path.as_uri()
        if isinstance(path, Path):
            return path.as_posix()
        return str(path)  # Fallback to string conversion

    @property
    def title(self) -> str | None:
        """Convenience property to access the 'title' from the `metadata` dictionary.

        Returns:
            str | None: The value of `self.metadata['title']` if it exists,
            otherwise `None`.

        """
        return self.metadata.get("title")

    def as_message(self, role: Literal["user", "assistant"] = "user") -> UserMessage | AssistantMessage:
        """Converts the `Record` into an Autogen `UserMessage` or `AssistantMessage`.

        This is useful for integrating Buttermilk records directly into Autogen
        conversational flows.

        Args:
            role (Literal["user", "assistant"]): The role to assign to the
                resulting message. Defaults to "user".

        Returns:
            UserMessage | AssistantMessage: An Autogen message object populated
            with the record's content and ID.

        """
        if role == "assistant":
            # Assistant message content should likely be string representation
            return AssistantMessage(content=self.as_markdown, source=self.record_id)

        # For user messages, content can be str or List[Union[str, Dict]] (for multimodal)
        message_content: str | list[Any]  # Use Any for list items to match Autogen's expectation for multimodal

        if isinstance(self.content, str):
            message_content = self.content
        elif isinstance(self.content, Sequence):
            # Convert content parts to a format suitable for Autogen UserMessage
            # (e.g., text strings or dicts for images like OpenAI's format)
            processed_parts: list[Any] = []
            for item in self.content:
                if isinstance(item, str):
                    processed_parts.append(item)
                elif isinstance(item, Image):  # Convert PIL Image to OpenAI format part
                    # This requires base64 encoding and determining MIME type
                    # For simplicity, this example might just use alt_text or skip if complex
                    # A more robust solution would involve a MediaObj-like conversion here
                    if self.alt_text:  # Use alt_text if image directly in content
                        processed_parts.append(f"[Image: {self.alt_text or 'Untitled Image'}]")
                    else:  # Fallback if no alt_text for direct image
                        processed_parts.append("[Image Content]")
                elif isinstance(item, MediaObj):
                    # Use MediaObj's own conversion logic
                    # Assuming OpenAI format for Autogen UserMessage by default here
                    processed_parts.append(item.as_content_part(model_type="openai"))
            message_content = processed_parts
        else:  # Fallback if content type is unexpected
            message_content = str(self.content)

        return UserMessage(content=message_content, source=self.record_id)  # type: ignore


# --- Flow Protocol Start signal ---
class RunRequest(BaseModel):
    """Input object to initiate an orchestrator run (a "flow").

    This Pydantic model defines the unified request structure for all Buttermilk
    entry points, including CLI interactions, API calls, and batch processing jobs.
    It encapsulates all necessary information to start and manage a flow execution.

    Attributes:
        flow (str): The name of the flow to be executed. This is a mandatory field.
        prompt (str | None): The main prompt, question, or instruction for the run.
            Can be aliased as "q". Defaults to an empty string.
        record_id (str | None): An optional ID of a specific record to look up
            and process.
        session_id (str): A unique identifier for this specific flow execution
            session. Defaults to a new short UUID.
        uri (str | None): An optional URI (e.g., URL, file path) from which to
            fetch initial data or a record.
        records (list[Record]): A list of `Record` objects to be used as input
            for the flow. This can include ground truth data. Defaults to an empty list.
        parameters (dict): A dictionary of additional parameters to customize
            the flow's execution. Defaults to an empty dict.
        callback_to_ui (Any | None): An optional callback function to send updates
            or messages back to a UI. Excluded from serialization.
        ui_type (str): The type of UI initiating the run (e.g., "cli", "api", "streamlit").
            This is a mandatory field, excluded from serialization.
        batch_id (str | None): If this run is part of a larger batch, this field
            holds the ID of the parent batch.
        status (str): The current status of this job/run (e.g., "pending",
            "running", "completed", "failed", "cancelled"). Defaults to "pending".
        created_at (str): Timestamp (ISO format UTC) of when the run request was created.
        started_at (str | None): Timestamp (ISO format UTC) of when the execution started.
        completed_at (str | None): Timestamp (ISO format UTC) of when the execution completed.
        error (str | None): Error message if the job/run failed.
        result_uri (str | None): URI pointing to the results of the job/run, if applicable.
        source (list[str]): List of source identifiers, potentially for API requests
            indicating data origins. Defaults to an empty list.
        mime_type (str | None): MIME type for input data, especially if `data` field is used.
        data (bytes | None): Raw binary data input, e.g., for file uploads via API.
        is_batch_job (bool): Computed property, True if `batch_id` is set.
        job_id (str): Computed property, generates a unique job ID, combining
            `batch_id` and `record_id` if available, otherwise a new UUID.
        tracing_attributes (dict): Computed property, provides a dictionary of
            attributes relevant for tracing this run.
        name (str): Computed property, generates a descriptive name for the run
            based on flow name, record ID, and criteria parameters.
        model_config (ConfigDict): Pydantic model configuration.
            - `extra`: "allow" - Allows extra fields for extensibility.
            - `populate_by_name`: True.

    """

    # Common flow execution fields
    flow: str = Field(..., description="The name of the Buttermilk flow to execute.")
    prompt: str | None = Field(
        default="",
        description="The main prompt, question, or instruction for the run.",
        validation_alias=AliasChoices("prompt", "q"),  # Allows using 'q' as an alias
    )
    record_id: str | None = Field(
        default="",
        description="Optional ID of a specific record to look up and process.",
    )
    session_id: str = Field(
        default_factory=shortuuid.uuid,  # Use factory for dynamic default
        description="A unique session ID for this specific flow execution.",
    )
    uri: str | None = Field(
        default="",
        description="Optional URI (e.g., URL, file path) to fetch initial data or a record from.",
    )
    records: list[Record] = Field(
        default_factory=list,
        description="Input `Record` objects for the flow, potentially including ground truth.",
    )
    parameters: dict[str, Any] = Field(  # Added type hint for dict value
        default_factory=dict,
        description="Additional parameters to customize flow execution.",
    )

    # Fields for client interaction, typically excluded from persisted state
    callback_to_ui: Any | None = Field(
        default=None,
        exclude=True,
        description="Optional callback function for sending updates to a UI.",
    )
    ui_type: str = Field(
        ...,
        description="Type of UI initiating the run (e.g., 'cli', 'api'). Excluded from serialization.",
        exclude=True,
    )

    # Batch processing specific fields
    batch_id: str | None = Field(
        default=None,
        description="ID of the parent batch, if this run is part of a batch.",
    )
    status: str = Field(
        default="pending",
        description="Current status of this job (e.g., 'pending', 'running', 'completed').",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat(),
        description="Timestamp (ISO format UTC) when the run request was created.",
    )
    started_at: str | None = Field(
        default=None,
        description="Timestamp (ISO format UTC) when execution started.",
    )
    completed_at: str | None = Field(
        default=None,
        description="Timestamp (ISO format UTC) when execution completed.",
    )
    error: str | None = Field(default=None, description="Error message if the job/run failed.")
    result_uri: str | None = Field(
        default=None,
        description="URI pointing to the results of the job/run, if applicable.",
    )

    # API-specific fields (can be used by other interfaces too)
    source: list[str] = Field(
        default_factory=list,
        description="List of source identifiers, e.g., for API requests indicating data origins.",
    )
    mime_type: str | None = Field(
        default=None,
        description="MIME type for input data, especially if 'data' field is used.",
    )
    data: bytes | None = Field(
        default=None,
        description="Raw binary data input, e.g., for file uploads via API.",
    )

    model_config = ConfigDict(
        extra="allow",  # Allows extra fields not defined in the model, for extensibility
        populate_by_name=True,  # Allows population by field name or alias
    )

    @field_validator("prompt", mode="before")
    @classmethod
    def sanitize_prompt(cls, v: Any) -> str | None:  # Allow None to pass through
        """Sanitizes the `prompt` field by stripping leading/trailing whitespace.

        Args:
            v: The input value for the `prompt`.

        Returns:
            str | None: The sanitized prompt string, or None if input was None.

        """
        if v is None:
            return None
        if isinstance(v, str):
            return v.strip()
        return str(v)  # Attempt to convert other types to string

    @property
    def is_batch_job(self) -> bool:
        """Checks if this `RunRequest` instance represents a batch job.

        Returns:
            bool: True if `batch_id` is not None, indicating it's part of a batch.

        """
        return self.batch_id is not None

    @property
    def job_id(self) -> str:
        """Generates or returns a unique job identifier for this run.

        If part of a batch (`batch_id` is set) and processing a specific record
        (`record_id` is set), it creates a composite ID: "{batch_id}:{record_id}".
        Otherwise, it generates a new short UUID.

        Returns:
            str: The unique job identifier.

        """
        if self.batch_id and self.record_id:
            return f"{self.batch_id}:{self.record_id}"
        # Fallback to a new unique ID if not part of a batch or no specific record_id
        return shortuuid.uuid()

    @computed_field
    @property
    def tracing_attributes(self) -> dict[str, Any]:
        """Generates a dictionary of attributes suitable for tracing systems like Weave.

        Includes key parameters from the `RunRequest` that are useful for
        identifying and filtering traces. Excludes None values.

        Returns:
            dict[str, Any]: A dictionary of tracing attributes.

        """
        attributes = {
            **self.parameters,  # Include all custom parameters
            "session_id": self.session_id,
            "flow_name": self.flow,
            "run_request_name": self.name,  # Use the computed name of the run request
            "record_id": self.record_id,
            "uri": self.uri,
            "batch_id": self.batch_id,
        }
        # Filter out any attributes that are None to keep traces clean
        return {k: v for k, v in attributes.items() if v is not None}

    @computed_field
    @property
    def name(self) -> str:
        """Generates a descriptive name for this run request.

        The name is constructed from the flow name, record ID (if any), and
        the value of a "criteria" parameter (if present in `self.parameters`).
        This aims to create a human-readable identifier for the specific run.

        Returns:
            str: A descriptive name for the run request.

        """
        parts = [self.flow]
        if self.record_id:
            parts.append(self.record_id)
        if criteria := self.parameters.get("criteria"):  # Safely get 'criteria'
            parts.append(str(criteria))

        # Join non-empty, non-None stringified parts
        display_name = " ".join(str(part) for part in parts if part is not None and str(part).strip())
        return display_name.strip() if display_name else "UnnamedRunRequest"
