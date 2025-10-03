"""Core data types and Pydantic models used throughout the Buttermilk framework.

This module defines fundamental data structures such as `Record` for representing
individual data items and `RunRequest` for encapsulating parameters for initiating
orchestrator runs. These types ensure consistent data handling and interfaces across
different components of Buttermilk.
"""

import datetime
import json  # For JSON parsing in validators
from collections.abc import Sequence  # For type hinting sequences
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Self  # Standard typing utilities

import shortuuid  # For generating short unique IDs

from autogen_core.models import AssistantMessage, UserMessage  # Autogen message types

from PIL.Image import Image  # For image manipulation with Pillow
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,  # For fields computed from other fields
    field_validator,  # For custom field validation
    model_validator,  # For model-level validation
)

# Conditional imports to avoid circular dependencies


class BaseRecord(BaseModel):
    """Base class for all records in pipelines and storage.

    This base model defines the common fields and validation logic for any record type
    that flows through pipelines or is stored in Buttermilk storage.
    """

    record_id: str = Field(
        default_factory=lambda: str(shortuuid.ShortUUID().uuid()),
        description="Unique identifier for the record.",
    )
    dataset_name: str | None = Field(default=None, description="Name of the dataset this record belongs to.")
    split_type: str | None = Field(default=None, description="Dataset split this record belongs to, e.g., 'train', 'test'.")
    content: str | None = Field(default=None, description="Main content of the record.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata associated with the record.",
    )
    error: list[Any] = Field(
        default_factory=list,
        description="List of errors or error events associated with this record.",
    )

    @field_validator("metadata", mode="before")
    @classmethod
    def parse_metadata(cls, v):
        """Parse metadata from JSON string if needed."""
        if isinstance(v, str):
            try:
                return json.loads(v) if v else {}
            except json.JSONDecodeError:
                # If it's not valid JSON, return empty dict
                return {}
        elif v is None:
            return {}
        return v

    @field_validator("error", mode="before")
    @classmethod
    def parse_error(cls, v):
        """Parse error from JSON string if needed."""
        if isinstance(v, str):
            try:
                return json.loads(v) if v else []
            except json.JSONDecodeError:
                # If it's not valid JSON, return empty list
                return []
        elif v is None:
            return []
        return v

    @model_validator(mode="before")
    @classmethod
    def set_defaults_for_required_fields(cls, values):
        """Set default values for required BaseRecord fields if missing.

        This runs before field validation to ensure required fields have defaults.
        """
        if isinstance(values, dict):
            # Convert record_id to string if it's an integer (common from BigQuery)
            if "record_id" in values and values["record_id"] is not None:
                values["record_id"] = str(values["record_id"])
            elif not values.get("record_id"):
                values["record_id"] = str(shortuuid.ShortUUID().uuid())

            # These can remain None but ensure the keys exist
            if "dataset_name" not in values:
                values["dataset_name"] = None
            if "split_type" not in values:
                values["split_type"] = None

            # Ensure error field exists as empty list if not present
            if "error" not in values:
                values["error"] = []

        return values

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for extensibility
        arbitrary_types_allowed=True,
        populate_by_name=True,
        frozen=True,  # Make records immutable
    )

    def as_message(self, role: Literal["user", "assistant"] = "user") -> Any:
        """Converts the `Record` into an Autogen `UserMessage` or `AssistantMessage`.

        Args:
            role (Literal["user", "assistant"]): The role to assign to the
                resulting message. Defaults to "user".

        Returns:
            Autogen message object populated with the record's content and ID.
            Returns None if Autogen is not available.

        """
        if role == "assistant":
            # Assistant message content should likely be string representation
            return AssistantMessage(content=self.as_markdown(), source=self.record_id)

        return UserMessage(content=self.as_markdown(), source=self.record_id)

    def as_text(self) -> str:
        """Unified text access for vector processing.

        Returns the best available text representation:
        1. content (if it's a string) - main content field
        2. alt_text (as fallback for non-text content)
        3. string representation of content (for multimodal)

        Returns:
            str: Text content suitable for vector processing.

        """
        return str(self.content)

    def as_markdown(self) -> str:
        """Combines metadata and text content into a single string.

        Returns:
            str: A string combining key information from the record.

        """
        parts: list[str] = []

        if self.metadata:
            for key, value in self.metadata.items():
                parts.append(f"**{key}**: {value!s}")

        # Handle content based on its type
        if self.content and isinstance(self.content, str):
            parts.append(self.content)
        elif isinstance(self.content, Sequence):  # It's a list of parts
            parts.append("\n".join(self.content))

        return "\n\n".join(p for p in parts if p)  # Join non-empty parts

    @computed_field
    @property
    def record_hash(self) -> str:
        """Computes SHA256 hash of the record's as_markdown() output.

        This enables detection of record changes by comparing hash values.
        The hash is automatically accessible via metadata['record_hash'].

        Returns:
            str: SHA256 hexdigest of the record's markdown representation.
        """
        from buttermilk._core.hashing import compute_record_hash

        markdown_content = self.as_markdown()
        hash_value = compute_record_hash(markdown_content)
        # Store in metadata for easy access
        self.metadata["record_hash"] = hash_value
        return hash_value



class Record(BaseRecord):
    """Represents a single data record within the Buttermilk framework.

    A `Record` encapsulates a piece of data to be processed, along with its
    metadata, alternative text representations, ground truth information (if any),
    and a unique identifier. It can handle both simple text content and more
    complex, potentially multimodal content (e.g., sequences of text and images).

    Inherits from BaseRecord:
        record_id (str): A unique identifier for the record. Defaults to a
            new short UUID.
        dataset_name (str): The name of the dataset this record belongs to.
        split_type (str | None): The dataset split this record belongs to, e.g., "train",
            "test", or "validation".
        metadata (dict[str, Any]): A dictionary for storing arbitrary metadata
            associated with the record (e.g., source, title, creation date, tags, uri).
        error (list[Any]): List of errors or error events associated with this record.

    Additional Attributes:
        alt_text (str | None): A textual description or transcript of the media
            objects contained in this record, especially useful for non-text content.
        ground_truth (dict | None): Optional dictionary containing ground truth
            data associated with this record, for evaluation or reference.
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

    """
    alt_text: str | None = Field(
        default=None,
        description="Textual description or transcript of media content within this record.",
    )
    ground_truth: dict[str, Any] | list[str | dict[str, str]] | str | None = Field(  # Added type hint for dict value
        default=None,
        description="Optional ground truth data associated with this record for evaluation.",
    )
    content: str | Sequence[str | Image] | None = Field(
        default=None,
        description="Main content of the record: a string, or a sequence of strings and Pillow Images.",
    )
    mime: str | None = Field(
        default="text/plain",
        description="Primary MIME type of the content.",
    )

    # Vector processing fields (optional, for enhanced functionality)
    file_path: str | None = Field(
        default=None,
        description="Source file path for the record (used in vector processing).",
    )
    chunks: list[Any] = Field(
        default_factory=list,
        description="Vector chunks created from this record (lazy-loaded).",
    )
    chunks_path: str | None = Field(
        default=None,
        description="Path to PyArrow file containing chunks and embeddings.",
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
        return images_found or None

    @computed_field
    @property
    def ground_truth_hash(self) -> str | None:
        """Computes SHA256 hash of the ground_truth values if they exist.
        
        This enables detection of ground truth changes by comparing hash values.
        The hash is automatically accessible via metadata['ground_truth_hash'].
        
        Returns:
            str | None: SHA256 hexdigest of ground_truth data, or None if no ground_truth.
        """
        from buttermilk._core.hashing import compute_ground_truth_hash
        
        hash_value = compute_ground_truth_hash(self.ground_truth)
        # Store in metadata for easy access
        self.metadata["ground_truth_hash"] = hash_value
        return hash_value

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Custom model_dump that excludes computed fields by default.
        """
        # Get current exclude set
        current_exclude = kwargs.get("exclude", set())
        if current_exclude is None:
            current_exclude = set()
        elif isinstance(current_exclude, (list, tuple)):
            current_exclude = set(current_exclude)
        elif not isinstance(current_exclude, set):
            current_exclude = {current_exclude}

        # Add computed fields to exclusion for simple text content
        if isinstance(self.content, str):
            current_exclude.update({"title", "images", "record_hash", "ground_truth_hash"})

        kwargs["exclude"] = current_exclude
        return super().model_dump(**kwargs)

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
        exclude={"title", "images", "record_hash", "ground_truth_hash"},  # Exclude computed properties from model_dump
        # positional_args=True, # Removed as it's less common and can be ambiguous
    )

    @field_validator("ground_truth", mode="before")
    @classmethod
    def parse_ground_truth(cls, v):
        """Parse ground_truth from JSON string if needed."""
        if isinstance(v, str):
            try:
                return json.loads(v) if v else None
            except json.JSONDecodeError:
                # If it's not valid JSON, keep as string
                return v
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v):
        """Validate that content is not empty or None."""
        if v is None:
            raise ValueError("Content cannot be None - Record must have meaningful content")

        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Content cannot be empty string - Record must have meaningful content")
        elif isinstance(v, Sequence):
            if not v:
                raise ValueError("Content sequence cannot be empty - Record must have meaningful content")
            # Check that at least one item in sequence is meaningful
            has_meaningful_content = False
            for item in v:
                if (isinstance(item, str) and item.strip()) or not isinstance(item, str):
                    has_meaningful_content = True
                    break
            if not has_meaningful_content:
                raise ValueError("Content sequence must contain at least one meaningful item")

        return v

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

    @property
    def title(self) -> str | None:
        """Convenience property to access the 'title' from the `metadata` dictionary.

        Returns:
            str | None: The value of `self.metadata['title']` if it exists,
            otherwise `None`.

        """
        return self.metadata.get("title")

    def as_message(self, role: Literal["user", "assistant"] = "user") -> Any:
        """Converts the `Record` into an Autogen `UserMessage` or `AssistantMessage`.

        This is useful for integrating Buttermilk records directly into Autogen
        conversational flows.

        Args:
            role (Literal["user", "assistant"]): The role to assign to the
                resulting message. Defaults to "user".

        Returns:
            Autogen message object populated with the record's content and ID.
            Returns None if Autogen is not available.

        """
        if role == "assistant":
            # Assistant message content should likely be string representation
            return AssistantMessage(content=self.as_markdown(), source=self.record_id)

        # For user messages, content can be str or List[Union[str, Dict]] (for multimodal)
        message_content: str | list[Any]  # Use Any for list items to match Autogen's expectation for multimodal

        if isinstance(self.content, str):
            message_content = self.content
        elif isinstance(self.content, Sequence):
            processed_parts: list[Any] = []
            for item in self.content:
                if isinstance(item, (str, Image)):
                    processed_parts.append(item)
            message_content = processed_parts
        else:  # Fallback if content type is unexpected
            message_content = str(self.content)

        return UserMessage(content=message_content, source=self.record_id)  # type: ignore

    @classmethod
    async def from_uri(cls, uri: str) -> Self:
        """Fetches a record based on `record_id`, `uri`, or a `prompt` containing a URI/ID.

        Args:
            uri (str | None): The URI (URL, file path) to fetch and convert.

        Returns:
            The uri content converted to a `Record` object.

        If the URI is a web page, it will be stripped of extraneous text.

        Raises:
            ProcessingError: If no record could be found or fetched.

        """
        from buttermilk._core.exceptions import ProcessingError
        from buttermilk.utils.media import download_and_convert  # Media utilities

        record: Record | None = None  # uri case
        record = await download_and_convert(uri)
        if record:  # Check if download_and_convert succeeded
            # Ensure metadata exists and add provenance
            if not record.metadata:
                record.metadata = {}
            record.metadata["fetch_source_uri"] = uri
            record.metadata["fetch_timestamp_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
            return record

        raise ProcessingError(f"Record not found for URI: {uri}")


# --- Flow Protocol Start signal ---
class RunRequest(BaseModel):
    """Input object to initiate an orchestrator run (a "flow").

    This Pydantic model defines the unified request structure for all Buttermilk
    entry points, including CLI interactions, API calls, and batch processing jobs.
    It encapsulates all necessary information to start and manage a flow execution.

    Attributes:
        flow (str): The name of the flow to be executed. This is a mandatory field.
        session_id (str): A unique identifier for this specific flow execution
            session. Defaults to a new short UUID.
        parameters (dict): A dictionary of parameters to customize the flow's execution.
        inputs (dict): A dictionary of inputs sent to agents.
            Common parameters include:
            - prompt (str | None): The main prompt, question, or instruction for the run.
            - record_id (str | None): An optional ID of a specific record to look up and process.
            - uri (str | None): An optional URI (e.g., URL, file path) from which to fetch initial data.
            - records (list[Record]): A list of `Record` objects to be used as input for the flow.
            Flow-specific parameters can also be included. Defaults to an empty dict.
        callback_to_ui (Any | None): An optional callback function to send updates
            or messages back to a UI. Excluded from serialization.
        batch_id (str | None): If this run is part of a larger batch, this field
            holds the ID of the parent batch.
        source (list[str]): List of source identifiers, potentially for API requests
            indicating data origins. Defaults to an empty list.
        mime_type (str | None): MIME type for input data, especially if `data` field is used.
        data (bytes | None): Raw binary data input, e.g., for file uploads via API.
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
    session_id: str = Field(
        default_factory=shortuuid.uuid,  # Use factory for dynamic default
        description="A unique session ID for this specific flow execution.",
    )
    parameters: dict[str, Any] = Field(  # Added type hint for dict value
        default_factory=dict,
        description="Additional parameter overrides to customize flow execution.",
    )
    inputs: dict[str, Any] = Field(  # Added type hint for dict value
        default_factory=dict,
        description="Additional inputs to pass to agents, potentially including 'prompt', 'record_id', 'uri', 'records', and other request specific data.",
    )

    # Fields for client interaction, typically excluded from persisted state
    callback_to_ui: Any | None = Field(
        default=None,
        exclude=True,
        description="Optional callback function for sending updates to a UI.",
    )

    # Batch processing specific fields
    batch_id: str | None = Field(
        default=None,
        description="ID of the parent batch, if this run is part of a batch.",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat(),
        description="Timestamp (ISO format UTC) when the run request was created.",
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
        (`record_id` in parameters is set), it creates a composite ID: "{batch_id}:{record_id}".
        Otherwise, it generates a new short UUID.

        Returns:
            str: The unique job identifier.

        """
        record_id = self.inputs.get("record_id")
        if self.batch_id and record_id:
            return f"{self.batch_id}:{record_id}"
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
            **self.inputs,  # Include all custom parameters
            "session_id": self.session_id,
            "flow_name": self.flow,
            "run_request_name": self.name,  # Use the computed name of the run request
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
        if record_id := self.inputs.get("record_id"):  # Get from parameters
            parts.append(record_id)
        if criteria := self.inputs.get("criteria"):  # Safely get 'criteria'
            parts.append(str(criteria))

        # Join non-empty, non-None stringified parts
        display_name = " ".join(str(part) for part in parts if part is not None and str(part).strip())
        return display_name.strip() if display_name else "UnnamedRunRequest"


class ProcessingResult(BaseModel):
    """Comprehensive result from record processing."""

    record: Record | None
    status: Literal["processed", "skipped", "failed"]
    reason: str = Field(default="", description="Reason for skipping or failure")
    chunks_created: int = Field(default=0, description="Number of chunks created during processing")
    embedding_model: str = Field(default="n/a", description="Embedding model used for processing")
    processing_time_ms: float = Field(default=-1, description="Time taken to process the record in milliseconds")
    metadata: dict[str, Any] = Field(default={}, description="Additional metadata about the processing result")


@dataclass
class BatchProcessingResult:
    """Result from batch processing operations."""

    total_records: int
    successful_count: int
    skipped_count: int
    failed_count: int
    processing_time_ms: float
    validation_result: dict[str, Any] | None
    failed_records: list[tuple[str, str]]  # (record_id, error_message)
    metadata: dict[str, Any]
