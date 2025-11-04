"""Provides the Describer agent for generating textual descriptions of media content.

This module defines the `Describer` agent, which is a specialized `LLMAgent`
designed to analyze media objects (images, videos, audio) within a `Record`
and generate a textual description (often an alt text or caption) using a
configured Language Model.
"""

from typing import Any  # For type hinting

from PIL.Image import Image  # For image type checking
from pydantic import BaseModel, Field

from buttermilk import logger  # Buttermilk's centralized logger
from buttermilk._core.agent import AgentInput  # Buttermilk AgentInput type
from buttermilk._core.contract import AgentOutput  # Buttermilk contract types
from buttermilk._core.exceptions import ProcessingError
from buttermilk.agents.llm import LLMAgent  # Base LLM Agent


class MediaDescription(BaseModel):
    """Structured output for media descriptions."""

    description: str = Field(
        ...,
        description="The textual description of the media content (alt text, caption, or transcript)",
    )
    media_type: str = Field(
        ...,
        description="Type of media described (image, video, audio, text)",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence level in the description accuracy",
    )

    def as_markdown(self, agent_id: str = None, call_id: str = None) -> str:
        """Returns a Markdown formatted string for insertion into templates.

        Format follows the standard: agent identifier on first line, followed by
        content-specific fields without empty lines between components.

        Args:
            agent_id: The agent identifier (e.g., "DESC-gpt4")
            call_id: The call identifier for this execution

        Returns:
            str: Formatted markdown string suitable for template insertion
        """
        header = ""
        if agent_id and call_id:
            # Use only the last 8 characters of call_id for brevity
            short_call_id = call_id[-8:] if len(call_id) > 8 else call_id
            header = f"**{agent_id} #{short_call_id}**\n"

        return f"{header}" f"{self.description}\n" f"Type: {self.media_type}\n" f"Confidence: {self.confidence:.2f}"

    def __str__(self) -> str:
        """Returns a Markdown formatted string representation.

        When agent context is available (via _agent_id and _call_id attributes),
        includes the full header. Otherwise returns just the description.
        """
        # Check if agent context is available (set by ExecutionTrace)
        agent_id = getattr(self, "_agent_id", None)
        call_id = getattr(self, "_call_id", None)

        if agent_id and call_id:
            return self.as_markdown(agent_id, call_id)

        # Fallback to just the description
        return self.description


class Describer(LLMAgent):
    """An agent that generates textual descriptions for media objects using an LLM.

    The `Describer` agent extends `LLMAgent` to specifically focus on tasks
    like creating alt text for images, transcribing audio, or summarizing video
    content. It checks if a description (`alt_text`) already exists or if the
    content is purely textual before invoking the LLM. It can also download
    media from a URI if necessary.

    Key Configuration Parameters (from `AgentConfig.parameters`):
        - `model` (str): **Required (inherited from LLMAgent)**. The name of the
          LLM to use for generating descriptions.
        - `template` (str): Name of the Jinja2 template to use for prompting.
          Defaults to "describer".
        - Additional parameters are passed to the LLM (e.g., `temperature`,
          `max_tokens`).

    Attributes:
        output_model: Set to MediaDescription for structured outputs

    """

    def __init__(self, **kwargs):
        """Initializes the Judge agent with its specific configuration and output model."""
        super().__init__(**kwargs)
        # Set the expected output model for the LLM's response
        self.output_model = MediaDescription

    async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput | None:
        """Process the input to generate a media description.

        This method checks if the record already has alt text or if it's purely
        textual content. If neither, it downloads media from a URI if needed,
        and invokes the LLM to generate a description.

        Args:
            message: The input message containing the record to describe.
            **kwargs: Additional keyword arguments passed to the LLM agent.

        Returns:
            AgentOutput | None: Contains the generated description or an appropriate
                message if no description was needed.

        Raises:
            ProcessingError: If no records provided or no content to describe.

        """
        if not message.record:
            raise ProcessingError("No record provided for description.")

        # Get the record to describe
        record = message.record

        # Check if alt_text already exists in metadata
        if hasattr(record, "metadata") and isinstance(record.metadata, dict) and record.metadata.get("alt_text"):
            logger.debug("Record already has alt_text", alt_text=record.metadata["alt_text"][:50])
            # Return structured output even for existing alt text
            existing_description = MediaDescription(
                description=record.metadata["alt_text"],
                media_type="unknown",  # We don't know the original media type
                confidence=1.0,
            )
            return AgentOutput(
                agent_id=self.agent_id,
                outputs=existing_description,
                metadata={"source": "existing_alt_text"},
            )

        # Check if the record is purely textual
        if record.media:
            # There's media content
            if isinstance(record.media, list) and not record.media:
                # Empty media list, fall back to text
                if record.text:
                    return self._create_text_response(record)
                raise ProcessingError("Record has no media or text content to describe.")
            # Process media content
            return await self._process_media(message, record, **kwargs)
        if record.text:
            # No media, just text
            return self._create_text_response(record)
        # Neither media nor text
        raise ProcessingError("Record has no content to describe.")

    def _create_text_response(self, record: Any) -> AgentOutput:
        """Create a response for text-only records."""
        logger.debug("Creating text-only response", record_id=record.id)
        raise NotImplementedError("Text-only response handling not implemented yet.")

    async def _process_media(self, message: AgentInput, record: Any, **kwargs: Any) -> AgentOutput | None:
        """Process media content and generate description."""
        # Check if we need to download from URI
        uri = record.metadata.get("uri") if hasattr(record, "metadata") else None
        if uri and not record.media:
            logger.debug("Downloading media from URI", uri=uri)
            try:
                # Import here to avoid circular imports
                from buttermilk.utils.media import download_and_convert

                downloaded_media = await download_and_convert(uri)
                if downloaded_media:
                    record.media = downloaded_media
                else:
                    raise ProcessingError(f"Failed to download media from URI: {uri}")
            except Exception as e:
                logger.error("Error downloading media", uri=uri, error=e)
                raise ProcessingError(f"Failed to download media: {e!s}") from e

        # Determine media type
        media_type = "unknown"
        if hasattr(record, "media") and record.media:
            if isinstance(record.media, list) and record.media:
                first_media = record.media[0]
                if isinstance(first_media, Image):
                    media_type = "image"
                # Could add more type detection here
            elif isinstance(record.media, Image):
                media_type = "image"

        # Now process with the parent LLMAgent's process method
        # which will use the template and model to generate a description
        result = await super()._process(message=message, **kwargs)

        # The parent's process should return structured MediaDescription
        # due to output_model setting
        return result
