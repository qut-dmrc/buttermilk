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
from buttermilk._core.contract import AgentTrace, ErrorEvent  # Buttermilk contract types
from buttermilk.agents.llm import LLMAgent  # Base LLM Agent
from buttermilk.utils.media import download_and_convert  # Utility for media handling


class MediaDescription(BaseModel):
    """Structured output for media descriptions."""
    
    description: str = Field(
        ...,
        description="The textual description of the media content (alt text, caption, or transcript)"
    )
    media_type: str = Field(
        ...,
        description="Type of media described (image, video, audio, text)"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence level in the description accuracy"
    )


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
        _output_model: Set to MediaDescription for structured outputs

    """
    
    # Force structured output for descriptions
    _output_model: type[BaseModel] | None = MediaDescription

    async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentTrace | ErrorEvent:
        """Process the input to generate a media description.

        This method checks if the record already has alt text or if it's purely
        textual content. If neither, it downloads media from a URI if needed,
        and invokes the LLM to generate a description.

        Args:
            message: The input message containing the record to describe.
            **kwargs: Additional keyword arguments passed to the LLM agent.

        Returns:
            AgentTrace: Contains the generated description or an appropriate
                message if no description was needed.
            ErrorEvent: If an error occurs during processing.

        """
        if not message.records:
            return ErrorEvent(
                source=self.agent_id,
                error="No records provided for description.",
                error_code="NO_RECORDS",
            )

        # Get the record to describe
        record = message.records[0]  # Use the first record if multiple

        # Check if alt_text already exists in metadata
        if (
            hasattr(record, "metadata")
            and isinstance(record.metadata, dict)
            and record.metadata.get("alt_text")
        ):
            logger.info(f"Record already has alt_text: {record.metadata['alt_text'][:50]}...")
            # Return structured output even for existing alt text
            existing_description = MediaDescription(
                description=record.metadata["alt_text"],
                media_type="unknown",  # We don't know the original media type
                confidence=1.0
            )
            return AgentTrace(
                agent_id=self.agent_id,
                agent_type="describer",
                agent_name=self.agent_name or "Describer",
                content=existing_description.description,
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
                else:
                    return ErrorEvent(
                        source=self.agent_id,
                        error="Record has no media or text content to describe.",
                        error_code="NO_CONTENT",
                    )
            # Process media content
            return await self._process_media(message, record, **kwargs)
        elif record.text:
            # No media, just text
            return self._create_text_response(record)
        else:
            # Neither media nor text
            return ErrorEvent(
                source=self.agent_id,
                error="Record has no content to describe.",
                error_code="NO_CONTENT",
            )

    def _create_text_response(self, record: Any) -> AgentTrace:
        """Create a response for text-only records."""
        text_description = MediaDescription(
            description=f"This is a text-only record. Content: {record.text[:200]}...",
            media_type="text",
            confidence=1.0
        )
        return AgentTrace(
            agent_id=self.agent_id,
            agent_type="describer",
            agent_name=self.agent_name or "Describer",
            content=text_description.description,
            outputs=text_description,
            metadata={"media_type": "text", "skipped_reason": "text_only"},
        )

    async def _process_media(self, message: AgentInput, record: Any, **kwargs: Any) -> AgentTrace | ErrorEvent:
        """Process media content and generate description."""
        # Check if we need to download from URI
        if hasattr(record, "uri") and record.uri and not record.media:
            logger.info(f"Downloading media from URI: {record.uri}")
            try:
                # Import here to avoid circular imports
                from buttermilk.utils.media import download_and_convert

                downloaded_media = await download_and_convert(record.uri)
                if downloaded_media:
                    record.media = downloaded_media
                else:
                    return ErrorEvent(
                        source=self.agent_id,
                        error=f"Failed to download media from URI: {record.uri}",
                        error_code="DOWNLOAD_FAILED",
                    )
            except Exception as e:
                logger.error(f"Error downloading media from {record.uri}: {e}", exc_info=True)
                return ErrorEvent(
                    source=self.agent_id,
                    error=f"Failed to download media: {str(e)}",
                    error_code="DOWNLOAD_ERROR",
                )

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
        # due to _output_model setting
        return result