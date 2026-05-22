from collections.abc import AsyncGenerator
from typing import Any

import pydantic
from pydantic import BaseModel

from buttermilk._core.contract import AgentInput
from buttermilk._core.log import logger
from buttermilk._core.processing_context import ProcessingContext
from buttermilk._core.types import Record
from buttermilk.agents.llm import LLMAgent

CITATION_TEXT_CHAR_LIMIT = 4000  # characters


class FormattedCitation(BaseModel):
    """Model for formatted citation."""

    title: str = pydantic.Field(..., description="Title of the work being cited")
    citation: str = pydantic.Field(..., description="Formatted citation text")
    style: str = pydantic.Field(..., description="Citation style used (e.g., APA, MLA)")
    error: str | None = pydantic.Field(None, description="Error message if citation generation failed")


class Citator(LLMAgent):
    """Generates a citation for a given text using an LLM."""

    def __init__(self, output_model: type[pydantic.BaseModel] = None, **kwargs: Any):
        # Extract model and template from kwargs if not already in parameters
        if "parameters" not in kwargs:
            kwargs["parameters"] = {}

        # Move model and template into parameters if they're at the top level
        for key in ["model", "template"]:
            if key in kwargs and key not in kwargs["parameters"]:
                kwargs["parameters"][key] = kwargs.pop(key)

        # Set default template if not provided
        if "template" not in kwargs["parameters"]:
            kwargs["parameters"]["template"] = "citator"

        # Set defaults for agent configuration
        kwargs["agent_id"] = kwargs.get("agent_id", "citator")
        kwargs["description"] = kwargs.get("description", "Generates a citation for a given text using an LLM.")

        # Set the expected output model for the LLM's response
        output_model = output_model or FormattedCitation

        # Initialize parent class - kwargs are passed through to AgentConfig
        super().__init__(output_model=output_model, **kwargs)

    async def process(self, context: ProcessingContext) -> AsyncGenerator[Record, None]:
        """
        Process a Record to generate a citation using the LLM.

        Args:
            context: Unified processing context

        Yields:
            Record: The updated Record with the generated citation (or with error metadata if processing failed).
        """
        item = context.record
        processor_stage = context.session_id

        # Take the first N characters for citation generation
        citation_text = item.content[:CITATION_TEXT_CHAR_LIMIT]

        input_data = AgentInput(inputs={"text_extract": citation_text})
        try:
            result = await self.invoke(input_data)
            if not result or not result.outputs:
                logger.error(f"No outputs from LLM for item {item.record_id}")
                # Don't yield anything - filter out this record
                return

            # Store it in the metadata (overwrites if 'citation' key already exists)
            updated_metadata = item.metadata.copy() if item.metadata else {}
            if result.outputs.citation:
                updated_metadata["citation"] = result.outputs.citation
            if result.outputs.title:
                updated_metadata["title"] = result.outputs.title

            logger.debug(
                f"Generated citation for doc {item.record_id}: '{result.outputs.citation[:100]}.'",
            )

            # Yield the updated record
            yield item.model_copy(update={"metadata": updated_metadata})

        except Exception as e:
            logger.error(
                f"Error generating citation for doc {item.record_id}: {e} {e.args=}",
            )
            # Yield record with error metadata rather than filtering it out
            updated_metadata = item.metadata.copy() if item.metadata else {}
            updated_metadata["citation"] = f"Error generating citation: {e!s}"
            yield item.model_copy(update={"metadata": updated_metadata})
