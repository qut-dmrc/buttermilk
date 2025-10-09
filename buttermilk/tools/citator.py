from typing import Any

import pydantic
from pydantic import BaseModel

from buttermilk._core.contract import AgentInput
from buttermilk._core.log import logger
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
        # Set defaults for agent configuration
        kwargs["agent_id"] = kwargs.get("agent_id", "citator")
        kwargs["description"] = kwargs.get("description", "Generates a citation for a given text using an LLM.")

        # Set the expected output model for the LLM's response
        output_model = output_model or FormattedCitation

        # Initialize parent class with all kwargs
        super().__init__(output_model=output_model, parameters=kwargs)

    async def process(self, item: Record, *, processor_stage: str = "cite", **kwargs) -> Record | None:
        """
        Process a Record to generate a citation using the LLM.
        Args:
            item (Record): The Record containing the text to cite.
            processor_stage: Stage name for metadata tracking (default: "cite")
            **kwargs: Additional keyword arguments (ignored, for compatibility)
        Returns:
            Record | None: The updated Record with the generated citation or None if processing failed.
        """

        # Take the first N characters for citation generation
        citation_text = item.content[:CITATION_TEXT_CHAR_LIMIT]

        input_data = AgentInput(
            inputs={"text_extract": citation_text}
        )
        try:
            result = await self.invoke(input_data)
            if not result or not result.outputs:
                logger.error(f"No outputs from LLM for item {item.record_id}")
                return None

            # Store it in the metadata (overwrites if 'citation' key already exists)
            if result.outputs.citation:
                item.metadata["citation"] = result.outputs.citation
            if result.outputs.title:
                item.metadata["title"] = result.outputs.title
            logger.debug(
                f"Generated citation for doc {item.record_id}: '{result.outputs.citation[:100]}.'",
            )
            return item
        except Exception as e:
            logger.error(
                f"Error generating citation for doc {item.record_id}: {e} {e.args=}",
            )
            item.metadata["citation"] = f"Error generating citation: {str(e)}"
            return item
