from typing import Self

import pydantic
from pydantic import BaseModel, PrivateAttr

from buttermilk._core.contract import AgentInput
from buttermilk.agents.llm import LLMAgent
from buttermilk.bm import logger
from buttermilk.data.vector import InputDocument

CITATION_TEXT_CHAR_LIMIT = 4000  # characters


class Citator(BaseModel):
    """Generates a citation for a given text using an LLM."""

    template: str = "citator"
    model: str
    _agent: LLMAgent = PrivateAttr()

    @pydantic.model_validator(mode="after")
    def _init_agent(self) -> Self:
        self._agent = LLMAgent(
            id="citator",
            role="Citator",
            description="Gets citation information from the first page or two.",
            parameters={"template": "citator", "model": self.model},
            inputs={"text_extract": "text_extract"},
            fail_on_unfilled_parameters=True,
        )
        return self

    async def process(self, item: InputDocument, **kwargs) -> InputDocument | None:
        try:
            # Take the first N characters for citation generation
            citation_text = item.full_text[:CITATION_TEXT_CHAR_LIMIT]
            input_data = AgentInput(
                agent_role="citator",
                agent_id="Citator",
                inputs={"text_extract": citation_text},
            )
            result = await self._agent(input_data, **kwargs)

            if not result or result.error:
                logger.error(
                    f"Error generating citation for doc {item.record_id}: {result.error}",
                )
                return item
            generated_citation = result.outputs["citation"]
            # Store it in the metadata (overwrites if 'citation' key already exists)
            item.metadata["citation"] = generated_citation
            logger.debug(
                f"Generated citation for doc {item.record_id}: '{generated_citation[:100]}...'",
            )
            return item
        except Exception as e:
            logger.error(
                f"Error generating citation for doc {item.record_id}: {e} {e.args=}",
            )
            return item
