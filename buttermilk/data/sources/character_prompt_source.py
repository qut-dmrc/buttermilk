"""Pipeline source that generates character-based prompts.

This module provides CharacterPromptSource, which yields BaseRecord objects
containing formatted prompts combining character descriptions (filtered by
mask_attributes) with scenarios.
"""

from typing import AsyncIterator

from pydantic import BaseModel
from shortuuid import ShortUUID

from buttermilk._core.types import BaseRecord
from buttermilk.tools.characters import CharacterGenerator


class CharacterPromptSource(BaseModel):
    """Pipeline source that generates character-based prompts.

    Yields BaseRecord objects containing formatted prompts combining
    character descriptions (filtered by mask_attributes) with scenarios.

    Attributes:
        mask_attributes: List of character attributes to keep (all others removed).
        scenarios: List of scenario descriptions to combine with characters.
        session_id: Unique identifier for this generation session (auto-generated if not provided).
    """

    mask_attributes: list[str]
    scenarios: list[str]
    session_id: str | None = None

    def model_post_init(self, __context: object) -> None:
        """Generate session_id if not provided."""
        if self.session_id is None:
            object.__setattr__(self, "session_id", ShortUUID().uuid()[:8])

    async def __aiter__(self) -> AsyncIterator[BaseRecord]:
        """Yield BaseRecord for each scenario with filtered character.

        Generates a single character identity, filters it to keep only
        specified attributes, then yields one BaseRecord per scenario
        combining the filtered character with each scenario.

        Yields:
            BaseRecord with prompt in content field and metadata including
            session_id, scenario, character, and scenario_index.
        """
        gen = CharacterGenerator()
        char = gen.generate_identity()
        filtered = gen.reverse_mask(char, keep=self.mask_attributes)

        for idx, scenario in enumerate(self.scenarios):
            prompt = gen._format_character_with_scenario(str(filtered), scenario)

            yield BaseRecord(
                record_id=f"{self.session_id}_scenario{idx}",
                content=prompt,
                metadata={
                    "session_id": self.session_id,
                    "scenario": scenario,
                    "character": filtered.model_dump(),
                    "scenario_index": idx,
                },
            )
