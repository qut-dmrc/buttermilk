"""Pipeline source that generates character-based prompts.

This module provides CharacterPromptSource, which yields BaseRecord objects
containing formatted prompts combining character descriptions (filtered by
mask_attributes) with scenarios.
"""

from typing import AsyncIterator

from pydantic import BaseModel

from buttermilk import bm
from buttermilk._core.types import BaseRecord
from buttermilk.tools.characters import CharacterGenerator


class CharacterPromptSource(BaseModel):
    """Pipeline source that generates character-based prompts.

    Yields BaseRecord objects containing formatted prompts combining
    character descriptions (filtered by mask_attributes) with scenarios.

    Attributes:
        mask_attributes: List of character attributes to keep (all others removed).
        scenarios: List of scenario descriptions to combine with characters.
                   If not provided, uses CharacterGenerator.generate_scenarios().
    """

    mask_attributes: list[str]
    scenarios: list[str] | None = None

    async def __aiter__(self) -> AsyncIterator[BaseRecord]:
        """Yield BaseRecord for each scenario with filtered character.

        Generates a single character identity, filters it to keep only
        specified attributes, then yields one BaseRecord per scenario
        combining the filtered character with each scenario.

        If scenarios not provided, generates them using CharacterGenerator.

        Uses bm.session_info.session_id for session tracking.

        Yields:
            BaseRecord with prompt in content field and metadata including
            session_id, scenario, character, and scenario_index.
        """
        gen = CharacterGenerator()
        char = gen.generate_identity()
        filtered = gen.reverse_mask(char, keep=self.mask_attributes)

        # Use provided scenarios or generate them
        scenarios = (
            self.scenarios if self.scenarios is not None else gen.generate_scenarios()
        )

        session_id = bm.session_info.session_id

        for idx, scenario in enumerate(scenarios):
            prompt = gen._format_character_with_scenario(str(filtered), scenario)

            yield BaseRecord(
                record_id=f"{session_id}_scenario{idx}",
                content=prompt,
                metadata={
                    "session_id": session_id,
                    "scenario": scenario,
                    "character": filtered.model_dump(),
                    "scenario_index": idx,
                },
            )
