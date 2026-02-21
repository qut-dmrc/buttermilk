"""Pipeline source that generates character-based prompts.

This module provides CharacterPromptSource, which yields BaseRecord objects
containing formatted prompts combining character descriptions (filtered by
mask_attributes) with scenarios.
"""

import uuid
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
        identity: List of identity strings to use directly (e.g., ["a lesbian", "a gay man"]).
                  If provided, overrides random character generation.
        mask_attributes: List of character attributes to keep (all others removed).
                         Only used when identity is NOT provided.
        scenarios: List of scenario descriptions to combine with characters.
                   If not provided, uses CharacterGenerator.generate_scenarios().
    """

    identity: list[str] | None = None
    mask_attributes: list[str] | None = None
    scenarios: list[str] | None = None

    async def __aiter__(self) -> AsyncIterator[BaseRecord]:
        """Yield BaseRecord for each (identity, scenario) combination.

        Two modes:
        1. Identity mode: Use provided identity strings directly
        2. Random mode: Generate random character and filter by mask_attributes

        If scenarios not provided, generates them using CharacterGenerator.

        Uses bm.session_info.session_id for session tracking.

        Yields:
            BaseRecord with prompt in content field and metadata including
            session_id, scenario, character (if random mode), and indices.
        """
        gen = CharacterGenerator()

        # Get scenarios (either provided or generated)
<<<<<<< HEAD
        scenarios = self.scenarios if self.scenarios is not None else gen.generate_scenarios()
=======
        scenarios = (
            self.scenarios if self.scenarios is not None else gen.generate_scenarios()
        )
>>>>>>> origin/stable

        # Get session_id (with fallback for testing)
        try:
            session_id = bm.session_info.session_id
        except RuntimeError:
            # Fallback for tests where bm is not initialized
            session_id = str(uuid.uuid4())

        # MODE 1: Identity strings provided - use them directly
        if self.identity is not None:
            record_idx = 0
            for identity_idx, identity_str in enumerate(self.identity):
                for scenario_idx, scenario in enumerate(scenarios):
                    prompt = gen._format_character_with_scenario(identity_str, scenario)

                    yield BaseRecord(
                        record_id=f"{session_id}_identity{identity_idx}_scenario{scenario_idx}",
                        content=prompt,
                        metadata={
                            "session_id": session_id,
                            "scenario": scenario,
                            "identity": identity_str,
                            "identity_index": identity_idx,
                            "scenario_index": scenario_idx,
                        },
                    )
                    record_idx += 1

        # MODE 2: Random character generation with filtering
        else:
            if self.mask_attributes is None:
<<<<<<< HEAD
                raise ValueError("Either 'identity' or 'mask_attributes' must be provided")
=======
                raise ValueError(
                    "Either 'identity' or 'mask_attributes' must be provided"
                )
>>>>>>> origin/stable

            char = gen.generate_identity()
            filtered = gen.reverse_mask(char, keep=self.mask_attributes)

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
