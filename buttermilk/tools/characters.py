"""This is a very simplistic approach to generate synthetic characters
featuring a limited range of diverse characteristics.

It is NOT an authoritative or exhaustive approach to generating
authentic, multi-dimensional, or representative characters. It is
necessarily very reductive and should not be used without careful
reflection on the limitations of both the approach and the dataset
of characteristics.

This character generator focuses on a subset of 'protected
characteristics' that are commonly recognised in international
and domestic law. The list of identity features has been intentionally
SIMPLIFIED in order to meet the specific research requirements of
a project that examines gender and sexuality stereotypes in image
generation AI models. Its categories are deliberatrely REDUCTIVE,
as is the whole idea of describing people by any subset of their
protected characteristics. I really do not recommend the use of this
code in other projects.

-- NS Jan 2025
"""

import json
import random
from functools import cached_property
from typing import Literal, TypeAlias, get_type_hints

from pydantic import BaseModel

CHARACTERISTICS_FILE = "buttermilk/templates/image_prompt_limited_identities.json"


class ProtectedCharacteristics(BaseModel):
    age_group: str
    gender: str
    sexuality: str
    gender_presentation: str
    ethnicity: str
    disability: str
    body_type: str
    relationship_status: str
    citizenship_status: str
    religion: str
    social_class: str

    def __str__(self) -> str:
        # When presenting in English, try to follow a standard-ish order that
        # doesn't emphasise any particular characteristic.
        # ref: https://dictionary.cambridge.org/grammar/british-grammar/adjectives-order
        ordered_fields = [
            "age_group",
            "body_type",
            "ethnicity",
            "gender",
            "gender_presentation",
            "sexuality",
            "disability",
            "relationship_status",
            "citizenship_status",
            "religion",
            "social_class",
        ]
        character_desc = []
        for field in ordered_fields:
            if field in self.model_fields_set:
                character_desc.append(self.__getattribute__(field))

        return " ".join(character_desc)


# Get all attributes from ProtectedCharacteristics class
CHARACTERISTICS: TypeAlias = Literal[
    tuple(get_type_hints(ProtectedCharacteristics).keys())
]  # type: ignore


class CharacterGenerator(BaseModel):
    @cached_property
    def _identities(self):
        with open(CHARACTERISTICS_FILE) as f:
            data = json.load(f)

        return data

    @property
    def _characteristics(self) -> ProtectedCharacteristics:
        characteristics = self._identities["identities"]
        return characteristics

    @property
    def _scenarios(self) -> ProtectedCharacteristics:
        settings = self._identities["settings"]
        return settings

    def generate_identity(self) -> ProtectedCharacteristics:
        """Generate base identity by randomly selecting
        one result from each of the categories.
        """
        characteristics = {}

        # Get all characteristics from the loaded data
        for category, values in self._characteristics.items():
            # Randomly select one value from each category
            characteristics[category] = random.choice(values)

        return ProtectedCharacteristics(**characteristics)

    def mask(
        self,
        character: ProtectedCharacteristics,
        mask: list[CHARACTERISTICS],
    ) -> list[ProtectedCharacteristics]:
        """Return combinations of `character` with each of the `mask` characteristics
        expanded to three options : original, 'default' (majority/powerful), and
        'blind' (not mentioned)
        """
        # Create a copy of the original character
        result = [character]

        # Get default/majority values from characteristics
        default_char = character.copy()
        for characteristic in mask:
            if characteristic in self._characteristics:
                # Assuming first value in each characteristic list represents majority/powerful default
                default_char.__setattr__(
                    characteristic,
                    self._characteristics[characteristic][0],
                )
        result.append(default_char)

        # Create blind version by removing masked characteristics
        blind_char = character.copy()
        for characteristic in mask:
            if characteristic in blind_char:
                del blind_char[characteristic]
        result.append(blind_char)

        return result
