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
    body_type: str
    social_class: str
    age_group: str
    gender: str
    sexuality: str
    gender_presentation: str
    ethnicity: str
    religion: str
    relationship_status: str
    citizenship_status: str
    disability: str

    def __str__(self) -> str:
        # When presenting in English, try to follow a standard-ish order that
        # doesn't emphasise any particular characteristic.
        # ref: https://dictionary.cambridge.org/grammar/british-grammar/adjectives-order
        ordered_fields = [
            "body_type",
            "social_class",
            "age_group",
            "ethnicity",
            "religion",
            "sexuality",
            "gender_presentation",
            "relationship_status",
            "citizenship_status",
            "gender",
            "disability",
        ]
        character_desc = []
        for field in ordered_fields:
            if field in self.model_fields_set:
                if x := self.__getattribute__(field):
                    character_desc.append(x)

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
    def _scenarios(self) -> dict:
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

    def generate_scenarios(self, multiple_scenarios: bool = True) -> list[str]:
        scenarios = []
        categories = list(self._scenarios.keys())
        random.shuffle(categories)
        # Get all scenario categories from the loaded data
        for category in categories:
            # Randomly select one value from each category
            scenarios.append(random.choice(self._scenarios[category]))
            if not multiple_scenarios:
                break
        return scenarios

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
        default_char = character.model_copy()
        for characteristic in mask:
            if characteristic in self._characteristics:
                # Assuming first value in each characteristic list represents majority/powerful default
                default_char.__setattr__(
                    characteristic,
                    self._characteristics[characteristic][0],
                )
        result.append(default_char)

        # Create blind version by removing masked characteristics
        blind_char = character.model_copy()
        for characteristic in mask:
            blind_char.__setattr__(characteristic, None)
        result.append(blind_char)

        return result

    def generate_variants(
        self,
        mask: list[str],
        prob: float = 0.6,
        multiple_scenarios: bool = True,
    ) -> list[str]:  # list[tuple[ProtectedCharacteristics, str]]:
        character = self.generate_identity()

        variants = self.mask(character=character, mask=mask)

        for i, v in enumerate(variants.copy()):
            # Remove fields with probability 1-prob
            for characteristic in v.model_fields_set:
                # don't remove fields we're explicitly masking
                if characteristic not in mask:
                    if random.random() > prob:
                        setattr(variants[i], characteristic, None)

        # variants = [list(x.model_dump().values()) for x in variants]
        scenarios = self.generate_scenarios(multiple_scenarios=multiple_scenarios)

        outputs = []
        for character in variants:
            for scene in scenarios:
                outputs.append(str(character) + " " + str(scene))

        return outputs


"""For testing, output ten generations."""
if __name__ == "__main__":
    idgen = CharacterGenerator()

    characters = idgen.generate_variants(
        mask=["sexuality", "gender", "ethnicity"],
        prob=0.3,
        multiple_scenarios=True,
    )
    try:
        from rich import print
    except:
        pass

    for prompt in characters:
        print(" ".join(prompt))
        print()
