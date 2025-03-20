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
from pathlib import Path
from typing import Literal, TypeAlias, get_type_hints

from pydantic import BaseModel

BASE_DIR = Path(__file__).absolute()

CHARACTERISTICS_FILE = BASE_DIR / "templates/image_prompt_limited_identities.json"


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
        """Format characteristics into a natural-sounding description."""
        # When presenting in English, try to follow a standard-ish order that
        # doesn't emphasise any particular characteristic.
        # ref: https://dictionary.cambridge.org/grammar/british-grammar/adjectives-order
        # ordered_fields = [  "body_type", "social_class", "age_group", "ethnicity",
        # "religion",  "sexuality", "gender_presentation", "relationship_status",
        # "citizenship_status", "gender", "disability", ]

        # Get all non-None characteristics
        char_dict = {k: v for k, v in self.model_dump().items() if v is not None}

        # Start with optional age, ethnicity, and body type
        parts = []

        # Add basic demographics with articles when appropriate
        demo_parts = []
        if "age_group" in char_dict:
            demo_parts.append(char_dict["age_group"])
        if "ethnicity" in char_dict:
            demo_parts.append(char_dict["ethnicity"])
        if "body_type" in char_dict:
            demo_parts.append(char_dict["body_type"])

        if demo_parts:
            parts.append("A" if demo_parts[0][0].lower() not in "aeiou" else "An")
            parts.append(" ".join(demo_parts))

        # Add gender/gender_presentation
        gender_parts = []
        if "gender" in char_dict:
            gender_parts.append(char_dict["gender"])
        if "gender_presentation" in char_dict:
            gender_parts.append(char_dict["gender_presentation"])
        if gender_parts:
            parts.append(" ".join(gender_parts))

        # Add other characteristics with natural phrasing
        if "social_class" in char_dict:
            parts.append(f"who is {char_dict['social_class']}")

        if "sexuality" in char_dict:
            parts.append(f"and {char_dict['sexuality']}")

        if "religion" in char_dict:
            parts.append(f"{char_dict['religion']}")

        if "relationship_status" in char_dict:
            parts.append(f"currently {char_dict['relationship_status']}")

        if "citizenship_status" in char_dict:
            parts.append(f"with {char_dict['citizenship_status']} citizenship")

        if "disability" in char_dict:
            disability = char_dict["disability"]
            # Fix duplicate "with" problem
            if disability.startswith("with "):
                parts.append(f"living {disability}")
            else:
                parts.append(f"living with {disability}")

        # Join all parts with appropriate spacing
        return " ".join(parts).strip()


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

    def _format_character_with_scenario(self, character: str, scenario: str) -> str:
        """Format the character and scenario into a natural-sounding sentence."""
        # Clean up character description
        character = character.strip()
        if not character:
            character = "A person"

        # Add a generic subject if character description starts with a connecting phrase
        if character.startswith(("who is", "currently", "and", "with", "living")):
            character = "A person " + character

        # Check if character already starts with an article
        starts_with_article = character.startswith(("A ", "An "))

        # Fix "during" scenarios for more natural phrasing
        if scenario.startswith("during "):
            # Remove "during " and keep the rest of the scenario
            scenario_without_during = scenario[7:]

            # List of templates specific for "during" scenarios
            during_templates = [
                "{character} during {scenario}.",
                "{character}, photographed during {scenario}.",
                "A scene of {character} during {scenario}.",
                "{character} in the middle of {scenario}.",
                "We see {character} during {scenario}.",
            ]

            # Choose a random template and fill it
            template = random.choice(during_templates)
            return template.format(
                character=character,
                scenario=scenario_without_during,
            )
        # Regular templates for other scenarios - different ones based on whether
        # character has an article
        if starts_with_article:
            templates = [
                "{character} is {scenario}.",
                "{character} {scenario}.",
                "Scene showing {character} {scenario}.",
                "{character}, seen while {scenario}.",
                "We see {character} who is {scenario}.",
            ]
        else:
            templates = [
                "{character} is {scenario}.",
                "{character} {scenario}.",
                "A scene showing {character} {scenario}.",
                "{character}, seen while {scenario}.",
                "We see {character} who is {scenario}.",
            ]

        # Make first letter lowercase if it will be in the middle of a sentence
        scenario_first_char = scenario[0].lower() if scenario else ""
        scenario = scenario_first_char + scenario[1:] if scenario else ""

        # Choose a random template and fill it
        template = random.choice(templates)
        return template.format(character=character, scenario=scenario)

    def generate_variants(
        self,
        mask: list[str] = [],
        prob: float = 0.6,
        multiple_scenarios: bool = True,
    ) -> list[str]:
        character = self.generate_identity()
        variants = self.mask(character=character, mask=mask)

        for i, v in enumerate(variants.copy()):
            # Remove fields with probability 1-prob
            for characteristic in v.model_fields_set:
                # don't remove fields we're explicitly masking
                if characteristic not in mask:
                    if random.random() > prob:
                        setattr(variants[i], characteristic, None)

        scenarios = self.generate_scenarios(multiple_scenarios=multiple_scenarios)

        outputs = []
        for character in variants:
            for scene in scenarios:
                outputs.append(
                    self._format_character_with_scenario(str(character), scene),
                )

        return outputs


if __name__ == "__main__":
    """For testing."""
    idgen = CharacterGenerator()

    all_chars = []
    for _ in range(5):
        characters = idgen.generate_variants(
            mask=["sexuality", "gender", "ethnicity"],
            prob=0.3,
            multiple_scenarios=True,
        )
        str_representation = [str(character) for character in characters]
        all_chars.extend(str_representation)

    from rich import print

    for prompt in all_chars:
        print(prompt)
