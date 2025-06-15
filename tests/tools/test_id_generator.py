import pytest

from buttermilk.tools.characters import CharacterGenerator


@pytest.fixture
def idgen():
    return CharacterGenerator()


def test_generator(idgen):
    character = idgen.generate_identity()
    assert set(character.model_fields_set) == set(character.__class__.model_fields)


def test_sexuality_iv(idgen):
    """We should receive sets of identity features with only the sexuality
    component differing.
    """
    character = idgen.generate_identity()
    variants = idgen.mask(character=character, mask=["sexuality"])
    assert variants[-1].sexuality is None
    assert str.lower(variants[1].sexuality) == "straight"
    assert str.lower(variants[0].sexuality) != "straight"


def test_scenario_with_gender_sexuality_ethnicity_ivs(idgen):
    """Expect the same character depiction with variance across
    gender, sexuality, and ethnicity, all presented in each
    category of scenario.
    """
    characters = idgen.generate_variants(
        mask=["sexuality", "gender", "ethnicity"],
        prob=0.6,
        multiple_scenarios=True,
    )

    assert len(characters) > 9


@pytest.mark.integration
def test_string_representation(idgen) -> list[str]:
    # this check looks to see whether the string representation of our
    # generated identity is parseable in English. It can only be checked
    # manually or with an llm, so we'll just output it for now.

    all_chars = []
    for _ in range(10):
        character = idgen.generate_identity()
        str_representation = str(character)
        print(str_representation)
        all_chars.append(str_representation)
    return all_chars


@pytest.mark.integration
def test_string_representation_with_scenario(idgen) -> list[str]:
    # this check looks to see whether the string representation of our
    # generated identity is parseable in English. It can only be checked
    # manually or with an llm, so we'll just output it for now.

    all_chars = []
    for _ in range(5):
        characters = idgen.generate_variants(
            prob=0.6,
            multiple_scenarios=True,
        )
        str_representation = [str(character) for character in characters]
        print(str_representation)
        all_chars.extend(str_representation)

    return all_chars
