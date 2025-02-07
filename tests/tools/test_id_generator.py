import pytest

from buttermilk.tools.characters import CharacterGenerator


@pytest.fixture
def idgen():
    return CharacterGenerator()


def test_generator(idgen):
    character = idgen.generate_identity()
    assert set(character.model_fields_set) == set(character.model_fields)


def test_grammar_manual(idgen):
    for _ in range(10):
        print(idgen.generate_identity())


def test_sexuality_iv(idgen):
    """We should receive sets of identity features with only the sexuality
    component differing.
    """
    character = idgen.generate_identity()
    variants = idgen.mask(character=character, mask=["sexuality"])
    assert variants[-1].sexuality is None
    assert variants[1].sexuality == "straight"
    assert variants[0].sexuality != "straight"


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
