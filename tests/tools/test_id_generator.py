import pytest

from buttermilk.tools.characters import CharacterGenerator


@pytest.fixture
def idgen():
    return CharacterGenerator()


def test_generator(idgen):
    character = idgen.generate_identity()
    assert set(character.model_fields_set) == set(character.model_fields)


def test_sexuality_iv(idgen):
    """We should receive sets of identity features with only the sexuality
    component differing.
    """
    character = idgen.generate_identity()
    variants = idgen.mask(character=character, mask=["sexuality"])
    assert "sexuality" not in variants[-1].model_fields
    assert variants[0].sexuality == "straight"
    assert variants[0].sexuality != "straight"
