"""Unit tests for CharacterGenerator."""

from buttermilk.tools.characters import CharacterGenerator, ProtectedCharacteristics


def test_generate_identity_creates_character():
    """Test that generate_identity creates a character with all attributes."""
    gen = CharacterGenerator()
    char = gen.generate_identity()

    assert isinstance(char, ProtectedCharacteristics)
    assert char.body_type is not None
    assert char.social_class is not None
    assert char.age_group is not None
    assert char.gender is not None
    assert char.sexuality is not None


def test_reverse_mask_keeps_only_specified_attributes():
    """Test that reverse_mask keeps ONLY specified attributes, sets all others to None."""
    gen = CharacterGenerator()
    char = gen.generate_identity()

    # Store original value for comparison
    original_sexuality = char.sexuality

    # Reverse mask to keep only sexuality
    filtered = gen.reverse_mask(char, keep=["sexuality"])

    assert filtered.sexuality == original_sexuality
    assert filtered.gender is None
    assert filtered.ethnicity is None
    assert filtered.body_type is None
    assert filtered.social_class is None
    assert filtered.age_group is None
    assert filtered.gender_presentation is None
    assert filtered.religion is None
    assert filtered.relationship_status is None
    assert filtered.citizenship_status is None
    assert filtered.disability is None


def test_reverse_mask_keeps_multiple_attributes():
    """Test that reverse_mask can keep multiple specified attributes."""
    gen = CharacterGenerator()
    char = gen.generate_identity()

    # Store original values
    original_sexuality = char.sexuality
    original_gender = char.gender
    original_ethnicity = char.ethnicity

    # Reverse mask to keep sexuality, gender, and ethnicity
    filtered = gen.reverse_mask(char, keep=["sexuality", "gender", "ethnicity"])

    assert filtered.sexuality == original_sexuality
    assert filtered.gender == original_gender
    assert filtered.ethnicity == original_ethnicity
    assert filtered.body_type is None
    assert filtered.social_class is None
    assert filtered.age_group is None
    assert filtered.gender_presentation is None
    assert filtered.religion is None
    assert filtered.relationship_status is None
    assert filtered.citizenship_status is None
    assert filtered.disability is None


def test_reverse_mask_empty_keep_list():
    """Test that reverse_mask with empty keep list sets all attributes to None."""
    gen = CharacterGenerator()
    char = gen.generate_identity()

    # Reverse mask with empty list - should set everything to None
    filtered = gen.reverse_mask(char, keep=[])

    assert filtered.sexuality is None
    assert filtered.gender is None
    assert filtered.ethnicity is None
    assert filtered.body_type is None
    assert filtered.social_class is None
    assert filtered.age_group is None
    assert filtered.gender_presentation is None
    assert filtered.religion is None
    assert filtered.relationship_status is None
    assert filtered.citizenship_status is None
    assert filtered.disability is None
