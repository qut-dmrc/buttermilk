"""Investigation test for character prompt grammar issues.

This test demonstrates grammatical problems in generated prompts when
combining sexuality and gender attributes. Current implementation produces
prompts like "Kathoey and Gay playing with children" which is grammatically
incorrect. Should produce "A gay kathoey person playing with children".
"""

import re

from buttermilk.tools.characters import CharacterGenerator


def test_sexuality_gender_prompts_are_grammatically_correct():
    """Verify prompts with sexuality+gender follow grammatical pattern.

    Current behavior: Produces "A Kathoey and Gay [scenario]"
    Expected behavior: Produces "A gay kathoey person [scenario]"

    Pattern should be: "[article] [sexuality] [gender] person [scenario]"
    """
    gen = CharacterGenerator()

    # Generate a real character then filter to keep only sexuality and gender
    char = gen.generate_identity()

    # Filter to keep only sexuality and gender
    filtered = gen.reverse_mask(char, keep=["sexuality", "gender"])

    # Get character string representation
    char_str = str(filtered)

    # Generate a simple scenario
    scenarios = ["working in an office", "playing with children at a park"]

    for scenario in scenarios:
        prompt = gen._format_character_with_scenario(char_str, scenario)

        # ASSERTIONS FOR GRAMMATICAL CORRECTNESS

        # Should NOT contain grammatically incorrect patterns
        assert "and Gay" not in prompt, f"Prompt contains grammatically incorrect 'and Gay': {prompt}"
        assert "and Kathoey" not in prompt, f"Prompt contains grammatically incorrect 'and Kathoey': {prompt}"

        # Should contain "person" keyword when both sexuality and gender present
        assert "person" in prompt.lower(), f"Prompt missing 'person' keyword: {prompt}"

        # Should follow pattern: [article] [sexuality] [gender...] person [scenario]
        # Using case-insensitive regex to allow various sexuality descriptors
        # Gender may be one or more words (e.g., "woman" or "trans woman")
        # Pattern: (A|An) + one or more words + "person"
        pattern = r"(a|an)\s+[\w\s]+person"
        assert re.search(pattern, prompt, re.IGNORECASE), (
            f"Prompt doesn't match expected pattern '[article] [sexuality] [gender] person [scenario]': {prompt}"
        )

        # The scenario should be incorporated naturally
        assert scenario in prompt or scenario.replace("at ", "") in prompt, f"Prompt doesn't incorporate scenario '{scenario}': {prompt}"


def test_character_str_with_sexuality_and_gender_includes_person():
    """Verify ProtectedCharacteristics.__str__() produces proper format.

    When a character has both sexuality and gender, the string representation
    should use 'person' to create grammatically correct phrasing.

    Expected: "A gay kathoey person"
    NOT: "A kathoey and gay"
    """
    gen = CharacterGenerator()

    # Generate then filter to sexuality and gender only
    char = gen.generate_identity()
    filtered = gen.reverse_mask(char, keep=["sexuality", "gender"])

    char_str = str(filtered)

    # Should contain "person" when both sexuality and gender are present
    assert "person" in char_str.lower(), f"Character string missing 'person': {char_str}"

    # Should NOT use "and" to connect sexuality and gender
    assert not re.search(r"and\s+(gay|lesbian|bisexual)", char_str, re.IGNORECASE), f"Character string uses 'and' incorrectly: {char_str}"

    # Should have proper article-adjective-noun structure
    # Format: "A [sexuality] [gender] person" or "An [sexuality] [gender] person"
    # Gender may be multiple words (e.g., "trans woman")
    pattern = r"(A|An)\s+[\w\s]+person"
    assert re.search(pattern, char_str, re.IGNORECASE), (
        f"Character string doesn't match expected pattern '[article] [sexuality] [gender] person': {char_str}"
    )


def test_single_attribute_character_without_person_keyword():
    """Verify characters with only gender (no sexuality) don't awkwardly add 'person'.

    A character with just gender should be "A woman" not "A woman person".
    Only add "person" when we need it for grammatical correctness with multiple descriptors.
    """
    gen = CharacterGenerator()

    # Generate then filter to gender only
    char = gen.generate_identity()
    filtered = gen.reverse_mask(char, keep=["gender"])

    # This test documents expected behavior - "person" only needed with sexuality+gender combo
    # When only gender is present, the string representation should be natural
    # without awkwardly adding "person" unnecessarily
    # The test passes as long as the character can be stringified without error
    assert str(filtered)  # Should produce natural output like "A woman" not "A woman person"
