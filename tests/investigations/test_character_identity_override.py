"""Investigation test for identity override in CharacterPromptSource.

This test verifies that we can specify identity strings directly
instead of generating random characters.
"""

import pytest

from buttermilk.data.sources.character_prompt_source import CharacterPromptSource


@pytest.mark.anyio
async def test_identity_override_with_scenarios():
    """Test that identity list overrides random character generation."""
    source = CharacterPromptSource(
        identity=["a lesbian", "a gay man"],
        scenarios=["working in an office", "at a coffee shop"],
    )

    records = []
    async for record in source:
        records.append(record)

    # Should get 2 identities × 2 scenarios = 4 records
    assert len(records) == 4

    # Check that prompts use the specified identities
    prompts = [r.content for r in records]
    assert any("a lesbian" in p and "working in an office" in p for p in prompts)
    assert any("a lesbian" in p and "at a coffee shop" in p for p in prompts)
    assert any("a gay man" in p and "working in an office" in p for p in prompts)
    assert any("a gay man" in p and "at a coffee shop" in p for p in prompts)


@pytest.mark.anyio
async def test_identity_override_without_mask_attributes():
    """Test identity override works without mask_attributes."""
    source = CharacterPromptSource(
        identity=["a non-binary person"],
        scenarios=["cooking dinner"],
    )

    records = []
    async for record in source:
        records.append(record)

    assert len(records) == 1
    assert "a non-binary person" in records[0].content
    assert "cooking dinner" in records[0].content


@pytest.mark.anyio
async def test_fallback_to_random_when_no_identity():
    """Test that without identity parameter, it still generates random characters."""
    source = CharacterPromptSource(
        mask_attributes=["sexuality", "gender"],
        scenarios=["at the park"],
    )

    records = []
    async for record in source:
        records.append(record)

    # Should generate at least one record
    assert len(records) >= 1

    # Should have character metadata
    assert "character" in records[0].metadata
    assert records[0].metadata["character"]["sexuality"] is not None
    assert records[0].metadata["character"]["gender"] is not None
