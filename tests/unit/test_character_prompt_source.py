"""Unit tests for CharacterPromptSource."""

import pytest

from buttermilk.data.sources.character_prompt_source import CharacterPromptSource


@pytest.mark.anyio
async def test_character_prompt_source_yields_base_records():
    """Test that source yields BaseRecord objects with prompts."""
    source = CharacterPromptSource(
        mask_attributes=["sexuality"],
        scenarios=["working in an office", "at a coffee shop"],
        session_id="test-session-001",
    )

    records = [r async for r in source]

    # Should yield one record per scenario
    assert len(records) == 2
    assert all(r.metadata["session_id"] == "test-session-001" for r in records)
    assert all(isinstance(r.content, str) for r in records)
    assert any("working in an office" in r.content.lower() for r in records)
    assert any("coffee shop" in r.content.lower() for r in records)


@pytest.mark.anyio
async def test_character_prompt_source_propagates_session_id():
    """Test that session_id propagates to all records."""
    source = CharacterPromptSource(
        mask_attributes=["gender", "sexuality"],
        scenarios=["cooking dinner"],
        session_id="unique-session",
    )

    records = [r async for r in source]

    assert all(r.metadata["session_id"] == "unique-session" for r in records)
    assert all("character" in r.metadata for r in records)


@pytest.mark.anyio
async def test_character_prompt_source_generates_session_id():
    """Test that session_id is auto-generated if not provided."""
    source = CharacterPromptSource(
        mask_attributes=["sexuality"],
        scenarios=["at a park"],
    )

    records = [r async for r in source]

    assert len(records) == 1
    # Session ID should be generated
    assert records[0].metadata["session_id"] is not None
    assert len(records[0].metadata["session_id"]) > 0


@pytest.mark.anyio
async def test_character_prompt_source_includes_scenario_metadata():
    """Test that scenario metadata is included in records."""
    source = CharacterPromptSource(
        mask_attributes=["gender"],
        scenarios=["giving a presentation", "playing sports"],
        session_id="test-123",
    )

    records = [r async for r in source]

    assert len(records) == 2
    assert records[0].metadata["scenario"] == "giving a presentation"
    assert records[0].metadata["scenario_index"] == 0
    assert records[1].metadata["scenario"] == "playing sports"
    assert records[1].metadata["scenario_index"] == 1
