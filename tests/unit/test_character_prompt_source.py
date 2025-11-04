"""Unit tests for CharacterPromptSource."""

import pytest

from buttermilk.data.sources.character_prompt_source import CharacterPromptSource


@pytest.mark.anyio
async def test_character_prompt_source_yields_base_records(real_bm):
    """Test that source yields BaseRecord objects with prompts."""
    source = CharacterPromptSource(
        mask_attributes=["sexuality"],
        scenarios=["working in an office", "at a coffee shop"],
    )

    records = [r async for r in source]

    # Should yield one record per scenario
    assert len(records) == 2
    # Session ID comes from bm.session_info.session_id
    assert all("session_id" in r.metadata for r in records)
    assert all(isinstance(r.content, str) for r in records)
    assert any("working in an office" in r.content.lower() for r in records)
    assert any("coffee shop" in r.content.lower() for r in records)


@pytest.mark.anyio
async def test_character_prompt_source_uses_bm_session_id(real_bm):
    """Test that session_id comes from bm.session_info.session_id."""
    source = CharacterPromptSource(
        mask_attributes=["gender", "sexuality"],
        scenarios=["cooking dinner"],
    )

    records = [r async for r in source]

    # All records should have same session_id from bm
    session_ids = {r.metadata["session_id"] for r in records}
    assert len(session_ids) == 1
    assert session_ids.pop() == real_bm.session_info.session_id


@pytest.mark.anyio
async def test_character_prompt_source_includes_scenario_metadata(real_bm):
    """Test that scenario metadata is included in records."""
    source = CharacterPromptSource(
        mask_attributes=["gender"],
        scenarios=["giving a presentation", "playing sports"],
    )

    records = [r async for r in source]

    assert len(records) == 2
    assert records[0].metadata["scenario"] == "giving a presentation"
    assert records[0].metadata["scenario_index"] == 0
    assert records[1].metadata["scenario"] == "playing sports"
    assert records[1].metadata["scenario_index"] == 1
    assert all("character" in r.metadata for r in records)
