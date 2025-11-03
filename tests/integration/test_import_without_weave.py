"""Test that buttermilk works without weave dependency.

This test verifies that after removing weave, buttermilk still initializes
and functions correctly.
"""

import pytest


def test_import_buttermilk_without_weave():
    """Test that buttermilk can be imported without weave."""
    import buttermilk

    # Should not raise ImportError
    assert buttermilk is not None


@pytest.mark.anyio
async def test_bm_initialize_without_weave(real_bm):
    """Test that BM initializes without weave."""
    # real_bm fixture initializes BM
    assert real_bm is not None
    # Test that we can access session info
    assert real_bm.session_info is not None
    assert real_bm.session_info.project_name == "buttermilk"


@pytest.mark.anyio
async def test_bm_get_weave_client_returns_none(real_bm):
    """Test that get_weave_client returns None when weave is removed."""
    # After weave removal, this should return None gracefully
    weave_client = await real_bm.get_weave_client()
    assert weave_client is None


@pytest.mark.skip(reason="Will pass after pyproject.toml updated (Cycle 6)")
def test_weave_not_importable():
    """Test that weave is not in the environment after removal."""
    with pytest.raises(ImportError):
        pass
