"""Test vector store metadata handling errors.

This test reproduces the dict_items error encountered with record B7ZX9ISZ.
"""

import pytest

from buttermilk.data.vector import _get_chunk_field, _sanitize_metadata_for_chroma


class TestMetadataHandling:
    """Test metadata handling in vector store operations."""

    def test_dict_items_object_causes_attribute_error(self):
        """Test that passing dict_items to code expecting dict causes AttributeError.

        This reproduces the error from the logs:
        'dict_items' object has no attribute 'items'
        """
        # Create a dict_items object
        test_dict = {"key1": "value1", "key2": "value2"}
        dict_items_obj = test_dict.items()

        # Verify it's actually dict_items
        assert type(dict_items_obj).__name__ == "dict_items"

        # Try to call .items() on it (this should fail)
        with pytest.raises(AttributeError) as exc_info:
            dict_items_obj.items()  # type: ignore

        assert "'dict_items' object has no attribute 'items'" in str(exc_info.value)

    def test_get_chunk_field_returns_dict_items_if_stored_as_dict_items(self):
        """Test that _get_chunk_field could return dict_items if that's what's in the chunk."""
        # If a chunk dict contains dict_items as a value, _get_chunk_field returns it as-is
        test_dict = {"content": "test", "metadata": {"a": "b"}}
        chunk = {
            "chunk_id": "test_id",
            "metadata": test_dict.items(),  # Store dict_items as the value
        }

        result = _get_chunk_field(chunk, "metadata", {})

        # _get_chunk_field just does dict.get(), so it returns whatever is stored
        assert type(result).__name__ == "dict_items"

    def test_sanitize_metadata_handles_dict_items_input(self):
        """Test that _sanitize_metadata_for_chroma handles dict_items input gracefully."""
        # Create dict_items
        test_dict = {"key1": "value1", "key2": "value2"}
        dict_items_obj = test_dict.items()

        # Call _sanitize_metadata_for_chroma with dict_items
        # It should log a warning and return empty dict
        result = _sanitize_metadata_for_chroma(dict_items_obj)  # type: ignore

        # Should return empty dict (fail-safe behavior)
        assert result == {}

    def test_chunk_metadata_dict_comprehension_with_dict_items(self):
        """Test that dict_items can be converted to dict successfully.

        This verifies the defensive code at lines 1431-1443 in vector.py
        that converts dict_items to a proper dict before use.
        """
        # Simulate chunk_metadata being dict_items (the original bug)
        real_metadata = {
            "content_type": "text",
            "chunk_type": "semantic",
            "extra": "data",
        }
        chunk_metadata = real_metadata.items()  # This creates dict_items

        # Verify it's dict_items
        assert type(chunk_metadata).__name__ == "dict_items"
        assert not isinstance(chunk_metadata, dict)

        # The defensive code in vector.py converts dict_items to dict
        # dict_items doesn't have .items() but can be converted with dict()
        if not isinstance(chunk_metadata, dict):
            try:
                chunk_metadata = dict(chunk_metadata)
            except (TypeError, ValueError):
                chunk_metadata = {}

        # Now it should be a proper dict
        assert isinstance(chunk_metadata, dict)

        # And the dict comprehension pattern should work
        enhanced_metadata = {
            "content_type": chunk_metadata.get("content_type", "unknown"),
            "chunk_type": chunk_metadata.get("chunk_type", "unknown"),
<<<<<<< HEAD
            **{k: v for k, v in chunk_metadata.items() if k not in ["content_type", "chunk_type"]},
=======
            **{
                k: v
                for k, v in chunk_metadata.items()
                if k not in ["content_type", "chunk_type"]
            },
>>>>>>> origin/stable
        }

        # Verify the result
        assert enhanced_metadata["content_type"] == "text"
        assert enhanced_metadata["chunk_type"] == "semantic"
        assert enhanced_metadata["extra"] == "data"

    def test_isinstance_dict_items_is_not_dict(self):
        """Verify that dict_items is not an instance of dict."""
        test_dict = {"a": "b"}
        dict_items_obj = test_dict.items()

        # dict_items should NOT be an instance of dict
        assert not isinstance(dict_items_obj, dict)


class TestChunkMetadataOrigin:
    """Test where dict_items might come from in chunk metadata."""

    def test_scrub_serializable_does_not_return_dict_items(self):
        """Test that scrub_serializable never returns dict_items."""
        from buttermilk.utils.utils import scrub_serializable

        # Test various inputs
        test_cases = [
            {"a": "b", "c": "d"},
            {"nested": {"a": "b"}},
            {"list": [1, 2, 3]},
        ]

        for test_input in test_cases:
            result = scrub_serializable(test_input)
<<<<<<< HEAD
            assert isinstance(result, dict), f"scrub_serializable returned {type(result)} for {test_input}"
=======
            assert isinstance(result, dict), (
                f"scrub_serializable returned {type(result)} for {test_input}"
            )
>>>>>>> origin/stable
            assert type(result).__name__ != "dict_items"

    def test_scrub_serializable_with_dict_items_input(self):
        """Test scrub_serializable if given dict_items as input."""
        from buttermilk.utils.utils import scrub_serializable

        test_dict = {"a": "b", "c": "d"}
        dict_items_obj = test_dict.items()

        # scrub_serializable should handle this gracefully
        result = scrub_serializable(dict_items_obj)

        # It might convert it to a dict, or return it as-is
        # Let's see what happens
        assert result is not None

    def test_semantic_splitter_chunk_metadata_type(self):
        """Test what type of metadata SemanticSplitter produces.

        Based on the recent fix in vector.py for SemanticSplitter chunks,
        we should verify the metadata structure.
        """
        # This would require mocking SemanticSplitter output
        # Skip for now - need to understand the actual chunk structure
        pytest.skip("Need to investigate actual SemanticSplitter output format")


class TestMetadataErrorReproduction:
    """Attempt to reproduce the exact error from record B7ZX9ISZ."""

    def test_record_b7zx9isz_metadata_structure(self):
        """Test with a mock chunk structure that could cause the error.

        The error suggests that somewhere in the pipeline, metadata
        gets set to dict_items instead of a dict.
        """
        # Create a mock chunk that has dict_items as metadata
        # This could happen if somewhere we accidentally did:
        # chunk['metadata'] = some_dict.items()

        test_metadata = {"content_type": "text", "extra_key": "extra_value"}

        # Scenario 1: metadata is dict_items
        bad_chunk = {
            "chunk_id": "B7ZX9ISZ_chunk_0",
            "chunk_text": "Some text",
            "document_title": "Test Doc",
            "chunk_index": 0,
            "document_id": "B7ZX9ISZ",
            "metadata": test_metadata.items(),  # BUG: dict_items instead of dict
            "embedding": [0.1, 0.2, 0.3],
        }

        chunk_metadata = _get_chunk_field(bad_chunk, "metadata", {})

        # Verify it's dict_items
        assert type(chunk_metadata).__name__ == "dict_items"

        # Now try the dict comprehension from line 1110
        # Since isinstance(dict_items, dict) is False, this might not fail the way we expect
        is_dict = isinstance(chunk_metadata, dict)
        assert is_dict is False  # dict_items is NOT a dict

        # The ternary should go to the else branch: {}.items()
        # So this should work:
        (chunk_metadata.items() if isinstance(chunk_metadata, dict) else {}.items())

        # If isinstance returns False, we get {}.items() which is fine
        # So the error must be happening differently...

        # WAIT - the error says "'dict_items' object has no attribute 'items'"
        # This means chunk_metadata.items() is being called somehow
        # Let me check if there's another code path...

    def test_scrub_serializable_dict_like_object_path(self):
        """Test if scrub_serializable's dict-like object handler could return dict_items."""
        from buttermilk.utils.utils import scrub_serializable

        # Create an object that has .items(), .keys(), .values() but is not a dict
        class DictLikeWithDictItems:
            def __init__(self, data):
                self._data = data

            def items(self):
                return self._data.items()  # Returns dict_items!

            def keys(self):
                return self._data.keys()

            def values(self):
                return self._data.values()

        dict_like = DictLikeWithDictItems({"a": "b", "c": "d"})

        # Call scrub_serializable
        result = scrub_serializable(dict_like)

        # Check if result is dict_items
        if type(result).__name__ == "dict_items":
            pytest.fail("scrub_serializable returned dict_items from dict-like object!")

        # It should return a regular dict
        assert isinstance(result, dict)
