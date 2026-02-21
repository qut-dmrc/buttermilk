"""Test that vector.py handles dict chunks from SemanticSplitter correctly.

This test reproduces the bug where vector.py fails with AttributeError when
chunks are dicts instead of objects (e.g., from SemanticSplitter).

The bug: Code accesses chunk.metadata, chunk.chunk_text, etc. directly
The fix: Use helper functions to handle both dict and object chunks
"""

import pytest

from buttermilk.data.vector import _get_chunk_embedding, _set_chunk_embedding

pytestmark = pytest.mark.slow


class TestVectorDictChunks:
    """Test that vector.py handles dict chunks from SemanticSplitter."""

    def test_dict_chunks_fail_with_direct_attribute_access(self):
        """Demonstrate that dict chunks fail when accessing attributes directly.

        This simulates the exact failure scenario from the Zotero pipeline.
        """
        # Create a dict chunk as returned by SemanticSplitter
        dict_chunk = {
            "chunk_id": "I2678GVC_0",
            "chunk_index": 0,
            "chunk_text": "This is the first chunk of text.",
            "chunk_title": "Section 1",
            "document_id": "I2678GVC",
            "document_title": "Test Document",
            "metadata": {"chunk_type": "content", "section": "intro"},
        }

        # This is what the code at line 923 tries to do:
        # chunk.metadata.update(...)
<<<<<<< HEAD
        with pytest.raises(AttributeError, match="'dict' object has no attribute 'metadata'"):
=======
        with pytest.raises(
            AttributeError, match="'dict' object has no attribute 'metadata'"
        ):
>>>>>>> origin/stable
            chunk = dict_chunk
            chunk.metadata.update({"new_key": "value"})  # Fails!

    def test_dict_chunks_fail_when_accessing_chunk_text(self):
        """Demonstrate failure at line 1404 when accessing chunk.chunk_text."""

        # This is what line 1404 tries to do:
        # text=chunk.chunk_text
<<<<<<< HEAD
        with pytest.raises(AttributeError, match="'dict' object has no attribute 'chunk_text'"):
=======
        with pytest.raises(
            AttributeError, match="'dict' object has no attribute 'chunk_text'"
        ):
>>>>>>> origin/stable
            pass

    def test_proper_dict_access_works(self):
        """Show the correct way to access dict chunk fields."""
        dict_chunk = {
            "chunk_id": "TEST_0",
            "chunk_text": "Test content",
            "chunk_title": "Test Title",
            "metadata": {"chunk_type": "content"},
        }

        # The correct way to access dict fields
        assert dict_chunk["chunk_text"] == "Test content"
        assert dict_chunk["chunk_title"] == "Test Title"
        assert dict_chunk["metadata"]["chunk_type"] == "content"

        # And updating metadata correctly
        dict_chunk["metadata"]["new_field"] = "new_value"
        assert dict_chunk["metadata"]["new_field"] == "new_value"

    def test_helper_functions_handle_both_dict_and_object(self):
        """Test that helper functions correctly handle both dict and object chunks."""
        # Test with dict chunk
        dict_chunk = {"embedding": [0.1, 0.2, 0.3]}
        assert _get_chunk_embedding(dict_chunk) == [0.1, 0.2, 0.3]

        _set_chunk_embedding(dict_chunk, [0.4, 0.5, 0.6])
        assert dict_chunk["embedding"] == [0.4, 0.5, 0.6]

        # Test with object chunk (mock)
        class ObjectChunk:
            def __init__(self):
                self.embedding = [0.7, 0.8, 0.9]

        obj_chunk = ObjectChunk()
        assert _get_chunk_embedding(obj_chunk) == [0.7, 0.8, 0.9]

        _set_chunk_embedding(obj_chunk, [1.0, 1.1, 1.2])
        assert obj_chunk.embedding == [1.0, 1.1, 1.2]

    def test_generic_chunk_field_helper_needed(self):
        """Demonstrate need for generic helper function for all chunk fields.

        We need something like _get_chunk_field(chunk, field_name) that works
        for both dict and object chunks.
        """
        # Dict chunk
        dict_chunk = {
            "chunk_id": "TEST_0",
            "chunk_text": "Test content",
            "chunk_title": "Test Title",
            "chunk_index": 0,
            "document_id": "TEST",
            "document_title": "Test Document",
            "metadata": {"chunk_type": "content"},
        }

        # Object chunk
        class ObjectChunk:
            def __init__(self):
                self.chunk_id = "TEST_0"
                self.chunk_text = "Test content"
                self.chunk_title = "Test Title"
                self.chunk_index = 0
                self.document_id = "TEST"
                self.document_title = "Test Document"
                self.metadata = {"chunk_type": "content"}

        obj_chunk = ObjectChunk()

        # We need a helper that works for both:
        def _get_chunk_field(chunk, field_name, default=None):
            """Get field from chunk whether it's dict or object."""
            if isinstance(chunk, dict):
                return chunk.get(field_name, default)
            else:
                return getattr(chunk, field_name, default)

        # Test that it works for both
        assert _get_chunk_field(dict_chunk, "chunk_text") == "Test content"
        assert _get_chunk_field(obj_chunk, "chunk_text") == "Test content"

        assert _get_chunk_field(dict_chunk, "metadata") == {"chunk_type": "content"}
        assert _get_chunk_field(obj_chunk, "metadata") == {"chunk_type": "content"}

        # And handles missing fields gracefully
        assert _get_chunk_field(dict_chunk, "missing_field", "default") == "default"
        assert _get_chunk_field(obj_chunk, "missing_field", "default") == "default"


class TestVectorDictChunksFix:
    """Test that the fix for dict chunks works correctly."""

    def test_get_chunk_field_works_correctly(self):
        """Test that _get_chunk_field helper function works as expected."""
        from buttermilk.data.vector import _get_chunk_field

        # Test with dict chunk
        dict_chunk = {
            "chunk_id": "TEST_0",
            "chunk_text": "Test content",
            "metadata": {"key": "value"},
        }

        assert _get_chunk_field(dict_chunk, "chunk_id") == "TEST_0"
        assert _get_chunk_field(dict_chunk, "chunk_text") == "Test content"
        assert _get_chunk_field(dict_chunk, "metadata") == {"key": "value"}
        assert _get_chunk_field(dict_chunk, "missing", "default") == "default"

        # Test with object chunk
        class ObjectChunk:
            def __init__(self):
                self.chunk_id = "OBJ_0"
                self.chunk_text = "Object content"
                self.metadata = {"obj_key": "obj_value"}

        obj_chunk = ObjectChunk()

        assert _get_chunk_field(obj_chunk, "chunk_id") == "OBJ_0"
        assert _get_chunk_field(obj_chunk, "chunk_text") == "Object content"
        assert _get_chunk_field(obj_chunk, "metadata") == {"obj_key": "obj_value"}
        assert _get_chunk_field(obj_chunk, "missing", "default") == "default"

    def test_metadata_update_works_with_dict_chunks(self):
        """Test that metadata update logic works with dict chunks."""
        # Simulate what the fixed code does
        dict_chunk = {"chunk_id": "TEST_0", "metadata": {"existing": "value"}}

        # Get metadata safely
        from buttermilk.data.vector import _get_chunk_field

        chunk_metadata = _get_chunk_field(dict_chunk, "metadata", {})

        # Update metadata
        chunk_metadata.update({"new_field": "new_value", "another": "field"})

        # Set it back (this is what the fix does)
        if isinstance(dict_chunk, dict):
            dict_chunk["metadata"] = chunk_metadata
        else:
            dict_chunk.metadata = chunk_metadata

        # Verify it worked
        assert dict_chunk["metadata"]["existing"] == "value"
        assert dict_chunk["metadata"]["new_field"] == "new_value"
        assert dict_chunk["metadata"]["another"] == "field"
