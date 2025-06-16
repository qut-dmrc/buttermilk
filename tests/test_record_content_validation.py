"""Test cases for Record content validation.

These tests ensure that Record objects enforce content requirements
and catch configuration issues like missing field mappings.
"""

import pytest
from pydantic import ValidationError

from buttermilk._core.types import Record


class TestRecordContentValidation:
    """Test Record content validation requirements."""

    def test_record_requires_content(self):
        """Test that Record requires non-empty content."""
        # Should fail with no content
        with pytest.raises(ValidationError, match="Field required"):
            Record()

    def test_record_rejects_none_content(self):
        """Test that Record rejects None content."""
        with pytest.raises(ValidationError, match="Input should be a valid string|Input should be an instance of Sequence"):
            Record(content=None)

    def test_record_rejects_empty_string_content(self):
        """Test that Record rejects empty string content."""
        with pytest.raises(ValidationError, match="Content cannot be empty string"):
            Record(content="")
        
        with pytest.raises(ValidationError, match="Content cannot be empty string"):
            Record(content="   ")  # Whitespace only

    def test_record_rejects_empty_sequence_content(self):
        """Test that Record rejects empty sequence content."""
        with pytest.raises(ValidationError, match="Content sequence cannot be empty"):
            Record(content=[])

    def test_record_rejects_meaningless_sequence_content(self):
        """Test that Record rejects sequences with only empty strings."""
        with pytest.raises(ValidationError, match="Content sequence must contain at least one meaningful item"):
            Record(content=["", "   ", ""])

    def test_record_accepts_valid_string_content(self):
        """Test that Record accepts valid string content."""
        record = Record(content="This is valid content")
        assert record.content == "This is valid content"

    def test_record_accepts_valid_sequence_content(self):
        """Test that Record accepts valid sequence content."""
        record = Record(content=["First part", "Second part"])
        assert record.content == ["First part", "Second part"]

    def test_record_accepts_mixed_sequence_with_meaningful_content(self):
        """Test that Record accepts sequences with some empty strings if there's meaningful content."""
        record = Record(content=["", "Valid content", ""])
        assert "Valid content" in record.content

    def test_osb_field_mapping_scenario(self):
        """Test the specific scenario we encountered with OSB data mapping.
        
        This reproduces the issue where content field was not mapped correctly,
        resulting in content being None or ending up in metadata instead.
        """
        # Simulate what happens when field mapping is incorrect
        # Case 1: Content mapped incorrectly, ends up as None
        with pytest.raises(ValidationError, match="Field required"):
            Record(
                record_id="test-id",
                metadata={
                    "title": "Test Document",
                    "fulltext": "This should have been mapped to content field"
                }
                # Missing content field due to incorrect mapping
            )
        
        # Case 2: Content field gets None due to wrong source field
        with pytest.raises(ValidationError, match="Input should be a valid string|Input should be an instance of Sequence"):
            Record(
                record_id="test-id", 
                content=None,  # Would happen if JSON field name is wrong
                metadata={
                    "title": "Test Document",
                    "fulltext": "This text is orphaned in metadata"
                }
            )
        
        # Case 3: Correct mapping should work
        record = Record(
            record_id="test-id",
            content="This should have been mapped to content field",  # Correct mapping
            metadata={
                "title": "Test Document"
            }
        )
        assert record.content == "This should have been mapped to content field"
        assert record.metadata["title"] == "Test Document"
        # Ensure orphaned fulltext doesn't exist in metadata
        assert "fulltext" not in record.metadata

    def test_text_content_property_with_valid_content(self):
        """Test that text_content property works correctly with valid content."""
        record = Record(content="Test content for chunking")
        assert record.text_content == "Test content for chunking"
        
        # Ensure it's long enough for meaningful chunking
        long_content = "This is a longer piece of content. " * 50  # 1750 chars
        record = Record(content=long_content)
        assert len(record.text_content) > 1200  # Should be chunkable with 1200 char chunks

    def test_content_validation_prevents_silent_failures(self):
        """Test that content validation prevents silent failures in vector processing."""
        # This would have caused the "only 1 chunk" issue we saw
        with pytest.raises(ValidationError):
            Record(
                record_id="OSB-123",
                content="",  # Empty content would cause chunking to fail silently
                metadata={
                    "title": "Some Document",
                    "summary": "Document summary"
                }
            )
        
        # Valid case should work
        record = Record(
            record_id="OSB-123", 
            content="This is substantial content that can be chunked into multiple pieces. " * 30,
            metadata={
                "title": "Some Document", 
                "summary": "Document summary"
            }
        )
        assert len(record.text_content) > 1200
        assert record.metadata["title"] == "Some Document"


class TestRecordFieldCounts:
    """Test that Record doesn't have unnecessary fields."""
    
    def test_record_has_reasonable_field_count(self):
        """Test that Record doesn't have too many fields."""
        record = Record(content="test")
        
        # Get all actual fields (not computed properties)
        actual_fields = set(Record.model_fields.keys())
        
        # Expected core fields for a data record
        expected_core_fields = {
            "record_id", "content", "metadata", "alt_text", "ground_truth", "uri", "mime"
        }
        
        # Expected vector processing fields (these were added for the vector workflow)
        expected_vector_fields = {
            "file_path", "full_text", "chunks", "chunks_path"
        }
        
        expected_all_fields = expected_core_fields | expected_vector_fields
        
        # Check that we don't have unexpected extra fields
        unexpected_fields = actual_fields - expected_all_fields
        assert unexpected_fields == set(), f"Unexpected fields found: {unexpected_fields}"
        
        # Check that we have the core fields
        missing_core_fields = expected_core_fields - actual_fields
        assert missing_core_fields == set(), f"Missing core fields: {missing_core_fields}"
        
        print(f"✅ Record has {len(actual_fields)} fields (expected ~{len(expected_all_fields)})")
        print(f"   Core fields: {expected_core_fields}")
        print(f"   Vector fields: {expected_vector_fields}")


class TestStructuredDataHandling:
    """Test that Record handles structured metadata properly (arrays, objects, etc.)."""
    
    def test_record_preserves_structured_metadata(self):
        """Test that Record can store and access structured metadata like OSB data."""
        # Create a record similar to OSB structure
        osb_like_record = Record(
            record_id="BUN-QBBLZ8WI",
            content="This is the fulltext content for vector processing",
            metadata={
                "title": "Mention of Al-Shabaab",
                "case_content": "The first post included a picture showing weapons...",
                "result": "leave up",
                "type": "summary",
                "location": "Somalia", 
                "case_date": "2023-11-22",
                "topics": ["War and conflict", "Dangerous individuals and organizations"],
                "standards": ["Dangerous Individuals and Organizations policy"],
                "reasons": [
                    "The policy prohibits content that 'praises' dangerous organizations...",
                    "In the first post, the caption describes a military operation..."
                ],
                "recommendations": [
                    "Enhance training and accuracy of reviewers...",
                    "Add criteria and illustrative examples..."
                ],
                "job_id": "2Luac3REAVKPnF52dZqtc4",
                "timestamp": 1732052347313
            }
        )
        
        # Verify structured data is preserved
        assert osb_like_record.metadata["topics"] == ["War and conflict", "Dangerous individuals and organizations"]
        assert len(osb_like_record.metadata["reasons"]) == 2
        assert len(osb_like_record.metadata["recommendations"]) == 2
        assert osb_like_record.metadata["timestamp"] == 1732052347313
        
        # Verify content is accessible for vector processing
        assert len(osb_like_record.text_content) > 100
        assert osb_like_record.content == "This is the fulltext content for vector processing"
        
    def test_record_metadata_types_preserved(self):
        """Test that different data types in metadata are preserved correctly."""
        record = Record(
            content="Test content",
            metadata={
                "string_field": "text value",
                "number_field": 42,
                "boolean_field": True,
                "array_field": ["item1", "item2", "item3"],
                "object_field": {"key1": "value1", "key2": "value2"},
                "null_field": None
            }
        )
        
        # Verify types are preserved
        assert isinstance(record.metadata["string_field"], str)
        assert isinstance(record.metadata["number_field"], int)
        assert isinstance(record.metadata["boolean_field"], bool)
        assert isinstance(record.metadata["array_field"], list)
        assert isinstance(record.metadata["object_field"], dict)
        assert record.metadata["null_field"] is None
        
        # Verify values are correct
        assert record.metadata["array_field"] == ["item1", "item2", "item3"]
        assert record.metadata["object_field"]["key1"] == "value1"