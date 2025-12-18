"""Unit tests for JSON schema transformation utilities."""

import pytest

from buttermilk._core.json_schema import resolve_json_schema_refs


class TestResolveJsonSchemaRefs:
    """Test suite for resolve_json_schema_refs functionality."""

    def test_ref_with_description_sibling_gets_inlined(self):
        """Test that $ref with description sibling gets inlined correctly.

        When a property has both a $ref and a description, the resolution should:
        1. Inline the definition from $defs
        2. Preserve the description
        3. Remove the $ref
        4. Remove the $defs section from the schema
        """
        schema = {
            "$defs": {
                "ErrorType": {
                    "enum": ["a", "b"],
                    "type": "string",
                }
            },
            "properties": {
                "error_type": {
                    "$ref": "#/$defs/ErrorType",
                    "description": "The error type",
                }
            },
            "type": "object",
        }

        result = resolve_json_schema_refs(schema)

        # Assert the $ref is gone
        assert "$ref" not in result["properties"]["error_type"]

        # Assert the enum values are inlined
        assert result["properties"]["error_type"]["enum"] == ["a", "b"]
        assert result["properties"]["error_type"]["type"] == "string"

        # Assert the description is preserved
        assert result["properties"]["error_type"]["description"] == "The error type"

        # Assert $defs is removed
        assert "$defs" not in result
