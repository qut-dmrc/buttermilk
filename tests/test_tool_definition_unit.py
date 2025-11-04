"""Unit tests for the tool definition system.

Tests the AgentToolDefinition, decorators, and schema validation utilities.
"""

import pytest

from buttermilk._core.schema_validation import (
    SchemaValidationError,
    SchemaValidator,
    coerce_to_schema,
    generate_example_from_schema,
    validate_tool_input,
)
from buttermilk._core.tool_definition import (
    AgentToolDefinition,
)


class TestAgentToolDefinition:
    """Test AgentToolDefinition class."""

    def test_basic_tool_definition(self):
        """Test creating a basic tool definition."""
        tool_def = AgentToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"output": {"type": "string"}}},
        )

        assert tool_def.name == "test_tool"
        assert tool_def.description == "A test tool"
        assert tool_def.permissions == []

    def test_tool_definition_with_mcp_route(self):
        """Test tool definition with MCP route."""
        tool_def = AgentToolDefinition(
            name="analyze",
            description="Analyze data",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            permissions=["read:data"],
        )

        assert tool_def.permissions == ["read:data"]

    def test_to_autogen_schema(self):
        """Test conversion to Autogen tool schema."""
        tool_def = AgentToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            output_schema={"type": "string"},
        )

        schema = tool_def.schema
        schema = tool_def.schema
        assert schema["name"] == "test_tool"
        assert schema["description"] == "A test tool"
        assert schema["parameters"] == tool_def.input_schema


class TestSchemaValidation:
    """Test schema validation utilities."""

    def test_schema_validator_valid(self):
        """Test validator with valid data."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name"]}

        validator = SchemaValidator(schema)
        validator.validate({"name": "John", "age": 30})
        assert validator.is_valid({"name": "Jane"})

    def test_schema_validator_invalid(self):
        """Test validator with invalid data."""
        schema = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}

        validator = SchemaValidator(schema)

        with pytest.raises(SchemaValidationError) as exc_info:
            validator.validate({"count": "not a number"})
        assert "count: " in str(exc_info.value)

        assert not validator.is_valid({})  # Missing required field

    def test_validate_partial(self):
        """Test partial validation (ignoring required)."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}

        validator = SchemaValidator(schema)
        # Full validation would fail without age
        with pytest.raises(SchemaValidationError):
            validator.validate({"name": "John"})

        # Partial validation should pass
        validator.validate_partial({"name": "John"})

    def test_coerce_to_schema(self):
        """Test data coercion to match schema."""
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}, "ratio": {"type": "number"}, "active": {"type": "boolean"}, "name": {"type": "string"}},
        }

        data = {"count": "123", "ratio": "3.14", "active": "true", "name": 42}

        coerced = coerce_to_schema(schema, data)
        assert coerced["count"] == 123
        assert coerced["ratio"] == 3.14
        assert coerced["active"] is True
        assert coerced["name"] == "42"

    def test_generate_example_from_schema(self):
        """Test example generation from schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
                "tags": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                "active": {"type": "boolean"},
            },
            "required": ["name", "age"],
        }

        example = generate_example_from_schema(schema)
        assert isinstance(example, dict)
        assert "name" in example
        assert "age" in example
        assert isinstance(example["name"], str)
        assert isinstance(example["age"], int)
        assert example["age"] >= 0

    def test_tool_input_validation(self):
        """Test tool input validation helper."""
        schema = {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}

        # Valid input
        validated = validate_tool_input(schema, {"x": 3.14})
        assert validated == {"x": 3.14}

        # Invalid input
        with pytest.raises(SchemaValidationError):
            validate_tool_input(schema, {"x": "not a number"})
