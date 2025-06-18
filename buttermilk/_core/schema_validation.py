"""Schema validation utilities for tool definitions.

This module provides utilities for validating data against JSON schemas
used in tool definitions.
"""

from typing import Any

import jsonschema
from jsonschema import Draft7Validator, ValidationError
from pydantic import BaseModel


class SchemaValidationError(Exception):
    """Raised when data fails schema validation."""
    
    def __init__(self, message: str, errors: list[ValidationError] | None = None):
        super().__init__(message)
        self.errors = errors or []


class SchemaValidator:
    """Validates data against JSON schemas."""
    
    def __init__(self, schema: dict[str, Any]):
        """Initialize validator with a JSON schema.
        
        Args:
            schema: JSON schema dictionary following Draft 7 specification.
        """
        self.schema = schema
        self._validator = Draft7Validator(schema)
    
    def validate(self, data: Any) -> None:
        """Validate data against the schema.
        
        Args:
            data: Data to validate.
            
        Raises:
            SchemaValidationError: If validation fails.
        """
        errors = list(self._validator.iter_errors(data))
        if errors:
            error_messages = []
            for error in errors:
                path = ".".join(str(p) for p in error.path) if error.path else "root"
                error_messages.append(f"{path}: {error.message}")
            
            raise SchemaValidationError(
                f"Schema validation failed: {'; '.join(error_messages)}",
                errors=errors
            )
    
    def is_valid(self, data: Any) -> bool:
        """Check if data is valid against the schema.
        
        Args:
            data: Data to validate.
            
        Returns:
            True if valid, False otherwise.
        """
        return self._validator.is_valid(data)
    
    def validate_partial(self, data: dict[str, Any]) -> None:
        """Validate a partial object (ignoring required fields).
        
        Useful for validating updates where not all fields are provided.
        
        Args:
            data: Partial data to validate.
            
        Raises:
            SchemaValidationError: If validation fails.
        """
        # Create a copy of the schema without required fields
        partial_schema = self.schema.copy()
        if "required" in partial_schema:
            partial_schema.pop("required")
        
        partial_validator = Draft7Validator(partial_schema)
        errors = list(partial_validator.iter_errors(data))
        
        if errors:
            error_messages = []
            for error in errors:
                path = ".".join(str(p) for p in error.path) if error.path else "root"
                error_messages.append(f"{path}: {error.message}")
            
            raise SchemaValidationError(
                f"Partial schema validation failed: {'; '.join(error_messages)}",
                errors=errors
            )


def validate_tool_input(
    tool_schema: dict[str, Any],
    input_data: dict[str, Any]
) -> dict[str, Any]:
    """Validate input data against a tool's input schema.
    
    Args:
        tool_schema: The tool's input schema.
        input_data: Input data to validate.
        
    Returns:
        The validated input data.
        
    Raises:
        SchemaValidationError: If validation fails.
    """
    validator = SchemaValidator(tool_schema)
    validator.validate(input_data)
    return input_data


def validate_tool_output(
    tool_schema: dict[str, Any],
    output_data: Any
) -> Any:
    """Validate output data against a tool's output schema.
    
    Args:
        tool_schema: The tool's output schema.
        output_data: Output data to validate.
        
    Returns:
        The validated output data.
        
    Raises:
        SchemaValidationError: If validation fails.
    """
    validator = SchemaValidator(tool_schema)
    validator.validate(output_data)
    return output_data


def coerce_to_schema(
    schema: dict[str, Any],
    data: dict[str, Any]
) -> dict[str, Any]:
    """Attempt to coerce data to match schema types.
    
    This function tries to convert data types to match the schema where possible.
    For example, converting string "123" to integer 123 if schema expects integer.
    
    Args:
        schema: JSON schema to coerce to.
        data: Data to coerce.
        
    Returns:
        Coerced data.
    """
    if schema.get("type") == "object" and isinstance(data, dict):
        properties = schema.get("properties", {})
        coerced = {}
        
        for key, value in data.items():
            if key in properties:
                prop_schema = properties[key]
                coerced[key] = _coerce_value(prop_schema, value)
            else:
                # Keep unknown properties as-is unless schema forbids them
                if not schema.get("additionalProperties", True):
                    continue
                coerced[key] = value
        
        return coerced
    
    return _coerce_value(schema, data)


def _coerce_value(schema: dict[str, Any], value: Any) -> Any:
    """Coerce a single value to match schema type."""
    schema_type = schema.get("type")
    
    if schema_type == "string":
        return str(value) if value is not None else None
    
    elif schema_type == "integer":
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return value
        elif isinstance(value, (int, float)):
            return int(value)
    
    elif schema_type == "number":
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value
        elif isinstance(value, (int, float)):
            return float(value)
    
    elif schema_type == "boolean":
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "1")
        return bool(value)
    
    elif schema_type == "array" and isinstance(value, list):
        items_schema = schema.get("items", {})
        return [_coerce_value(items_schema, item) for item in value]
    
    elif schema_type == "object" and isinstance(value, dict):
        return coerce_to_schema(schema, value)
    
    # Handle nullable types
    if schema.get("nullable") and value is None:
        return None
    
    # Handle anyOf/oneOf
    if "anyOf" in schema:
        for sub_schema in schema["anyOf"]:
            try:
                validator = SchemaValidator(sub_schema)
                if validator.is_valid(value):
                    return _coerce_value(sub_schema, value)
            except:
                continue
    
    return value


def merge_schemas(*schemas: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple JSON schemas into one.
    
    Useful for combining schemas from multiple tools or creating
    composite schemas.
    
    Args:
        *schemas: Variable number of schema dictionaries to merge.
        
    Returns:
        Merged schema.
    """
    if not schemas:
        return {}
    
    if len(schemas) == 1:
        return schemas[0].copy()
    
    # Start with the first schema as base
    merged = schemas[0].copy()
    
    for schema in schemas[1:]:
        # Merge type
        if "type" in schema:
            if "type" in merged and merged["type"] != schema["type"]:
                # Different types - use anyOf
                merged = {
                    "anyOf": [
                        {"type": merged["type"]},
                        {"type": schema["type"]}
                    ]
                }
            else:
                merged["type"] = schema["type"]
        
        # Merge properties (for object types)
        if "properties" in schema:
            if "properties" not in merged:
                merged["properties"] = {}
            merged["properties"].update(schema["properties"])
        
        # Merge required fields
        if "required" in schema:
            if "required" not in merged:
                merged["required"] = []
            merged["required"].extend(
                field for field in schema["required"]
                if field not in merged["required"]
            )
        
        # Merge other fields
        for key, value in schema.items():
            if key not in ["type", "properties", "required"]:
                merged[key] = value
    
    return merged


def generate_example_from_schema(schema: dict[str, Any]) -> Any:
    """Generate an example value that matches the given schema.
    
    Useful for documentation and testing.
    
    Args:
        schema: JSON schema to generate example for.
        
    Returns:
        Example value matching the schema.
    """
    schema_type = schema.get("type", "string")
    
    if schema_type == "string":
        if "enum" in schema:
            return schema["enum"][0]
        elif "pattern" in schema:
            return f"string matching {schema['pattern']}"
        return "example string"
    
    elif schema_type == "integer":
        if "minimum" in schema:
            return schema["minimum"]
        return 42
    
    elif schema_type == "number":
        if "minimum" in schema:
            return float(schema["minimum"])
        return 3.14
    
    elif schema_type == "boolean":
        return True
    
    elif schema_type == "array":
        items_schema = schema.get("items", {"type": "string"})
        min_items = schema.get("minItems", 1)
        return [
            generate_example_from_schema(items_schema)
            for _ in range(min_items)
        ]
    
    elif schema_type == "object":
        example = {}
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        # Add all required properties
        for prop in required:
            if prop in properties:
                example[prop] = generate_example_from_schema(properties[prop])
        
        # Add some optional properties for illustration
        for prop, prop_schema in properties.items():
            if prop not in example and len(example) < 3:
                example[prop] = generate_example_from_schema(prop_schema)
        
        return example
    
    elif schema_type == "null":
        return None
    
    # Handle anyOf by using first option
    if "anyOf" in schema:
        return generate_example_from_schema(schema["anyOf"][0])
    
    return None