"""JSON schema transformation utilities."""

from __future__ import annotations

import copy
from typing import Any


def resolve_json_schema_refs(schema: dict) -> dict:
    """Resolve $ref references in a JSON schema by inlining definitions.

    Takes a JSON schema with $defs and $ref nodes, and returns a new schema
    with all references inlined. Preserves sibling keys (like description)
    that appear alongside $ref nodes.

    Args:
        schema: JSON schema dictionary containing $defs and $ref nodes

    Returns:
        New schema with all $refs resolved and $defs removed

    Raises:
        ValueError: If a $ref points to a missing definition or has unsupported format
    """
    # Deep copy to avoid mutating input
    result = copy.deepcopy(schema)

    # Extract definitions
    defs = result.get("$defs", {})

    # Recursively resolve all $refs
    _resolve_refs_recursive(result, defs)

    # Remove $defs from result
    if "$defs" in result:
        del result["$defs"]

    return result


def _resolve_refs_recursive(node: Any, defs: dict[str, Any]) -> None:
    """Recursively find and resolve $ref nodes in place.

    Args:
        node: Current node in the schema tree
        defs: Dictionary of definitions from $defs section

    Raises:
        ValueError: If $ref format is unsupported or definition is missing
    """
    if not isinstance(node, dict):
        return

    # If this node has a $ref, resolve it
    if "$ref" in node:
        ref_value = node["$ref"]

        # Parse the $ref - expect format "#/$defs/DefinitionName"
        if not ref_value.startswith("#/$defs/"):
            raise ValueError(f"Unsupported $ref format: {ref_value}")

        def_name = ref_value.split("/")[-1]

        # Check if definition exists
        if def_name not in defs:
            raise ValueError(f"Missing definition: {def_name}")

        # Get the definition
        definition = defs[def_name]

        # Merge definition into node (preserving siblings like description)
        # Store siblings first
        siblings = {k: v for k, v in node.items() if k != "$ref"}

        # Clear node and add definition content
        node.clear()
        node.update(definition)

        # Re-add siblings (they override definition if there are conflicts)
        node.update(siblings)

    # Recurse into child nodes
    for value in node.values():
        if isinstance(value, dict):
            _resolve_refs_recursive(value, defs)
        elif isinstance(value, list):
            for item in value:
                _resolve_refs_recursive(item, defs)


def make_all_properties_required(schema: dict) -> dict:
    """Make all properties in a JSON schema required.

    Azure OpenAI strict mode requires ALL properties to be in the required array,
    even those with defaults. This function recursively finds all objects with
    properties and ensures their required array contains all property names.

    Args:
        schema: JSON schema dictionary

    Returns:
        New schema with all properties marked as required
    """
    # Deep copy to avoid mutating input
    result = copy.deepcopy(schema)

    # Recursively make all properties required
    _make_properties_required_recursive(result)

    return result


def _make_properties_required_recursive(node: Any) -> None:
    """Recursively find objects with properties and make all properties required.

    Args:
        node: Current node in the schema tree
    """
    if not isinstance(node, dict):
        return

    # If this node has properties, set required to all property names
    if "properties" in node:
        property_names = list(node["properties"].keys())
        node["required"] = property_names

    # Recurse into child nodes
    for value in node.values():
        if isinstance(value, dict):
            _make_properties_required_recursive(value)
        elif isinstance(value, list):
            for item in value:
                _make_properties_required_recursive(item)


def convert_enum_values_to_strings(obj: Any) -> Any:
    """Recursively convert integer enum values to strings for Vertex AI compatibility.

    Vertex AI requires enum values to be strings, not integers.
    This converts any integer values in enum arrays to their string representation.
    Also updates 'type' to 'string' if it was 'integer' when an enum is present.

    Args:
        obj: JSON schema node to process

    Returns:
        Processed schema with string enum values
    """
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if k == "enum" and isinstance(v, list):
                new_obj[k] = [str(item) for item in v]
            else:
                new_obj[k] = convert_enum_values_to_strings(v)

        # If we have an enum, ensure type is string if it was integer or number
        if "enum" in new_obj:
            if new_obj.get("type") in ("integer", "number"):
                new_obj["type"] = "string"

        return new_obj
    elif isinstance(obj, list):
        return [convert_enum_values_to_strings(item) for item in obj]
    return obj


def prepare_schema_for_vertex(schema: type, is_gemini: bool = False) -> dict[str, Any]:
    """Prepare a Pydantic model's JSON schema for use with Vertex AI APIs.

    Applies all necessary transforms: resolve $refs, make all properties required,
    and convert enum values to strings (for Gemini models).

    Args:
        schema: Pydantic model class with model_json_schema()
        is_gemini: If True, also convert enum values to strings

    Returns:
        Transformed JSON schema dictionary ready for Vertex AI
    """
    schema_dict = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema.schema()
    schema_dict = resolve_json_schema_refs(schema_dict)
    schema_dict = make_all_properties_required(schema_dict)
    if is_gemini:
        schema_dict = convert_enum_values_to_strings(schema_dict)
    return schema_dict
