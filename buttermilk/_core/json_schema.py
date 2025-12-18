"""JSON schema transformation utilities."""

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
