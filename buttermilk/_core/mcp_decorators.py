"""Decorators and utilities for MCP route generation.

This module provides decorators that enable automatic MCP route generation
from agent methods.
"""

import functools
import inspect
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from buttermilk._core.tool_definition import AgentToolDefinition

F = TypeVar("F", bound=Callable[..., Any])


class MCPRoute:
    """Decorator for marking agent methods as MCP-exposed routes.
    
    Usage:
        @MCPRoute("/analyze", permissions=["read:data"])
        def analyze_data(self, dataset: str, query: str) -> dict:
            ...
    """
    
    def __init__(
        self, 
        path: str | None = None,
        *,
        permissions: list[str] | None = None,
        description: str | None = None,
        include_in_tools: bool = True
    ):
        """Initialize MCPRoute decorator.
        
        Args:
            path: MCP route path (e.g., "/analyze"). If None, uses method name.
            permissions: Required permissions for accessing this route.
            description: Description of what this route does.
            include_in_tools: Whether to include this in agent's tool list.
        """
        self.path = path
        self.permissions = permissions if permissions is not None else []
        self.description = description
        self.include_in_tools = include_in_tools
    
    def __call__(self, func: F) -> F:
        """Apply decorator to function."""
        # Store MCP metadata on the function
        # Use explicit path or generate from function name
        path = self.path if self.path is not None else f"/{func.__name__}"
        
        # Use provided description, then fall back to docstring, then require it
        description = self.description
        if description is None:
            description = func.__doc__
        if not description:
            raise ValueError(f"MCP route for function '{func.__name__}' must have a description or docstring")
        
        func._mcp_route = {
            "path": path,
            "permissions": self.permissions,
            "description": description,
            "include_in_tools": self.include_in_tools
        }
        
        # Important: Don't wrap the function, just add metadata
        # This preserves async functions properly
        return func


def tool(
    func: F | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    include_in_mcp: bool = True
) -> F | Callable[[F], F]:
    """Decorator for marking agent methods as tools.
    
    This is a simpler alternative to MCPRoute that focuses on tool definition
    without specifying MCP routes.
    
    Usage:
        @tool
        def analyze_data(self, dataset: str, query: str) -> dict:
            ...
        
        @tool(name="custom_analyze", description="Analyze dataset")
        def analyze(self, dataset: str) -> dict:
            ...
    """
    def decorator(func: F) -> F:
        # Store tool metadata on the function
        func._tool_metadata = {
            "name": name or func.__name__,
            "description": description or func.__doc__ or "",
            "include_in_mcp": include_in_mcp
        }
        return func
    
    if func is None:
        # Called with arguments: @tool(name="...", ...)
        return decorator
    else:
        # Called without arguments: @tool
        return decorator(func)


def extract_tool_definitions(obj: Any) -> list[AgentToolDefinition]:
    """Extract tool definitions from an object (typically an Agent instance).
    
    This function inspects the object for methods decorated with @tool or @MCPRoute
    and generates AgentToolDefinition objects for each.
    """
    tool_definitions = []
    
    for name, method in inspect.getmembers(obj, inspect.ismethod):
        # Skip private methods
        if name.startswith("_"):
            continue
            
        # Check for MCP route decoration
        if hasattr(method, "_mcp_route"):
            mcp_meta = method._mcp_route
            if not mcp_meta["include_in_tools"]:
                continue
                
            tool_def = _create_tool_definition_from_method(
                method=method,
                name=name,
                description=mcp_meta["description"],
                mcp_route=mcp_meta["path"],
                permissions=mcp_meta["permissions"]
            )
            tool_definitions.append(tool_def)
            
        # Check for tool decoration
        elif hasattr(method, "_tool_metadata"):
            tool_meta = method._tool_metadata
            
            tool_def = _create_tool_definition_from_method(
                method=method,
                name=tool_meta["name"],
                description=tool_meta["description"],
                mcp_route=f"/{tool_meta['name']}" if tool_meta["include_in_mcp"] else None,
                permissions=[]
            )
            tool_definitions.append(tool_def)
    
    return tool_definitions


def _create_tool_definition_from_method(
    method: Callable,
    name: str,
    description: str,
    mcp_route: str | None = None,
    permissions: list[str] | None = None
) -> AgentToolDefinition:
    """Create an AgentToolDefinition from a method using introspection."""
    # Get method signature
    sig = inspect.signature(method)
    
    # Build input schema from parameters
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        # Skip 'self' parameter
        if param_name == "self":
            continue
            
        # Determine parameter type and schema
        param_schema = {"type": "string"}  # Default to string
        
        if param.annotation != inspect.Parameter.empty:
            # Try to infer JSON schema from type annotation
            param_schema = _type_to_json_schema(param.annotation)
        
        # Add description from docstring if available
        # This is a simplified version - could be enhanced with docstring parsing
        param_schema["description"] = f"Parameter {param_name}"
        
        properties[param_name] = param_schema
        
        # Check if parameter is required
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
    
    input_schema = {
        "type": "object",
        "properties": properties,
        "required": required
    }
    
    # Build output schema from return type annotation
    output_schema = {"type": "object"}
    if sig.return_annotation != inspect.Signature.empty:
        output_schema = _type_to_json_schema(sig.return_annotation)
    
    return AgentToolDefinition(
        name=name,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        mcp_route=mcp_route,
        permissions=permissions or []
    )


def _type_to_json_schema(type_hint: Any) -> dict[str, Any]:
    """Convert Python type hint to JSON schema.
    
    This is a simplified implementation that handles common cases.
    Could be enhanced with more comprehensive type mapping.
    """
    import types
    from typing import get_args, get_origin, Union
    
    # Handle None type
    if type_hint is type(None):
        return {"type": "null"}
    
    # Handle basic types
    basic_type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }
    
    if type_hint in basic_type_map:
        return basic_type_map[type_hint]
    
    # Handle generic types (List[str], Dict[str, Any], etc.)
    origin = get_origin(type_hint)
    if origin is not None:
        if origin is list:
            args = get_args(type_hint)
            if args:
                return {
                    "type": "array",
                    "items": _type_to_json_schema(args[0])
                }
            return {"type": "array"}
        
        elif origin is dict:
            return {"type": "object"}
        
        elif origin is Union or (hasattr(types, "UnionType") and origin is getattr(types, "UnionType")):
            # Handle Union types (works for both typing.Union and types.UnionType in 3.10+)
            args = get_args(type_hint)
            if len(args) == 2 and type(None) in args:
                # Optional type
                non_none_type = args[0] if args[1] is type(None) else args[1]
                schema = _type_to_json_schema(non_none_type)
                return {**schema, "nullable": True}
            else:
                # General union - use anyOf
                return {
                    "anyOf": [_type_to_json_schema(arg) for arg in args]
                }
    
    # Handle Pydantic models
    if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
        return type_hint.model_json_schema()
    
    # Default fallback
    return {"type": "string"}