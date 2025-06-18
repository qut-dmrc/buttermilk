"""Tool definition system for structured agent tool invocation and MCP routes.

This module provides the base classes and utilities for defining structured tool
definitions that can be used for both LLM tool invocation and MCP route generation.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentToolDefinition(BaseModel):
    """Base class for agent tool definitions.
    
    Each agent can generate its own structured tool definition that serves multiple purposes:
    - LLM tool definition for structured invocation
    - MCP route specification for remote access
    - Input validation schema
    - Documentation and description
    """
    
    name: str = Field(
        ...,
        description="The unique name of the tool, should be snake_case",
        pattern="^[a-z][a-z0-9_]*$"
    )
    description: str = Field(
        ...,
        description="A clear description of what the tool does"
    )
    input_schema: dict[str, Any] = Field(
        ...,
        description="JSON Schema format for input validation"
    )
    output_schema: dict[str, Any] = Field(
        ...,
        description="JSON Schema format for output validation"
    )
    mcp_route: str | None = Field(
        default=None,
        description="Optional MCP route path (e.g., '/analyze')"
    )
    permissions: list[str] = Field(
        default_factory=list,
        description="Required permissions for accessing this tool"
    )
    
    def to_autogen_tool_schema(self) -> dict[str, Any]:
        """Convert to Autogen-compatible tool schema format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema
        }
    
    def to_openai_function_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema
        }
    
    def to_mcp_route_definition(self) -> dict[str, Any] | None:
        """Convert to MCP route definition if mcp_route is specified."""
        if not self.mcp_route:
            return None
            
        return {
            "path": self.mcp_route,
            "method": "POST",
            "handler": self.name,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "permissions": self.permissions,
            "description": self.description
        }


class MCPServerConfig(BaseModel):
    """Configuration for MCP server operation modes."""
    
    mode: Literal["embedded", "daemon"] = Field(
        default="embedded",
        description="Server operation mode"
    )
    port: int = Field(
        default=8787,
        description="Port for MCP server (daemon mode only)"
    )
    auth_required: bool = Field(
        default=True,
        description="Whether authentication is required for MCP routes"
    )
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed CORS origins for MCP routes"
    )


class UnifiedRequest(BaseModel):
    """Single request format for all agent/tool invocations.
    
    This consolidates AgentInput, ToolInput, and StepRequest into a single structure.
    """
    
    target: str = Field(
        ...,
        description="Target in format 'agent_name.tool_name' or just 'agent_name'"
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Input data validated against tool's input_schema"
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Shared context across agents"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Request metadata (auth, trace_id, etc.)"
    )
    
    @property
    def agent_name(self) -> str:
        """Extract agent name from target."""
        return self.target.split(".")[0]
    
    @property
    def tool_name(self) -> str | None:
        """Extract tool name from target if specified."""
        parts = self.target.split(".")
        return parts[1] if len(parts) > 1 else None