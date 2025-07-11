"""Tool definition system for structured agent tool invocation and MCP routes.

This module provides the base classes and utilities for defining structured tool
definitions that can be used for both LLM tool invocation and MCP route generation.
"""

from typing import Any, Literal

from autogen_core import CancellationToken
from autogen_core.tools import ToolSchema
from pydantic import BaseModel, Field


class AgentToolDefinition(BaseModel):
    """Base class for agent tool definitions that implements the Tool protocol.
    
    Each agent can generate its own structured tool definition that serves multiple purposes:
    - LLM tool definition for structured invocation
    - MCP route specification for remote access
    - Input validation schema
    - Documentation and description
    
    This class implements the autogen Tool protocol, making it directly usable
    with LLM create() calls while remaining non-executable (execution is handled
    by the host agent routing to actual agents).
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
    
    # Implement Tool protocol properties and methods
    
    @property
    def schema(self) -> ToolSchema:
        """Return the tool schema in OpenAI format for the Tool protocol."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema
            }
        }  # type: ignore
    
    def args_type(self) -> type:
        """Return the args type (dict for schema-based tools)."""
        return dict
    
    def return_type(self) -> type:
        """Return the return type (dict for schema-based tools)."""
        return dict
    
    def state_type(self) -> type:
        """Return the state type (None for stateless tools)."""
        return type(None)
    
    async def run_json(self, args_json: str, cancellation_token: CancellationToken) -> str:
        """This method should never be called.
        
        The host agent intercepts tool calls before execution.
        """
        raise NotImplementedError(
            f"AgentToolDefinition '{self.name}' is not directly executable. "
            "Tool calls should be intercepted and routed by the host agent."
        )
    
    def return_value_as_string(self, value: Any) -> str:
        """Convert return value to string."""
        import json
        if isinstance(value, str):
            return value
        return json.dumps(value)
    
    async def save_state_json(self) -> str:
        """No state to save for stateless tools."""
        return "{}"
    
    async def load_state_json(self, state_json: str) -> None:
        """No state to load for stateless tools."""
        pass
    
    
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
    
    This consolidates AgentInput, ToolInput, and StepRequest into a single structure
    and supports both groupchat and MCP invocation contexts.
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
    
    @property
    def is_mcp_request(self) -> bool:
        """Check if this request originated from MCP."""
        return self.metadata.get("source") == "mcp" or "mcp_route" in self.metadata
    
    @property
    def is_groupchat_request(self) -> bool:
        """Check if this request originated from groupchat."""
        return self.metadata.get("source") == "groupchat" or "step_request" in self.metadata
    
    def to_agent_input(self) -> dict[str, Any]:
        """Convert to AgentInput format for legacy compatibility."""
        return {
            "inputs": self.inputs,
            "context": self.context,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_mcp_call(
        cls, 
        tool_name: str, 
        parameters: dict[str, Any], 
        agent_name: str | None = None
    ) -> "UnifiedRequest":
        """Create UnifiedRequest from MCP tool call.
        
        Args:
            tool_name: Name of the tool being called
            parameters: Tool parameters
            agent_name: Optional agent name (will be inferred if not provided)
            
        Returns:
            UnifiedRequest instance
        """
        target = f"{agent_name}.{tool_name}" if agent_name else tool_name
        return cls(
            target=target,
            inputs=parameters,
            metadata={"source": "mcp", "tool_name": tool_name}
        )
    
    @classmethod
    def from_groupchat_step(
        cls,
        agent_role: str,
        inputs: dict[str, Any],
        tool_name: str | None = None
    ) -> "UnifiedRequest":
        """Create UnifiedRequest from groupchat step request.
        
        Args:
            agent_role: Role of the target agent
            inputs: Input parameters
            tool_name: Optional specific tool name
            
        Returns:
            UnifiedRequest instance
        """
        target = f"{agent_role}.{tool_name}" if tool_name else agent_role
        return cls(
            target=target,
            inputs=inputs,
            metadata={"source": "groupchat", "agent_role": agent_role}
        )