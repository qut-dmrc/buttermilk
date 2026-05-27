"""Tool definition system for structured agent tool invocation.

This module provides the base classes and utilities for defining structured tool
definitions that can be used for LLM tool invocation.
"""

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

from buttermilk._core.tool_types import CancellationToken, ToolSchema


class AgentToolDefinition(BaseModel):
    """Base class for agent tool definitions that implements the Tool protocol.

    Each agent can generate its own structured tool definition that serves multiple purposes:
    - LLM tool definition for structured invocation
    - Input validation schema
    - Documentation and description

    This class implements the Tool protocol, making it directly usable
    with LLM create() calls while remaining non-executable (execution is handled
    by the host agent routing to actual agents).
    """

    name: str = Field(
        ...,
        description="The unique name of the tool",
    )
    description: str = Field(
        ...,
        description="A clear description of what the tool does",
    )
    input_schema: dict[str, Any] = Field(
        ...,
        description="JSON Schema format for input validation",
    )
    output_schema: dict[str, Any] = Field(
        ...,
        description="JSON Schema format for output validation",
    )
    permissions: list[str] = Field(
        default_factory=list,
        description="Required permissions for accessing this tool",
    )

    @field_validator("name", mode="before")
    def validate_agent_name(cls, value: str) -> str:
        """Ensure tool name only has alphanumeric, underscore, or hyphen chars."""
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", value):
            raise ValueError("Tool name must contain only letters, digits, underscores, or hyphens")
        return value

    # Implement Tool protocol properties and methods

    @property
    def schema(self) -> ToolSchema:
        """Return the tool schema in Tool protocol format for the Tool protocol."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.input_schema,
        )

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
            f"AgentToolDefinition '{self.name}' is not directly executable. Tool calls should be intercepted and routed by the host agent.",
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
