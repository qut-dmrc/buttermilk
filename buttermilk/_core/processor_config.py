"""Configuration models for Processors in the Unified Architecture.

This module defines the base `ProcessorConfig` and concrete subclasses for specific
processor types. It uses Pydantic's discriminated unions to enable polymorphic
deserialization and type-safe configuration.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ProcessorConfig(BaseModel):
    """Base configuration for all processors.

    Attributes:
        type: Discriminator field for polymorphic deserialization.
        name: Optional unique name for the processor instance (useful for debugging/tracing).
        enabled: specific flag to disable a processor without removing it from config.
    """
    type: str = Field(..., description="Processor type identifier (e.g., 'llm', 'filter').")
    name: Optional[str] = Field(default=None, description="Optional instance name.")
    enabled: bool = Field(default=True, description="Whether this processor is active.")

    model_config = ConfigDict(
        extra="forbid",  # Strict config validation
        populate_by_name=True,
    )


class LLMProcessorConfig(ProcessorConfig):
    """Configuration for LLM-based processors."""
    type: Literal["llm"] = "llm"
    model: str = Field(..., description="Model identifier (e.g., 'gpt-4').")
    prompt_template: str = Field(..., description="Prompt template string or path.")
    temperature: float = Field(default=0.0, description="Sampling temperature.")
    max_tokens: Optional[int] = Field(default=None, description="Max output tokens.")
    
    # Parameters to inject into the prompt
    input_variables: dict[str, Any] = Field(default_factory=dict, description="Static variables for the prompt.")


class FilterProcessorConfig(ProcessorConfig):
    """Configuration for filtering processors."""
    type: Literal["filter"] = "filter"
    criteria: str = Field(..., description="Filtering criteria or expression.")


class ShellProcessorConfig(ProcessorConfig):
    """Configuration for shell command processors."""
    type: Literal["shell"] = "shell"
    command: str = Field(..., description="Shell command to execute.")
    timeout_seconds: int = Field(default=60, description="Command execution timeout.")


class GroupchatProcessorConfig(ProcessorConfig):
    """Configuration for groupchat processors."""
    type: Literal["groupchat"] = "groupchat"
    participants: list[str] = Field(..., description="List of agent roles/names.")
    max_rounds: int = Field(default=10, description="Maximum number of chat rounds.")


class ExpanderProcessorConfig(ProcessorConfig):
    """Configuration for expanding one record into multiple (1:N)."""
    type: Literal["expander"] = "expander"
    field_to_expand: str = Field(..., description="Field containing list to expand.")


class TransformProcessorConfig(ProcessorConfig):
    """Configuration for transform processors like JMESPath."""
    type: Literal["transform"] = "transform"
    expression: str = Field(..., description="JMESPath expression to apply.")
    output_field: str = Field(default="transformed", description="Field to store result in metadata.")


# Discriminated Union for type-safe parsing
# Add new processor config types here
ProcessorConfigUnion = LLMProcessorConfig | FilterProcessorConfig | ShellProcessorConfig | GroupchatProcessorConfig | ExpanderProcessorConfig | TransformProcessorConfig

