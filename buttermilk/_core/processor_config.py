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
    output_col: str = Field(default="output", description="Field name for LLM output in record.")

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
    flow_name: str = Field(..., description="Name of the flow to execute.")
    flow_config: Any = Field(..., description="Flow configuration (OrchestratorProtocol).")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Parameters for orchestrator.")
    collect_traces: bool = Field(default=True, description="Collect ExecutionTrace outputs.")


class ExpanderProcessorConfig(ProcessorConfig):
    """Configuration for expanding one record into multiple (1:N)."""
    type: Literal["expander"] = "expander"
    field_to_expand: str = Field(..., description="Field containing list to expand.")


class TransformProcessorConfig(ProcessorConfig):
    """Configuration for transform processors like JMESPath."""
    type: Literal["transform"] = "transform"
    expression: str = Field(..., description="JMESPath expression to apply.")
    output_field: str = Field(default="transformed", description="Field to store result in metadata.")


class BatchProcessorConfig(ProcessorConfig):
    """Base config for batch processors.

    Batch processors accumulate records and process them in batches for efficiency.
    Used for operations like GPU inference, bulk API calls, etc.
    """
    batch_size: int = Field(default=32, description="Number of records to batch together.")


class EmbeddingProcessorConfig(BatchProcessorConfig):
    """Configuration for embedding processors."""
    type: Literal["embedding"] = "embedding"
    embedding_model: str = Field(..., description="Embedding model identifier (e.g., 'gemini-embedding-001').")
    dimensionality: int = Field(default=3072, description="Embedding vector dimensionality.")
    task: str = Field(default="RETRIEVAL_DOCUMENT", description="Task type for embeddings.")
    embedding_max_retries: int = Field(default=5, description="Max retries for embedding API calls.")
    embedding_min_wait_seconds: float = Field(default=1.0, description="Min wait between embedding retries.")
    embedding_max_wait_seconds: float = Field(default=120.0, description="Max wait between embedding retries.")
    embedding_cooldown_seconds: float = Field(default=0.1, description="Cooldown between successful embedding calls.")


class ChromaDBProcessorConfig(ProcessorConfig):
    """Configuration for ChromaDB upload processors."""
    type: Literal["chromadb"] = "chromadb"
    collection_name: str = Field(..., description="ChromaDB collection name")
    persist_directory: str = Field(..., description="ChromaDB persist directory (can be remote)")
    sync_batch_size: int = Field(default=50, description="Sync to remote every N records")
    sync_interval_minutes: int = Field(default=10, description="Sync to remote every N minutes")
    disable_auto_sync: bool = Field(default=False, description="Disable automatic syncing (manual only)")
    upsert_batch_size: int = Field(default=1000, description="Batch size for ChromaDB upserts")


# Discriminated Union for type-safe parsing
# Add new processor config types here
ProcessorConfigUnion = LLMProcessorConfig | FilterProcessorConfig | ShellProcessorConfig | GroupchatProcessorConfig | ExpanderProcessorConfig | TransformProcessorConfig | EmbeddingProcessorConfig | ChromaDBProcessorConfig | BatchProcessorConfig

