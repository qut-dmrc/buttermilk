"""ProcessingContext for Unified Processor Architecture.

This module defines the ProcessingContext, which serves as the unified state container
passed between processors in a pipeline. It aggregates:
- Session resources (global state, database connections)
- Accumulated data (records, artifacts)
- Execution traces (OpenTelemetry spans, execution logs)
- Interaction hooks (UI callbacks)

The ProcessingContext replaces fragmented state management (global vars, disparate
config objects) with a single, type-safe context object.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from opentelemetry.trace import Span

from buttermilk._core.types import BaseRecord


@dataclass
class ProcessingContext:
    """Unified context passed through the processor chain.

    Holds all state required for processing, including session-scoped resources,
    per-request accumulated data, and observability handles.

    Attributes:
        session_id (str): Unique identifier for the processing session.
        batch_id (Optional[str]): Identifier for the batch if part of a batch job.
        record (BaseRecord): The current record being processed.
        metadata (dict[str, Any]): Mutable dictionary for accumulating intermediate results
            and state shared across processors in the chain.
        span (Optional[Span]): Current OpenTelemetry span for the active processing stage.
        ui_callback (Optional[Callable[[Any], None]]): Callback function for sending
            updates to a user interface (e.g., streaming tokens, progress updates).
        resources (dict[str, Any]): Shared resources like database connections or clients,
            typically initialized once per session.
    """

    # Identity
    session_id: str
    record: BaseRecord
    batch_id: Optional[str] = None

    # State
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Observability
    span: Optional[Span] = None
    
    # Interaction
    ui_callback: Optional[Callable[[Any], None]] = None
    
    # Shared Resources (injected by Executor)
    resources: dict[str, Any] = field(default_factory=dict)

    def update_metadata(self, key: str, value: Any) -> None:
        """Update metadata with a new key-value pair."""
        self.metadata[key] = value

    def get_resource(self, key: str) -> Any:
        """Retrieve a shared resource by key."""
        return self.resources[key]
