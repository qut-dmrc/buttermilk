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
        input (Optional[Any]): Captured input for tracing (set by capture_input).
        outputs (list[Any]): Captured outputs for tracing (accumulated by capture_output).
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

    # I/O Capture for Tracing (typed data flow)
    input: Optional[Any] = None
    outputs: list[Any] = field(default_factory=list)

    def update_metadata(self, key: str, value: Any) -> None:
        """Update metadata with a new key-value pair."""
        self.metadata[key] = value

    def get_resource(self, key: str) -> Any:
        """Retrieve a shared resource by key."""
        return self.resources[key]

    def capture_input(self, input_obj: Any) -> None:
        """Capture the input object for tracing.

        Args:
            input_obj: The input being processed (any BaseModel).
        """
        self.input = input_obj

    def capture_output(self, output_obj: Any) -> None:
        """Capture an output object for tracing.

        Supports 1:N processors by accumulating outputs.

        Args:
            output_obj: An output yielded by the processor (any BaseModel).
        """
        self.outputs.append(output_obj)
