"""Context variables for simplified session-centric observability.

These context variables support a simplified architecture:
- Session (individual task execution) - primary identifier
- Batch (optional grouping of related sessions)
- Agent (specific worker within a session)

The context variables (`session_id_var`, `batch_id_var`, `agent_id_var`)
enable clean logging correlation without artificial hierarchy complexity.

The `set_logging_context` function is provided as a convenient utility
to set these variables at appropriate lifecycle points.

NOTE: Developers should integrate calls to `set_logging_context`
at relevant points in the application's lifecycle to ensure that log messages
can be correlated with specific sessions, batches, and agent activities.
"""

from contextvars import ContextVar

# Simplified session-centric context variables
session_id_var: ContextVar[str | None] = ContextVar("session_id_var", default=None)
batch_id_var: ContextVar[str | None] = ContextVar("batch_id_var", default=None)
agent_id_var: ContextVar[str | None] = ContextVar("agent_id_var", default=None)


def set_logging_context(
    session_id: str | None = None,
    batch_id: str | None = None,
    agent_id: str | None = None,
) -> None:
    """Sets the simplified logging context.

    Args:
        session_id: Unique identifier for the current session.
        batch_id: Optional batch identifier for grouping related sessions.
        agent_id: Identifier for the specific agent/worker.
    """
    if session_id is not None:
        session_id_var.set(session_id)
    if batch_id is not None:
        batch_id_var.set(batch_id)
    if agent_id is not None:
        agent_id_var.set(agent_id)


def get_logging_context() -> dict[str, str | None]:
    """Get the current logging context values.

    Returns:
        Dict containing all current context variable values.
    """
    return {
        "session_id": session_id_var.get(),
        "batch_id": batch_id_var.get(),
        "agent_id": agent_id_var.get(),
    }


def clear_logging_context() -> None:
    """Clear all logging context variables."""
    session_id_var.set(None)
    batch_id_var.set(None)
    agent_id_var.set(None)
