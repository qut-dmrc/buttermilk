"""Context variables for session and agent identification in logging.

These context variables (`session_id_var`, `agent_id_var`) are designed to hold
session-specific and agent-specific identifiers. They are intended to be set
by the application at appropriate points, for example:
- When a new user session begins.
- When a specific agent or worker starts processing a task.

The `set_logging_context` function is provided as a convenient utility
to set these variables.

NOTE: Developers should integrate calls to `set_logging_context`
at relevant points in the application's lifecycle to ensure that log messages
can be correlated with specific sessions or agent activities.
"""
from contextvars import ContextVar

session_id_var: ContextVar[str | None] = ContextVar(
    "session_id_var", default=None,
)
agent_id_var: ContextVar[str | None] = ContextVar("agent_id_var", default=None)


def set_logging_context(session_id: str | None, agent_id: str | None = None) -> None:
    """Sets the session_id and agent_id for logging context."""
    session_id_var.set(session_id)
    agent_id_var.set(agent_id)
