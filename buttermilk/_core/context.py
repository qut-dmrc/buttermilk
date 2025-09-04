"""Context variables for three-tier observability in logging.

These context variables support Buttermilk's three-tier architecture:
- ExecutionContext (process-level infrastructure)
- ResearchRun (logical grouping of related research tasks)
- Session (individual task execution)
- Agent (specific worker within a session)

The context variables (`execution_context_id_var`, `research_run_id_var`, 
`session_id_var`, `agent_id_var`) enable hierarchical logging correlation 
across all levels of the system.

The `set_logging_context` function is provided as a convenient utility
to set these variables at appropriate lifecycle points.

NOTE: Developers should integrate calls to `set_logging_context`
at relevant points in the application's lifecycle to ensure that log messages
can be correlated with specific execution contexts, research runs, sessions,
and agent activities.
"""
from contextvars import ContextVar

# Three-tier architecture context variables
execution_context_id_var: ContextVar[str | None] = ContextVar(
    "execution_context_id_var", default=None
)
research_run_id_var: ContextVar[str | None] = ContextVar(
    "research_run_id_var", default=None
)
session_id_var: ContextVar[str | None] = ContextVar(
    "session_id_var", default=None
)
agent_id_var: ContextVar[str | None] = ContextVar(
    "agent_id_var", default=None
)

# Legacy context variable for run_id (session-level identifier)
run_id_var: ContextVar[str | None] = ContextVar(
    "run_id_var", default=None
)


def set_logging_context(
    session_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    execution_context_id: str | None = None,
    research_run_id: str | None = None
) -> None:
    """Sets the logging context for three-tier observability.
    
    Args:
        session_id: Unique identifier for the current session.
        agent_id: Identifier for the specific agent/worker.
        run_id: Legacy run identifier (now session-level).
        execution_context_id: Process-level execution context identifier.
        research_run_id: Research run identifier for grouping related sessions.
    """
    if session_id is not None:
        session_id_var.set(session_id)
    if agent_id is not None:
        agent_id_var.set(agent_id)
    if run_id is not None:
        run_id_var.set(run_id)
    if execution_context_id is not None:
        execution_context_id_var.set(execution_context_id)
    if research_run_id is not None:
        research_run_id_var.set(research_run_id)


def get_logging_context() -> dict[str, str | None]:
    """Get the current logging context values.
    
    Returns:
        Dict containing all current context variable values.
    """
    return {
        "execution_context_id": execution_context_id_var.get(),
        "research_run_id": research_run_id_var.get(),
        "session_id": session_id_var.get(),
        "run_id": run_id_var.get(),
        "agent_id": agent_id_var.get(),
    }


def clear_logging_context() -> None:
    """Clear all logging context variables."""
    execution_context_id_var.set(None)
    research_run_id_var.set(None)
    session_id_var.set(None)
    run_id_var.set(None)
    agent_id_var.set(None)


# Legacy function for backward compatibility
def set_session_context(session_id: str | None, agent_id: str | None = None) -> None:
    """Legacy function for backward compatibility.
    
    Args:
        session_id: Session identifier.
        agent_id: Agent identifier.
    """
    set_logging_context(session_id=session_id, agent_id=agent_id)
