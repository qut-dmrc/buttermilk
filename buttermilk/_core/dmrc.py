from __future__ import annotations

from .bm_init import BM, create_session_bm
from .execution_context import ExecutionContext, get_or_create_execution_context

# This is a singleton pattern for the BM class.
# The bm variable is initialized to None and will be set to an instance of BM
# when the module is imported. This allows other modules to access the same
# instance of BM without creating a new one.
_bm_instance: BM = None  # Private storage  # noqa:


def get_bm() -> BM:
    """Return the singleton BM instance."""
    if _bm_instance is None:
        raise RuntimeError("BM singleton not initialized. Make sure CLI or nb.init() has been run.")
    return _bm_instance


def set_bm(instance: BM) -> None:
    """Set the singleton BM instance."""
    global _bm_instance
    _bm_instance = instance


def initialize_session_bm(
    name: str, job: str, session_id: str | None = None, execution_context: ExecutionContext | None = None, platform: str = "local", **kwargs
) -> BM:
    """Initialize and set a session-scoped BM instance as the global singleton.

    This function provides backward compatibility by creating a session-scoped BM
    and setting it as the global singleton for existing code to access via get_bm().

    Args:
        name: User-defined name for the current session or project.
        job: User-defined name for the specific job or task.
        session_id: Research run identifier. If None, a new unique ID is generated.
        execution_context: ExecutionContext to use. If None, creates a new one.
        platform: Platform where the session is running.
        **kwargs: Additional arguments for SessionInfo.

    Returns:
        BM: The newly created and set session-scoped BM instance.
    """
    # Create execution context if not provided
    if execution_context is None:
        try:
            from .execution_context import get_execution_context
            execution_context = get_execution_context()
        except RuntimeError:
            # No execution context exists, create one with minimal config
            execution_context = get_or_create_execution_context()
    
    # Create session-scoped BM
    bm_instance = create_session_bm(name=name, job=job, session_id=session_id, execution_context=execution_context, platform=platform, **kwargs)
    
    # Set as global singleton for backward compatibility
    set_bm(bm_instance)
    
    return bm_instance
