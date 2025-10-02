# # Import silence_logs early to suppress noisy log messages - conditional to avoid circular imports
# try:
#     from buttermilk.utils.silence_logs import silence_task_logs
#     # Suppress logs as early as possible during import
#     silence_task_logs()
# except ImportError:
#     # If we can't import silence_task_logs, create a no-op function
#     def silence_task_logs():
#         pass
from typing import TYPE_CHECKING

import structlog
from opentelemetry import trace

# Conditional import of BM for type checking only
if TYPE_CHECKING:
    from ._core.bm_init import BM
else:
    # Create a placeholder class for runtime
    class BM:
        """Placeholder for BM type - actual class is in _core.bm_init"""

        pass


from ._core.constants import _LOGGER_NAME, _TRACER_NAME, BASE_DIR, BQ_SCHEMA_DIR, COL_PREDICTION, TEMPLATES_PATH

tracer = trace.get_tracer(_TRACER_NAME)
logger = structlog.get_logger(_LOGGER_NAME)


class BMAccessor:
    """Descriptor that provides access to the singleton BM instance."""

    @property
    def __class__(self):
        """Make isinstance(bm, BM) work correctly."""
        from ._core.bm_init import BM

        return BM

    def __class_getitem__(cls, item):
        """Support type hints like bm: BM."""
        from ._core.bm_init import BM

        return BM

    def __getattr__(self, name):  # -> Any:
        from ._core.dmrc import get_bm

        return getattr(get_bm(), name)

    def __get__(self, obj, objtype=None) -> "BM":
        from ._core.dmrc import get_bm

        if get_bm() is None:
            raise RuntimeError("BM singleton not initialized. Make sure CLI has been run.")
        return get_bm()

    def __set__(self, obj, value: "BM") -> None:
        from ._core.dmrc import set_bm

        set_bm(value)


# Create a singleton accessor
bm = BMAccessor()


def __getattr__(name):
    """Module-level attribute access to handle runtime BM imports.

    This allows 'from buttermilk import BM' to work at runtime
    without causing circular dependencies during module initialization.
    """
    if name == "BM":
        from ._core.bm_init import BM

        return BM
    if name == "bm":
        return BMAccessor()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from ._core.bm_init import create_session_bm_async
from ._core.config import AgentConfig as AgentConfig, AgentVariants as AgentVariants
from ._core.config_bootstrap import init, init_async
from ._core.contract import (
    AgentInput as AgentInput,
    AllMessages as AllMessages,
    ConductorRequest as ConductorRequest,
    ExecutionTrace as ExecutionTrace,
    FlowMessage as FlowMessage,
    GroupchatMessageTypes as GroupchatMessageTypes,
    HeartBeat as HeartBeat,
    OOBMessages as OOBMessages,
    ProceedToNextTaskSignal as ProceedToNextTaskSignal,
    StepRequest as StepRequest,
    SystemPromptMessage as SystemPromptMessage,
    TaskProcessingComplete as TaskProcessingComplete,
    ToolOutput as ToolOutput,
    UserResponseMessage as UserResponseMessage,
)
from ._core.exceptions import FatalError, ProcessingError
from ._core.execution_context import ExecutionContext, create_execution_context, get_or_create_execution_context
from ._core.llm_core import LLMCore

__all__ = [
    "BASE_DIR",
    "BQ_SCHEMA_DIR",
    "COL_PREDICTION",
    "TEMPLATES_PATH",
    "init",  # Sync wrapper (deprecated)
    "init_async",  # PRIMARY async initialization
    "bm",  # Export the singleton accessor (deprecated)
    "BM",  # Export the BM class for type hints
    "logger",
    "create_session_bm_async",
    "initialize_session_bm",  # Initialize session-scoped BM as singleton
    # New session-scoped API
    "ExecutionContext",  # Execution context class
    "create_execution_context",  # Factory for execution context
    "get_or_create_execution_context",  # Safe factory for execution context
    # Agent contracts
    "AgentConfig",
    "AgentVariants",
    "StepRequest",
    "FlowMessage",
    "AgentInput",
    "ExecutionTrace",
    "UserResponseMessage",
    "SystemPromptMessage",
    "TaskProcessingComplete",
    "OOBMessages",
    "ToolOutput",
    "AllMessages",
    "GroupchatMessageTypes",
    "OOBMessages",
    "ProceedToNextTaskSignal",
    "ConductorRequest",
    "HeartBeat",
    "LLMCore",
    "tracer",
    # Exceptions
    "FatalError",
    "ProcessingError",
    "_LOGGER_NAME",
]

# Replace the placeholder BM with the real class now that all imports are complete
if not TYPE_CHECKING:
    from ._core.bm_init import BM as _RealBM

    BM = _RealBM
