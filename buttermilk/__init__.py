# # Import silence_logs early to suppress noisy log messages - conditional to avoid circular imports
# try:
#     from buttermilk.utils.silence_logs import silence_task_logs
#     # Suppress logs as early as possible during import
#     silence_task_logs()
# except ImportError:
#     # If we can't import silence_task_logs, create a no-op function
#     def silence_task_logs():
#         pass

from ._core.bm_init import BM, create_batch_session_bm, create_session_bm, tracer
from ._core.config import AgentConfig as AgentConfig, AgentVariants as AgentVariants
from ._core.config_bootstrap import init
from ._core.constants import BASE_DIR, BQ_SCHEMA_DIR, COL_PREDICTION, TEMPLATES_PATH
from ._core.contract import (
    AgentInput as AgentInput,
    ExecutionTrace as ExecutionTrace,
    AllMessages as AllMessages,
    ConductorRequest as ConductorRequest,
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
from ._core.log import logger


class BMAccessor:
    """Descriptor that provides access to the singleton BM instance."""

    def __getattr__(self, name):  # -> Any:
        from ._core.dmrc import get_bm
        return getattr(get_bm(), name)

    def __get__(self, obj, objtype=None) -> BM:
        from ._core.dmrc import get_bm
        if get_bm() is None:
            raise RuntimeError("BM singleton not initialized. Make sure CLI has been run.")
        return get_bm()

    def __set__(self, obj, value: BM) -> None:
        from ._core.dmrc import set_bm
        set_bm(value)


# Create a singleton accessor
bm = BMAccessor()

__all__ = [
    "BASE_DIR",
    "BM",
    "BQ_SCHEMA_DIR",
    "COL_PREDICTION",
    "TEMPLATES_PATH",
    "init",
    "bm",  # Export the singleton accessor (deprecated)
    "logger",
    "initialize_session_bm",  # Initialize session-scoped BM as singleton
    # New session-scoped API
    "create_session_bm",  # Factory for session-scoped BM instances
    "create_batch_session_bm",  # Factory for batch session BM instances
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
    "tracer",
    # Exceptions
    "FatalError",
    "ProcessingError",
]
