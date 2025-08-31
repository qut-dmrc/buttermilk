# # Import silence_logs early to suppress noisy log messages - conditional to avoid circular imports
# try:
#     from buttermilk.utils.silence_logs import silence_task_logs
#     # Suppress logs as early as possible during import
#     silence_task_logs()
# except ImportError:
#     # If we can't import silence_task_logs, create a no-op function
#     def silence_task_logs():
#         pass

from ._core.bm_init import BM, logger, tracer
from ._core.config import AgentConfig as AgentConfig, AgentVariants as AgentVariants
from ._core.constants import BASE_DIR, BQ_SCHEMA_DIR, COL_PREDICTION, TEMPLATES_PATH
from ._core.contract import (
    AgentInput as AgentInput,
    AgentTrace as AgentTrace,
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
from ._core.dmrc import get_bm, set_bm

get_buttermilk_instance = get_bm


class BMAccessor:
    """Descriptor that provides access to the singleton BM instance."""

    def __getattr__(self, name):  # -> Any:
        return getattr(get_bm(), name)

    def __get__(self, obj, objtype=None) -> BM:
        if get_bm() is None:
            raise RuntimeError("BM singleton not initialized. Make sure CLI has been run.")
        return get_bm()

    def __set__(self, obj, value: BM) -> None:
        set_bm(value)


# Create a singleton accessor
bm = BMAccessor()

__all__ = [
    "BASE_DIR",
    "BM",
    "BQ_SCHEMA_DIR",
    "COL_PREDICTION",
    "TEMPLATES_PATH",
    "bm",  # Export the singleton accessor
    "get_bm",  # Export the getter function
    "get_buttermilk_instance",  # Export the alias for get_bm
    "logger",
    "set_bm",  # Export the setter function
    "AgentConfig",
    "AgentVariants",
    "StepRequest",
    "FlowMessage",
    "AgentInput",
    "AgentTrace",
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
]
