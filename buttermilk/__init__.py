import importlib.metadata
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


from ._core.constants import (
    _LOGGER_NAME,
    _TRACER_NAME,
    BASE_DIR,
    BQ_SCHEMA_DIR,
    COL_PREDICTION,
    TEMPLATES_PATH,
)

tracer = trace.get_tracer(_TRACER_NAME)
logger = structlog.get_logger(_LOGGER_NAME)

try:
    __version__ = importlib.metadata.version("buttermilk")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"


class BMAccessor:
    """Descriptor that provides access to the singleton BM instance."""

    @property
    def __class__(self):
        """Make isinstance(bm, BM) work correctly."""
        from ._core.bm_init import BM  # noqa: PLC0415

        return BM

    def __class_getitem__(cls, item):
        """Support type hints like bm: BM."""
        from ._core.bm_init import BM  # noqa: PLC0415

        return BM

    def __getattr__(self, name):  # -> Any:
        from ._core.dmrc import get_bm  # noqa: PLC0415

        return getattr(get_bm(), name)

    def __get__(self, obj, objtype=None) -> "BM":
        from ._core.dmrc import get_bm  # noqa: PLC0415

        if get_bm() is None:
            msg = "BM singleton not initialized. Make sure CLI has been run."
            raise RuntimeError(msg)
        return get_bm()

    def __set__(self, obj, value: "BM") -> None:
        from ._core.dmrc import set_bm  # noqa: PLC0415

        set_bm(value)


# Create a singleton accessor
bm = BMAccessor()


def __getattr__(name):
    """Module-level attribute access to handle runtime BM imports.

    This allows 'from buttermilk import BM' to work at runtime
    without causing circular dependencies during module initialization.
    """
    if name == "BM":
        from ._core.bm_init import BM  # noqa: PLC0415

        return BM
    if name == "bm":
        return BMAccessor()
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


from ._core.bm_init import create_session_bm_async  # noqa: E402
from ._core.config import AgentConfig, AgentVariants  # noqa: E402
from ._core.config_bootstrap import init, init_async  # noqa: E402
from ._core.contract import (  # noqa: E402
    AgentInput,
    AllMessages,
    ConductorRequest,
    ExecutionTrace,
    FlowMessage,
    GroupchatMessageTypes,
    HeartBeat,
    OOBMessages,
    StepRequest,
    SystemPromptMessage,
    TaskProcessingComplete,
    ToolOutput,
    UserResponseMessage,
)
from ._core.exceptions import FatalError, ProcessingError  # noqa: E402
from ._core.execution_context import (  # noqa: E402
    ExecutionContext,
    create_execution_context,
    get_or_create_execution_context,
)
from ._core.llm_core import LLMCore  # noqa: E402

# MCP server utilities - import these FIRST in MCP servers
from .utils.suppress_stdout import (  # noqa: E402
    configure_for_mcp,
    redirect_stdout_to_stderr,
    stdout_redirected_to_stderr,
    stdout_suppressed,
    suppress_stdout_completely,
)

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
    "ConductorRequest",
    "HeartBeat",
    "LLMCore",
    "tracer",
    # Exceptions
    "FatalError",
    "ProcessingError",
    "_LOGGER_NAME",
    # MCP server utilities
    "configure_for_mcp",
    "redirect_stdout_to_stderr",
    "stdout_redirected_to_stderr",
    "stdout_suppressed",
    "suppress_stdout_completely",
    "__version__",
]

# Replace the placeholder BM with the real class now that all imports are complete
if not TYPE_CHECKING:
    from ._core.bm_init import BM as _RealBM

    BM = _RealBM
