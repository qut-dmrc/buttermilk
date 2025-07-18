# Import silence_logs early to suppress noisy log messages - conditional to avoid circular imports
try:
    from buttermilk.utils.silence_logs import silence_task_logs
    # Suppress logs as early as possible during import
    silence_task_logs()
except ImportError:
    # If we can't import silence_task_logs, create a no-op function
    def silence_task_logs():
        pass

from buttermilk._core.dmrc import get_bm, set_bm

from ._core.bm_init import BM
from ._core.constants import BASE_DIR, BQ_SCHEMA_DIR, COL_PREDICTION, TEMPLATES_PATH
from ._core.log import logger

get_buttermilk_instance = get_bm


class BMAccessor:
    """Descriptor that provides access to the singleton BM instance."""

    def __getattr__(self, name):# -> Any:
        return getattr(get_bm(), name)

    def __get__(self, obj, objtype=None) -> BM:
        if get_bm() is None:
            raise RuntimeError("BM singleton not initialized. Make sure CLI has been run.")
        return get_bm()

    def __set__(self, obj, value: BM) -> None:
        set_bm(value)


# Create a singleton accessor
buttermilk = BMAccessor()

__all__ = [
    "BASE_DIR",
    "BM",
    "BQ_SCHEMA_DIR",
    "COL_PREDICTION",
    "TEMPLATES_PATH",
    "buttermilk",         # Export the singleton accessor
    "get_bm",             # Export the getter function
    "get_buttermilk_instance",  # Export the alias for get_bm
    "logger",
    "set_bm",             # Export the setter function
    "silence_task_logs",  # Export the utility for explicit use
]
