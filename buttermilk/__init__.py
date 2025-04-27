# Import silence_logs early to suppress noisy log messages
from .utils.silence_logs import silence_task_logs
# Suppress logs as early as possible during import
silence_task_logs()

from ._core.defaults import BASE_DIR, BQ_SCHEMA_DIR, COL_PREDICTION, TEMPLATES_PATH
from ._core.log import logger
from .bm import BM

__all__ = [
    "BASE_DIR",
    "BM",
    "BQ_SCHEMA_DIR",
    "COL_PREDICTION",
    "TEMPLATES_PATH",
    "logger",
    "silence_task_logs",  # Export the utility for explicit use
]
