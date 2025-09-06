import json
import logging
from pathlib import Path

import structlog
from google.cloud import logging as gcp_logging
from google.cloud.logging_v2.handlers import CloudLoggingHandler
from rich.logging import RichHandler

from buttermilk._core.context import get_logging_context

# Single logger for the entire application
_LOGGER_NAME = "buttermilk"
logger = structlog.get_logger(_LOGGER_NAME)


# Configure structlog for structured JSON logging
def configure_structlog() -> None:
    """Configure structlog for structured JSON logging."""
    structlog.configure(
        processors=[
            # Add context variables automatically
            structlog.contextvars.merge_contextvars,
            # Add log level
            structlog.processors.add_log_level,
            # Add timestamp
            structlog.processors.TimeStamper(fmt="iso"),
            # Capture exception info if present
            structlog.processors.format_exc_info,
            # Output as JSON
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


class StructlogRichHandler(RichHandler):
    def format(self, record):
        # Convert structlog JSON back to rich format
        if hasattr(record, "msg") and isinstance(record.msg, str):
            try:
                data = json.loads(record.msg)
                record.msg = data.get("event", "")
                # Add structured data as extra context
            except json.JSONDecodeError:
                pass
        return super().format(record)


def setup_console_logging(verbose: bool = False) -> None:
    """Set up beautiful console logging with Rich.

    Args:
        verbose: If True, shows DEBUG level logs on console
    """

    # Clear existing handlers to avoid conflicts
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    # Set logging levels to reduce noise from other libraries
    # root_logger.setLevel(logging.WARNING)
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        if isinstance(logging.Logger.manager.loggerDict[logger_name], logging.Logger):
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Create Rich handler for beautiful console output
    rich_handler = StructlogRichHandler(show_time=True, show_level=True, show_path=False, markup=True, rich_tracebacks=True)

    # Keep buttermilk logger at INFO level
    console_level = logging.INFO  # logging.DEBUG if verbose else logging.INFO
    rich_handler.setLevel(console_level)

    # Add the handler to the root logger so all structlog messages go through it
    logging.getLogger().addHandler(rich_handler)

    # Also ensure structlog is configured for proper integration
    configure_structlog()


def setup_file_logging(execution_context_id: str, verbose: bool = False) -> list[str]:
    """Set up structured JSON logging to files.

    Args:
        execution_context_id: Unique execution context identifier for log file naming
        verbose: If True, creates both INFO and DEBUG log files

    Returns:
        List of log file paths created
    """
    # Configure structlog if not already done
    configure_structlog()

    log_files = []

    # Set up context for this session using simplified architecture
    context = get_logging_context()
    # Bind all available context variables
    non_null_context = {k: v for k, v in context.items() if v is not None}
    if non_null_context:
        structlog.contextvars.bind_contextvars(**non_null_context)

    # Always create an INFO JSON log file
    info_log_path = Path(f"/tmp/buttermilk_{execution_context_id}_info.jsonl")
    info_handler = logging.FileHandler(info_log_path, mode="w")
    info_handler.setLevel(logging.INFO)

    # Use structlog formatter for JSON output with full processing pipeline
    structlog_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            # Add context variables automatically
            structlog.contextvars.merge_contextvars,
            # Add log level
            structlog.processors.add_log_level,
            # Add timestamp
            structlog.processors.TimeStamper(fmt="iso"),
            # Output as JSON
            structlog.processors.JSONRenderer(),
        ],
    )
    info_handler.setFormatter(structlog_formatter)

    # Add to both standard logger and structlog
    logging.getLogger().addHandler(info_handler)
    log_files.append(str(info_log_path))

    # Add debug file logging when verbose is True
    if verbose:
        debug_log_path = Path(f"/tmp/buttermilk_{execution_context_id}_debug.jsonl")
        debug_handler = logging.FileHandler(debug_log_path, mode="w")
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(structlog_formatter)

        logging.getLogger().addHandler(debug_handler)
        log_files.append(str(debug_log_path))

    return log_files


def setup_cloud_logging(logger_cfg, cloud_manager, session_info) -> None:
    """Set up Google Cloud Logging with structured JSON.

    Uses the same structlog JSON format as file logging for consistency.

    Args:
        logger_cfg: Logger configuration object
        cloud_manager: Cloud manager instance for GCS client access
        session_info: Session information
    """
    if logger_cfg and logger_cfg.type == "gcp" and cloud_manager:
        try:
            # Ensure structlog is configured
            configure_structlog()

            cloud_logging_resource = gcp_logging.Resource(
                type="generic_task",
                labels={
                    "project": logger_cfg.project_id,
                    "location": logger_cfg.location,
                    "namespace": session_info.name,
                    "job": session_info.job,
                    "task_id": session_info.session_id,
                },
            )

            # Filter out None values from labels as protobuf doesn't accept them
            raw_labels = session_info.model_dump(include={"session_id", "name", "job", "platform"})
            labels = {k: str(v) for k, v in raw_labels.items() if v is not None}
            
            cloud_handler = CloudLoggingHandler(
                client=cloud_manager.gcs_log_client(logger_cfg),
                resource=cloud_logging_resource,
                name=session_info.name,
                labels=labels,
            )
            cloud_handler.setLevel(logging.INFO)

            # Use structlog JSON formatter for consistency with file logging
            structlog_formatter = structlog.stdlib.ProcessorFormatter(
                processors=[
                    # Add context variables automatically
                    structlog.contextvars.merge_contextvars,
                    # Add log level
                    structlog.processors.add_log_level,
                    # Add timestamp
                    structlog.processors.TimeStamper(fmt="iso"),
                    # Output as JSON
                    structlog.processors.JSONRenderer(),
                ],
            )
            cloud_handler.setFormatter(structlog_formatter)

            # Bind session context for automatic inclusion (simplified architecture)
            context_vars = {
                "job": session_info.job,
                "project": session_info.name,
                "session_id": "unknown",
                "batch_id": "unknown",
            }

            # Add batch context if available
            if session_info.session_id:
                context_vars["session_id"] = session_info.session_id[-12:]  # Last 12 chars for brevity
            if session_info.batch_id:
                context_vars["batch_id"] = session_info.batch_id[-12:]  # Last 12 chars
                
            structlog.contextvars.bind_contextvars(**context_vars)

            # Add to root logger so all log messages go to cloud
            logging.getLogger().addHandler(cloud_handler)
            logger.info("Cloud logging handler added")

        except Exception as e:
            logger.exception(
                "Cloud logging setup failed",
                extra={
                    "error": str(e),
                    "logger_type": logger_cfg.type,
                    "project_id": logger_cfg.project_id,
                    "location": logger_cfg.location,
                },
            )
