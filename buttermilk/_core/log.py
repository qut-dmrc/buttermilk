import json
import logging
from pathlib import Path
from typing import Any

import structlog
from google.cloud import logging as gcp_logging
from google.cloud.logging_v2.handlers import CloudLoggingHandler
from rich.logging import RichHandler

from buttermilk._core.context import get_logging_context

# Single logger for the entire application
_LOGGER_NAME = "buttermilk"
logger = structlog.get_logger(_LOGGER_NAME)

# Global state tracking for logging initialization protection
_console_logging_configured = False
_file_logging_configured = False
_structlog_configured = False
_cloud_logging_sessions = set()  # Track which sessions have cloud logging configured


# Configure structlog for structured JSON logging
def configure_structlog(min_level) -> None:
    """Configure structlog for structured JSON logging.

    Ensures structlog is configured only once to prevent reconfiguration issues
    that can cause mixed formatted output.
    """
    global _structlog_configured

    if _structlog_configured:
        return  # Already configured, skip to prevent breaking existing setup

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
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),  # Use DEBUG to support all handlers
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _structlog_configured = True


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
        
    Raises:
        RuntimeError: If console logging has already been configured
    """
    global _console_logging_configured
    
    if _console_logging_configured:
        raise RuntimeError(
            "Console logging has already been configured. "
            "Multiple calls to setup_console_logging() can break verbose logging functionality. "
            "This indicates a problematic initialization sequence."
        )

    # Clear existing handlers to avoid conflicts
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    # Set logging levels to reduce noise from other libraries
    root_logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        if isinstance(logging.Logger.manager.loggerDict[logger_name], logging.Logger):
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Ensure buttermilk logger respects verbose setting
    logging.getLogger(_LOGGER_NAME).setLevel(logging.DEBUG if verbose else logging.INFO)

    # Create Rich handler for beautiful console output
    rich_handler = StructlogRichHandler(show_time=True, show_level=True, show_path=False, markup=True, rich_tracebacks=True)

    # Keep buttermilk logger at INFO level
    console_level = logging.INFO  # logging.DEBUG if verbose else logging.INFO
    rich_handler.setLevel(console_level)

    # Add handler only to buttermilk logger for structured output
    logging.getLogger(_LOGGER_NAME).addHandler(rich_handler)

    # Also ensure structlog is configured for proper integration
    struct_level = logging.DEBUG if verbose else logging.INFO
    configure_structlog(min_level=struct_level)
    
    # Mark console logging as configured
    _console_logging_configured = True
    logger.debug(f"Console logging configured with verbose={verbose}")


def setup_file_logging(execution_context_id: str, verbose: bool = False) -> list[str]:
    """Set up structured JSON logging to files.

    Args:
        execution_context_id: Unique execution context identifier for log file naming
        verbose: If True, creates both INFO and DEBUG log files

    Returns:
        List of log file paths created
        
    Raises:
        RuntimeError: If file logging has already been configured
    """
    global _file_logging_configured
    
    if _file_logging_configured:
        raise RuntimeError(
            "File logging has already been configured. "
            "Multiple calls to setup_file_logging() can break verbose logging functionality "
            "and create conflicting log file handlers. This indicates a problematic initialization sequence."
        )
    # Configure structlog if not already done
    struct_level = logging.DEBUG if verbose else logging.INFO
    configure_structlog(min_level=struct_level)
    
    # Ensure root logger level allows DEBUG messages when verbose
    root_logger = logging.getLogger()
    if verbose:
        root_logger.setLevel(logging.DEBUG)
    
    # Ensure buttermilk logger respects verbose setting
    logging.getLogger(_LOGGER_NAME).setLevel(logging.DEBUG if verbose else logging.INFO)

    log_files = []

    # Set up context for this session using simplified architecture
    context = get_logging_context()
    # Bind all available context variables
    non_null_context = {k: v for k, v in context.items() if v is not None}
    if non_null_context:
        structlog.contextvars.bind_contextvars(**non_null_context)

    # Create single JSON log file with level based on verbose setting
    log_path = Path(f"/tmp/buttermilk_{execution_context_id}.jsonl")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    # Since all logs come through structlog's stdlib bridge, they're already processed
    # Just extract the pre-formatted message from the LogRecord
    file_handler.setFormatter(logging.Formatter('%(message)s'))

    # Add handler only to buttermilk logger, not root logger
    # This ensures we only get structured logs from our code
    logging.getLogger(_LOGGER_NAME).addHandler(file_handler)
    log_files.append(str(log_path))

    logger.info("Log file created", log_path=str(log_path), verbose=verbose)
    if verbose:
        logger.debug("Verbose logging enabled.")
    
    # Mark file logging as configured
    _file_logging_configured = True

    return log_files


def setup_cloud_logging(logger_cfg, cloud_manager, session_info) -> None:
    """Set up Google Cloud Logging with structured JSON.

    Uses the same structlog JSON format as file logging for consistency.
    Prevents duplicate handlers for the same session.

    Args:
        logger_cfg: Logger configuration object
        cloud_manager: Cloud manager instance for GCS client access
        session_info: Session information
    """
    global _cloud_logging_sessions
    
    # Check if cloud logging is already configured for this session
    session_key = f"{session_info.session_id}:{logger_cfg.project_id}"
    if session_key in _cloud_logging_sessions:
        logger.debug(
            "Cloud logging already configured for this session",
            session_id=session_info.session_id,
            project_id=logger_cfg.project_id
        )
        return
    if logger_cfg and logger_cfg.type == "gcp" and cloud_manager:
        try:
            cloud_logging_resource = gcp_logging.Resource(
                type="generic_task",
                labels={
                    "project": logger_cfg.project_id,
                    "location": logger_cfg.location,
                    "namespace": session_info.project_name,
                    "job": session_info.job,
                    "task_id": session_info.session_id,
                },
            )

            # Filter out None values from labels as protobuf doesn't accept them
            raw_labels = session_info.model_dump(include={"session_id", "project_name", "job", "platform"})
            labels = {k: str(v) for k, v in raw_labels.items() if v is not None}
            
            cloud_handler = CloudLoggingHandler(
                client=cloud_manager.gcs_log_client(logger_cfg),
                resource=cloud_logging_resource,
                name=session_info.project_name,
                labels=labels,
            )
            cloud_handler.setLevel(logging.INFO)

            # Since all logs come through structlog's stdlib bridge, they're already processed
            # Just extract the pre-formatted message from the LogRecord
            cloud_handler.setFormatter(logging.Formatter('%(message)s'))

            # Bind session context for automatic inclusion (simplified architecture)
            context_vars = {
                "job": session_info.job,
                "project": session_info.project_name,
                "session_id": "unknown",
                "batch_id": "unknown",
            }

            # Add batch context if available
            if session_info.session_id:
                context_vars["session_id"] = session_info.session_id[-12:]  # Last 12 chars for brevity
            if session_info.batch_id:
                context_vars["batch_id"] = session_info.batch_id[-12:]  # Last 12 chars
                
            structlog.contextvars.bind_contextvars(**context_vars)

            # Check for existing cloud handlers to prevent duplicates
            root_logger = logging.getLogger()
            existing_cloud_handlers = [
                h for h in root_logger.handlers if isinstance(h, CloudLoggingHandler) and getattr(h, "name", "") == session_info.name
            ]
            
            if existing_cloud_handlers:
                logger.debug(
                    "Cloud logging handler already exists for this session",
                    session_id=session_info.session_id,
                    existing_handlers=len(existing_cloud_handlers)
                )
            else:
                # Add handler only to buttermilk logger for structured logs
                logging.getLogger(_LOGGER_NAME).addHandler(cloud_handler)
                logger.info(
                    "Cloud logging handler added",
                    session_id=session_info.session_id,
                    project_id=logger_cfg.project_id
                )
            
            # Mark this session as having cloud logging configured
            _cloud_logging_sessions.add(session_key)

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


def validate_logging_state(verbose_expected: bool = None) -> dict[str, Any]:
    """Validate the current logging configuration state.
    
    This function checks that logging is properly configured and hasn't been
    tampered with in ways that would break verbose logging functionality.
    
    Args:
        verbose_expected: If provided, validates that verbose logging is configured correctly
        
    Returns:
        dict: Validation results with status and details
        
    Raises:
        RuntimeError: If critical logging configuration issues are detected
    """
    global _console_logging_configured, _file_logging_configured
    
    validation_results = {
        "console_configured": _console_logging_configured,
        "file_configured": _file_logging_configured,
        "root_logger_level": logging.getLogger().getEffectiveLevel(),
        "buttermilk_logger_level": logging.getLogger(_LOGGER_NAME).getEffectiveLevel(),
        "issues": []
    }
    
    # Check if basic logging setup has been done
    if not _console_logging_configured:
        validation_results["issues"].append("Console logging not configured")
        
    if not _file_logging_configured:
        validation_results["issues"].append("File logging not configured")
    
    # Check root logger level for verbose mode
    root_level = logging.getLogger().getEffectiveLevel()
    if verbose_expected is True and root_level > logging.DEBUG:
        validation_results["issues"].append(
            f"Verbose mode expected but root logger level is {logging.getLevelName(root_level)}, "
            "should be DEBUG. This will prevent DEBUG messages from being logged."
        )
    
    # Check buttermilk logger level
    buttermilk_level = logging.getLogger(_LOGGER_NAME).getEffectiveLevel()
    if verbose_expected is True and buttermilk_level > logging.DEBUG:
        validation_results["issues"].append(
            f"Verbose mode expected but buttermilk logger level is {logging.getLevelName(buttermilk_level)}, "
            "should be DEBUG. This will prevent verbose logging from working."
        )
    
    # Check for handler count anomalies
    root_handlers = len(logging.getLogger().handlers)
    if root_handlers == 0:
        validation_results["issues"].append("No logging handlers configured on root logger")
    elif root_handlers > 5:  # Arbitrary threshold for too many handlers
        validation_results["issues"].append(
            f"Unusually high number of handlers ({root_handlers}) on root logger, "
            "may indicate duplicate handler registration"
        )
    
    validation_results["handler_count"] = root_handlers
    validation_results["valid"] = len(validation_results["issues"]) == 0
    
    # Log validation results
    if validation_results["issues"]:
        logger.warning(
            "Logging configuration validation failed",
            issues=validation_results["issues"],
            root_level=logging.getLevelName(root_level),
            buttermilk_level=logging.getLevelName(buttermilk_level)
        )
    else:
        logger.debug(
            "Logging configuration validation passed",
            root_level=logging.getLevelName(root_level),
            buttermilk_level=logging.getLevelName(buttermilk_level),
            handler_count=root_handlers
        )
    
    return validation_results


def ensure_logging_properly_initialized() -> None:
    """Ensure logging is properly initialized and fail fast if not.
    
    This function should be called at critical points to ensure the logging
    system is in a valid state before proceeding with operations that depend
    on proper logging functionality.
    
    Raises:
        RuntimeError: If logging is not properly initialized or is in an invalid state
    """
    validation = validate_logging_state()
    
    if not validation["valid"]:
        error_msg = (
            "Logging system is not properly initialized or is in an invalid state. "
            f"Issues found: {'; '.join(validation['issues'])}. "
            "This indicates a problem with the logging initialization sequence that "
            "could break verbose logging functionality."
        )
        raise RuntimeError(error_msg)


def reset_logging_configuration() -> None:
    """Reset logging configuration state for testing.
    
    This function resets all global logging state and removes handlers,
    allowing tests to call setup functions multiple times without conflicts.
    
    WARNING: This is intended for testing only and should not be used in
    production code as it can break the fail-fast logging protection.
    """
    global _console_logging_configured, _file_logging_configured, _structlog_configured, _cloud_logging_sessions

    # Reset global state flags
    _console_logging_configured = False
    _file_logging_configured = False
    _structlog_configured = False
    _cloud_logging_sessions.clear()
    
    # Remove all handlers from buttermilk logger
    buttermilk_logger = logging.getLogger(_LOGGER_NAME)
    for handler in buttermilk_logger.handlers[:]:
        buttermilk_logger.removeHandler(handler)
    
    # Reset structlog configuration to default state
    structlog.reset_defaults()
