import asyncio
import copy
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from rich.console import Console

# Lazy imports for google.cloud.logging (heavy dependency)
# Only imported when cloud logging is actually configured
if TYPE_CHECKING:
<<<<<<< HEAD
    pass
=======
    from google.cloud import logging as gcp_logging
    from google.cloud.logging_v2.handlers import CloudLoggingHandler
>>>>>>> origin/stable
from rich.logging import RichHandler
from structlog.processors import CallsiteParameter, CallsiteParameterAdder

from buttermilk._core.context import get_logging_context

from .constants import _LOGGER_NAME

try:
    # Optional: OpenTelemetry trace context for log correlation
    from opentelemetry.trace import get_current_span
except Exception:  # pragma: no cover
    get_current_span = None  # type: ignore

# Single logger for the entire application
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

    def _inject_trace_ids(logger, method_name, event_dict):
        """Inject OTEL trace_id/span_id into logs if available."""
        try:
            if get_current_span is None:
                return event_dict
            span = get_current_span()
            if span is None:
                return event_dict
            ctx = span.get_span_context()
            # Some SDKs expose is_valid attribute; guard usage
            if getattr(ctx, "trace_id", 0):
                event_dict["trace_id"] = f"{ctx.trace_id:032x}"
                event_dict["span_id"] = f"{ctx.span_id:016x}"
                # Add sampled flag when available
                sampled = getattr(getattr(ctx, "trace_flags", None), "sampled", None)
                if sampled is not None:
                    event_dict["trace_sampled"] = bool(sampled)
        except Exception:
            # Never break logging due to telemetry issues
            return event_dict
        return event_dict

    def _add_runtime_context(logger, method_name, event_dict):
        """Inject common runtime context onto every log."""
        try:
            event_dict.setdefault("pid", os.getpid())
            event_dict.setdefault(
                "process_name",
<<<<<<< HEAD
                getattr(os, "getppid", lambda: None)() and logging.getLogger().name or "python",
=======
                getattr(os, "getppid", lambda: None)()
                and logging.getLogger().name
                or "python",
>>>>>>> origin/stable
            )
            event_dict.setdefault("thread_name", threading.current_thread().name)
            # Async task name/id if available
            task_name = None
            try:
                task = asyncio.current_task()
                if task:
                    task_name = task.get_name()
            except Exception:
                pass
            if task_name:
                event_dict.setdefault("task", task_name)
        except Exception:
            return event_dict
        return event_dict

    def _extract_exception_fields(logger, method_name, event_dict):
        """When exc_info is present, add standardized exception fields."""
        try:
            exc_info = event_dict.get("exc_info")
            exc = None
            if exc_info is True:
                _, exc, _ = sys.exc_info()
            elif isinstance(exc_info, tuple) and len(exc_info) == 3:
                exc = exc_info[1]
            elif isinstance(exc_info, BaseException):
                exc = exc_info

            if exc is not None:
                event_dict.setdefault("error_type", exc.__class__.__name__)
                event_dict.setdefault("error_message", str(exc))
                # Root cause (walk __cause__ / __context__)
                cause = exc
                while getattr(cause, "__cause__", None) is not None:
                    cause = cause.__cause__
                if cause is exc and getattr(exc, "__context__", None) is not None:
                    cause = exc.__context__
                if cause is not None and cause is not exc:
                    event_dict.setdefault("root_cause_type", cause.__class__.__name__)
                    event_dict.setdefault("root_cause_message", str(cause))
        except Exception:
            return event_dict
        return event_dict

    structlog.configure(
        processors=[
            # Add context variables automatically
            structlog.contextvars.merge_contextvars,
            # Add log level
            structlog.processors.add_log_level,
            # Add timestamp
            structlog.processors.TimeStamper(fmt="iso"),
            # Inject OTEL trace/span IDs for correlation (no-op if not available)
            _inject_trace_ids,
            # Common runtime fields on every log
            _add_runtime_context,
            # Add callsite info (module, function, line)
            CallsiteParameterAdder(
                {
                    CallsiteParameter.MODULE,
                    CallsiteParameter.FUNC_NAME,
                    CallsiteParameter.LINENO,
                    CallsiteParameter.PATHNAME,
                }
            ),
            # Normalize exception fields before rendering
            _extract_exception_fields,
            # Capture exception info if present (adds "exception" with traceback)
            structlog.processors.format_exc_info,
            # Output as JSON
            structlog.processors.JSONRenderer(),
        ],
<<<<<<< HEAD
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),  # Use DEBUG to support all handlers
=======
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.DEBUG
        ),  # Use DEBUG to support all handlers
>>>>>>> origin/stable
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _structlog_configured = True


class StructlogRichHandler(RichHandler):
    def format(self, record):
        # Convert structlog JSON back to rich format
        # CRITICAL: Create a copy to avoid modifying original record for other handlers
        display_record = copy.copy(record)

        if hasattr(display_record, "msg") and isinstance(display_record.msg, str):
            try:
                data = json.loads(display_record.msg)
                display_record.msg = data.get("event", "")
                # Add structured data as extra context
            except json.JSONDecodeError:
                pass
        return super().format(display_record)


def setup_console_logging(verbose: bool = False, enable_console: bool = True) -> None:
    """Set up beautiful console logging with Rich.

    Args:
        verbose: If True, shows DEBUG level logs on console
        enable_console: If False, disables console logging entirely

    Raises:
        RuntimeError: If console logging has already been configured

    Note:
        Logs are written to stderr (not stdout) following Python best practices.
        This makes buttermilk compatible with MCP servers and other stdio protocols
        without requiring special configuration. MCP clients automatically capture
        stderr for debugging.
    """
    global _console_logging_configured

    if _console_logging_configured:
        raise RuntimeError(
            "Console logging has already been configured. "
            "Multiple calls to setup_console_logging() can break verbose logging functionality. "
            "This indicates a problematic initialization sequence."
        )

    if not enable_console:
        # Still configure structlog but without console handler
        struct_level = logging.DEBUG if verbose else logging.INFO
        configure_structlog(min_level=struct_level)
        _console_logging_configured = True
        return

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

    # Create Rich handler for beautiful console output (write to stderr, not stdout)
    # This follows Python best practices and is required for MCP servers
    stderr_console = Console(stderr=True)
    rich_handler = StructlogRichHandler(
        console=stderr_console,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )

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


<<<<<<< HEAD
def setup_file_logging(execution_context_id: str, verbose: bool = False, project_name: str | None = None) -> list[str]:
=======
def setup_file_logging(
    execution_context_id: str, verbose: bool = False, project_name: str | None = None
) -> list[str]:
>>>>>>> origin/stable
    """Set up structured JSON logging to files.

    Args:
        execution_context_id: Unique execution context identifier for log file naming
        verbose: If True, creates both INFO and DEBUG log files
        project_name: Project name to include in log file name (REQUIRED)

    Returns:
        List of log file paths created

    Raises:
        RuntimeError: If file logging has already been configured
        ValueError: If project_name is None or empty
    """
    global _file_logging_configured

    if _file_logging_configured:
        raise RuntimeError(
            "File logging has already been configured. "
            "Multiple calls to setup_file_logging() can break verbose logging functionality "
            "and create conflicting log file handlers. This indicates a problematic initialization sequence."
        )

    # Fail fast if project_name is not provided
    if not project_name:
        raise ValueError(
            "project_name is required for log file naming. "
            "Log files must follow the format: bm_{project_name}_{execution_context_id}.jsonl. "
            "This ensures proper identification and filtering of Buttermilk log files."
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

    # Create single JSON log file with bm_ prefix for easy identification
    log_filename = f"bm_{project_name}_{execution_context_id}.jsonl"

    log_path = Path(f"/tmp/{log_filename}")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    # Since all logs come through structlog's stdlib bridge, they're already processed
    # Just extract the pre-formatted message from the LogRecord
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    # Add handler only to buttermilk logger, not root logger
    # This ensures we only get structured logs from our code
    logging.getLogger(_LOGGER_NAME).addHandler(file_handler)
    log_files.append(str(log_path))

<<<<<<< HEAD
    logger.info(f"Log file created at {log_path}", log_path=str(log_path), verbose=verbose)
=======
    logger.info(
        f"Log file created at {log_path}", log_path=str(log_path), verbose=verbose
    )
>>>>>>> origin/stable
    if verbose:
        logger.debug(
            f"Verbose logging enabled for {log_path}.",
            log_path=str(log_path),
            verbose=verbose,
        )

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
    # Lazy import to avoid loading google.cloud at module load time
    from google.cloud import logging as gcp_logging
    from google.cloud.logging_v2.handlers import CloudLoggingHandler

    global _cloud_logging_sessions

    # Check if cloud logging is already configured for this session
    session_key = f"{session_info.session_id}:{logger_cfg.project_id}"
    if session_key in _cloud_logging_sessions:
        logger.debug(
            "Cloud logging already configured for this session",
            session_id=session_info.session_id,
            project_id=logger_cfg.project_id,
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
<<<<<<< HEAD
            raw_labels = session_info.model_dump(include={"session_id", "project_name", "job", "platform"})
=======
            raw_labels = session_info.model_dump(
                include={"session_id", "project_name", "job", "platform"}
            )
>>>>>>> origin/stable
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
            cloud_handler.setFormatter(logging.Formatter("%(message)s"))

            # Bind session context for automatic inclusion (simplified architecture)
            context_vars = {
                "job": session_info.job,
                "project": session_info.project_name,
                "session_id": "unknown",
                "batch_id": "unknown",
            }

            # Add batch context if available
            if session_info.session_id:
<<<<<<< HEAD
                context_vars["session_id"] = session_info.session_id[-12:]  # Last 12 chars for brevity
=======
                context_vars["session_id"] = session_info.session_id[
                    -12:
                ]  # Last 12 chars for brevity
>>>>>>> origin/stable
            if session_info.batch_id:
                context_vars["batch_id"] = session_info.batch_id[-12:]  # Last 12 chars

            structlog.contextvars.bind_contextvars(**context_vars)

            # Check for existing cloud handlers to prevent duplicates
            root_logger = logging.getLogger()
            existing_cloud_handlers = [
<<<<<<< HEAD
                h for h in root_logger.handlers if isinstance(h, CloudLoggingHandler) and getattr(h, "name", "") == session_info.project_name
=======
                h
                for h in root_logger.handlers
                if isinstance(h, CloudLoggingHandler)
                and getattr(h, "name", "") == session_info.project_name
>>>>>>> origin/stable
            ]

            if existing_cloud_handlers:
                logger.debug(
                    "Cloud logging handler already exists for this session",
                    session_id=session_info.session_id,
                    existing_handlers=len(existing_cloud_handlers),
                )
            else:
                # Add handler only to buttermilk logger for structured logs
                logging.getLogger(_LOGGER_NAME).addHandler(cloud_handler)
                logger.info(
                    "Cloud logging handler added",
                    session_id=session_info.session_id,
                    project_id=logger_cfg.project_id,
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
        "issues": [],
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

    # Check for handler count anomalies on buttermilk logger (where handlers are actually added)
    root_handlers = len(logging.getLogger().handlers)
    buttermilk_handlers = len(logging.getLogger(_LOGGER_NAME).handlers)

    # Check both root and buttermilk logger for handlers
    if root_handlers == 0 and buttermilk_handlers == 0:
<<<<<<< HEAD
        validation_results["issues"].append("No logging handlers configured on root logger")
    elif root_handlers > 5:  # Arbitrary threshold for too many handlers
        validation_results["issues"].append(
            f"Unusually high number of handlers ({root_handlers}) on root logger, may indicate duplicate handler registration"
=======
        validation_results["issues"].append(
            "No logging handlers configured on root logger"
        )
    elif root_handlers > 5:  # Arbitrary threshold for too many handlers
        validation_results["issues"].append(
            f"Unusually high number of handlers ({root_handlers}) on root logger, "
            "may indicate duplicate handler registration"
>>>>>>> origin/stable
        )

    validation_results["handler_count"] = root_handlers
    validation_results["buttermilk_handler_count"] = buttermilk_handlers
    validation_results["valid"] = len(validation_results["issues"]) == 0

    # Log validation results
    if validation_results["issues"]:
        logger.warning(
            "Logging configuration validation failed",
            issues=validation_results["issues"],
            root_level=logging.getLevelName(root_level),
            buttermilk_level=logging.getLevelName(buttermilk_level),
        )
    else:
        logger.debug(
            "Logging configuration validation passed",
            root_level=logging.getLevelName(root_level),
            buttermilk_level=logging.getLevelName(buttermilk_level),
            handler_count=root_handlers,
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
<<<<<<< HEAD
    global _console_logging_configured, _file_logging_configured, _structlog_configured, _cloud_logging_sessions
=======
    global \
        _console_logging_configured, \
        _file_logging_configured, \
        _structlog_configured, \
        _cloud_logging_sessions
>>>>>>> origin/stable

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
<<<<<<< HEAD


def flush_logging() -> None:
    """Flush and close all logging handlers.

    CRITICAL for CloudLoggingHandler to ensure logs are sent before exit.
    This should be called during graceful shutdown.
    """
    bm_logger = logging.getLogger(_LOGGER_NAME)
    handlers = list(bm_logger.handlers)

    if handlers:
        logger.debug(f"Flushing {len(handlers)} logging handlers...")

    for handler in handlers:
        try:
            # Check if it's a CloudLoggingHandler (lazy check to avoid import)
            handler_type = type(handler).__name__
            if handler_type == "CloudLoggingHandler":
                logger.debug("Closing CloudLoggingHandler...")

            handler.flush()
            handler.close()
            bm_logger.removeHandler(handler)
        except Exception:
            # Don't let logging failures block shutdown
            pass
=======
>>>>>>> origin/stable
