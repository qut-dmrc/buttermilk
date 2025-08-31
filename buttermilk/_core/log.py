import json
import logging
import os
from datetime import datetime, timezone
from logging import getLogger

from buttermilk._core.context import agent_id_var, session_id_var

_LOGGER_NAME = "buttermilk"


HIGHLIGHT_CODE = "\033[1;34m"  # Blue color for highlight
RESET_CODE = "\033[0m"  # Reset

# Max message length (characters). Set BUTTERMILK_LOG_MAX_LEN=0 to disable truncation.
MAX_LOG_MESSAGE_LENGTH = int(os.getenv("BUTTERMILK_LOG_MAX_LEN", "2000"))
TRUNCATION_SUFFIX = "… [truncated]"


class BMLogger(logging.Logger):
    """Buttermilk extended logger with custom functionality."""

    def highlight(self, msg: str, *args, stacklevel=2, **kwargs) -> None:
        """Log a highlighted INFO message that stands out visually."""
        # Add visual separators to make the message stand out
        highlighted_msg = HIGHLIGHT_CODE + f"▶ {msg}" + RESET_CODE
        self.info(highlighted_msg, *args, stacklevel=stacklevel, **kwargs)


# Set custom logger class before creating logger instance
logging.setLoggerClass(BMLogger)
logger: BMLogger = getLogger(_LOGGER_NAME)  # type: ignore[assignment]
logging.setLoggerClass(logging.Logger)  # Reset to default for other loggers


class ContextFilter(logging.Filter):
    def filter(self, record):
        # Store original values
        original_session_id = session_id_var.get()
        original_agent_id = agent_id_var.get()

        # Add original values to the record (for most handlers)
        record.session_id = original_session_id
        record.agent_id = original_agent_id

        # Add condensed string attributes for the console format
        record.short_context = (
            f"{original_session_id[-4:] if original_session_id else None}" + f":{original_agent_id}"
            if original_agent_id
            else ""
        )

        return True


class TruncatingFilter(logging.Filter):
    """Filter that truncates very long log messages to keep output manageable."""

    def __init__(self, max_length: int | None = None) -> None:
        super().__init__()
        self.max_length = max_length if max_length is not None else MAX_LOG_MESSAGE_LENGTH

    def filter(self, record: logging.LogRecord) -> bool:
        max_len = self.max_length
        if max_len and max_len > 0:
            try:
                message = record.getMessage()
            except Exception:
                # If formatting fails, don't interfere with the record
                return True
            if len(message) > max_len:
                keep = max(0, max_len - len(TRUNCATION_SUFFIX))
                truncated = message[:keep] + TRUNCATION_SUFFIX
                # Ensure ANSI colors don't leak if truncation occurs mid-sequence
                if "\033[" in message and not truncated.endswith(RESET_CODE):
                    truncated += RESET_CODE
                # Replace the original (possibly format-string) message with the truncated text
                record.msg = truncated
                record.args = ()
                record.truncated = True  # Optional: can be used by formatters
        return True


class CloudJSONFormatter(logging.Formatter):
    """JSON formatter specifically for Google Cloud Logging.
    
    Formats log records as JSON with structured fields while preserving
    all context information from ContextFilter. Designed to work with
    Google Cloud Logging for better log querying and analysis.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.
        
        Args:
            record: The log record to format
            
        Returns:
            JSON-formatted string with structured log data
        """
        # Create base log entry with standard fields
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "filename": record.filename,
            "lineno": record.lineno,
            "funcName": record.funcName,
        }
        
        # Add context information if available (from ContextFilter)
        if hasattr(record, "session_id") and record.session_id:
            log_entry["session_id"] = record.session_id
        if hasattr(record, "agent_id") and record.agent_id:
            log_entry["agent_id"] = record.agent_id
        if hasattr(record, "short_context") and record.short_context:
            log_entry["short_context"] = record.short_context
            
        # Add any extra fields from the log record
        if hasattr(record, "run_details"):
            log_entry["run_details"] = record.run_details
            
        # Handle exceptions if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None,
            }
            
        # Handle stack traces
        if record.stack_info:
            log_entry["stack_info"] = record.stack_info
            
        # Add truncation info if present
        if hasattr(record, "truncated") and record.truncated:
            log_entry["truncated"] = True
            
        return json.dumps(log_entry, ensure_ascii=False)


# Attach filters (idempotently) to the buttermilk logger
if not any(isinstance(f, ContextFilter) for f in logger.filters):
    logger.addFilter(ContextFilter())
if not any(isinstance(f, TruncatingFilter) for f in logger.filters):
    logger.addFilter(TruncatingFilter())
