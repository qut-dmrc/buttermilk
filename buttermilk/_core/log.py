import logging
from logging import getLogger

from buttermilk._core.context import agent_id_var, session_id_var

_LOGGER_NAME = "buttermilk"


HIGHLIGHT_CODE = "\033[1;34m"  # Blue color for highlight
RESET_CODE = "\033[0m"  # Reset


class BMLogger(logging.Logger):
    """Buttermilk extended logger with custom functionality."""

    def highlight(self, msg: str, *args, **kwargs) -> None:
        """Log a highlighted INFO message that stands out visually."""
        # Add visual separators to make the message stand out
        highlighted_msg = HIGHLIGHT_CODE + f"▶ {msg}" + RESET_CODE
        self.info(highlighted_msg, *args, **kwargs)


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
