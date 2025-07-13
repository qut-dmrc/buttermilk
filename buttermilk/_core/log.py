import logging
from logging import getLogger
from typing import Any

from buttermilk._core.context import agent_id_var, session_id_var

_LOGGER_NAME = "buttermilk"


class HighlightLogger(logging.Logger):
    """Extended logger with highlight capability."""
    
    def highlight(self, msg: str, *args, **kwargs) -> None:
        """Log a highlighted INFO message that stands out visually."""
        # Add visual separators to make the message stand out
        highlighted_msg = f"\n{'═' * 80}\n▶ {msg}\n{'═' * 80}"
        self.info(highlighted_msg, *args, **kwargs)


# Set custom logger class before creating logger instance
logging.setLoggerClass(HighlightLogger)
logger: HighlightLogger = getLogger(_LOGGER_NAME)  # type: ignore[assignment]
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
