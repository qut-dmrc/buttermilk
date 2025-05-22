import logging
from logging import getLogger

from buttermilk._core.context import agent_id_var, session_id_var

_LOGGER_NAME = "buttermilk"
logger = getLogger(_LOGGER_NAME)


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
