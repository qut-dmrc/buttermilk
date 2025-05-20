import logging
from logging import getLogger

from buttermilk._core.context import agent_id_var, session_id_var

_LOGGER_NAME = "buttermilk"
logger = getLogger(_LOGGER_NAME)


class ContextFilter(logging.Filter):
    def filter(self, record):
        record.session_id = session_id_var.get()
        record.agent_id = agent_id_var.get()
        return True
