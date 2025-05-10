"""User Interface (UI) implementations for Buttermilk agents.

This module provides different UI implementations that can be dynamically selected
at runtime. The UIProxyAgent serves as an intermediary that delegates to the
appropriate concrete UI implementation.
"""

from buttermilk import logger
from buttermilk.agents.ui.proxy import UIProxyAgent
from buttermilk.agents.ui.registry import register_ui

# Import UI implementations
# Use try/except to handle potential missing implementations
try:
    from buttermilk.agents.ui.web import WebUIAgent
    web_available = True
except ImportError:
    web_available = False
    logger.warning("WebUIAgent not available")

try:
    from buttermilk.agents.ui.console import CLIUserAgent
    console_available = True
except ImportError:
    console_available = False
    logger.warning("ConsoleUIAgent not available")

try:
    from buttermilk.agents.ui.slackthreadchat import SlackUIAgent
    slack_available = True
except ImportError:
    slack_available = False
    logger.warning("SlackUIAgent not available")

# Register available UI implementations with the registry
# This makes them available for dynamic selection by the UIProxyAgent
try:
    # Register web UI if available
    if web_available:
        register_ui("web", WebUIAgent, default=True)
        logger.debug("Registered 'web' UI implementation")

    # Register console UI if available
    if console_available:
        register_ui("console", CLIUserAgent)
        logger.debug("Registered 'console' UI implementation")

    # Register Slack UI if available
    if slack_available:
        register_ui("slack", SlackUIAgent)
        logger.debug("Registered 'slack' UI implementation")

except Exception as e:
    logger.error(f"Error registering UI implementations: {e}")

# Export the UIProxyAgent for use in flow configurations
__all__ = ["UIProxyAgent"]
