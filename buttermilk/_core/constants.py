"""Constants used throughout the Buttermilk framework.

This module contains all the constant values used by various components to avoid
circular dependencies and provide a single source of truth.
"""

CONFIG_CACHE_PATH = ".cache/buttermilk/models.json"
MODELS_CFG_KEY = "models_secret"
SHARED_CREDENTIALS_KEY = "credentials_secret"

# https://cloud.google.com/bigquery/pricing
GOOGLE_BQ_PRICE_PER_BYTE = 5 / 10e12  # $5 per tb.

# Standard Agent Roles
CONDUCTOR = "HOST"  # Role name often used for the agent directing the flow
MANAGER = "MANAGER"  # Role name often used for the user interface agent
CLOSURE = "COLLECTOR"  # Role name for the collector agent
CONFIRM = "CONFIRM"  # Special agent/topic name used for handling ManagerResponse

# Special Symbols / States
COMMAND_SYMBOL = "!"  # Prefix used to identify command messages
END = "END"  # Signal used to indicate the flow should terminate
WAIT = "WAIT"  # Signal used to indicate pausing/waiting

# Add any other constants used across the codebase here
