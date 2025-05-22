"""Constants used throughout the Buttermilk framework.

This module centralizes constant values used by various components of Buttermilk,
helping to avoid circular dependencies and providing a single source of truth
for these values. Constants include paths, configuration keys, pricing information,
standardized agent roles, and special symbols or states recognized by the framework.
"""

CONFIG_CACHE_PATH = ".cache/buttermilk/models.json"
"""Path to the local cache file for storing LLM (Language Model) configurations.
This cache helps in faster startup by avoiding repeated fetching of configurations.
"""

MODELS_CFG_KEY = "models_secret"
"""Key used to retrieve LLM (Language Model) configurations from a secret manager.
This allows sensitive or complex model configurations to be stored securely.
"""

SHARED_CREDENTIALS_KEY = "credentials_secret"
"""Key used to retrieve shared system credentials (e.g., API keys for various
services) from a secret manager.
"""

# Reference for pricing: https://cloud.google.com/bigquery/pricing
GOOGLE_BQ_PRICE_PER_BYTE = 5 / 10e12
"""Estimated cost per byte for Google BigQuery operations.
Based on a rate of $5 per Terabyte (TB). Used for cost estimation purposes.
1 TB = 10^12 bytes.
"""

# --- Standard Agent Roles ---
# These constants define conventional role names used within Buttermilk flows,
# particularly in multi-agent setups or when specific agent functionalities
# are standardized.

CONDUCTOR = "HOST"
"""Role name typically assigned to an agent responsible for directing the overall
flow of execution, making decisions about next steps, or managing other agents.
Often synonymous with a "host" or "orchestrator" agent within a specific flow logic.
"""

MANAGER = "MANAGER"
"""Role name typically assigned to an agent that interfaces with a human user
or an external UI. This agent handles user input, feedback, and confirmations.
"""

CLOSURE = "COLLECTOR"
"""Role name for an agent that collects, aggregates, or finalizes results
at the end of a flow or a specific phase of processing.
"""

CONFIRM = "CONFIRM"
"""A special agent role or topic name often used in conjunction with the `MANAGER`
role for handling user confirmations or responses to `UIMessage` prompts.
"""

# --- Special Symbols / States ---
# These constants define special string values that have specific meanings
# within the Buttermilk framework, often used in messages or state management.

COMMAND_SYMBOL = "!"
"""Prefix character used to identify messages that should be interpreted as
commands rather than regular content. For example, `!reset` or `!help`.
"""

END = "END"
"""Signal string used to indicate that a flow, a conversation, or a specific
processing sequence should terminate.
"""

WAIT = "WAIT"
"""Signal string used to indicate that an agent or a flow should pause,
wait for further input, or await the completion of another process.
"""

# Add any other constants used across the codebase here.
# Ensure they are well-documented with their purpose and usage context.
