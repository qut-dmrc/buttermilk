# Implementation Plan for Resolving Circular Dependencies

This plan outlines the specific steps needed to refactor the Buttermilk codebase to eliminate circular dependencies while maintaining a logical organization.

## Step 1: Create a constants.py file

Create a new file `buttermilk/_core/constants.py` that will contain all constants currently defined in various files:

```python
"""Constants used throughout the Buttermilk framework.

This module contains all the constant values used by various components to avoid
circular dependencies and provide a single source of truth.
"""

# Standard Agent Roles
CONDUCTOR = "HOST"  # Role name often used for the agent directing the flow
MANAGER = "MANAGER"  # Role name often used for the user interface agent
CLOSURE = "COLLECTOR"  # Role name for the collector agent
CONFIRM = "CONFIRM"  # Special agent/topic name used for handling ManagerResponse
END = "END"  # Signal used to indicate the flow should terminate
WAIT = "WAIT"  # Signal used to indicate pausing/waiting
COMMAND_SYMBOL = "!"  # Prefix used to identify command messages
```

## Step 2: Reorganize types.py

Refactor `buttermilk/_core/types.py` to also include the `SessionInfo` class from `job.py` to avoid circularity:

```python
# Add all of the current types.py code and
# Move SessionInfo from job.py to types.py

# Current SessionInfo imports:
import asyncio
import platform
from pathlib import Path
from tempfile import mkdtemp

import psutil
import pydantic
from cloudpathlib import AnyPath, CloudPath
from pydantic import ConfigDict, Field, field_validator, model_validator

class SessionInfo(pydantic.BaseModel):
    # Copy the existing implementation from job.py
    # ...
```

## Step 3: Update contract.py to use forward references

Modify `buttermilk/_core/contract.py` to use forward references for `AgentConfig`:

```python
# At the top of the file:
from typing import Any, Union, TYPE_CHECKING

# Add TYPE_CHECKING block for imports used only in type hints:
if TYPE_CHECKING:
    from .config import AgentConfig

# Modify the AgentTrace class:
class AgentTrace(FlowMessage):
    # ...
    agent_info: "AgentConfig" = Field(..., description="Configuration info from the agent (required)")
    # ...
```

## Step 4: Update job.py

Update `buttermilk/_core/job.py` to import SessionInfo from types.py:

```python
# Remove SessionInfo from this file
# Add:
from .types import SessionInfo
```

## Step 5: Update config.py

Modify `buttermilk/_core/config.py` to import SessionInfo from types.py:

```python
# Replace:
from .job import SessionInfo
# With:
from .types import SessionInfo
```

## Step 6: Update imports in agent.py

In `buttermilk/_core/agent.py`, organize imports to avoid circular dependencies:

```python
# Use a TYPE_CHECKING block for imports only used in type hints
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import AgentConfig

# Move the AgentConfig import to a runtime import for the _cfg property if needed
```

## Step 7: Update all import references to constants

Update all files to import constants from the new constants.py module:

```python
# In contract.py, remove these lines:
CONDUCTOR = "HOST"
MANAGER = "MANAGER"
CLOSURE = "COLLECTOR"
CONFIRM = "CONFIRM"
COMMAND_SYMBOL = "!"
END = "END"
WAIT = "WAIT"

# And replace with:
from .constants import (
    CONDUCTOR,
    MANAGER,
    CLOSURE,
    CONFIRM,
    COMMAND_SYMBOL,
    END,
    WAIT,
)

# Do similar updates in agent.py and any other file using these constants
```

## Step 8: Update message_data.py and other utilities

Ensure utility files like `message_data.py` only depend on types.py and constants.py, not higher-level modules:

```python
# Remove any imports of higher-level components
# Add TYPE_CHECKING blocks for type hints if needed
```

## Step 9: Update specialized agents

Ensure specialized agent implementations like `llm.py` only import what they need:

```python
# In agents/llm.py and other agent implementations:
from buttermilk._core.agent import Agent, AgentResponse
from buttermilk._core.constants import COMMAND_SYMBOL
from buttermilk._core.contract import AgentInput, ErrorEvent, LLMMessage
from buttermilk._core.exceptions import ProcessingError 
from buttermilk._core.types import Record
```

## Dependency Graph After Changes

After these changes, the dependency graph should look like:

```
constants.py       <-- No dependencies
exceptions.py      <-- No dependencies 
types.py           <-- Import exceptions.py (optional)
config.py          <-- Import types.py, constants.py, exceptions.py
contract.py        <-- Import types.py, constants.py, config.py (via TYPE_CHECKING)
message_data.py    <-- Import types.py, constants.py
agent.py           <-- Import all of the above
LLMAgent, etc.     <-- Import all of the above + agent.py 
orchestrator.py    <-- Import all of the above
```

## Testing Plan

After making these changes:

1. Run all unit tests to ensure functionality remains the same
2. Verify import paths work correctly from other parts of the codebase
3. Check that no new circular dependencies were introduced
4. Ensure type hints and IDE autocomplete still function properly
