# Files Needing Import Updates

The following files currently import constants from `buttermilk/_core/contract.py` and should be updated to import from `buttermilk/_core/constants.py` instead:

## 1. buttermilk/runner/slackbot.py
```python
# Current:
from buttermilk._core.contract import MANAGER

# Change to:
from buttermilk._core.constants import MANAGER
```

## 2. buttermilk/runner/groupchat.py
```python
# Current:
from buttermilk._core.contract import (
    CONDUCTOR,
    MANAGER,
    ...
)

# Change to:
from buttermilk._core.constants import CONDUCTOR, MANAGER
from buttermilk._core.contract import (
    # Remove CONDUCTOR and MANAGER from this import
    ...
)
```

## 3. buttermilk/runner/selector.py
```python
# Current:
from buttermilk._core.contract import (
    CONDUCTOR,
    END,
    WAIT,
    ...
)

# Change to:
from buttermilk._core.constants import CONDUCTOR, END, WAIT
from buttermilk._core.contract import (
    # Remove CONDUCTOR, END, and WAIT from this import
    ...
)
```

## 4. buttermilk/agents/ui/web.py
```python
# Current:
from buttermilk._core.contract import (
    MANAGER,
    ...
)

# Change to:
from buttermilk._core.constants import MANAGER
from buttermilk._core.contract import (
    # Remove MANAGER from this import
    ...
)
```

## 5. buttermilk/agents/fetch.py
```python
# Current:
from buttermilk._core.contract import (
    COMMAND_SYMBOL,
    ...
)

# Change to:
from buttermilk._core.constants import COMMAND_SYMBOL
from buttermilk._core.contract import (
    # Remove COMMAND_SYMBOL from this import
    ...
)
```

## 6. buttermilk/agents/flowcontrol/explorer.py
```python
# Current:
from buttermilk._core.contract import (
    END,
    WAIT,
    ...
)

# Change to:
from buttermilk._core.constants import END, WAIT
from buttermilk._core.contract import (
    # Remove END and WAIT from this import
    ...
)
```

## Other Files To Update

We also need to modify `buttermilk/_core/contract.py` to import all the constants it needs:

```python
# Current:
from .constants import (
    MANAGER,
)

# Change to:
from .constants import (
    CLOSURE,
    COMMAND_SYMBOL,
    CONDUCTOR,
    CONFIRM,
    END,
    MANAGER,
    WAIT,
)
```

## Verification Steps

After making these changes:

1. Run unit tests to verify functionality remains the same
2. Check for any import errors that might indicate missed references 
3. Look for linting warnings about unused imports after removing them from contract.py imports

This will ensure a clean transition to the new structure while maintaining compatibility.
