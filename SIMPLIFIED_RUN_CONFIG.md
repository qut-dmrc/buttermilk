# Run Configuration Simplification

## Summary

The Buttermilk run configuration has been simplified from a discriminated union pattern with separate classes for each mode to a simple enum-based design. This change removes unnecessary complexity while maintaining type safety and validation.

## What Changed

### Before: Discriminated Union (Over-engineered)

```python
# run_config.py - OLD (220 lines)
class BaseRunConfig(BaseModel):
    mode: str
    ui: str | None = None
    human_in_loop: bool = False


class ConsoleRunConfig(BaseRunConfig):
    mode: Literal["console"] = "console"
    ui: Literal["console"] = "console"


class APIRunConfig(BaseRunConfig):
    mode: Literal["api"] = "api"
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    # ... more API-specific fields


# ... 7 total classes


def create_run_config(config_dict: dict) -> RunConfig:
    """Factory function to create the appropriate RunConfig based on mode."""
    # Complex mapping logic...
```

**Problems:**

- Mode-specific parameters (host, port) were duplicated in both RunConfig AND at root level
- You accessed them via `config.host`, not `config.run.host`, making the nested structure misleading
- Half the classes were essentially empty (just a mode field)
- Factory pattern added unnecessary complexity
- Pretended to have nested structure when params were at root level anyway

### After: Simple Enum (Right-sized)

```python
# run_config.py - NEW (~100 lines)
class RunMode(str, Enum):
    """Valid execution modes for Buttermilk."""

    CONSOLE = "console"
    BATCH = "batch"
    BATCH_RUN = "batch_run"
    BATCH_ALL = "batch_all"
    API = "api"
    PIPELINE = "pipeline"
    STREAMLIT = "streamlit"
    PUBSUB = "pub/sub"
    SLACKBOT = "slackbot"


class RunConfig(BaseModel):
    """Configuration for Buttermilk execution mode.

    This is intentionally simple - it just validates the mode.
    Mode-specific parameters (host, port, max_jobs, etc.) are at
    the root level of ButtermilkConfig.
    """

    mode: RunMode
    ui: str | None = None
    human_in_loop: bool = False
```

**Benefits:**

- Truth in advertising: params are where you access them (root level)
- No pretense of nested structure when there isn't one
- Simple validation using enum
- Easy to extend with new modes
- 50% less code

## Usage Changes

### Before: Complex Type Guards

```python
from buttermilk._core.run_config import (
    APIRunConfig,
    BatchRunConfig,
    ConsoleRunConfig,
    create_run_config,
)

typed_cfg = create_config_from_hydra(conf)
run_config = typed_cfg.get_run_config()

# Type guards for mode-specific logic
if isinstance(run_config, APIRunConfig):
    host = run_config.host  # Misleading - this field doesn't exist
    host = typed_cfg.host  # This is where it actually is
```

### After: Simple Enum Checks

```python
from buttermilk._core.run_config import RunMode

typed_cfg = create_config_from_hydra(conf)

# Simple mode check
if typed_cfg.run.mode == RunMode.API:
    host = typed_cfg.host  # Clear - params are at root level
    port = typed_cfg.port

# Or use match statement
match typed_cfg.run.mode:
    case RunMode.API:
        start_api_server(typed_cfg.host, typed_cfg.port)
    case RunMode.BATCH | RunMode.BATCH_RUN | RunMode.BATCH_ALL:
        process_batch(max_jobs=typed_cfg.max_jobs)
```

## Files Changed

1. **`buttermilk/_core/run_config.py`**: Complete rewrite
   - Before: 192 lines with 7 classes + factory
   - After: ~100 lines with 1 enum + 1 class
   - Removed: `BaseRunConfig`, all mode-specific classes, `create_run_config()`
   - Added: `RunMode` enum

2. **`buttermilk/_core/main_config.py`**: Simplified
   - Changed import: `from buttermilk._core.run_config import RunConfig` (no BaseRunConfig, no factory)
   - Changed field: `run: RunConfig | dict[str, Any]`
   - Simplified validator: Direct `RunConfig(**v)` instead of factory
   - Removed: `get_run_config()` method (just use `config.run` directly)

3. **`examples/typed_config_example.py`**: Updated examples
   - Import: `from buttermilk._core.run_config import RunMode`
   - Pattern: `if typed_cfg.run.mode == RunMode.API:` instead of `isinstance(...)`
   - Clearer: Access params from root level, not through run config

4. **`CONFIGURATION_DESIGN.md`**: Updated design doc
   - Documented simpler approach
   - Removed factory pattern discussion
   - Added truth-in-advertising principle

5. **`docs/configuration.md`**: Updated user guide
   - Clarified where params live (root level)
   - Updated code examples
   - Simplified usage patterns

## Migration Guide

If you have existing code using the old pattern:

### Change 1: Imports

```python
# Before
from buttermilk._core.run_config import (
    APIRunConfig,
    BatchRunConfig,
    ConsoleRunConfig,
    create_run_config,
)

# After
from buttermilk._core.run_config import RunMode
```

### Change 2: Mode Checking

```python
# Before
run_config = typed_cfg.get_run_config()
if isinstance(run_config, APIRunConfig):
    # ...

# After
if typed_cfg.run.mode == RunMode.API:
    # ...
```

### Change 3: Remove get_run_config() calls

```python
# Before
run_config = typed_cfg.get_run_config()
mode = run_config.mode

# After
mode = typed_cfg.run.mode  # Access directly
```

## Why This is Better

### 1. Honest Design

The old design pretended mode-specific params were nested in the run config, but they were actually at root level. Now the design matches reality.

### 2. Simpler Code

- 50% less code in run_config.py
- No factory pattern
- No complex inheritance hierarchy
- No misleading type guards

### 3. Easier to Understand

New developers can immediately see:

- What modes are valid (enum values)
- Where params are accessed (root level)
- How to check mode (simple enum comparison)

### 4. Easier to Extend

Adding a new mode:

```python
# Before: Create new class, update factory, update type union
# After: Add one line to enum
class RunMode(str, Enum):
    # ...
    NEW_MODE = "new_mode"  # Done!
```

### 5. Better Validation Error Messages

```python
# Before
ValidationError: 1 validation error for ButtermilkConfig
run
  Input should be a valid instance of ConsoleRunConfig, BatchRunConfig, ...

# After
ValidationError: 1 validation error for RunConfig
mode
  Invalid run mode: invalid_mode. Must be one of: console, batch, api, ...
```

## Validation Still Works

The simplified design maintains all validation:

```python
# Valid
config = RunConfig(mode="console", ui="console")  # ✓

# Invalid mode
config = RunConfig(mode="invalid")  # ✗ ValueError

# Mode is required
config = RunConfig()  # ✗ ValidationError
```

## Backward Compatibility

YAML configs don't need to change:

```yaml
# This still works exactly the same
run:
  mode: api
  ui: web

host: 0.0.0.0
port: 8000
```

The only changes are in Python code that checks modes.

## Conclusion

This simplification removes over-engineering while maintaining type safety and validation. The design now honestly reflects how the configuration is structured and accessed, making the codebase easier to understand and maintain.

**Key principle:** Don't create abstractions that don't match reality. Mode-specific params were never nested in run config, so we shouldn't pretend they were.
