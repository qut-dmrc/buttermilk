# Buttermilk Configuration Type System - Design Document

## Executive Summary

This document describes the cleaned-up, strongly-typed Pydantic configuration system for Buttermilk. The new design simplifies the configuration structure by:

1. Keeping mode INSIDE run config (loaded via Hydra config groups like `run=api`)
2. Consolidating ALL execution parameters into RunConfig
3. Removing the BMConfig wrapper (direct session access)
4. Unifying max_records/max_jobs into a single `limit` parameter
5. Removing deprecated parameters (enqueue_only, process_only, dataset_key, prompt)

The system makes it clear what configuration options control what, while preserving the flexibility of Hydra's composition system.

## Problem Statement

The Buttermilk configuration system previously used loose dictionaries for the `run` object and other configuration sections, making it difficult to:

1. Know what configuration options are available
2. Understand what values are valid for each option
3. Get IDE autocomplete and type checking
4. Catch configuration errors before runtime
5. Track configuration changes and their impacts

## Solution Architecture

### Core Components

The solution adds four new modules to `/home/nic/src/writing/projects/buttermilk/buttermilk/_core/`:

1. **`run_config.py`** - Typed models for all run modes
2. **`pipeline_config.py`** - Typed models for pipeline processing
3. **`main_config.py`** - Root configuration model tying everything together
4. **`docs/configuration.md`** - Comprehensive documentation

### Design Principles

1. **Preserve Hydra Flexibility**: Continue using Hydra's composition, interpolation, and override system
2. **Add Type Safety**: Convert to Pydantic models at the entry point for validation
3. **Clear Structure**: Make the configuration hierarchy explicit and discoverable
4. **Backward Compatible**: Existing YAML configs work without changes

## Configuration Hierarchy (NEW SIMPLIFIED)

```
ButtermilkConfig (root)
├── Universal Essentials (Root Level)
│   ├── project_name: str
│   ├── job: str
│   └── verbose: bool
├── run: RunConfig                       # ALL execution parameters (including mode)
│   ├── mode: RunMode                    # Loaded via run=api config group
│   ├── flow: str | None
│   ├── limit: int | None                # Unified from max_records/max_jobs
│   ├── record_id: str | None
│   ├── host: str                        # API mode
│   ├── port: int                        # API mode
│   ├── workers: int                     # API mode
│   ├── reload: bool                     # API mode
│   ├── log_level: str                   # API mode
│   ├── pipeline: PipelineConfig | None  # Pipeline mode
│   └── storage_config: dict | None      # Batch mode override
├── session: SessionInfo                 # Direct access (no wrapper)
├── infrastructure: InfrastructureConfig
│   ├── clouds: list[CloudProvider]
│   ├── llms: dict[str, Any]
│   ├── tracing: TracingConfig
│   │   ├── weave: TracingProviderConfig
│   │   ├── traceloop: TracingProviderConfig
│   │   └── otel: TracingProviderConfig
│   └── logging: LoggerConfig
├── flows: dict[str, Any]                # Configuration, not execution
└── storage: dict[str, StorageConfig]
```

### Key Simplifications

1. **Root Level**: Only 3 fields (project_name, job, verbose)
2. **Run Config**: Consolidates ALL execution parameters including mode
3. **No Wrappers**: Direct session access (removed BMConfig)
4. **Hydra Config Groups**: Mode loaded via `run=api` from `conf/run/api.yaml`
5. **Unified Limit**: One parameter for both records and jobs

## Key Design Decisions

### 1. Mode Inside Run Config (via Hydra Config Groups)

Mode is inside RunConfig and loaded via Hydra config groups:

```python
class RunConfig(BaseModel):
    mode: RunMode = Field(
        default=RunMode.CONSOLE,
        description="Execution mode (set via run=api, run=batch, etc.)"
    )
    # All other execution params
    flow: str | None = None
    limit: int | None = None
    # ...

class ButtermilkConfig(BaseModel):
    # Root level essentials only
    project_name: str
    job: str
    verbose: bool

    # All execution params in run config (including mode)
    run: RunConfig
    session: SessionInfo
    # ...
```

**Benefits**:
- CLI uses Hydra config groups: `run=api` loads `conf/run/api.yaml`
- Mode is part of the run configuration it controls
- Each run config file (api.yaml, batch.yaml) has its own mode setting
- Clear separation: root = project essentials, run = execution details

### 2. Consolidated Run Config

ALL execution parameters are in RunConfig, including mode:

```python
class RunConfig(BaseModel):
    # Execution mode
    mode: RunMode = RunMode.CONSOLE

    # Flow execution
    flow: str | None = None
    limit: int | None = None  # Unified
    record_id: str | None = None

    # API mode
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"

    # Pipeline mode
    pipeline: PipelineConfig | None = None

    # Batch mode
    storage_config: dict | None = None
```

**Benefits**:
- Single source for ALL execution parameters (including mode)
- Clear separation: root = project essentials, run = execution
- Easy to see what affects program behavior
- Mode-specific params clearly documented

### 3. Remove BMConfig Wrapper

Session is now directly accessible:

```python
# Old structure
class BMConfig(BaseModel):
    session_info: SessionInfo

class ButtermilkConfig(BaseModel):
    bm: BMConfig  # Extra wrapper

# Access: typed_cfg.bm.session_info

# New structure
class ButtermilkConfig(BaseModel):
    session: SessionInfo  # Direct

# Access: typed_cfg.session
```

**Benefits**:
- Simpler structure, less nesting
- More intuitive access pattern
- Backward compatible via validator

### 4. Unified Limit Parameter

Replace max_records and max_jobs with single `limit`:

```python
# Old structure
class ButtermilkConfig(BaseModel):
    max_records: int | None = None
    max_jobs: int | None = None

# New structure
class RunConfig(BaseModel):
    limit: int | None = None  # Unified
```

**Benefits**:
- Single parameter for all limit scenarios
- Clearer intent - how many things to process
- Works for records, jobs, or any countable unit
- Automatic migration from old params

### 5. Backward Compatibility

The `create_config_from_hydra()` function provides automatic migration:

```python
def create_config_from_hydra(cfg: DictConfig) -> ButtermilkConfig:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Handle bm.session_info -> session
    if "bm" in cfg_dict:
        cfg_dict["session"] = cfg_dict["bm"]["session_info"]

    # Handle run.mode -> mode at root
    if "run" in cfg_dict and "mode" in cfg_dict["run"]:
        cfg_dict["mode"] = cfg_dict["run"]["mode"]

    # Unify max_records/max_jobs -> limit
    if "max_records" in cfg_dict or "max_jobs" in cfg_dict:
        cfg_dict["run"]["limit"] = cfg_dict.get("max_records") or cfg_dict.get("max_jobs")

    return ButtermilkConfig(**cfg_dict)
```

**Benefits**:
- Existing YAML configs continue to work
- Incremental migration path
- No breaking changes to deployed systems

## Implementation Details

### Run Configuration (`run_config.py`)

**Run Mode Enum** (unchanged):
```python
class RunMode(str, Enum):
    CONSOLE = "console"
    BATCH = "batch"
    BATCH_RUN = "batch_run"
    BATCH_ALL = "batch_all"
    API = "api"
    PIPELINE = "pipeline"
    STREAMLIT = "streamlit"
    PUBSUB = "pub/sub"
    SLACKBOT = "slackbot"
```

**Run Config Class** (NEW - consolidated execution params):
```python
class RunConfig(BaseModel):
    # Flow execution
    flow: str | None = None
    limit: int | None = None
    record_id: str | None = None

    # API mode
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"

    # Pipeline mode
    pipeline: PipelineConfig | None = None

    # Batch mode
    storage_config: dict | None = None

    model_config = {
        "extra": "allow",
        "arbitrary_types_allowed": True
    }
```

**Design Philosophy**:
- Consolidates ALL execution parameters in one place
- Clear defaults for each parameter
- Mode determines which params are relevant
- Self-documenting with clear field names

### Pipeline Configuration (`pipeline_config.py`)

```python
class PipelineConfig(BaseModel):
    source: StorageConfig | dict | None = None
    output: StorageConfig | dict | None = None
    tmdb: TMDBProcessorConfig | dict | bool | None = None
    concurrency: int = Field(default=1, ge=1)
    max_records: int | None = None
    buffer_size: int = Field(default=10, ge=1)
    flush_interval: int = Field(default=30, ge=1)

    def get_source_config(self) -> StorageConfig | None: ...
    def get_output_config(self) -> StorageConfig | None: ...
    def get_tmdb_config(self) -> TMDBProcessorConfig | None: ...
```

**Benefits**:
- Clear structure for pipeline stages
- Helper methods for safe config access
- Validation of concurrency and buffer sizes

### Main Configuration (`main_config.py`) - NEW SIMPLIFIED

```python
class ButtermilkConfig(BaseModel):
    # Root-level essentials only
    project_name: str
    job: str
    verbose: bool = False

    # All execution params in run config (including mode)
    run: RunConfig = Field(default_factory=RunConfig)

    # Direct session access (no wrapper)
    session: SessionInfo

    # Infrastructure and configuration
    infrastructure: InfrastructureConfig = Field(default_factory=InfrastructureConfig)
    flows: dict[str, Any] = Field(default_factory=dict)
    storage: dict[str, StorageConfig] = Field(default_factory=dict)

    # Validators
    @field_validator("session", mode="before")
    @classmethod
    def parse_session_info(cls, v) -> SessionInfo:
        # Handles both direct and old bm.session_info format
        ...

    # Helper methods
    def get_storage_config(self, name: str) -> StorageConfig | None: ...
```

**Benefits**:
- Clear root level: only universal essentials (no mode)
- Single source for all execution params including mode (run config)
- Mode loaded via Hydra config groups
- Direct session access
- Backward compatible validators
- Simpler structure, easier to understand

## Hydra Integration

### Interpolation Support

Hydra's interpolation works seamlessly:

```yaml
# config.yaml
verbose: true
project_name: buttermilk
job: test

bm:
  session_info:
    project_name: ${project_name}  # Interpolates
    job: ${job}                    # Interpolates

infrastructure:
  logging:
    verbose: ${verbose}            # Interpolates
```

Pydantic receives the resolved values after Hydra processes interpolations.

### Composition Support

Hydra's composition system continues to work:

```yaml
# config.yaml
defaults:
  - local
  - run: console
  - _self_

# testing.yaml
defaults:
  - run: batch
  - llms: debug
  - _self_
```

The typed system validates the composed result.

### Override Support

Command-line overrides work with Hydra config groups:

```bash
# Load run config group (includes mode inside)
python -m buttermilk.runner.cli run=api run.port=8080
```

## Usage Patterns (NEW)

### Pattern 1: Mode Inside Run Config

```python
typed_cfg = create_config_from_hydra(conf)

# Mode is inside run config (loaded from run=api, etc.)
match typed_cfg.run.mode:
    case RunMode.API:
        # All API params in run config
        start_api_server(
            host=typed_cfg.run.host,
            port=typed_cfg.run.port,
            workers=typed_cfg.run.workers
        )
    case RunMode.BATCH | RunMode.BATCH_RUN | RunMode.BATCH_ALL:
        # Unified limit parameter
        process_batch(
            flow=typed_cfg.run.flow,
            limit=typed_cfg.run.limit
        )
    case RunMode.PIPELINE:
        # Pipeline config in run
        run_pipeline(typed_cfg.run.pipeline)
```

### Pattern 2: Direct Session Access

```python
# Old way
session = typed_cfg.bm.session_info

# New way
session = typed_cfg.session  # Direct access

print(f"Session: {session.session_id}")
print(f"Project: {session.project_name}")
```

### Pattern 3: Unified Limit

```python
# Old way - different params for different modes
if mode == "batch":
    limit = typed_cfg.max_jobs
else:
    limit = typed_cfg.max_records

# New way - single unified parameter
limit = typed_cfg.run.limit  # Works for all modes
```

### Pattern 4: CLI Override

```bash
# Old way - mode at root
mode=api host=0.0.0.0 port=8080

# New way - use Hydra config groups
run=api run.host=0.0.0.0 run.port=8080

# Config group loads mode inside run
run=api run.port=8080  # Other params use defaults from api.yaml
```

## Validation Examples

### Example 1: Invalid Mode

```python
# YAML
run:
  mode: invalid_mode  # Not in valid modes

# Result
ValidationError: 1 validation error for ButtermilkConfig
run.mode
  Input should be 'console', 'batch', 'api', ...
```

### Example 2: Type Mismatch

```python
# YAML
run:
  mode: api
  port: "not_a_number"  # Should be int

# Result
ValidationError: 1 validation error for APIRunConfig
run.port
  Input should be a valid integer
```

### Example 3: Missing Required Field

```python
# YAML
bm:
  session_info:
    # Missing project_name (required)
    job: test

# Result
ValidationError: 1 validation error for SessionInfo
bm.session_info.project_name
  Field required
```

## Migration Path

### Phase 1: Add Typed Models (Completed)

✅ Created typed models for:
- Run configurations (all modes)
- Pipeline configuration
- Infrastructure configuration
- Main root configuration

### Phase 2: Update Entry Points (Next)

Update `cli.py` to use typed configs:

```python
# Before
mode = conf.run.get("mode", "console")
if mode == "api":
    host = conf.get("host", "0.0.0.0")

# After
typed_cfg = create_config_from_hydra(conf)
if typed_cfg.run.mode == RunMode.API:
    host = typed_cfg.host  # From root level, not run config
```

### Phase 3: Update Config Bootstrap (Next)

Update `config_bootstrap.py` to work with typed configs:

```python
# In ConfigurationBootstrapper
def get_run_mode(self) -> RunMode:
    """Get typed run mode."""
    from buttermilk._core.main_config import create_config_from_hydra
    typed_cfg = create_config_from_hydra(self.config)
    return typed_cfg.run.mode
```

### Phase 4: Testing (Next)

Add tests for:
- Configuration validation
- Mode-specific behavior
- Hydra integration
- Error handling

## Benefits Summary

### For Developers

1. **IDE Support**: Autocomplete, type hints, and inline documentation
2. **Early Errors**: Catch configuration errors at startup, not runtime
3. **Clear API**: Know exactly what fields are available for each mode
4. **Refactoring Safety**: Type checker catches breaking changes

### For Operations

1. **Validation**: Invalid configurations fail fast with clear errors
2. **Documentation**: Configuration structure is self-documenting
3. **Consistency**: Shared validation rules across all deployments

### For Maintenance

1. **Discoverability**: Easy to find where configs are used
2. **Impact Analysis**: Type system shows what depends on what
3. **Safe Evolution**: Can change configs with confidence

## Future Enhancements

### Short Term

1. Add validation rules for field combinations
   ```python
   @model_validator(mode="after")
   def validate_batch_modes(self):
       if self.enqueue_only and self.process_only:
           raise ValueError("Cannot specify both enqueue_only and process_only")
   ```

2. Add computed fields for derived values
   ```python
   @computed_field
   def full_project_path(self) -> str:
       return f"{self.project_name}/{self.job}"
   ```

### Medium Term

1. Add JSON schema export for documentation
   ```python
   schema = ButtermilkConfig.model_json_schema()
   # Generate docs from schema
   ```

2. Add configuration versioning
   ```python
   class ButtermilkConfig(BaseModel):
       version: Literal["1.0"] = "1.0"
       # Version-specific fields
   ```

### Long Term

1. Add configuration migrations
   ```python
   def migrate_config_v1_to_v2(old_cfg: dict) -> dict:
       # Handle breaking changes
   ```

2. Add configuration profiles
   ```python
   # profiles/production.yaml
   # profiles/development.yaml
   ```

## Files Created

### Core Implementation

1. `/home/nic/src/writing/projects/buttermilk/buttermilk/_core/run_config.py`
   - RunMode enum with 9 modes
   - Simple RunConfig class
   - ~100 lines (simplified from 220)

2. `/home/nic/src/writing/projects/buttermilk/buttermilk/_core/pipeline_config.py`
   - Pipeline configuration
   - TMDB processor config
   - Helper methods
   - ~130 lines

3. `/home/nic/src/writing/projects/buttermilk/buttermilk/_core/main_config.py`
   - Root configuration model
   - Infrastructure config
   - Tracing config
   - Helper methods
   - ~350 lines

### Documentation

4. `/home/nic/src/writing/projects/buttermilk/docs/configuration.md`
   - Complete usage guide
   - Examples for all modes
   - Troubleshooting
   - Best practices
   - ~600 lines

5. `/home/nic/src/writing/projects/buttermilk/CONFIGURATION_DESIGN.md`
   - This design document
   - Architecture decisions
   - Migration plan

### Examples

6. `/home/nic/src/writing/projects/buttermilk/examples/typed_config_example.py`
   - 8 runnable examples
   - Demonstrates all patterns
   - ~300 lines

## Next Steps

1. ✅ Review design with team
2. ⏭️ Update `cli.py` to use typed configs
3. ⏭️ Update `config_bootstrap.py` to use typed configs
4. ⏭️ Add unit tests
5. ⏭️ Add integration tests
6. ⏭️ Update existing code to use type-safe access
7. ⏭️ Add configuration validation to CI/CD

## Conclusion

The typed configuration system adds strong type safety to Buttermilk while preserving all the flexibility of Hydra's composition system. It makes the codebase more maintainable, catches errors earlier, and provides better developer experience through IDE support.

The design is modular, backward compatible, and ready for incremental adoption. No existing YAML files need to change - the type system validates what's already there and makes it easier to work with in code.
