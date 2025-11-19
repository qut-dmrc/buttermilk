# Buttermilk Configuration Type System

## Overview

Buttermilk uses strongly-typed Pydantic configuration with Hydra composition for flexibility and type safety. This design simplifies the configuration structure while preserving Hydra's powerful composition system.

## Design Principles

1. **Preserve Hydra Flexibility**: Continue using composition, interpolation, and override system
2. **Add Type Safety**: Convert to Pydantic models at entry point for validation
3. **Clear Structure**: Make configuration hierarchy explicit and discoverable
4. **Backward Compatible**: Existing YAML configs work without changes

## Configuration Hierarchy

```
ButtermilkConfig (root)
├── project_name: str
├── job: str
├── verbose: bool
├── run: RunConfig                      # All execution parameters
│   ├── mode: RunMode                   # Loaded via run=api config group
│   ├── flows: dict[str, Any]           # Flow definitions
│   ├── flow: str | None
│   ├── limit: int | None               # Unified from max_records/max_jobs
│   ├── host, port, workers, reload     # API mode params
│   ├── pipeline: PipelineConfig        # Pipeline mode
│   └── storage_config: dict            # Batch mode override
├── session: SessionInfo                # Direct access (no BMConfig wrapper)
├── infrastructure: InfrastructureConfig
│   ├── clouds: list[CloudProvider]
│   ├── llms: dict[str, Any]
│   ├── tracing: TracingConfig
│   └── logging: LoggerConfig
└── storage: dict[str, StorageConfig]
```

## Key Simplifications

1. **Root Level**: Only universal essentials (project_name, job, verbose)
2. **Run Config**: ALL execution parameters including mode
3. **No Wrappers**: Direct session access (removed BMConfig)
4. **Hydra Config Groups**: Mode loaded via `run=api` from `conf/run/api.yaml`
5. **Unified Limit**: Single parameter replaces max_records/max_jobs

## Run Modes

Mode is inside RunConfig and loaded via Hydra config groups:

```bash
# CLI usage - mode is loaded from config group
run=api run.port=8000              # Loads conf/run/api.yaml (mode: api)
run=console run.flow=trans         # Loads conf/run/console.yaml (mode: console)
run=batch run.limit=10             # Loads conf/run/batch.yaml (mode: batch)
```

## Backward Compatibility

The `create_config_from_hydra()` function provides automatic migration:
- `bm.session_info` → `session`
- `mode` at root → `run.mode`
- `max_records`/`max_jobs` → `run.limit`
- Root-level execution params → `run` config

Existing YAML configs continue to work without changes.

## Usage

```python
from buttermilk._core.main_config import create_config_from_hydra
from buttermilk._core.run_config import RunMode

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(conf: DictConfig) -> None:
    typed_cfg = create_config_from_hydra(conf)

    # Type-safe access with autocomplete
    if typed_cfg.run.mode == RunMode.API:
        start_api_server(
            host=typed_cfg.run.host,
            port=typed_cfg.run.port,
            workers=typed_cfg.run.workers
        )
    elif typed_cfg.run.mode == RunMode.CONSOLE:
        run_console(typed_cfg.run.flow, typed_cfg.run.record_id)
```

See `docs/configuration.md` for complete usage guide.

## Implementation Files

- `buttermilk/_core/run_config.py` - Run mode configurations (~100 lines)
- `buttermilk/_core/pipeline_config.py` - Pipeline configurations (~130 lines)
- `buttermilk/_core/main_config.py` - Root configuration model (~350 lines)
- `docs/configuration.md` - Comprehensive user guide (~600 lines)

## Migration Status

### Flows Configuration Migration (Completed 2025-10-20)

The migration of flows from root level to `run.flows` has been completed:

**YAML Changes**:
- ✅ All config files updated to use `run.flows`
- ✅ Hydra composition paths fixed (`runs.flows` → `run.flows`)
- ✅ Invalid run modes corrected

**Python Changes**:
- ✅ `buttermilk/runner/cli.py` - Uses `conf.run.flows` directly
- ✅ `buttermilk/runner/batch_cli.py` - All references updated
- ✅ All test files updated to use `run.flows`

**Backward Compatibility**:
- ✅ Validator in `main_config.py` (lines 162-167) migrates old configs automatically
- ✅ Root-level `flows` automatically moved to `run.flows` during config loading

All tests pass successfully. The backward compatibility validator can be removed in a future cleanup phase once all external configurations are confirmed to use `run.flows`.

## Benefits

**For Developers**:
- IDE autocomplete and type hints
- Catch configuration errors at startup
- Clear API for each run mode
- Type checker catches breaking changes

**For Operations**:
- Invalid configurations fail fast with clear errors
- Self-documenting configuration structure
- Consistent validation across deployments

**For Maintenance**:
- Easy to find where configs are used
- Type system shows dependencies
- Safe evolution of configuration schema
