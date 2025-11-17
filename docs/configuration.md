# Buttermilk Configuration Guide

## Overview

Buttermilk uses a strongly-typed configuration system built on **Hydra** for composition and **Pydantic** for validation. This combination provides:

- **Flexibility**: Compose configurations from multiple YAML files
- **Type Safety**: Catch configuration errors at startup, not runtime
- **IDE Support**: Get autocomplete and type hints when working with configs
- **Clear Structure**: Know exactly what options control what

## Configuration Structure (NEW SIMPLIFIED)

The main configuration is represented by `ButtermilkConfig`, with this simplified hierarchy:

```
ButtermilkConfig (root)
├── project_name: str                # Project identifier
├── job: str                         # Job identifier
├── verbose: bool                    # Global verbose logging
├── run: RunConfig                   # All execution parameters (including mode)
│   ├── mode: RunMode                # Execution mode (loaded via run=api, run=batch, etc.)
│   ├── flow: str                    # Which flow to execute
│   ├── limit: int                   # Unified limit for records/jobs
│   ├── record_id: str               # For console mode testing
│   ├── host: str                    # API server host
│   ├── port: int                    # API server port
│   ├── workers: int                 # API worker count
│   ├── reload: bool                 # API hot reload
│   ├── log_level: str               # API logging level
│   ├── pipeline: PipelineConfig     # Pipeline configuration
│   └── storage_config: dict         # Storage override for batch modes
├── session: SessionInfo             # Session tracking (direct, no wrapper)
├── infrastructure: InfrastructureConfig
│   ├── clouds: list[CloudProvider]  # GCP, AWS, Azure configurations
│   ├── llms: dict                   # LLM model configurations
│   ├── tracing: TracingConfig       # Traceloop, OTEL (Google Cloud Trace)
│   └── logging: LoggerConfig        # Logging configuration
├── flows: dict                      # Flow definitions
└── storage: dict[str, StorageConfig] # Named storage configurations
```

### Key Changes

- **Root Level**: Only universal essentials (project_name, job, verbose)
- **Run Config**: ALL execution parameters including mode
- **Direct Session**: No BMConfig wrapper - session is directly accessible
- **Hydra Config Groups**: Mode loaded via `run=api` from `conf/run/api.yaml`
- **Unified Limit**: Single `limit` replaces max_records/max_jobs

## Run Modes

The `mode` field inside `run` config determines how Buttermilk executes. Mode is loaded via Hydra config groups (e.g., `run=api` loads `conf/run/api.yaml` which contains `mode: api`).

### Console Mode

Run a single flow interactively:

```yaml
run:
  mode: console
  flow: my_flow
  record_id: "test-123" # Optional: test specific record

session:
  project_name: ${project_name}
  job: ${job}
```

**CLI Examples**:

```bash
# Load console run config (contains mode: console)
run=console run.flow=trans

# Test specific record
run=console run.flow=trans run.record_id=abc123

# Verbose mode
verbose=true run=console run.flow=trans
```

### Batch Mode

Create and/or process batch jobs:

```yaml
run:
  mode: batch_all # or batch, batch_run
  flow: trans
  limit: 50 # Process up to 50 jobs
```

**Batch mode variants**:

- `batch`: Create jobs only
- `batch_run`: Process existing jobs
- `batch_all`: Create and process (default)

**CLI Examples**:

```bash
# Process up to 10 jobs (run=batch loads conf/run/batch.yaml with mode: batch_run)
run=batch run.flow=trans run.limit=10

# Create jobs without processing (old enqueue_only)
run=batch run.flow=trans

# Process existing jobs (old process_only)
run=batch_run run.flow=trans
```

**Note**: Old `enqueue_only` and `process_only` flags are replaced by explicit mode selection.

### API Mode

Start a FastAPI server:

```yaml
run:
  mode: api
  host: 0.0.0.0
  port: 8000
  workers: 4
  reload: false # Set true for development
  log_level: info

flows: ${flows} # Expose all flows
```

**CLI Examples**:

```bash
# Basic API server (run=api loads conf/run/api.yaml with mode: api)
run=api

# Custom port and reload for development
run=api run.port=9000 run.reload=true

# Production with multiple workers
run=api run.workers=4 run.log_level=warning
```

### Pipeline Mode

Run multi-stage data processing:

```yaml
run:
  mode: pipeline
  pipeline:
    source:
      type: bigquery
      full_table_id: "project.dataset.table"
      custom_query: "SELECT * FROM {table} WHERE year > 2020"

    output:
      type: bigquery
      full_table_id: "project.dataset.output_table"

    tmdb:
      region: US
      language: en-US

    concurrency: 20
    buffer_size: 500
    flush_interval: 30
```

**CLI Examples**:

```bash
# Basic pipeline (run=pipeline loads conf/run/pipeline.yaml with mode: pipeline)
run=pipeline

# Override concurrency
run=pipeline run.pipeline.concurrency=50

# Limit records processed
run=pipeline run.limit=1000
```

## Infrastructure Configuration

The `infrastructure` section configures all external services:

### Cloud Providers

```yaml
infrastructure:
  clouds:
    - type: gcp
      project_id: my-project
      region: us-central1
      storage_bucket: my-bucket
      secrets:
        models_secret: dev__llm__connections
        credentials_secret: dev__shared_credentials
      pubsub:
        jobs_topic: jobs
        jobs_subscription: jobs-sub
```

**Type**: `InfrastructureConfig` with `GCPConfig`

### Tracing

```yaml
infrastructure:
  tracing:
    weave:
      enabled: true
      project_id: my-entity
      api_key: ${oc.env:WANDB_API_KEY}

    otel:
      enabled: true
      endpoint: https://telemetry.googleapis.com
```

**Type**: `TracingConfig` with `TracingProviderConfig` for each provider

### Logging

```yaml
infrastructure:
  logging:
    type: gcp # or local, aws, azure
    project_id: my-project
    verbose: ${verbose}
    console: true # Log to stderr for MCP compatibility
```

**Type**: `LoggerConfig`

## Storage Configuration

Storage configurations are strongly typed with a discriminated union:

```yaml
storage:
  observations:
    type: bigquery
    full_table_id: "project.dataset.observations"
    batch_size: 1000
    randomize: true

  vector_store:
    type: chromadb
    collection_name: my_collection
    persist_directory: gs://bucket/chromadb
    embedding_model: gemini-embedding-001
    dimensionality: 3072

  local_files:
    type: file
    path: /data/files
    glob: "**/*.json"
```

**Types**: `BigQueryStorageConfig`, `VectorStorageConfig`, `FileStorageConfig`, etc.

## Configuration Composition

Buttermilk uses Hydra's composition system to build configurations from multiple files:

### Base Configuration

`conf/config.yaml`:

```yaml
defaults:
  - local
  - _self_

verbose: false
flows: {}

bm:
  _target_: buttermilk._core.bm_init.BM
  session_info:
    project_name: ${project_name}
    job: ${job}
```

### Environment-Specific Overrides

`conf/testing.yaml`:

```yaml
defaults:
  - env: testing
  - flows:
      - trans
  - llms: debug
  - _self_

verbose: true

storage:
  observations:
    full_table_id: "test-project.testing.observations"
```

### Run Mode Overrides

`conf/run/batch.yaml`:

```yaml
#@package _global_
defaults:
  - _self_

run:
  mode: batch_run
  ui: console

max_jobs: 10
```

### Storage Configs

`conf/storage/observations.yaml`:

```yaml
type: bigquery
full_table_id: "project.dataset.observations"
batch_size: 1000
dataset_name: observations
split_type: train
```

## Using Typed Configurations in Code

### In CLI Entry Point

```python
from buttermilk._core.main_config import create_config_from_hydra
from buttermilk._core.run_config import RunMode


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(conf: DictConfig) -> None:
    # Convert to typed config
    typed_cfg = create_config_from_hydra(conf)

    # Type-safe access with IDE autocomplete
    mode = typed_cfg.run.mode  # RunMode enum (inside run config)
    verbose = typed_cfg.verbose  # bool

    # Access session info directly
    session = typed_cfg.session  # SessionInfo
    print(f"Session: {session.session_id}")

    # Check mode and access run config params
    if typed_cfg.run.mode == RunMode.API:
        host = typed_cfg.run.host  # str
        port = typed_cfg.run.port  # int
        start_api_server(host, port, typed_cfg.run.workers)

    elif typed_cfg.run.mode == RunMode.CONSOLE:
        flow_name = typed_cfg.run.flow  # str | None
        record_id = typed_cfg.run.record_id  # str | None
        run_console(flow_name, record_id)
```

### In Pipeline Mode

```python
# Get pipeline config from run config
if typed_cfg.run.mode == RunMode.PIPELINE:
    pipeline_cfg = typed_cfg.run.pipeline
    if pipeline_cfg:
        concurrency = pipeline_cfg.concurrency  # int

        # Get TMDB config
        tmdb_cfg = pipeline_cfg.get_tmdb_config()
        if tmdb_cfg:
            region = tmdb_cfg.region  # str
```

### Accessing Storage Configs

```python
# Get typed storage config
storage_cfg = typed_cfg.get_storage_config("observations")
if isinstance(storage_cfg, BigQueryStorageConfig):
    table_id = storage_cfg.full_table_id  # str | None
    batch_size = storage_cfg.batch_size  # int
```

## Validation

Pydantic validates all configurations at startup:

```python
# This will raise ValidationError if invalid
typed_cfg = create_config_from_hydra(conf)

# Example validation errors:
# - run.mode must be one of: console, batch, api, pipeline, ...
# - run.port must be an integer
# - storage.observations.type must be specified
# - infrastructure.clouds[0].project_id is required for GCP
```

## Interpolation

Hydra supports variable interpolation that Pydantic respects:

```yaml
# Root-level variables
verbose: true
project_name: buttermilk
job: test

# Reference in nested configs
bm:
  session_info:
    project_name: ${project_name} # Interpolates to "buttermilk"
    job: ${job} # Interpolates to "test"

infrastructure:
  logging:
    verbose: ${verbose} # Interpolates to true
```

## Command-Line Overrides

Override any configuration from the command line:

```bash
# Load run config group (contains mode inside run section)
run=api

# Override run config parameters
run=api run.port=9000 run.workers=4

# Override multiple values
verbose=true run=console run.flow=trans run.record_id=abc123

# Use a different environment config
-cn testing run=batch run.limit=10

# Override infrastructure
run=api infrastructure.tracing.weave.enabled=true
```

## Migration from Old Structure

The configuration system provides backward compatibility:

### Old Structure (Still Supported)

```yaml
# Old: mode at root
mode: api

# Old: params at root
host: 0.0.0.0
port: 8000
max_jobs: 10

# Old: BMConfig wrapper
bm:
  session_info:
    project_name: ${project_name}
    job: ${job}
```

### New Structure (Recommended)

```yaml
# New: mode inside run (loaded via run=api config group)
run:
  mode: api
  host: 0.0.0.0
  port: 8000
  limit: 10 # Unified from max_jobs/max_records

# New: direct session
session:
  project_name: ${project_name}
  job: ${job}
```

### Automatic Migration

The `create_config_from_hydra()` function automatically handles:

- `bm.session_info` → `session`
- `mode` at root → `run.mode`
- `max_records`/`max_jobs` → `run.limit`
- Root-level execution params → `run` config
- Removed deprecated params (enqueue_only, process_only, dataset_key, prompt)

````
## Migration Guide (Code)

### From Loose Dicts to Typed Configs

**Before**:
```python
mode = conf.run.get("mode", "console")  # Dict access
if mode == "api":
    host = conf.get("host", "0.0.0.0")  # May not exist
    session = conf.bm.session_info  # Nested wrapper
````

**After**:

```python
from buttermilk._core.run_config import RunMode

typed_cfg = create_config_from_hydra(conf)
if typed_cfg.run.mode == RunMode.API:  # Mode inside run config
    host = typed_cfg.run.host  # Type-safe, in run config
    session = typed_cfg.session  # Direct access, no wrapper
```

### Benefits

1. **Type Safety**: Catch errors at startup, not runtime
2. **IDE Support**: Autocomplete and type hints
3. **Clear Structure**: Mode inside run, all execution params together
4. **Direct Access**: No BMConfig wrapper for session
5. **Unified Limits**: Single parameter for records/jobs
6. **Backward Compatible**: Old configs still work

## Best Practices

1. **Use Type Hints**: Let your IDE help you
   ```python
   from buttermilk._core.main_config import ButtermilkConfig

   def process_config(cfg: ButtermilkConfig) -> None:
       # IDE will autocomplete cfg.mode, cfg.run.flow, etc.
   ```

2. **Validate Early**: Convert to typed config at entry point
   ```python
   typed_cfg = create_config_from_hydra(conf)  # Validates immediately
   ```

3. **Check Mode First**: Use the enum for type safety
   ```python
   from buttermilk._core.run_config import RunMode

   if typed_cfg.run.mode == RunMode.PIPELINE:
       # Access pipeline config from run
       pipeline = typed_cfg.run.pipeline
   ```

4. **Access Session Directly**: No more wrapper
   ```python
   # Old way
   session = typed_cfg.bm.session_info

   # New way
   session = typed_cfg.session
   ```

5. **Use Unified Limit**: Single parameter for all modes
   ```python
   # Old way
   max_records = typed_cfg.max_records
   max_jobs = typed_cfg.max_jobs

   # New way
   limit = typed_cfg.run.limit  # Works for both records and jobs
   ```

## Troubleshooting

### "Field required" Error

**Problem**: Pydantic complains about missing required field

**Solution**: Either provide the field or make it optional:

```python
field: str = Field(description="Required field")
# or
field: str | None = Field(default=None, description="Optional field")
```

### "Extra fields not permitted" Error

**Problem**: Pydantic rejects unexpected fields

**Solution**: Add `extra="allow"` to model_config:

```python
model_config = {
    "extra": "allow",  # Allow additional fields
}
```

### Type Mismatch Error

**Problem**: Field type doesn't match Hydra config

**Solution**: Use union types or validators:

```python
field: str | int  # Allow either type
# or use validator to coerce types
```

## Examples

See example configurations in `conf/`:

- `conf/testing.yaml` - Testing environment
- `conf/local.yaml` - Local development
- `conf/run/api.yaml` - API server mode
- `conf/run/batch.yaml` - Batch processing mode
- `conf/run/mmtmdb.yaml` - Pipeline mode example
