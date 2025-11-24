## Buttermilk Configuration Type System - Design Document

### Design Principles

1. **Preserve Hydra Flexibility**: Continue using Hydra's composition, interpolation, and override system
2. **Add Type Safety**: Convert to Pydantic models at the entry point for validation
3. **Clear Structure**: Make the configuration hierarchy explicit and discoverable
4. **Backward Compatible**: Existing YAML configs work without changes

### Configuration Hierarchy

```
ButtermilkConfig (root)
├── Universal Essentials (Root Level)
│   ├── project_name: str
│   ├── job: str
│   └── verbose: bool
├── run: RunConfig                       # ALL execution parameters (including mode)
│   ├── mode: RunMode                    # Loaded via run=api config group
│   ├── flows: dict[str, Any]            # Flow definitions (MOVED from root)
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
│   └── flows: dict[str, OrchestratorProtocol | Any]  # Flow definitions (MOVED from root)
└── storage: dict[str, StorageConfig]
```

### Key Simplifications

1. **Root Level**: Only 3 fields (project_name, job, verbose) - flows MOVED to run.flows
2. **Run Config**: Consolidates ALL execution parameters including mode AND flow definitions
3. **No Wrappers**: Direct session access (removed BMConfig)
4. **Hydra Config Groups**: Mode loaded via `run=api` from `conf/run/api.yaml`
5. **Unified Limit**: One parameter for both records and jobs
6. **Flows in run.flows**: Flow definitions now properly grouped with execution params
