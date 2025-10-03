# Buttermilk Configuration System

## Overview
Buttermilk uses Hydra with OmegaConf for configuration management. All configuration is done through YAML files - **NEVER** use manual dictionary configuration.

## Critical Rules

### DO
- ✅ Use YAML files exclusively
- ✅ Leverage interpolation
- ✅ Validate early with Pydantic
- ✅ Keep secrets in environment/secrets manager
- ✅ Document complex configurations

### DON'T
- ❌ Create manual dictionaries
- ❌ Hardcode values in code
- ❌ Change validation to suppress errors
- ❌ Commit sensitive data
- ❌ Ignore type mismatches

Remember: Configuration drives behavior. When debugging, always check the composed configuration first with `-c job`.

## Core Concepts

### 1. Hydra Basics
- **Composition**: Build configs from multiple files
- **Overrides**: Modify any value from command line 
- **Interpolation**: Reference other config values with `${}`
- **Validation**: Early fail with clear error messages

### 2. Configuration Structure
```
conf/
├── config.yaml          # Base configuration
├── local.yaml          # Local overrides (gitignored)
├── agents/             # Agent configurations
├── flows/              # Flow definitions
├── llms/               # Model configurations
├── run/                # Execution modes
└── storage/            # Storage backends
```

### 3. Configuration Issues

#### Check Composed Configuration
```bash
# View full configuration
uv run python -m buttermilk.runner.cli -c job

# Check specific values
uv run python -m buttermilk.runner.cli -c job | grep -A 10 "agents:"
```

## LLM Configuration Details

The LLM configuration is loaded from a GCP Secret named `models.json`. Authentication with GCP is required for tests to run correctly.




## Storage configuration

Define storage objects in Hydra YAML config:

```yaml
storage:
    my_bigquery_storage:
        type: bigquery
        full_table_id: my-gcp-project.my_dataset.my_table
        auto_create: true
        schema_path: "conf/schemas/my_schema.json"
```

Storage configs are automatically converted to Pydantic models (BigQueryStorageConfig) when loaded by Hydra. 

## How to access configured storage objects in your code

- The Hydra configurations are accessed through Pydantic models. 
- The `buttermilk._core.storage_config.BigQueryStorageConfig` class is the Pydantic model that represents the BigQuery storage configuration.
- When Hydra loads the configuration, it uses this Pydantic model to validate and create a `BigQueryStorageConfig` object. 
- The `buttermilk._core.storage_config.StorageFactory` class is responsible for instantiating storage objects from the configuration. Its `create_storage` method takes a configuration object and returns the appropriate storage implementation.

### Example: saving data to bigquery from within an orchestrated flow

The workflow is: YAML config → Pydantic model → StorageFactory → BigQuery storage instance → save data.

Storage configurations are defined in YAML files. For BigQuery, the `type` field must be set to `bigquery`. You can also provide a `full_table_id` for convenience, which will be parsed into its constituent parts.

Here is an example of a BigQuery storage configuration in a YAML file:

```yaml
# buttermilk/conf/flows/my_flow.yaml
storage:
    my_bigquery_storage:
        type: bigquery
        full_table_id: my-gcp-project.my_dataset.my_table
        auto_create: true
        schema_path: "conf/schemas/my_schema.json"
```
Within a flow, storage objects can be accessed through the orchestrator:

```python
  # In your orchestrator/flow
  storage_config = orchestrator.storage["my_bigquery_storage"]
  assert isinstance(storage_config, BigQueryStorageConfig)
```

Use the StorageFactory to create storage instances:

```python
from buttermilk._core.storage_config import StorageFactory

# Create storage instance from config
storage = StorageFactory.create_storage(storage_config)
```

The `buttermilk.storage.bigquery.BigQueryStorage` class provides a `save` method to write data to BigQuery. This method takes a list of Pydantic models or dictionaries and handles the serialization and upload
process. To save to BigQuery, use the storage object's save method:

```python
# Save Pydantic models or dictionaries
records = [{"id": 1, "name": "test"}]
storage.save(records)
```

### Example: Initialization with init()

The pattern is: init() → bm object → get_storage() → use storage for operations.
Use the main init() function to bootstrap a Buttermilk session:

```python
from buttermilk._core.config_bootstrap import init

## Initialize with job name and project
bm = init(job="my_analysis_job", project="my_project")

Accessing Storage Objects

After initialization, use the bm object to get storage instances:

# Method 1: Using get_storage() with config
storage_config = BigQueryStorageConfig(
    type="bigquery",
    project_id="my-gcp-project",
    dataset_id="my_dataset",
    table_id="my_table"
)
storage = bm.get_storage(storage_config)

# Method 2: Direct BigQuery storage convenience method
bq_storage = bm.get_bigquery_storage()

# Method 3: Direct client access
bq_client = bm.bq
gcs_client = bm.gcs
```

Complete Usage Example

```python
def run_my_flow():
    # 1. Initialize
    bm = init(job="data_analysis", project="my_project")

    # 2. Get storage from config
    storage = bm.get_storage(storage_config)

    # 3. Use storage for I/O operations
    records = [{"id": 1, "data": "example"}]
    storage.save(records)

    # 4. Or use clients directly
    query_job = bm.bq.query("SELECT * FROM my_table LIMIT 10")
    for row in query_job:
        print(row)
```
