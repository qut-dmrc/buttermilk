# Buttermilk Unified Storage - Usage Guide

## Overview

Buttermilk provides a unified storage framework that handles both reading and writing operations across different backends (BigQuery, files, etc.) through the BM singleton pattern.

## Recommended Usage Pattern

### ✅ Use BM Factory Methods (Recommended)

```python
from buttermilk import get_bm

bm = get_bm()

# Simple BigQuery storage with dataset name
storage = bm.get_bigquery_storage("my_dataset")

# Custom configuration via BM factory
from buttermilk.storage import StorageConfig

config = StorageConfig(
    type="bigquery", dataset_name="my_dataset", randomize=False, batch_size=500
)
storage = bm.get_storage(config)

# File storage
config = StorageConfig(type="file", path="data.jsonl")
storage = bm.get_storage(config)
```

### ❌ Direct Instantiation (Avoid)

```python
# Don't do this - bypasses BM singleton benefits
from buttermilk.storage import BigQueryStorage, StorageConfig

config = StorageConfig(type="bigquery", dataset_name="test")
storage = BigQueryStorage(config, bm=None)  # Missing BM integration
```

## When to Use Each Pattern

### Use `bm.get_bigquery_storage(dataset_name)` when:

- Simple BigQuery operations with default settings
- Dataset name is the only required parameter
- You want the simplest API

### Use `bm.get_storage(config)` when:

- Custom configuration is needed
- Advanced parameters like filtering, batch sizes, etc.
- Cross-backend compatibility (files, BigQuery, etc.)

### Use direct `StorageConfig` instantiation when:

- Creating configuration objects to pass to BM factory methods
- Testing or library development
- Need to validate configuration without creating storage instances

## Benefits of BM Factory Pattern

1. **Integrated Cloud Clients**: Automatic access to BigQuery, GCS clients via BM
1. **Default Configuration**: Inherits project-level storage defaults
1. **Resource Management**: Proper client lifecycle and connection pooling
1. **Configuration Merging**: Combines user config with BM defaults
1. **Session Context**: Access to run IDs, logging, tracing integration

## Agent Independence Principle

- **Agent configuration is STRICTLY independent from flow configuration**.
- Required flow configuration should be passed in through composable configuration at run time.
- This separation allows agents to be reused across different flows.

## Examples

### Reading Data

```python
from buttermilk import bm, logger, BM

# Read from BigQuery
storage = bm.get_bigquery_storage("my_dataset")
for record in storage:
    print(record.content)

# Read from file with filtering
config = StorageConfig(type="file", path="data.jsonl", limit=100)
storage = bm.get_storage(config)
records = list(storage)
```

### Writing Data

```python
from buttermilk import bm, logger, BM

# Write to BigQuery
storage = bm.get_bigquery_storage("my_dataset")
storage.save(records)

# Write to file
config = StorageConfig(type="file", path="output.jsonl")
storage = bm.get_storage(config)
storage.save(records)
```

### Migration Between Backends

```python
bm = get_bm()

# Source: File
source_config = StorageConfig(type="file", path="input.jsonl")
source = bm.get_storage(source_config)

# Target: BigQuery
target = bm.get_bigquery_storage("my_dataset")

# Migrate
records = list(source)
target.save(records)
```

## Legacy Compatibility

Old patterns are supported but deprecated:

```python
# DEPRECATED - use bm.get_bigquery_storage() instead
from buttermilk.data.bigquery_loader import BigQueryRecordLoader

loader = BigQueryRecordLoader(dataset_name="test")

# DEPRECATED - use bm.get_storage() instead
from buttermilk.storage.compat import BigQueryRecordLoader

loader = BigQueryRecordLoader(config=config)
```
