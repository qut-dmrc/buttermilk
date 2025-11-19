# Minimal Buttermilk Configuration

## Overview

The `minimal.yaml` configuration provides a way to use Buttermilk without requiring any cloud credentials or external services. This is useful for:

- **Local development** without GCP access
- **CI/CD testing** without secret management
- **Containerized deployments** (like MCP servers) that only need local functionality
- **Learning and experimentation** with Buttermilk's core features

## What's Included

### Enabled Features

- Local logging
- Session management
- Record handling and manipulation
- Template rendering
- Configuration loading
- Local file storage

### Disabled Features

- Cloud storage (GCS, BigQuery)
- Cloud logging
- Secret Manager
- Vertex AI
- LLM integrations (no API keys)
- Tracing (Traceloop, OpenTelemetry)

## Usage

### In Code

```python
from hydra import compose, initialize
from buttermilk._core.config_bootstrap import bootstrap_session_with_config

# Initialize with minimal config
with initialize(version_base=None, config_path="../buttermilk/conf"):
    cfg = compose(config_name="minimal")

bm, resolved_conf = bootstrap_session_with_config(config=cfg)

# Now use bm for local-only operations
from buttermilk._core.types import Record

record = Record(
    content="Test content", mime="text/plain", metadata={"source": "local_test"}
)
```

### In Tests

A `local_bm` fixture is available in `tests/endtoend/conftest.py`:

```python
def test_my_local_feature(local_bm):
    """Test that works without cloud credentials."""
    assert local_bm is not None
    # ... test local functionality
```

## Testing

Integration tests for minimal configuration are in:

```
tests/endtoend/test_local_minimal_init.py
```

Run them with:

```bash
uv run pytest tests/endtoend/test_local_minimal_init.py -v --tb=short
```

## Configuration Structure

```yaml
bm:
  session_info:
    project_name: buttermilk
    job: local_test
    cache_dir: ".cache/buttermilk"
    sessions_dir: "data/sessions"

infrastructure:
  clouds: [] # No cloud configuration
  llms: {} # No LLM configuration

  tracing:
    traceloop: { enabled: false }
    otel: { enabled: false }

  logging:
    type: local # Only local logging
    verbose: false

storage: {} # No storage by default
pipeline: null # No pipeline by default
```

## Adding Features

To add specific features to the minimal config:

### Local Storage

```yaml
storage:
  my_data:
    type: file
    path: "data/my_data.json"
    record_class: buttermilk.storage.base.BaseRecord
```

### ChromaDB (local)

```yaml
storage:
  vectors:
    type: chromadb
    collection_name: my_collection
    persist_directory: ".cache/chromadb"
    embedding_model: "local-model" # Use local embeddings
    dimensionality: 384
```

## Use Cases

### MCP Servers

The osbchatmcp and zotmcp servers use this pattern - they need buttermilk to BUILD the ChromaDB databases (with full cloud access), but at RUNTIME they only use the local database files without any cloud dependencies.

### Testing

Tests can use `local_bm` fixture to verify functionality that doesn't require external services.

### Development

Developers can work on local features without needing GCP credentials configured.

## Limitations

Without cloud access, you cannot:

- Call LLMs (Gemini, OpenAI, etc.)
- Access BigQuery datasets
- Read from/write to GCS buckets
- Use Secret Manager for credentials
- Use Vertex AI services
- Send telemetry to cloud tracing services

However, you can still build, test, and deploy applications that use Buttermilk's core functionality with pre-built local resources (like ChromaDB databases).
