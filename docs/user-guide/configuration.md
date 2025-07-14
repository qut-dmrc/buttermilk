# Configuration Guide

This comprehensive guide explains how to configure Buttermilk using Hydra, covering everything from basic setup to advanced patterns.

## Table of Contents
- [Understanding Hydra Configuration](#understanding-hydra-configuration)
- [Configuration Structure](#configuration-structure)
- [Basic Usage](#basic-usage)
- [Advanced Patterns](#advanced-patterns)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Best Practices](#best-practices)

## Understanding Hydra Configuration

Buttermilk uses [Hydra](https://hydra.cc/) for configuration management, providing:
- **Hierarchical composition** - Build configs from multiple files
- **Command-line overrides** - Modify any config value at runtime
- **Interpolation** - Reference other config values
- **Type safety** - Structured configs with validation

### Key Concepts

#### 1. Configuration Composition
Hydra builds configurations by merging multiple YAML files:

```yaml
# Base config (config.yaml)
defaults:
  - local              # Includes conf/local.yaml
  - llms: lite        # Includes conf/llms/lite.yaml at 'llms' key
  - flows: []         # Empty by default
```

#### 2. Package Directives
The `@` symbol controls where configs are placed:

```yaml
defaults:
  # Places content at root level
  - run: api
  
  # Places content under 'bm.llms'
  - llms@bm.llms: lite
  
  # Places agents under 'osb.agents'
  - /agents/rag@osb.agents: researcher
```

#### 3. Interpolation
Reference other config values with `${}`:

```yaml
run:
  name: ${bm.name}      # References bm.name
  flows: ${flows}       # References root-level flows
  
agents:
  model: ${llms.general}  # References llms.general
```

## Configuration Structure

```
conf/
├── config.yaml          # Base configuration
├── local.yaml           # Local environment settings
├── flows/              # Flow definitions
│   ├── osb.yaml        # OSB flow
│   ├── tox.yaml        # Toxicity flow
│   └── trans.yaml      # Trans flow
├── agents/             # Agent configurations
│   ├── rag/            # RAG agents
│   │   ├── researcher.yaml
│   │   └── policy_analyst.yaml
│   └── host/           # Host agents
│       ├── llm_host.yaml
│       └── sequence_host.yaml
├── llms/               # LLM configurations
│   ├── lite.yaml       # Lightweight models
│   └── full.yaml       # Full model suite
├── run/                # Run mode configurations
│   ├── api.yaml        # API server mode
│   ├── console.yaml    # Console mode
│   └── batch.yaml      # Batch processing mode
├── data/               # Data source configurations
│   ├── local_files.yaml
│   ├── gsheet.yaml
│   └── csv.yaml
└── storage/            # Storage configurations
    ├── local.yaml
    ├── bigquery.yaml
    └── gcs.yaml
```

## Basic Usage

### Running with Default Configuration

```bash
# Use default configuration
uv run python -m buttermilk.runner.cli

# Check what configuration is loaded
uv run python -m buttermilk.runner.cli -c job
```

### Selecting Configuration Groups

```bash
# Select run mode
uv run python -m buttermilk.runner.cli run=api

# Select flow
uv run python -m buttermilk.runner.cli +flow=osb

# Select LLM configuration
uv run python -m buttermilk.runner.cli llms=full

# Combine selections
uv run python -m buttermilk.runner.cli run=api +flow=osb llms=full
```

### Command Line Overrides

```bash
# Override specific values
uv run python -m buttermilk.runner.cli run=api server.port=8080

# Add new values with +
uv run python -m buttermilk.runner.cli +debug=true

# Override nested values
uv run python -m buttermilk.runner.cli agents.0.model=gpt-4
```

## Advanced Patterns

### Pattern 1: Multi-Flow Configuration

```yaml
# conf/flows/multi_flow.yaml
defaults:
  - _self_
  # Load multiple flows
  - flows:
    - osb
    - tox
    - trans
  - llms: full
  
run:
  mode: api
  flows: ${flows}  # References all loaded flows
```

### Pattern 2: Environment-Specific Configs

```yaml
# conf/envs/production.yaml
defaults:
  - _self_
  - /base
  
bm:
  platform: gcp
  clouds:
    - type: gcp
      project: ${oc.env:GCP_PROJECT}  # From environment variable
      
logging:
  level: INFO
  
storage:
  type: bigquery
  project: ${oc.env:GCP_PROJECT}
```

### Pattern 3: Reusable Components

```yaml
# conf/components/rag_agents.yaml
defaults:
  - /agents@agents.researcher: rag/researcher
  - /agents@agents.analyst: rag/policy_analyst
  
# Can be included in flows
defaults:
  - /components/rag_agents
```

### Pattern 4: Flow with Agents

```yaml
# conf/flows/analysis_flow.yaml
defaults:
  - _self_
  # Load multiple agents under 'agents' key
  - /agents@agents:
    - assistant
    - researcher
  # Load observers
  - /agents@observers:
    - host/llm_host

analysis_flow:
  orchestrator: buttermilk.orchestrators.groupchat.AutogenOrchestrator
  name: "Analysis Flow"
  # These will be populated by defaults
  agents: {}      # Filled by defaults
  observers: {}   # Filled by defaults
```

## Working with Data Sources

### Local Files

```yaml
# conf/data/local_files.yaml
data:
  source: local
  path: "/path/to/data"
  format: csv
  columns:
    id: "record_id"
    text: "content"
```

### Google Sheets

```yaml
# conf/data/gsheet.yaml
data:
  source: gsheet
  spreadsheet_id: "your-spreadsheet-id"
  range: "Sheet1!A:C"
  credentials_path: "path/to/service-account.json"
```

### CSV Files

```yaml
# conf/data/csv.yaml
data:
  source: csv
  path: "/path/to/file.csv"
  delimiter: ","
  encoding: "utf-8"
  columns:
    id: "id"
    text: "text"
    metadata: "meta"
```

## LLM Configuration

### Lightweight Configuration

```yaml
# conf/llms/lite.yaml
llms:
  general: gemini-flash
  models:
    gemini-flash:
      provider: google
      model: gemini-1.5-flash
      temperature: 0.7
      max_tokens: 1000
```

### Full Configuration

```yaml
# conf/llms/full.yaml
llms:
  general: gemini-pro
  coding: claude-3-5-sonnet
  fast: gemini-flash
  
  models:
    gemini-pro:
      provider: google
      model: gemini-1.5-pro
      temperature: 0.7
      max_tokens: 8192
      
    claude-3-5-sonnet:
      provider: anthropic
      model: claude-3-5-sonnet-20241022
      temperature: 0.5
      max_tokens: 4096
      
    gemini-flash:
      provider: google
      model: gemini-1.5-flash
      temperature: 0.9
      max_tokens: 2048
```

## Storage Configuration

### Local Storage

```yaml
# conf/storage/local.yaml
storage:
  type: local
  path: "/tmp/buttermilk-data"
  format: json
  compression: gzip
```

### BigQuery Storage

```yaml
# conf/storage/bigquery.yaml
storage:
  type: bigquery
  project: ${oc.env:GCP_PROJECT}
  dataset: "buttermilk_data"
  table: "results"
  schema:
    - name: "id"
      type: "STRING"
      mode: "REQUIRED"
    - name: "content"
      type: "STRING"
      mode: "NULLABLE"
```

### Google Cloud Storage

```yaml
# conf/storage/gcs.yaml
storage:
  type: gcs
  bucket: "buttermilk-storage"
  prefix: "results/"
  format: jsonl
  compression: gzip
```

## Common Issues and Solutions

### Error: "Could not override 'X'. No match in defaults"

**Cause**: Trying to override a config that doesn't exist.

**Fix**: Use `+` to add new configs:
```bash
# Wrong
uv run python -m buttermilk.runner.cli flows=osb

# Correct
uv run python -m buttermilk.runner.cli +flows=osb
```

### Error: "Interpolation key 'X' not found"

**Cause**: Config references a value that doesn't exist.

**Fix**: Ensure the referenced config is loaded:
```yaml
defaults:
  - llms: lite  # Must be loaded for ${llms.general} to work
```

### Error: "Key 'X' is not in struct"

**Cause**: Structured configs don't allow arbitrary keys.

**Fix**: Either:
1. Use `+` prefix to add keys
2. Define the structure in the config class
3. Set `struct: false` in the config

### Error: "MissingConfigException: Missing @package directive"

**Cause**: Config group doesn't specify where to place content.

**Fix**: Add `# @package` directive:
```yaml
# @package agents.researcher
name: "Research Assistant"
type: "RAGAgent"
```

## Best Practices

### 1. Use Explicit Defaults

Always specify what configs you need:

```yaml
defaults:
  - _self_            # Current config takes precedence
  - local             # Environment settings
  - flows: osb        # Specific flow
  - llms: lite        # LLM configuration
```

### 2. Create Reusable Components

```yaml
# conf/components/rag_agents.yaml
defaults:
  - /agents@agents.researcher: rag/researcher
  - /agents@agents.analyst: rag/policy_analyst
```

### 3. Document Interpolations

```yaml
# Document what values are expected
run:
  name: ${bm.name}        # From bm config
  flows: ${flows}         # From flows config group
  model: ${llms.general}  # From llms config
```

### 4. Test Configuration Composition

```bash
# View the composed configuration
uv run python -m buttermilk.runner.cli --config-name=myconfig --cfg job

# Save composed config to file
uv run python -m buttermilk.runner.cli --config-name=myconfig --cfg job > composed.yaml
```

### 5. Use Local Overrides

Create `conf/local.yaml` for environment-specific settings:

```yaml
# conf/local.yaml
defaults:
  - _self_

# Override for development
llms:
  general: gemini-flash  # Use faster model for testing

logging:
  level: DEBUG

storage:
  type: local
  path: "/tmp/dev-data"
```

## Debugging Configuration Issues

### 1. Enable Debug Output

```bash
export HYDRA_FULL_ERROR=1
uv run python -m buttermilk.runner.cli run=console
```

### 2. Check Composition

```bash
# See what configs are being loaded
uv run python -m buttermilk.runner.cli --config-name=X --cfg hydra

# Show search paths
uv run python -m buttermilk.runner.cli --info searchpath
```

### 3. Validate Interpolations

```python
# In Python
from omegaconf import OmegaConf

try:
    OmegaConf.resolve(cfg)
except Exception as e:
    print(f"Interpolation error: {e}")
```

### 4. Use Structured Configs

```python
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

@dataclass
class RunConfig:
    mode: str
    ui: str
    flows: dict

cs = ConfigStore.instance()
cs.store(name="run_schema", node=RunConfig)
```

## Environment Variables

### Using Environment Variables

```yaml
# Reference environment variables
database:
  host: ${oc.env:DB_HOST}
  port: ${oc.env:DB_PORT,5432}  # Default value
  password: ${oc.env:DB_PASSWORD}

# Boolean environment variables
debug: ${oc.env:DEBUG,false}
```

### Setting Environment Variables

```bash
# Set variables before running
export GCP_PROJECT=my-project
export DEBUG=true
uv run python -m buttermilk.runner.cli run=api
```

## Advanced Configuration Techniques

### Conditional Includes

```yaml
defaults:
  - _self_
  - optional local: local  # Only included if exists
  - optional cloud: ${oc.env:CLOUD_CONFIG}  # From environment
```

### Override Specific Values

```yaml
defaults:
  - base
  # Override specific agent parameters
  - override /agents/researcher:
      parameters:
        max_results: 20
```

### Dynamic Configuration

```python
# In Python code
from hydra import compose, initialize
from omegaconf import OmegaConf

with initialize(config_path="conf"):
    cfg = compose(
        config_name="config",
        overrides=[
            "+flows=[osb]",
            "run.mode=api",
            f"run.name=run_{timestamp}"  # Dynamic value
        ]
    )
```

## Real-World Examples

### Development Environment

```yaml
# conf/envs/development.yaml
defaults:
  - _self_
  - llms: lite
  - storage: local

bm:
  environment: development
  debug: true
  
logging:
  level: DEBUG
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Production Environment

```yaml
# conf/envs/production.yaml
defaults:
  - _self_
  - llms: full
  - storage: bigquery

bm:
  environment: production
  debug: false
  
logging:
  level: INFO
  
security:
  enable_auth: true
  token_expiry: 3600
```

### Testing Configuration

```yaml
# conf/envs/test.yaml
defaults:
  - _self_
  - llms: lite
  - storage: local

bm:
  environment: test
  
data:
  source: inline
  records:
    - id: "test_1"
      text: "Test content"
```

## References

- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
- [Hydra Patterns](https://hydra.cc/docs/patterns/)
- [Buttermilk Flow Examples](../getting-started/first-flow.md)
- [CLI Reference](cli-reference.md)