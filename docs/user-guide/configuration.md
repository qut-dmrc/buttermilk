# Configuration Guide

This guide explains how to configure Buttermilk using Hydra, focusing on practical usage for HASS researchers.

> **💡 For LLM Developers**: See [docs/bots/config.md](../bots/config.md) for detailed configuration internals and debugging.

## Quick Start

### Running with Configuration
```bash
# View current configuration
uv run python -m buttermilk.runner.cli -c job

# Run with specific flow
uv run python -m buttermilk.runner.cli +flow=osb

# Override values
uv run python -m buttermilk.runner.cli +flow=osb llms=full

# Multiple flows
uv run python -m buttermilk.runner.cli "+flows=[osb,trans]"
```

### Basic Configuration Structure
```
conf/
├── config.yaml          # Base configuration
├── local.yaml          # Local overrides (gitignored)
├── flows/              # Flow definitions (osb.yaml, trans.yaml)
├── agents/             # Agent configurations
├── llms/               # Model configurations (lite.yaml, full.yaml)
├── run/                # Execution modes (api.yaml, console.yaml)
└── storage/            # Storage backends (local.yaml, bigquery.yaml)
```

## Flow Configuration

### Selecting Flows
```bash
# Single flow
uv run python -m buttermilk.runner.cli +flow=trans

# Multiple flows
uv run python -m buttermilk.runner.cli "+flows=[trans,osb,zot]"

# Flow with custom settings
uv run python -m buttermilk.runner.cli +flow=osb llms=full storage=bigquery
```

### Understanding Flow Files
Flow files define complete research pipelines:

```yaml
# conf/flows/example.yaml
defaults:
  - _self_
  - /agents@agents:
    - researcher    # Load researcher agent
    - analyst      # Load analyst agent

example_flow:
  name: "Example Analysis"
  orchestrator: buttermilk.orchestrators.groupchat.AutogenOrchestrator
  agents: ${agents}  # References loaded agents
```

## Model Configuration

### Model Selection
```bash
# Use lightweight models (faster, cheaper)
uv run python -m buttermilk.runner.cli llms=lite

# Use full model suite (more capable)
uv run python -m buttermilk.runner.cli llms=full
```

### Model Files
- **`conf/llms/lite.yaml`**: Gemini Flash for quick testing
- **`conf/llms/full.yaml`**: Complete model suite (Gemini Pro, Claude, GPT)

> **📝 Note**: Model connections are configured in `models.json`, stored as a GCP secret.

## Storage Configuration

### Local Development
```bash
# Use local storage (default)
uv run python -m buttermilk.runner.cli storage=local
```

### Cloud Storage
```bash
# Use BigQuery for data
uv run python -m buttermilk.runner.cli storage=bigquery

# Use Google Cloud Storage
uv run python -m buttermilk.runner.cli storage=gcs
```

## Common Patterns

### Development Setup
```bash
# Local development with fast models
uv run python -m buttermilk.runner.cli run=console llms=lite storage=local +flow=trans
```

### Production Setup
```bash
# API server with full capabilities
uv run python -m buttermilk.runner.cli run=api llms=full storage=bigquery "+flows=[osb,trans,zot]"
```

### Experimentation
```bash
# Try different model configurations
uv run python -m buttermilk.runner.cli +flow=trans llms.general=claude-3-5-sonnet
```

## Environment-Specific Configuration

### Local Overrides
Create `conf/local.yaml` for your environment:

```yaml
# conf/local.yaml (gitignored)
defaults:
  - _self_

# Fast models for local testing
llms:
  general: gemini-flash

# Local storage
storage:
  type: local
  path: "/tmp/buttermilk-dev"

# Debug logging
logging:
  level: DEBUG
```

### Environment Variables
```bash
# Required for cloud features
export GOOGLE_CLOUD_PROJECT=your-project
export OPENAI_API_KEY=your-key
export ANTHROPIC_API_KEY=your-key

# Optional
export BUTTERMILK_LOG_LEVEL=DEBUG
```

## Troubleshooting

### Common Issues

#### "Could not override 'X'"
Use `+` prefix for new keys:
```bash
# Wrong
uv run python -m buttermilk.runner.cli flows=osb

# Correct
uv run python -m buttermilk.runner.cli +flows=osb
```

#### Configuration Not Loading
Check the configuration is valid:
```bash
# View composed configuration
uv run python -m buttermilk.runner.cli -c job

# Enable detailed errors
export HYDRA_FULL_ERROR=1
```

#### Missing Models
Ensure you're authenticated:
```bash
gcloud auth login
gcloud auth application-default login
```

## Next Steps

- **First Flow**: See [getting-started/first-flow.md](../getting-started/first-flow.md)
- **Running Flows**: See [user-guide/flows.md](flows.md)
- **Creating Agents**: See [developer-guide/creating-agents.md](../developer-guide/creating-agents.md)

## Advanced Configuration

For detailed configuration internals, debugging, and LLM development patterns, see:
- **[docs/bots/config.md](../bots/config.md)** - Complete configuration reference
- **[docs/bots/debugging.md](../bots/debugging.md)** - Configuration debugging
- **[developer-guide/contributing.md](../developer-guide/contributing.md)** - Development guidelines