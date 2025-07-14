# CLI Reference

Complete command-line interface reference for Buttermilk, covering all commands, options, and usage patterns.

## Basic Usage

```bash
uv run python -m buttermilk.runner.cli [OPTIONS] [HYDRA_OPTIONS]
```

## Command Structure

The CLI follows Hydra's configuration system:

```bash
uv run python -m buttermilk.runner.cli [run_mode] [config_groups] [overrides]
```

### Components:
- **run_mode**: How to execute (console, api, batch, notebook)
- **config_groups**: Configuration selections (flows, llms, etc.)
- **overrides**: Runtime parameter modifications

## Run Modes

### Console Mode
Interactive terminal execution:

```bash
# Basic console mode
uv run python -m buttermilk.runner.cli run=console

# With specific flow
uv run python -m buttermilk.runner.cli run=console flow=trans_clean

# With custom prompt
uv run python -m buttermilk.runner.cli run=console flow=trans_clean +prompt="Custom analysis"
```

**Options:**
- `run.human_in_loop=true|false` - Enable/disable human interaction
- `run.timeout=SECONDS` - Set execution timeout
- `run.debug=true|false` - Enable debug mode

### API Mode
Start FastAPI server:

```bash
# Basic API server
uv run python -m buttermilk.runner.cli run=api

# With specific flows
uv run python -m buttermilk.runner.cli "+flows=[trans,tox,osb]" run=api

# With full LLM configuration
uv run python -m buttermilk.runner.cli run=api llms=full

# Custom port
uv run python -m buttermilk.runner.cli run=api server.port=8080
```

**Options:**
- `server.host=HOST` - Server host (default: 0.0.0.0)
- `server.port=PORT` - Server port (default: 8000)
- `server.reload=true|false` - Enable auto-reload for development

### Batch Mode
Automated processing of multiple records:

```bash
# Basic batch processing
uv run python -m buttermilk.runner.cli run=batch flow=trans_clean

# With custom data source
uv run python -m buttermilk.runner.cli run=batch flow=trans_clean \
  data.source=csv \
  data.path=/path/to/data.csv

# Parallel processing
uv run python -m buttermilk.runner.cli run=batch flow=trans_clean \
  run.parallel=true \
  run.max_workers=4
```

**Options:**
- `run.parallel=true|false` - Enable parallel processing
- `run.max_workers=N` - Maximum worker threads
- `run.batch_size=N` - Records per batch
- `run.resume=true|false` - Resume from checkpoint

### Notebook Mode
Jupyter notebook integration:

```bash
# Start notebook mode
uv run python -m buttermilk.runner.cli run=notebook flow=trans_clean

# With specific kernel
uv run python -m buttermilk.runner.cli run=notebook flow=trans_clean \
  notebook.kernel=python3
```

## Configuration Groups

### Flow Selection

```bash
# Single flow
uv run python -m buttermilk.runner.cli +flow=trans_clean

# Multiple flows
uv run python -m buttermilk.runner.cli "+flows=[trans,tox,osb]"

# Flow with parameters
uv run python -m buttermilk.runner.cli flow=trans_clean \
  flow.timeout=60 \
  flow.max_retries=3
```

### LLM Configuration

```bash
# Lightweight models
uv run python -m buttermilk.runner.cli llms=lite

# Full model suite
uv run python -m buttermilk.runner.cli llms=full

# Custom model
uv run python -m buttermilk.runner.cli llms=custom \
  llms.general=gpt-4 \
  llms.fast=gemini-flash
```

### Data Sources

```bash
# Local CSV file
uv run python -m buttermilk.runner.cli data=csv \
  data.path=/path/to/data.csv

# Google Sheets
uv run python -m buttermilk.runner.cli data=gsheet \
  data.spreadsheet_id=SHEET_ID

# Inline data
uv run python -m buttermilk.runner.cli data=inline \
  +prompt="Text to analyze"
```

### Storage Configuration

```bash
# Local storage
uv run python -m buttermilk.runner.cli storage=local \
  storage.path=/tmp/results

# BigQuery
uv run python -m buttermilk.runner.cli storage=bigquery \
  storage.project=PROJECT_ID \
  storage.dataset=DATASET

# Google Cloud Storage
uv run python -m buttermilk.runner.cli storage=gcs \
  storage.bucket=BUCKET_NAME
```

## Common Options

### Input Parameters

```bash
# Custom prompt
uv run python -m buttermilk.runner.cli +prompt="Analyze this content"

# Specific record
uv run python -m buttermilk.runner.cli +record_id="record_123"

# Input text
uv run python -m buttermilk.runner.cli +text="Content to analyze"

# Input URL
uv run python -m buttermilk.runner.cli +uri="https://example.com/article"
```

### Output Control

```bash
# Verbose output
uv run python -m buttermilk.runner.cli +verbose=true

# Quiet mode
uv run python -m buttermilk.runner.cli +quiet=true

# Custom output format
uv run python -m buttermilk.runner.cli output.format=json

# Save results
uv run python -m buttermilk.runner.cli output.save=true \
  output.path=/path/to/results
```

### Debugging Options

```bash
# Enable debug mode
uv run python -m buttermilk.runner.cli +debug=true

# Full error traces
HYDRA_FULL_ERROR=1 uv run python -m buttermilk.runner.cli [options]

# Enable profiling
uv run python -m buttermilk.runner.cli +profile=true
```

## Configuration Inspection

### View Configuration

```bash
# Show resolved configuration
uv run python -m buttermilk.runner.cli --cfg job

# Show Hydra configuration
uv run python -m buttermilk.runner.cli --cfg hydra

# Show all configuration
uv run python -m buttermilk.runner.cli --cfg all
```

### List Available Options

```bash
# Show configuration groups
uv run python -m buttermilk.runner.cli --info searchpath

# Show available flows
uv run python -m buttermilk.runner.cli --info flows

# Show available models
uv run python -m buttermilk.runner.cli --info llms
```

## Override Syntax

### Basic Overrides

```bash
# Simple value override
uv run python -m buttermilk.runner.cli agents.0.model=gpt-4

# Nested value override
uv run python -m buttermilk.runner.cli run.server.port=8080

# Boolean override
uv run python -m buttermilk.runner.cli run.debug=true
```

### Add New Values

```bash
# Add new top-level value
uv run python -m buttermilk.runner.cli +new_param=value

# Add new nested value
uv run python -m buttermilk.runner.cli +custom.setting=value
```

### List Overrides

```bash
# Override list items
uv run python -m buttermilk.runner.cli "agents=[agent1,agent2]"

# Add to existing list
uv run python -m buttermilk.runner.cli "+agents=[new_agent]"
```

### Complex Overrides

```bash
# JSON-style override
uv run python -m buttermilk.runner.cli \
  'agents.0={name: custom_agent, model: gpt-4}'

# Multiple related overrides
uv run python -m buttermilk.runner.cli \
  agents.0.model=gpt-4 \
  agents.0.temperature=0.7 \
  agents.0.max_tokens=2000
```

## Environment Variables

### Configuration

```bash
# Set environment variables
export GOOGLE_CLOUD_PROJECT=your-project
export BUTTERMILK_LOG_LEVEL=DEBUG

# Use in configuration
uv run python -m buttermilk.runner.cli \
  storage.project='${oc.env:GOOGLE_CLOUD_PROJECT}'
```

### Debug Options

```bash
# Enable full error output
export HYDRA_FULL_ERROR=1

# Set log level
export BUTTERMILK_LOG_LEVEL=DEBUG

# Enable profiling
export BUTTERMILK_PROFILE=1
```

## Advanced Usage

### Multi-Stage Processing

```bash
# Stage 1: Data preparation
uv run python -m buttermilk.runner.cli run=batch flow=prepare_data \
  data.source=csv \
  data.path=raw_data.csv \
  output.path=prepared_data.json

# Stage 2: Analysis
uv run python -m buttermilk.runner.cli run=batch flow=analyze_data \
  data.source=json \
  data.path=prepared_data.json \
  output.path=analysis_results.json
```

### Conditional Execution

```bash
# Run only if condition is met
uv run python -m buttermilk.runner.cli run=console flow=trans_clean \
  run.condition="record_count > 0"

# Skip if results exist
uv run python -m buttermilk.runner.cli run=batch flow=trans_clean \
  run.skip_existing=true
```

### Resource Management

```bash
# Limit memory usage
uv run python -m buttermilk.runner.cli run=batch flow=trans_clean \
  run.memory_limit=8GB

# Set timeout
uv run python -m buttermilk.runner.cli run=batch flow=trans_clean \
  run.timeout=3600

# Control concurrency
uv run python -m buttermilk.runner.cli run=batch flow=trans_clean \
  run.max_concurrent=2
```

## Integration Commands

### Docker Integration

```bash
# Run in Docker
docker run -it buttermilk:latest \
  uv run python -m buttermilk.runner.cli run=api

# Mount configuration
docker run -it -v $(pwd)/conf:/app/conf buttermilk:latest \
  uv run python -m buttermilk.runner.cli run=api
```

### Cloud Integration

```bash
# Google Cloud Run
gcloud run deploy buttermilk-api \
  --image=buttermilk:latest \
  --platform=managed \
  --region=us-central1 \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID}"
```

## Makefile Commands

Buttermilk includes a Makefile with common commands:

```bash
# View default configuration
make config

# Start API server
make api

# Start in debug mode
make debug

# Run tests
make test

# Format code
make format

# Lint code
make lint

# Run with coverage
make coverage
```

## Examples by Use Case

### Content Analysis

```bash
# Single article analysis
uv run python -m buttermilk.runner.cli run=console flow=trans_clean \
  +prompt="Analyze this article for bias and sentiment" \
  +uri="https://example.com/article"

# Batch analysis of CSV data
uv run python -m buttermilk.runner.cli run=batch flow=trans_clean \
  data.source=csv \
  data.path=articles.csv \
  run.parallel=true \
  output.format=jsonl
```

### Model Comparison

```bash
# Compare different models
uv run python -m buttermilk.runner.cli run=console flow=model_comparison \
  +text="Test content" \
  "models=[gpt-4,claude-3,gemini-pro]"
```

### Quality Assurance

```bash
# Run QA flow with human review
uv run python -m buttermilk.runner.cli run=console flow=qa_review \
  data.source=json \
  data.path=content_to_review.json \
  run.human_in_loop=true
```

### Research Pipeline

```bash
# Multi-step research workflow
uv run python -m buttermilk.runner.cli run=batch flow=research_pipeline \
  data.source=gsheet \
  data.spreadsheet_id=SHEET_ID \
  storage.type=bigquery \
  storage.project=PROJECT_ID \
  run.checkpoint=true
```

## Troubleshooting Commands

### Configuration Issues

```bash
# Check configuration composition
uv run python -m buttermilk.runner.cli --cfg job | less

# Find configuration files
uv run python -m buttermilk.runner.cli --info searchpath

# Validate configuration
uv run python -m buttermilk.runner.cli --cfg validate
```

### Runtime Issues

```bash
# Enable debug logging
uv run python -m buttermilk.runner.cli run=console flow=trans_clean \
  +debug=true \
  logging.level=DEBUG

# Check system resources
uv run python -m buttermilk.runner.cli --info system

# Test connectivity
uv run python -m buttermilk.runner.cli --info health
```

### Performance Issues

```bash
# Enable profiling
uv run python -m buttermilk.runner.cli run=console flow=trans_clean \
  +profile=true \
  +text="Test content"

# Monitor resources
uv run python -m buttermilk.runner.cli run=batch flow=trans_clean \
  +monitor=true \
  run.max_workers=1
```

## Error Handling

### Common Errors

**Flow not found:**
```bash
# Error: Flow 'invalid_flow' not found
# Solution: List available flows
uv run python -m buttermilk.runner.cli --info flows
```

**Configuration error:**
```bash
# Error: Missing required configuration
# Solution: Check configuration
uv run python -m buttermilk.runner.cli --cfg job
```

**Permission error:**
```bash
# Error: Permission denied
# Solution: Check authentication
gcloud auth list
gcloud auth application-default login
```

### Debug Commands

```bash
# Full error traces
HYDRA_FULL_ERROR=1 uv run python -m buttermilk.runner.cli [command]

# Verbose output
uv run python -m buttermilk.runner.cli [command] +verbose=true

# Debug mode
uv run python -m buttermilk.runner.cli [command] +debug=true
```

## Best Practices

### 1. Use Configuration Files

Instead of long command lines, create configuration files:

```yaml
# conf/my_analysis.yaml
defaults:
  - _self_
  - flow: trans_clean
  - data: csv
  - storage: local

data:
  path: /path/to/data.csv

output:
  path: /path/to/results
  format: json
```

```bash
# Use the configuration
uv run python -m buttermilk.runner.cli --config-name=my_analysis
```

### 2. Environment-Specific Configs

```bash
# Development
uv run python -m buttermilk.runner.cli env=dev run=console

# Production
uv run python -m buttermilk.runner.cli env=prod run=api
```

### 3. Script Integration

```bash
#!/bin/bash
# analysis.sh

set -e

echo "Starting analysis..."
uv run python -m buttermilk.runner.cli run=batch flow=trans_clean \
  data.source=csv \
  data.path="$1" \
  output.path="$2" \
  run.parallel=true

echo "Analysis complete!"
```

### 4. Logging Configuration

```bash
# Structured logging
uv run python -m buttermilk.runner.cli run=batch flow=trans_clean \
  logging.format=json \
  logging.level=INFO \
  logging.file=/path/to/logs/analysis.log
```

## Related Documentation

- [Configuration Guide](configuration.md) - Detailed configuration options
- [API Reference](api-reference.md) - HTTP API endpoints
- [Flow Guide](flows.md) - Flow execution patterns
- [Troubleshooting](../reference/troubleshooting.md) - Common issues and solutions