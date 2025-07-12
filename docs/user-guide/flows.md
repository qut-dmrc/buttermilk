# Running Flows

This guide covers everything you need to know about running Buttermilk flows, from basic execution to advanced deployment scenarios.

## Quick Start

### Example Flows

Buttermilk includes two example flows to demonstrate different configuration approaches:

#### Trans Flow (Hierarchical Composition)
```bash
# Console mode (interactive)
uv run python -m buttermilk.runner.cli run=console flow=trans_clean

# API mode (web interface)
uv run python -m buttermilk.runner.cli run=api flow=trans_clean

# Batch mode (automated processing)
uv run python -m buttermilk.runner.cli run=batch flow=trans_clean
```

#### Tox Flow (All-in-One)
```bash
# Console mode
uv run python -m buttermilk.runner.cli run=console flow=tox_allinone

# API mode
uv run python -m buttermilk.runner.cli run=api flow=tox_allinone

# Batch mode
uv run python -m buttermilk.runner.cli run=batch flow=tox_allinone
```

## Run Modes

### Console Mode (Interactive)

Interactive terminal mode for testing and development:

```bash
# Basic usage
uv run python -m buttermilk.runner.cli run=console flow=<flow_name>

# With custom prompt
uv run python -m buttermilk.runner.cli run=console flow=<flow_name> +prompt="Your custom prompt"

# With specific record
uv run python -m buttermilk.runner.cli run=console flow=<flow_name> +record_id="record_123"
```

**Features:**
- Interactive terminal interface
- Human-in-the-loop processing
- Real-time output and logging
- Good for testing and debugging

### API Mode (Server)

Start a FastAPI server for HTTP access:

```bash
# Start API server
uv run python -m buttermilk.runner.cli run=api

# With specific flows
uv run python -m buttermilk.runner.cli "+flows=[trans,zot,osb]" +run=api llms=full

# Debug mode with enhanced logging
make debug
```

**Server Details:**
- Runs on http://localhost:8000
- API documentation at http://localhost:8000/docs
- WebSocket support for real-time updates

**API Endpoints:**
- `GET /` - API documentation
- `POST /flow/{flow_name}` - Run a specific flow
- `GET /flows` - List available flows
- `WebSocket /ws` - Real-time flow execution

**Example API Request:**
```bash
curl -X POST http://localhost:8000/flow/trans_clean \
  -H "Content-Type: application/json" \
  -d '{
    "flow": "trans_clean",
    "prompt": "Analyze this article about trans issues",
    "text": "Article content here..."
  }'
```

### Batch Mode (Automated)

Process multiple records automatically:

```bash
# Basic batch processing
uv run python -m buttermilk.runner.cli run=batch flow=<flow_name>

# With custom data source
uv run python -m buttermilk.runner.cli run=batch flow=<flow_name> \
  data.source=csv \
  data.path=/path/to/data.csv

# Disable human interaction
uv run python -m buttermilk.runner.cli run=batch flow=<flow_name> run.human_in_loop=false
```

**Features:**
- Automated processing of large datasets
- No user interaction required
- Parallel processing support
- Progress tracking and resumption

### Notebook Mode (Jupyter)

For interactive development and analysis:

```bash
uv run python -m buttermilk.runner.cli run=notebook flow=<flow_name>
```

**Features:**
- Jupyter notebook integration
- Interactive data exploration
- Visualization support
- Step-by-step execution

## Advanced Run Modes

### Pub/Sub Worker

For distributed processing with message queues:

```bash
uv run python -m buttermilk.runner.cli ui=pub/sub
```

**Requirements:**
- Message broker configuration (Google Pub/Sub, RabbitMQ, etc.)
- Configuration in `conf/pubsub/default.yaml`
- Worker process management

### Slackbot Integration

Run Buttermilk as a Slack bot:

```bash
uv run python -m buttermilk.runner.cli ui=slackbot
```

**Requirements:**
- Slack API tokens
- Bot configuration in `conf/slack/default.yaml`
- Slack workspace permissions

## Configuration Approaches

### Hierarchical Composition (Recommended)

Uses Hydra's defaults system for reusable components:

```yaml
# conf/flows/my_flow.yaml
defaults:
  - /agents/llm: gemini_simple
  - /data: local_files
  - /flows/criteria: analysis_criteria
  - _self_

flow:
  name: my_flow
  description: "Custom analysis flow"
```

**Pros:**
- Reusable components
- DRY principle
- Easy to maintain
- Consistent across flows

**Best for:**
- Production flows
- Team environments
- Standardized processes

### All-in-One Configuration

Everything defined in a single file:

```yaml
# conf/flows/simple_flow.yaml
flow:
  name: simple_flow
  description: "Self-contained flow"
  
agents:
  - name: analyzer
    type: LLMAgent
    model: gemini-pro
    system_prompt: "Your analysis prompt here"
    
data:
  source: inline
  records: []
```

**Pros:**
- Self-contained
- Easy to understand
- Good for experiments
- Quick to set up

**Best for:**
- Quick experiments
- One-off flows
- Learning and testing

## Command Line Options

### Basic Structure
```bash
uv run python -m buttermilk.runner.cli [run_mode] [flow] [options]
```

### Common Options
- `run=MODE` - Execution mode (console, api, batch, notebook)
- `flow=NAME` - Flow to run
- `+prompt="text"` - Add a prompt for the flow
- `+record_id="ID"` - Process a specific record
- `run.human_in_loop=false` - Disable human interaction
- `+verbose=true` - Enable verbose output
- `llms=full` - Use full LLM configuration

### Configuration Overrides

Use Hydra's override syntax to customize behavior:

```bash
# Override model
uv run python -m buttermilk.runner.cli run=console flow=trans_clean agents.0.model=gpt-4

# Override data source
uv run python -m buttermilk.runner.cli run=console flow=trans_clean data.source=gsheet

# Multiple overrides
uv run python -m buttermilk.runner.cli run=console flow=trans_clean \
  agents.0.model=gpt-4 \
  data.source=csv \
  data.path=/new/path.csv
```

### Debug and Inspection

```bash
# View resolved configuration
uv run python -m buttermilk.runner.cli run=console flow=trans_clean --cfg job

# Show configuration tree
uv run python -m buttermilk.runner.cli --info searchpath

# Enable full error traces
HYDRA_FULL_ERROR=1 uv run python -m buttermilk.runner.cli run=console flow=trans_clean
```

## Configuration Hierarchy

Understanding the configuration structure:

```
conf/
├── config.yaml              # Main entry point
├── run/                     # Execution modes
│   ├── api.yaml            # API server config
│   ├── console.yaml        # Console mode config
│   ├── batch.yaml          # Batch mode config
│   └── notebook.yaml       # Notebook mode config
├── flows/                   # Flow definitions
│   ├── trans_clean.yaml    # Hierarchical example
│   ├── tox_allinone.yaml   # All-in-one example
│   └── criteria/           # Reusable criteria
├── agents/                  # Reusable agent configs
│   ├── llm/                # LLM agent configs
│   ├── rag/                # RAG agent configs
│   └── host/               # Host agent configs
├── data/                    # Data source configs
│   ├── local_files.yaml    # Local file sources
│   ├── gsheet.yaml         # Google Sheets
│   └── csv.yaml            # CSV files
├── llms/                    # LLM model configs
│   ├── lite.yaml           # Lightweight models
│   └── full.yaml           # Full model suite
└── storage/                 # Storage configurations
    ├── local.yaml          # Local storage
    ├── bigquery.yaml       # BigQuery
    └── gcs.yaml            # Google Cloud Storage
```

## Working with Data

### Data Sources

```bash
# Local CSV file
uv run python -m buttermilk.runner.cli run=console flow=my_flow \
  data.source=csv \
  data.path=/path/to/data.csv

# Google Sheets
uv run python -m buttermilk.runner.cli run=console flow=my_flow \
  data.source=gsheet \
  data.spreadsheet_id=your_sheet_id

# Inline data
uv run python -m buttermilk.runner.cli run=console flow=my_flow \
  data.source=inline \
  +prompt="Custom text to analyze"
```

### Storage Options

```bash
# Local storage
uv run python -m buttermilk.runner.cli run=console flow=my_flow \
  storage.type=local \
  storage.path=/tmp/results

# BigQuery
uv run python -m buttermilk.runner.cli run=console flow=my_flow \
  storage.type=bigquery \
  storage.project=your_project \
  storage.dataset=results
```

## Performance and Scaling

### Parallel Processing

```bash
# Enable parallel processing
uv run python -m buttermilk.runner.cli run=batch flow=my_flow \
  run.parallel=true \
  run.max_workers=4
```

### Resource Management

```bash
# Limit memory usage
uv run python -m buttermilk.runner.cli run=batch flow=my_flow \
  run.memory_limit=8GB

# Set timeout
uv run python -m buttermilk.runner.cli run=batch flow=my_flow \
  run.timeout=3600
```

## Troubleshooting

### Common Issues

**Flow not found:**
```bash
# List available flows
uv run python -m buttermilk.runner.cli --info searchpath
```

**Configuration errors:**
```bash
# Show resolved configuration
uv run python -m buttermilk.runner.cli run=console flow=my_flow --cfg job
```

**Server won't start:**
```bash
# Check for existing processes
ps aux | grep buttermilk
pkill -f buttermilk.runner.cli
```

**Authentication errors:**
```bash
# Re-authenticate
gcloud auth login
gcloud auth application-default login
```

### Debug Mode

Start the API server with enhanced logging:

```bash
# Stop existing server
pkill -f buttermilk.runner.cli

# Start in debug mode
make debug

# Check logs
tail -f /tmp/buttermilk_*.log
```

### Getting Help

- Check the [configuration guide](configuration.md)
- Review [troubleshooting guide](../reference/troubleshooting.md)
- Examine [API reference](api-reference.md)
- Contact support at [nic@suzor.com](mailto:nic@suzor.com)

## Best Practices

1. **Start Simple**: Begin with console mode for testing
2. **Use Hierarchical Config**: For production and team environments
3. **Test Incrementally**: Validate each component before combining
4. **Monitor Performance**: Watch resource usage in batch mode
5. **Document Flows**: Add clear descriptions and examples
6. **Version Control**: Track changes to flow configurations
7. **Error Handling**: Plan for failures and recovery
8. **Security**: Use proper authentication and access controls

## Next Steps

- Learn about [configuration management](configuration.md)
- Explore [API integration](api-reference.md)
- Understand [CLI commands](cli-reference.md)
- Study [flow architecture](../developer-guide/architecture.md)