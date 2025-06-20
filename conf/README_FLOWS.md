# How to Run Buttermilk Flows

## Quick Start

### 1. Running the Trans Flow (Hierarchical Composition)

```bash
# Console mode (interactive)
uv run python -m buttermilk.runner.cli run=console flow=trans_clean

# API mode (web interface)
uv run python -m buttermilk.runner.cli run=api flow=trans_clean

# Batch mode (automated processing)
uv run python -m buttermilk.runner.cli run=batch flow=trans_clean
```

### 2. Running the Tox Flow (All-in-One)

```bash
# Console mode
uv run python -m buttermilk.runner.cli run=console flow=tox_allinone

# API mode
uv run python -m buttermilk.runner.cli run=api flow=tox_allinone

# Batch mode
uv run python -m buttermilk.runner.cli run=batch flow=tox_allinone
```

## Understanding the Two Approaches

### Hierarchical Composition (trans_clean.yaml)

- **Pros**: Reusable components, DRY principle, easy to maintain
- **How it works**: Uses Hydra's `defaults` to pull in configurations from:
  - `/agents/` - Reusable agent definitions
  - `/data/` - Data source configurations
  - `/flows/criteria/` - Criteria definitions
- **Best for**: Production flows where you want consistency

### All-in-One (tox_allinone.yaml)

- **Pros**: Everything in one place, self-contained, easy to understand
- **How it works**: All configuration defined directly in the file
- **Best for**: Quick experiments, custom one-off flows

## Running the API Server

### Start the API server:

```bash
# Create a configuration file first (see docs/CONFIGURATION_GUIDE.md)
# Then run:
uv run python -m buttermilk.runner.cli --config-name=your_api_config

# Or use the example OSB API config:
uv run python -m buttermilk.runner.cli --config-name=osb_api
```

This will:
1. Start a FastAPI server on http://localhost:8000
2. Load all configured flows
3. Provide endpoints for running flows via HTTP

### API Endpoints:

- `GET /` - API documentation  
- `POST /flow/{flow_name}` - Run a specific flow
- `GET /flows` - List available flows
- `WebSocket /ws` - Real-time flow execution

### Example API Request:

```bash
curl -X POST http://localhost:8000/flow/trans_clean \
  -H "Content-Type: application/json" \
  -d '{
    "flow": "trans_clean",
    "prompt": "Analyze this article about trans issues",
    "text": "Article content here..."
  }'
```

## Command Line Options

### Basic Structure:
```bash
uv run python -m buttermilk.runner.cli [run_mode] [flow] [options]
```

### Run Modes:
- `run=console` - Interactive terminal mode
- `run=api` - Start API server
- `run=batch` - Batch processing mode
- `run=notebook` - Jupyter notebook mode

### Common Options:
- `flow=FLOW_NAME` - Specify which flow to run
- `+prompt="Your prompt"` - Add a prompt for the flow
- `+record_id="ID"` - Process a specific record
- `run.human_in_loop=false` - Disable human interaction

### Examples:

```bash
# Run trans flow with a specific prompt
uv run python -m buttermilk.runner.cli run=console flow=trans_clean +prompt="Analyze this news article"

# Run tox flow in batch mode without human interaction
uv run python -m buttermilk.runner.cli run=batch flow=tox_allinone run.human_in_loop=false

# Start API server with specific flow
uv run python -m buttermilk.runner.cli run=api +flow=trans_clean
```

## Configuration Hierarchy

```
conf/
├── config.yaml          # Main entry point
├── run/                 # Execution modes
│   ├── api.yaml        # API server config
│   ├── console.yaml    # Console mode config
│   └── batch.yaml      # Batch mode config
├── flows/              # Flow definitions
│   ├── trans_clean.yaml    # Hierarchical example
│   └── tox_allinone.yaml   # All-in-one example
├── agents/             # Reusable agent configs
├── data/               # Data source configs
└── llms/               # LLM model configs
```

## Troubleshooting

### Flow not found?
```bash
# List all available flows
uv run python -m buttermilk.runner.cli --info searchpath
```

### Configuration errors?
```bash
# Show resolved configuration
uv run python -m buttermilk.runner.cli run=console flow=trans_clean --cfg job
```

### Need to debug?
```bash
# Enable full error traces
HYDRA_FULL_ERROR=1 uv run python -m buttermilk.runner.cli run=console flow=trans_clean
```