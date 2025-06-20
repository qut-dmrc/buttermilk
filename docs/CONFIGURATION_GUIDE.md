# Buttermilk Configuration Guide

This guide explains how to configure and run Buttermilk flows using Hydra configuration management.

## Table of Contents
- [Understanding Hydra Configuration](#understanding-hydra-configuration)
- [Configuration Structure](#configuration-structure)
- [Running Flows](#running-flows)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [OSB Flow Example](#osb-flow-example)

## Understanding Hydra Configuration

Buttermilk uses [Hydra](https://hydra.cc/) for configuration management. This provides:
- Hierarchical configuration composition
- Command-line overrides
- Configuration interpolation
- Type safety with structured configs

### Key Concepts

1. **Defaults Lists**: Configurations are composed from multiple files
2. **Interpolation**: Values can reference other config values using `${path.to.value}`
3. **Overrides**: Command-line arguments can override any config value
4. **Package Directives**: Control where configs are placed in the final structure

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
│   └── host/           # Host agents
├── llms/               # LLM configurations
│   ├── lite.yaml       # Lightweight models
│   └── full.yaml       # Full model suite
├── run/                # Run mode configurations
│   ├── api.yaml        # API server mode
│   ├── console.yaml    # Console mode
│   └── batch.yaml      # Batch processing mode
└── storage/            # Storage configurations
```

## Running Flows

### Method 1: Create a Custom Config File (Recommended)

Create a config file that combines all necessary components:

```yaml
# conf/my_flow_config.yaml
defaults:
  - _self_
  - local                # Environment settings
  - flows: 
    - osb               # Include OSB flow
  - llms: lite          # Use lite LLM config
  - run: api           # Run in API mode

bm:
  name: my_api
  job: flows

run:
  mode: api
  ui: web
  human_in_loop: true
  flows: ${flows}
  name: ${bm.name}
  job: ${bm.job}
```

Then run:
```bash
uv run python -m buttermilk.runner.cli --config-name=my_flow_config
```

### Method 2: Command Line Overrides

For simple cases, you can override values directly:

```bash
# Run OSB flow in console mode
uv run python -m buttermilk.runner.cli \
  +flows=[osb] \
  +run.mode=console \
  +run.ui=console \
  flow=osb \
  prompt="Your question here"
```

### Method 3: Using Existing Configs

Some pre-configured setups exist:

```bash
# Run the testing configuration
uv run python -m buttermilk.runner.cli --config-name=testing
```

## Common Issues and Solutions

### Issue 1: "Could not override 'run'. No match in the defaults list"

**Problem**: The `run` configuration group isn't included in the defaults.

**Solution**: Use `+run=api` to add it, or create a config file that includes it in defaults.

### Issue 2: "Interpolation key 'llms.general' not found"

**Problem**: Agents reference `${llms.general}` but the LLM config doesn't define it.

**Solution**: Ensure your LLM config (e.g., `lite.yaml`) defines the `general` key:
```yaml
general:
  - gemini2flashlite
  - gpt41mini
```

### Issue 3: "0 agents and 0 observers"

**Problem**: Flow configuration isn't loading agent definitions.

**Solution**: Ensure the flow's defaults section properly references agents:
```yaml
defaults:
  - _self_
  - /agents/rag@osb.agents: 
    - researcher
    - policy_analyst
```

### Issue 4: API Server Won't Start

**Problem**: Missing `_background_init` method or other initialization issues.

**Solution**: Ensure BM is instantiated correctly. The API expects certain methods that may not exist when created via Hydra's instantiate.

## OSB Flow Example

### Running OSB in Console Mode

```bash
# Create a console config for OSB
cat > conf/osb_console.yaml << 'EOF'
defaults:
  - _self_
  - local
  - flows: 
    - osb
  - llms: lite

bm:
  name: osb_console
  job: flows

run:
  mode: console
  ui: console
  human_in_loop: true
  flows: ${flows}
  name: ${bm.name}
  job: ${bm.job}
EOF

# Run it
uv run python -m buttermilk.runner.cli \
  --config-name=osb_console \
  flow=osb \
  prompt="What is the definition of hate speech for meta?"
```

### Running OSB API Server

```bash
# The osb_api.yaml config we created earlier
uv run python -m buttermilk.runner.cli --config-name=osb_api
```

### What Happens When OSB Runs

1. **Flow Loading**: The OSB flow configuration is loaded from `conf/flows/osb.yaml`
2. **Agent Creation**: 
   - 4 RAG agents are created (researcher, policy_analyst, fact_checker, explorer)
   - 1 HOST observer is created
3. **Agent Announcements**: Each agent announces itself with capabilities
4. **Orchestration**: AutogenOrchestrator manages the group chat
5. **Vector Store**: Agents access the OSB vector store for document retrieval

## Best Practices

1. **Create Named Configs**: For production use, create named configuration files rather than using command-line overrides
2. **Use Interpolation**: Reference other config values to keep things DRY
3. **Test Incrementally**: Start with console mode before moving to API mode
4. **Check Logs**: Enable verbose logging to debug configuration issues
5. **Validate Configs**: Use `--cfg job` to see the composed configuration without running

## Configuration Reference

### BM Configuration
```yaml
bm:
  name: str              # Instance name
  job: str               # Job type
  platform: str          # Platform (local, gcp, etc.)
  tracing: dict          # Tracing configuration
  clouds: list           # Cloud provider configs
  secret_provider: dict  # Secret management
  logger_cfg: dict       # Logging configuration
```

### Run Configuration
```yaml
run:
  mode: str              # console, api, batch
  ui: str                # console, web
  human_in_loop: bool    # Enable human interaction
  flows: dict            # Flow configurations
  name: str              # Run name
  job: str               # Job name
```

### Flow Configuration
```yaml
flowname:
  orchestrator: str      # Orchestrator class path
  name: str              # Display name
  description: str       # Flow description
  parameters: dict       # Flow-specific parameters
  agents: dict           # Agent configurations
  observers: dict        # Observer configurations
```

## Debugging Tips

1. **View Composed Config**: Add `--cfg job` to see the final configuration
2. **Check Hydra Output**: Look for "Hydra config" in logs
3. **Validate Interpolations**: Ensure all `${...}` references resolve
4. **Test Minimal Config**: Start with the simplest possible configuration
5. **Use HYDRA_FULL_ERROR=1**: Get complete stack traces for configuration errors

## Additional Resources

- [Hydra Documentation](https://hydra.cc/)
- [Buttermilk Flow README](../conf/README_FLOWS.md)
- [Agent Configuration Guide](AGENT_CONFIGURATION.md)
- [OSB Flow Analysis](../OSB_FLOW_ANALYSIS.md)