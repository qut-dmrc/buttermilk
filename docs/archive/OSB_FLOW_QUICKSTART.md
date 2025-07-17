# OSB Flow Quick Start Guide

This guide provides quick instructions for running the OSB (Oversight Board) flow in different modes.

## Prerequisites

- Buttermilk environment set up
- Access to GCP credentials (for vector store access)
- Python environment with dependencies installed

## Quick Commands

### Console Mode (Interactive Terminal)

```bash
# Quick run with defaults
uv run python -m buttermilk.runner.cli \
  --config-name=osb_api \
  run.mode=console \
  run.ui=console \
  flow=osb \
  prompt="What is Meta's hate speech definition?"
```

### API Mode (Web Interface)

```bash
# Start API server with OSB flow
uv run python -m buttermilk.runner.cli --config-name=osb_api

# Server will be available at http://localhost:8000
```

### Batch Mode (Automated Processing)

```bash
# Process multiple queries
uv run python -m buttermilk.runner.cli \
  --config-name=osb_api \
  run.mode=batch \
  flow=osb
```

## Configuration File

The OSB flow requires a configuration file. Here's a minimal example:

```yaml
# Save as conf/osb_quickstart.yaml
defaults:
  - _self_
  - local
  - flows: [osb]
  - llms: lite

bm:
  name: osb_quickstart
  job: flows

run:
  mode: console  # Change to 'api' for web interface
  ui: console    # Change to 'web' for API mode
  human_in_loop: true
  flows: ${flows}
  name: ${bm.name}
  job: ${bm.job}
```

## What the OSB Flow Does

The OSB flow provides multi-agent analysis of Meta's Oversight Board decisions:

1. **ENHANCED_RESEARCHER**: Searches for relevant policies and cases
2. **POLICY_ANALYST**: Analyzes policy implications and precedents  
3. **FACT_CHECKER**: Verifies claims against official documents
4. **RESEARCH_EXPLORER**: Discovers related themes and connections
5. **HOST**: Coordinates the agents and manages workflow

## Example Queries

```bash
# Meta's hate speech definition
prompt="What is the definition of hate speech according to Meta?"

# Specific case analysis
prompt="How does Meta handle COVID-19 misinformation?"

# Policy comparison
prompt="Compare Meta's approach to political speech across different regions"

# Oversight Board decisions
prompt="What are the key principles in Oversight Board case 2021-001?"
```

## Troubleshooting

### "No agents loaded"
- Ensure the OSB flow configuration includes agent definitions
- Check that `conf/flows/osb.yaml` exists and is properly formatted

### "Vector store connection failed"
- Verify GCP credentials are configured
- Check that the OSB vector store is accessible
- Ensure `data: ${...parameters.data}` is properly set in agent configs

### "LLM configuration error"
- Verify `conf/llms/lite.yaml` exists and defines required models
- Check that API keys are configured in secrets

## Advanced Usage

### Custom Agent Selection

Modify which agents participate:

```yaml
# In your config file, override the agents
flows:
  osb:
    agents:
      researcher: ${agents.rag.researcher}
      # Exclude other agents for faster responses
```

### Adjusting Search Parameters

Configure agent search behavior:

```yaml
flows:
  osb:
    agents:
      researcher:
        parameters:
          max_results_per_strategy: 10  # More results
          confidence_threshold: 0.7      # Higher quality threshold
```

### Using Different LLMs

Switch to more powerful models:

```bash
uv run python -m buttermilk.runner.cli \
  --config-name=osb_api \
  llms=full \  # Use full model suite
  flow=osb
```

## Next Steps

- Read the [full configuration guide](CONFIGURATION_GUIDE.md)
- Explore [agent customization](AGENT_CONFIGURATION.md)
- Learn about [flow development](../conf/README_FLOWS.md)
- Check [OSB flow implementation details](../OSB_FLOW_ANALYSIS.md)