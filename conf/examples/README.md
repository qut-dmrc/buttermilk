# Example Configuration Files

This directory contains example configuration files for common Buttermilk use cases.

## Files

### osb_api.yaml
Configuration for running the OSB flow with API server mode. Includes:
- OSB flow with all RAG agents
- Lite LLM configuration for cost-effective operation
- Web UI enabled with human-in-the-loop

### osb_console.yaml  
Configuration for running the OSB flow in console/terminal mode. Includes:
- Same OSB flow configuration
- Console UI for terminal interaction
- Human-in-the-loop enabled for interactive sessions

### multi_flow_api.yaml
Configuration for running multiple flows in API mode. Includes:
- OSB, Toxicity, and Trans flows
- Full LLM suite for maximum capability
- API server configuration

### batch_processing.yaml
Configuration for batch processing mode. Includes:
- Batch mode settings
- No human-in-the-loop (automated)
- Optimized for throughput

## Usage

Copy any example to the main conf directory and customize:

```bash
# Copy example
cp conf/examples/osb_api.yaml conf/my_config.yaml

# Edit as needed
vim conf/my_config.yaml

# Run with your config
uv run python -m buttermilk.runner.cli --config-name=my_config
```

## Creating Your Own

Use these examples as templates. Key sections to customize:

1. **defaults**: Which components to include
2. **bm**: Buttermilk instance configuration  
3. **run**: Execution mode and UI settings
4. **flows**: Which flows to make available

See [Configuration Guide](../../docs/CONFIGURATION_GUIDE.md) for detailed documentation.