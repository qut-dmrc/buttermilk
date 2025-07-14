# Quick Start Guide

Get up and running with Buttermilk in 5 minutes! This guide assumes you've already completed the [installation](installation.md).

## ✅ Two Working Examples

Buttermilk provides two example flows to demonstrate different configuration approaches:

### 1. Trans Flow (Hierarchical Composition)

This flow demonstrates modular configuration using Hydra's composition system:

```bash
# Console mode (interactive)
uv run python -m buttermilk.runner.cli run=console +flow=trans_clean

# With a custom prompt
uv run python -m buttermilk.runner.cli run=console +flow=trans_clean +prompt="Analyze this article about trans issues"
```

### 2. Tox Flow (All-in-One)

This flow shows everything defined in a single configuration file:

```bash
# Console mode
uv run python -m buttermilk.runner.cli run=console +flow=tox_allinone

# With a custom prompt
uv run python -m buttermilk.runner.cli run=console +flow=tox_allinone +prompt="Check this content for toxicity"
```

## 🚀 Start the API Server

Launch the API server to access flows via HTTP:

```bash
# Start API server (runs on http://localhost:8000)
uv run python -m buttermilk.runner.cli run=api

# Or with specific flows
uv run python -m buttermilk.runner.cli "+flows=[trans,zot,osb]" +run=api llms=full
```

### Using the API

Once the server is running, you can:

1. **View API documentation**: Visit http://localhost:8000/docs
2. **Make HTTP requests**:

```bash
# Example API request
curl -X POST http://localhost:8000/flow/trans_clean \
  -H "Content-Type: application/json" \
  -d '{
    "flow": "trans_clean",
    "prompt": "Analyze this article",
    "text": "Your content here..."
  }'
```

3. **Use the web interface**: Navigate to http://localhost:8000 in your browser

## 🔧 Understanding the Configuration

### Hierarchical Composition (trans_clean)

Configuration file: `conf/flows/trans_clean.yaml`

```yaml
# Uses Hydra's defaults to compose from multiple sources
defaults:
  - /agents/rag: simple_rag
  - /data: local_files
  - /flows/criteria: trans_analysis

# Flow-specific settings
flow:
  name: trans_clean
  description: "Analysis of trans-related content"
```

**Pros:**
- Reusable components
- DRY principle
- Easy to maintain
- Consistent across flows

### All-in-One (tox_allinone)

Configuration file: `conf/flows/tox_allinone.yaml`

```yaml
# Everything defined in one place
flow:
  name: tox_allinone
  description: "Toxicity detection flow"
  
agents:
  - name: toxicity_detector
    type: LLMAgent
    model: gpt-4
    prompt: "Check this content for toxicity..."
    
data:
  source: inline
  records: []
```

**Pros:**
- Self-contained
- Easy to understand
- Good for experiments
- Quick to set up

## 📚 Key Concepts

Understanding these concepts will help you work with Buttermilk:

- **Flows**: Complete processing pipelines
- **Agents**: Components that perform specific tasks
- **Records**: Individual pieces of data being processed
- **Jobs**: Units of work that process one record
- **Orchestrators**: Coordinate the execution of flows

For detailed explanations, see [Core Concepts](../reference/concepts.md).

## 🎯 Common Run Modes

### Console Mode (Interactive)

```bash
uv run python -m buttermilk.runner.cli run=console flow=trans_clean
```

- Interactive terminal interface
- Good for testing and debugging
- Allows human-in-the-loop processing

### API Mode (Server)

```bash
uv run python -m buttermilk.runner.cli run=api flow=trans_clean
```

- Starts FastAPI server
- HTTP endpoints for all flows
- Web interface included

### Batch Mode (Automated)

```bash
uv run python -m buttermilk.runner.cli run=batch flow=trans_clean
```

- Automated processing
- Good for large datasets
- No user interaction required

## 🔧 Command Line Options

### Basic Structure

```bash
uv run python -m buttermilk.runner.cli [run_mode] [flow] [options]
```

### Common Options

- `run=MODE` - Execution mode (console, api, batch)
- `flow=NAME` - Flow to run
- `+prompt="text"` - Add a prompt
- `+record_id="id"` - Process specific record
- `run.human_in_loop=false` - Disable user interaction
- `+verbose=true` - Enable verbose output

### Examples

```bash
# Run with specific model
uv run python -m buttermilk.runner.cli run=console flow=trans_clean llms=full

# Disable human interaction
uv run python -m buttermilk.runner.cli run=console flow=trans_clean run.human_in_loop=false

# Add custom prompt
uv run python -m buttermilk.runner.cli run=console flow=trans_clean +prompt="Custom analysis request"
```

## 🐛 Debug Mode

Start the API server in debug mode with enhanced logging:

```bash
# Stop any existing server
pkill -f buttermilk.runner.cli

# Start in debug mode
make debug

# Check logs
tail -f /tmp/buttermilk_*.log
```

## 📖 Next Steps

Now that you have Buttermilk running:

1. **Learn the basics**: [Your First Flow](first-flow.md)
2. **Explore configuration**: [Configuration Guide](../user-guide/configuration.md)
3. **Try the API**: [API Reference](../user-guide/api-reference.md)
4. **Build custom flows**: [Creating Agents](../developer-guide/creating-agents.md)

## 🆘 Troubleshooting

### Flow not found?
```bash
# List available flows
uv run python -m buttermilk.runner.cli --info searchpath
```

### Configuration errors?
```bash
# Show resolved configuration
uv run python -m buttermilk.runner.cli run=console flow=trans_clean --cfg job
```

### Server won't start?
```bash
# Check for existing processes
ps aux | grep buttermilk
pkill -f buttermilk.runner.cli
```

### Need more help?
- Check the [full troubleshooting guide](../reference/troubleshooting.md)
- Review [command reference](../user-guide/cli-reference.md)
- Contact support at [nic@suzor.com](mailto:nic@suzor.com)