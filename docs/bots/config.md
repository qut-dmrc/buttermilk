# Buttermilk Configuration System

## Overview
Buttermilk uses Hydra with OmegaConf for configuration management. All configuration is done through YAML files - **NEVER** use manual dictionary configuration.

## Core Concepts

### 1. Hydra Basics
- **Composition**: Build configs from multiple files
- **Overrides**: Modify any value from command line
- **Interpolation**: Reference other config values with `${}`
- **Validation**: Early fail with clear error messages

### 2. Configuration Structure
```
conf/
├── config.yaml          # Base configuration
├── local.yaml          # Local overrides (gitignored)
├── agents/             # Agent configurations
├── flows/              # Flow definitions
├── llms/               # Model configurations
├── run/                # Execution modes
└── storage/            # Storage backends
```

### 3. Key Files

#### config.yaml (Base Configuration)
```yaml
defaults:
  - _self_
  - local              # Local environment settings
  - llms: lite        # Default to lightweight models
  - flows: []         # No flows by default
  
# Global settings
bm:
  name: buttermilk
  platform: local
  debug: false
```

#### local.yaml (Environment Overrides)
```yaml
# Local development settings
defaults:
  - _self_

# Override for local development
llms:
  general: gemini-flash  # Use faster model locally

storage:
  type: local
  path: "/tmp/buttermilk-data"
```

## Command Line Usage

### Basic Commands
```bash
# View configuration
uv run python -m buttermilk.runner.cli -c job

# Run with specific flow
uv run python -m buttermilk.runner.cli +flow=osb

# Override values
uv run python -m buttermilk.runner.cli +flow=osb llms=full

# Multiple flows
uv run python -m buttermilk.runner.cli "+flows=[osb,trans]"
```

### Override Syntax
- **Replace**: `key=value` (must exist in defaults)
- **Add**: `+key=value` (new key)
- **Remove**: `~key` (remove key)
- **Nested**: `parent.child=value`

## Configuration Patterns

### 1. Flow Configuration
```yaml
# conf/flows/example_flow.yaml
defaults:
  - _self_
  # Load agents with package directive
  - /agents@agents:
    - researcher
    - analyst
  # Load host agent
  - /agents/host@observers: llm_host

example_flow:
  name: "Example Analysis Flow"
  orchestrator: buttermilk.orchestrators.groupchat.AutogenOrchestrator
  description: "Analyzes content using multiple agents"
  # References loaded agents
  agents: ${agents}
  observers: ${observers}
```

### 2. Agent Configuration
```yaml
# conf/agents/researcher.yaml
researcher:
  role: RESEARCHER
  agent_obj: buttermilk.agents.rag.RAGAgent
  description: "Searches and retrieves relevant information"
  parameters:
    max_results: 10
    search_depth: 3
  # Model reference from llms config
  model: ${llms.general}
  # Input mappings
  inputs:
    query: "${user.question}"
    context: "${flow.context}"
```

### 3. Model Configuration
```yaml
# conf/llms/full.yaml
llms:
  # Named model references
  general: gemini-pro
  fast: gemini-flash
  coding: claude-3-5-sonnet
  
  # Model definitions
  models:
    gemini-pro:
      provider: google
      model: gemini-1.5-pro-002
      temperature: 0.7
      max_tokens: 8192
    
    claude-3-5-sonnet:
      provider: anthropic
      model: claude-3-5-sonnet-20241022
      temperature: 0.5
      max_tokens: 4096
```

### 4. Package Directives
```yaml
defaults:
  # Place at root level
  - run: api
  
  # Place under specific key
  - llms@bm.llms: full
  
  # Multiple items under key
  - /agents@flow.agents:
    - researcher
    - analyst
```

## Interpolation

### Basic Interpolation
```yaml
# Reference other values
server:
  host: localhost
  port: 8000
  url: "http://${server.host}:${server.port}"
```

### Cross-File References
```yaml
# In agent config
agent:
  model: ${llms.general}  # References llms config
  storage: ${storage}     # References storage config
```

### Environment Variables
```yaml
# With defaults
database:
  host: ${oc.env:DB_HOST,localhost}
  port: ${oc.env:DB_PORT,5432}
  
# Required
api_key: ${oc.env:OPENAI_API_KEY}
```

## Advanced Patterns

### 1. Conditional Configuration
```yaml
# Optional includes
defaults:
  - optional local: local  # Only if exists
  - optional override: ${oc.env:CONFIG_OVERRIDE}
```

### 2. List Composition
```yaml
# Base config
agents:
  - name: base_agent
    type: basic

# Override to add
defaults:
  - _self_
  - override /agents:
    - ${agents}  # Keep existing
    - name: new_agent
      type: advanced
```

### 3. Structured Configs
```python
# In Python code
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

@dataclass
class AgentConfig:
    role: str
    agent_obj: str
    description: str
    parameters: dict = None

# Register schema
cs = ConfigStore.instance()
cs.store(name="agent_schema", node=AgentConfig)
```

## Common Issues & Solutions

### 1. "Could not override 'X'"
**Problem**: Key doesn't exist in defaults
**Solution**: Use `+` prefix
```bash
# Wrong
uv run python -m buttermilk.runner.cli flows=osb

# Correct
uv run python -m buttermilk.runner.cli +flows=osb
```

### 2. "Interpolation key 'X' not found"
**Problem**: Referenced value not loaded
**Solution**: Ensure source is in defaults
```yaml
defaults:
  - llms: lite  # Must load this for ${llms.general}
```

### 3. "Key 'X' is not in struct"
**Problem**: Structured config doesn't allow key
**Solution**: Use `+` or set `struct: false`

### 4. Validation Errors
**Problem**: Invalid data type or missing required field
**Solution**: Check Pydantic model requirements

## Best Practices

### 1. Configuration Organization
- One concept per file
- Clear naming conventions
- Logical directory structure
- Document complex configs

### 2. Use Interpolation
```yaml
# Good - DRY principle
model_name: gemini-pro
agents:
  - model: ${model_name}
  - model: ${model_name}

# Bad - Repetition
agents:
  - model: gemini-pro
  - model: gemini-pro
```

### 3. Environment-Specific Settings
```yaml
# conf/envs/production.yaml
defaults:
  - _self_
  - llms: full
  - storage: bigquery

bm:
  platform: gcp
  debug: false
```

### 4. Testing Configurations
```bash
# Validate without running
uv run python -m buttermilk.runner.cli --cfg job

# Check specific values
uv run python -m buttermilk.runner.cli --cfg job | grep model
```

## LLM Configuration Details

### Model Loading
The LLM configuration is loaded from a GCP Secret named `models.json`. Authentication with GCP is required for tests to run correctly.

### Model Selection Strategy
```yaml
# Define model roles
llms:
  general: gemini-pro      # General purpose
  fast: gemini-flash      # Quick responses
  coding: claude-3-5      # Code generation
  judgers: ${llms.models} # All models for judging
```

### Dynamic Model Lists
```yaml
# For judge agents that try multiple models
judge:
  variants:
    model: ${llms.judgers}  # List of all models
```

## Storage Configuration

### Local Storage
```yaml
storage:
  type: local
  path: "/tmp/buttermilk"
  format: json
```

### Cloud Storage
```yaml
storage:
  type: bigquery
  project: ${oc.env:GCP_PROJECT}
  dataset: buttermilk_data
  table: results
```

## Debugging Configuration

### Enable Debug Output
```bash
export HYDRA_FULL_ERROR=1
uv run python -m buttermilk.runner.cli
```

### Configuration Search Path
```bash
uv run python -m buttermilk.runner.cli --info searchpath
```

### Override Precedence
1. Command line overrides (highest)
2. Local.yaml
3. Composed configs
4. Defaults (lowest)

## Critical Rules

### DO
- ✅ Use YAML files exclusively
- ✅ Leverage interpolation
- ✅ Validate early with Pydantic
- ✅ Keep secrets in environment/secrets manager
- ✅ Document complex configurations

### DON'T
- ❌ Create manual dictionaries
- ❌ Hardcode values in code
- ❌ Change validation to suppress errors
- ❌ Commit sensitive data
- ❌ Ignore type mismatches

Remember: Configuration drives behavior. When debugging, always check the composed configuration first with `-c job`.