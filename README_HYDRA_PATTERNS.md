# Hydra Configuration Patterns in Buttermilk

This document explains common Hydra patterns used in Buttermilk and how to avoid configuration errors.

## Key Hydra Concepts

### 1. Configuration Composition

Hydra builds configurations by merging multiple YAML files:

```yaml
# Base config (config.yaml)
defaults:
  - local              # Includes conf/local.yaml
  - llms: lite        # Includes conf/llms/lite.yaml at 'llms' key
  - flows: []         # Empty by default
```

### 2. Package Directives

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

### 3. Interpolation

Reference other config values with `${}`:

```yaml
run:
  name: ${bm.name}      # References bm.name
  flows: ${flows}       # References root-level flows
  
agents:
  model: ${llms.general}  # References llms.general
```

## Common Patterns

### Pattern 1: Flow with Agents

```yaml
# conf/flows/myflow.yaml
defaults:
  - _self_
  # Load multiple agents under 'agents' key
  - /agents@agents:
    - assistant
    - researcher
  # Load observers
  - /agents@observers:
    - host/llm_host

myflow:
  orchestrator: buttermilk.orchestrators.groupchat.AutogenOrchestrator
  name: "My Flow"
  # These will be populated by defaults
  agents: {}      # Filled by defaults
  observers: {}   # Filled by defaults
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
```

### Pattern 3: Multi-Flow Configuration

```yaml
# conf/multiflow.yaml
defaults:
  - _self_
  - local
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

## Error Patterns and Solutions

### Error: "Could not override 'X'. No match in defaults"

**Cause**: Trying to override a config that doesn't exist.

**Fix**: Use `+` to add new configs:
```bash
# Wrong
uv run python -m cli.py flows=osb

# Correct
uv run python -m cli.py +flows=osb
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

## Advanced Patterns

### Conditional Includes

```yaml
defaults:
  - _self_
  - optional local: local  # Only included if exists
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

## Debugging Hydra Issues

1. **Enable Debug Output**:
   ```bash
   export HYDRA_FULL_ERROR=1
   ```

2. **Check Composition**:
   ```bash
   # See what configs are being loaded
   uv run python -m cli.py --config-name=X --cfg hydra
   ```

3. **Validate Interpolations**:
   ```python
   # In Python
   from omegaconf import OmegaConf
   
   try:
       OmegaConf.resolve(cfg)
   except Exception as e:
       print(f"Interpolation error: {e}")
   ```

4. **Use Structured Configs**:
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

## References

- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
- [Hydra Patterns](https://hydra.cc/docs/patterns/)