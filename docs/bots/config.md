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

### 3. Configuration Issues

#### Check Composed Configuration
```bash
# View full configuration
uv run python -m buttermilk.runner.cli -c job

# Check specific values
uv run python -m buttermilk.runner.cli -c job | grep -A 10 "agents:"
```

## LLM Configuration Details

The LLM configuration is loaded from a GCP Secret named `models.json`. Authentication with GCP is required for tests to run correctly.

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