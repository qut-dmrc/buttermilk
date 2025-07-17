# External Configuration Guide for Buttermilk

This guide shows HASS researchers how to create their own configurations without modifying the Buttermilk source code.

## Quick Start

### 1. Set up your configuration directory

Create a configuration directory structure for your research project:

```bash
mkdir -p my_research_project/conf/{flows,agents,data,infrastructure,criteria}
cd my_research_project
touch conf/__init__.py  # Required for Python package discovery
```

### 2. Set environment variable

```bash
export BUTTERMILK_CONFIG_DIR="/path/to/my_research_project/conf"
```

### 3. Create your infrastructure config

```yaml
# conf/infrastructure/my_university.yaml
# @package bm.infrastructure

clouds:
  - type: gcp
    project_id: my-university-research
    region: us-west1
    storage_bucket: my-research-data
    
secret_provider:
  type: gcp
  project_id: my-university-research
  models_secret: research-llm-keys
  
run_info:
  save_dir_base: gs://my-research-data/results/
```

### 4. Create your data configuration

```yaml
# conf/data/my_dataset.yaml
defaults:
  - /storage: bigquery_default

dataset_name: my_study
source_jsonl: gs://my-research-data/input/study_data.jsonl
dataset_id: research_datasets
table_id: study_records
```

### 5. Create your flow

```yaml
# conf/flows/my_analysis.yaml
defaults:
  - _self_
  - /data: my_dataset
  - /agents@agents.judge: judge
  - /agents@observers.spy: spy
  
orchestrator: buttermilk.orchestrators.groupchat.AutogenOrchestrator
name: "my_analysis"
description: "Custom analysis for my research"

parameters:
  human_in_loop: false
  criteria:
    - my_custom_criteria
```

### 6. Run your flow

```bash
# In your project directory
uv run python -m buttermilk.runner.cli \
  infrastructure=my_university \
  flow=my_analysis \
  run.mode=batch
```

## Advanced Patterns

### Custom Agent Development

Create custom agents in your configuration:

```yaml
# conf/agents/my_custom_agent.yaml
role: custom_analyzer
name: "🔬 My Analyzer"
description: "Custom domain-specific analysis agent"
agent_obj: LLMAgent

parameters:
  template: my_custom_template
  
variants:
  model: ${bm.llms.general}
```

### Multi-Institution Setup

For collaborative research across institutions:

```yaml
# conf/infrastructure/multi_institution.yaml
defaults:
  - /infrastructure: my_university  # Base config
  - _self_

# Override specific settings for collaboration
clouds:
  - type: gcp
    project_id: shared-research-project
    storage_bucket: collaborative-data

secret_provider:
  type: gcp
  project_id: shared-research-project
```

### Environment-Specific Configs

Use Hydra's interpolation for different environments:

```yaml
# conf/config.yaml
defaults:
  - infrastructure: ${oc.env:RESEARCH_ENV,local_dev}
  - flow: ${oc.env:RESEARCH_FLOW,my_analysis}
```

## Configuration Package Distribution

To share configurations across your research team:

### 1. Create a Python package

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="my_research_configs",
    packages=find_packages(),
    package_data={
        "my_research_configs": ["conf/**/*.yaml"],
    },
)
```

### 2. Package structure

```
my_research_configs/
├── setup.py
├── my_research_configs/
│   ├── __init__.py
│   └── conf/
│       ├── __init__.py
│       ├── flows/
│       ├── agents/
│       ├── data/
│       └── infrastructure/
```

### 3. Install and use

```bash
pip install -e .  # Development install
# Configs automatically discovered by Buttermilk
```

## Best Practices

### 1. Infrastructure Separation
- Keep infrastructure configs separate from flow configs
- Use environment variables for sensitive information
- Create reusable infrastructure templates

### 2. Agent Independence
- Design agents to be flow-independent
- Let the BM singleton provide storage/save configuration
- Use variants for model selection

### 3. Data Organization
- Use storage templates for consistency
- Separate data source configs from flow configs
- Support both local and cloud storage

### 4. Version Control
- Version control your configuration packages
- Use git tags for stable releases
- Document configuration changes

### 5. Documentation
- Document your custom criteria and templates
- Provide examples for other researchers
- Include validation and testing configs

## Troubleshooting

### Configuration Not Found
1. Check `BUTTERMILK_CONFIG_DIR` environment variable
2. Verify `__init__.py` files exist in all config directories
3. Use `--info searchpath` to debug search paths

### Interpolation Errors
1. Check variable references use correct syntax: `${variable}`
2. Verify referenced configs exist
3. Use `--cfg job` to see resolved configuration

### Agent Configuration Issues
1. Ensure agent classes are importable
2. Check template files exist in templates directory
3. Verify LLM model configurations are valid