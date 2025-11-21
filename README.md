# buttermilk: opinionated data tools for HASS scholars

**AI and data tools for HASS researchers.**

Developed by and for @QUT-DMRC scholars, this repo provides standard **flows** and **pipelines** that help humanities scholars develop rigorous, traceable systems to collect data and use machine learning, generative AI, and computational techniques as part of analysis and experimentation driven by theory and cultural context. We aim to:

- Provide research-backed analysis tools that help scholars bring cultural expertise to computational methods
- Help HASS scholars with well-documented tools for data collection and analysis
- Make scholarly workflows easier with opinionated defaults for logging and archiving in standard formats
- Create a space for collaboration, experimentation, and evaluation of computational methods for HASS scholars

The tools are tested, documented, and versioned. We use our own research projects as development guides, test cases, and ongoing measures of reliability. The goal is to make it easy for HASS scholars to use AI tools in a way that is understandable, traceable, and reproducible.

## Core Concepts

Buttermilk is built around a few core concepts that help structure your research and data processing:

- **Flows**: Complete research or data processing pipelines
- **Records**: Immutable data structures with rich metadata
- **Pipelines**: Composeable, extensible chains of **processors** with **full caching** at each step.
- **Processors**: asynchronous iterators that consume a BaseRecord and yield zero or more BaseRecords.
- **Orchestrators**: Coordinate and manage flow execution within a composable and programmable groupchat paradigm involving agents and potentially humans
- **Agents**: Specialized components for specific tasks (AI models, data collection)
- **Configuration (Hydra)**: Flexible, hierarchical configuration management

For detailed explanations, see **[Core Concepts](docs/reference/concepts.md)**.

## Features

Buttermilk provides several components and features to facilitate HASS research:

- Multimodal support for current-generation foundation models (Gemini, Claude, Llama, GPT) and plug-in support for other analysis tool APIs.
- A prompt templating system for evaluating, improving, and reusing prompt components.
- Standard cloud logging, flexible data storage options, secure credential management (e.g., Azure KeyVault, Google Secrets), built-in database storage (e.g., BigQuery), and tracing capabilities (e.g., Promptflow, Langchain).
- An API and CLI for integrating components and orchestrating complex workflows.
- Support for running code locally, on remote GPUs, or in cloud compute environments (Azure/Google Compute, with AWS Lambda planned).
- A distributed queue system (e.g., pub/sub) for managing batch runs.

## Contributing

Buttermilk is actively under development. We welcome contributions and feedback. If you're interested in getting involved, please contact [nic](mailto:n.suzor@qut.edu.au).

## Installation

Create a new environment and install using uv:

```shell
pip install uv
uv install
```

Authenticate to cloud providers, where your relevant secrets are stored.

```shell
GOOGLE_CLOUD_PROJECT=<project>
gcloud auth login --update-adc --enable-gdrive-access --project ${GOOGLE_CLOUD_PROJECT} --billing-project ${GOOGLE_CLOUD_PROJECT}
gcloud auth application-default set-quota-project ${GOOGLE_CLOUD_PROJECT}
gcloud config set project ${GOOGLE_CLOUD_PROJECT}
```

Configurations are stored as YAML files in `conf/`. You can select options at runtime using [hydra](https://hydra.cc).

## Usage

### Command Line Interface

Run flows using the `bm` command with Hydra configuration:

```shell
# Run a single flow interactively
bm run.mode=console run.flow=trans

# Use different LLM configurations
bm run.mode=console llms=debug      # Fast, cheap models for testing
bm run.mode=console llms=full       # Production-quality models

# Batch processing
bm run.mode=batch run.flow=trans run.limit=100        # Create batch jobs
bm run.mode=batch_run run.limit=5                     # Process queued jobs
bm run.mode=batch_all run.flow=trans run.limit=100   # Create and process

# Start API server
bm run.mode=api

# Run data pipeline
bm run.mode=pipeline run.limit=100
```

Available modes: `console`, `batch`, `batch_run`, `batch_all`, `api`, `pipeline`, `streamlit`, `slackbot`

Available LLM configurations: `debug`, `lite`, `full`, `expensive` (see `conf/llms/` for details)

#### Using from Third-Party Projects

Install buttermilk as a dependency and point to your project's config directory:

```shell
# Use bm with custom config path
bm --config-path=./conf run.mode=console run.flow=your_flow

# Or use Python module
uv run python -m buttermilk.runner.cli --config-path=./conf run.mode=batch run.flow=your_flow
```

Create a `conf/` directory in your project with `config.yaml` and your flow definitions in `conf/flows/`

### Python API

```python
from pathlib import Path
from buttermilk import init, init_async

script_dir = Path(__file__).parent
bm = await init_async(config_dir=str(script_dir / "../conf"), job="my job")
logger = bm.logger
config = bm.cfg
logger.info("structured logging available", job=bm.cfg.job)
```
