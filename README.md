# buttermilk: opinionated data tools for HASS scholars

**AI and data tools for HASS researchers, putting culture first.**

Developed by and for @QUT-DMRC scholars, this repo aims to provide standard **flows** that help scholars make their own **pipelines** to collect data and use machine learning, generative AI, and computational techniques as part of rich analysis and experimentation that is driven by theory and deep understanding of cultural context. We try to:

* Provide a set of research-backed analysis tools that help scholars bring cultural expertise to computational methods.
* Help HASS scholars with easy, well-documented, and proven tools for data collection and analysis.
* Make ScholarOps™ easier with opinionated defaults that take care of logging and archiving in standard formats.
* Create a space for collaboration, experimentation, and evaluation of computational methods for HASS scholars.

```md
Q: Why 'buttermilk'??
A: It's cultured and flows...
```

```md
Q: What's MLOps?
A: A general term for standardised approaches to machine learning workflows that 
helps you organize your project, collaborate, iteratively improve your analysis 
and track versioned changes, monitor onging performance, reproduce experiments, 
and verify and compare results. 
```

The "pipeline" we are building is documented and versioned. We're aiming to make it easy for HASS scholars to use AI tools in a way that is understandable, traceable, and reproducible.

## Core Concepts

Buttermilk is built around a few core concepts that help structure your research and data processing:

*   **Flows**: Complete research or data processing pipelines
*   **Jobs**: Basic units of work for processing individual records
*   **Records**: Immutable data structures with rich metadata
*   **Agents**: Specialized components for specific tasks (AI models, data collection)
*   **Orchestrators**: Coordinate and manage flow execution
*   **Configuration (Hydra)**: Flexible, hierarchical configuration management

For detailed explanations, see **[Core Concepts](docs/reference/concepts.md)**.

## Features

Buttermilk provides several components and features to facilitate HASS research:


*   Multimodal support for current-generation foundation models (Gemini, Claude, Llama, GPT) and plug-in support for other analysis tool APIs.
*   A prompt templating system for evaluating, improving, and reusing prompt components.
*   Standard cloud logging, flexible data storage options, secure credential management (e.g., Azure KeyVault, Google Secrets), built-in database storage (e.g., BigQuery), and tracing capabilities (e.g., Promptflow, Langchain).
*   An API and CLI for integrating components and orchestrating complex workflows.
*   Support for running code locally, on remote GPUs, or in cloud compute environments (Azure/Google Compute, with AWS Lambda planned).
*   A distributed queue system (e.g., pub/sub) for managing batch runs.


## 📚 Examples and tutorials

Currently available:
*   A web interface and example notebooks for assessing, tracking, and comparing performance.

## Contributing and Current Status

Buttermilk is actively under development. We welcome contributions and feedback! If you're interested in getting involved, please contact [nic](mailto:n.suzor@qut.edu.au) to discuss ideas, planning, or how to contribute.

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

```python
from pathlib import Path

script_dir = Path(__file__).parent
bm = init(config_dir=str(script_dir / "../conf"), job="my job")
logger = bm.logger
logger.info("structured logging available", job=bm.cfg.job)
```