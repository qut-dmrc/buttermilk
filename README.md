# buttermilk

[![Tests](https://github.com/qut-dmrc/buttermilk/actions/workflows/tests.yml/badge.svg)](https://github.com/qut-dmrc/buttermilk/actions/workflows/tests.yml)
[![License: GPL v3+](https://img.shields.io/badge/License-GPLv3+-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/buttermilk.svg)](https://pypi.org/project/buttermilk/)

**Opinionated AI and data tools for HASS scholars — putting culture first.**

Developed by and for [@QUT-DMRC](https://research.qut.edu.au/dmrc/) researchers, buttermilk provides standard *flows*, *agents*, and *pipelines* that help humanities and social-science scholars use machine learning, generative AI, and computational techniques in a way that is **understandable, traceable, and reproducible**.

We aim to:

- Bring cultural and theoretical expertise to computational methods.
- Provide easy, well-documented, and proven tools for data collection and analysis.
- Make ScholarOps easier with opinionated defaults for logging, tracing, and archiving.
- Create space for collaboration, experimentation, and evaluation of computational methods for HASS research.

```md
Q: Why "buttermilk"?  A: It's cultured and flows...
```

## Core concepts

- **Flows** — complete research or data-processing pipelines.
- **Records** — immutable data structures with rich metadata.
- **Pipelines** — composable, extensible chains of *processors* with full caching at each step.
- **Processors** — async iterators that consume a `Record` and yield zero or more `Record`s.
- **Orchestrators** — coordinate flow execution in a composable groupchat paradigm involving agents and humans.
- **Agents** — specialised components for specific tasks (LLMs, scrapers, classifiers, data collection).
- **Configuration (Hydra)** — flexible, hierarchical YAML config.

## Features

- Multimodal support for current-generation foundation models (Gemini, Claude, GPT, Llama) and pluggable APIs.
- Prompt templating system for evaluating, improving, and reusing prompt components.
- Standard cloud logging, flexible storage, secure credential management (Azure KeyVault, Google Secret Manager), BigQuery-backed storage, and tracing (OpenTelemetry, Traceloop).
- API and CLI for orchestrating complex multi-agent workflows.
- Run locally, on remote GPUs, or in cloud compute (GCP / Azure; AWS planned).
- Batch processing for large-scale data work.

## Installation

> **Note.** Buttermilk's default Hydra config targets Google Cloud (Vertex AI, BigQuery). Running the full quickstart with default config requires a configured GCP project. The core library is provider-agnostic; you can configure other backends by editing `conf/`.

```bash
pip install buttermilk
# or, for development:
git clone https://github.com/qut-dmrc/buttermilk.git
cd buttermilk
uv sync --extra dev --upgrade
```

For the default GCP setup, authenticate before first run:

```bash
export GOOGLE_CLOUD_PROJECT=<your-project>
gcloud auth application-default login --project ${GOOGLE_CLOUD_PROJECT}
gcloud config set project ${GOOGLE_CLOUD_PROJECT}
```

Configs live as YAML under [`conf/`](conf/) and are composed with [Hydra](https://hydra.cc).

## Quickstart

The minimal async initialisation, using buttermilk's public Python API:

```python
import asyncio
from pathlib import Path
from buttermilk import init_async

async def main():
    # Point at a directory of YAML configs (here, the buttermilk default conf/)
    bm = await init_async(
        config_dir=str(Path(__file__).parent / "conf"),
        job="my-first-job",
    )
    bm.logger.info("buttermilk initialised", job=bm.cfg.job, version=bm.__version__)

asyncio.run(main())
```

For a worked example, see [`examples/typed_config_example.py`](examples/typed_config_example.py).

### Command-line interface

```bash
# Run a single flow interactively
bm run.mode=console run.flow=trans

# Use different LLM profiles
bm run.mode=console llms=debug      # Fast, cheap models for testing
bm run.mode=console llms=full       # Production-quality models

# Batch processing
bm run.mode=batch run.flow=trans run.limit=100

# Start the API server
bm run.mode=api
```

Available modes: `console`, `batch`, `api`, `pipeline`, `streamlit`, `slackbot`.
Available LLM profiles: `debug`, `lite`, `full`, `expensive` (see [`conf/llms/`](conf/llms/)).

### Using from a third-party project

Install buttermilk as a dependency and point at your project's config:

```bash
pip install buttermilk
bm --config-path=./conf run.mode=console run.flow=your_flow
```

Create a `conf/` directory in your project with `config.yaml` and your flow definitions under `conf/flows/`.

## Documentation

The full documentation site is on the roadmap. For now:

- **[`examples/`](examples/)** — runnable examples and notebooks.
- **[`conf/`](conf/)** — every flow, LLM profile, and pipeline mode is configurable from here.
- **[`docs/`](docs/)** — placeholder index; will fill in over time.

## Contributing

Buttermilk is actively under development. We welcome contributions and feedback — including bug reports, design discussions, and code.

- Read [**CONTRIBUTING.md**](CONTRIBUTING.md) for development setup, tests, and the PR process.
- Read the [**Code of Conduct**](CODE_OF_CONDUCT.md).
- Report security issues privately — see [**SECURITY.md**](SECURITY.md).

## How to cite

If you use buttermilk in academic work, please cite it as described in [`CITATION.cff`](CITATION.cff). GitHub renders a "Cite this repository" button in the repo sidebar.

## License

Buttermilk is released under **GPL-3.0-or-later** — see [`LICENSE`](LICENSE).

GPL-3.0 covers *distribution* of buttermilk and its derivatives. Running buttermilk as part of a hosted service does **not** by itself trigger copyleft obligations — see the License FAQ in [`CONTRIBUTING.md`](CONTRIBUTING.md#license-faq).
