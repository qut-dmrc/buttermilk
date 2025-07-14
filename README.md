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

## 📚 Documentation

**[→ Complete Documentation](docs/README.md)**

### Quick Start
- **[Installation Guide](docs/getting-started/installation.md)** - Set up your environment
- **[Quick Start](docs/getting-started/quickstart.md)** - Get running in 5 minutes
- **[Your First Flow](docs/getting-started/first-flow.md)** - Build a custom flow

### User Guide
- **[Running Flows](docs/user-guide/flows.md)** - Complete flow execution guide
- **[Configuration](docs/user-guide/configuration.md)** - Hydra configuration management
- **[API Reference](docs/user-guide/api-reference.md)** - REST API documentation
- **[CLI Reference](docs/user-guide/cli-reference.md)** - Command-line interface

### Core Concepts

Buttermilk is built around a few core concepts that help structure your research and data processing:

*   **Flows**: Complete research or data processing pipelines
*   **Jobs**: Basic units of work for processing individual records
*   **Records**: Immutable data structures with rich metadata
*   **Agents**: Specialized components for specific tasks (AI models, data collection)
*   **Orchestrators**: Coordinate and manage flow execution
*   **Configuration (Hydra)**: Flexible, hierarchical configuration management

For detailed explanations, see **[Core Concepts](docs/reference/concepts.md)**.

## Usage

Buttermilk provides several components and features to facilitate HASS research:

Currently available:

*   Multimodal support for current-generation foundation models (Gemini, Claude, Llama, GPT) and plug-in support for other analysis tool APIs.
*   A prompt templating system for evaluating, improving, and reusing prompt components.
*   Standard cloud logging, flexible data storage options, secure credential management (e.g., Azure KeyVault, Google Secrets), built-in database storage (e.g., BigQuery), and tracing capabilities (e.g., Promptflow, Langchain).
*   An API and CLI for integrating components and orchestrating complex workflows.
*   Support for running code locally, on remote GPUs, or in cloud compute environments (Azure/Google Compute, with AWS Lambda planned).

Future Development:

*   Tutorial workbooks demonstrating complete research pipeline examples.
*   A distributed queue system (e.g., pub/sub) for managing batch runs.
*   A web interface and example notebooks for assessing, tracking, and comparing performance.

## Contributing and Current Status

Buttermilk is actively under development. We welcome contributions and feedback! If you're interested in getting involved, please contact [nic](mailto:n.suzor@qut.edu.au) to discuss ideas, planning, or how to contribute.

### For Contributors
- **[Contributing Guide](docs/developer-guide/contributing.md)** - Development process and standards
- **[Architecture Guide](docs/developer-guide/architecture.md)** - System architecture and design
- **[Creating Agents](docs/developer-guide/creating-agents.md)** - Build custom agents
- **[Testing Guide](docs/developer-guide/testing.md)** - Testing best practices

## Contributing to Documentation

We warmly welcome contributions to improve Buttermilk's documentation! Clear, concise, and up-to-date documentation is crucial for helping HASS scholars and developers effectively use and contribute to the project.

### Documentation Style

*   **Docstrings (Python Code)**: Please follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for all docstrings within the Python code. This includes clear descriptions of modules, classes, functions, methods, arguments, and return values.
*   **Markdown Files (e.g., README.md, docs/*.md)**: Aim for clarity, conciseness, and accuracy. Use standard Markdown formatting. Ensure that examples are easy to follow and reproduce.
*   **General Principles**:
    *   Write for the target audience (HASS scholars, developers).
    *   Be explicit and avoid jargon where possible, or explain it clearly.
    *   Keep documentation consistent with the current state of the codebase.

### Keeping Documentation Up-to-Date

As features are added or modified, please ensure that corresponding documentation is also updated. This includes:
*   Updating module, class, and function docstrings.
*   Revising relevant sections in `README.md` or other documentation files in the `docs/` directory.
*   Ensuring examples and command-line usage instructions are still accurate.

### Process for Documentation Changes

*   **Identify Areas for Improvement**: This could be missing information, unclear explanations, outdated instructions, or typos.
*   **Make Your Changes**: Edit the relevant files. For new concepts or substantial additions, consider discussing them in an issue first.
*   **Submit Changes**: Documentation changes should be submitted via Pull Requests (PRs) to the main repository. Please clearly describe the documentation changes made in your PR description.

We appreciate your help in making Buttermilk more accessible and understandable!

## Development

### Dependencies
You might need to install some tools. If you're running debian:

```bash
# Set the timezone to Australia/Brisbane
ln -sf /usr/share/zoneinfo/Australia/Brisbane /etc/localtime &&  echo "Australia/Brisbane" | tee /etc/timezone

# Install base packages
sudo apt update && sudo apt install -y --no-install-recommends python3-pip build-essential neovim rsync gzip jq less htop git zsh fonts-roboto fonts-noto && sudo apt-get autoremove -y && sudo apt-get -y clean

# Install google cloud sdk
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && sudo apt-get update -y && sudo apt-get install google-cloud-cli -y

# azure CLI tools
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
    
# Install Node and npm
export NODE_MAJOR=20
curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | sudo tee /etc/apt/sources.list.d/nodesource.list && sudo apt update && sudo apt install --no-install-recommends -y nodejs && sudo npm install -g node-pty @devcontainers/cli diff-so-fancy && sudo npm rebuild
```

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

## Instructions for bots
Read [CLAUDE.md](CLAUDE.md)