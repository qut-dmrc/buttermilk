# Installation Guide

This guide will help you set up Buttermilk for development and usage.

## System Requirements

- **Python 3.10+** (Python 3.12 recommended)
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ free space
- **Network**: Internet access for downloading dependencies and cloud services

## Quick Installation

### 1. Install uv (Recommended)

```bash
pip install uv
```

### 2. Clone the Repository

```bash
git clone https://github.com/qut-dmrc/buttermilk.git
cd buttermilk
```

### 3. Install Dependencies

```bash
uv install
```

### 4. Verify Installation

```bash
uv run python -m buttermilk.runner.cli --help
```

## Cloud Authentication

Buttermilk integrates with cloud services for configuration and storage. You'll need to authenticate with at least one cloud provider:

### Google Cloud Platform

```bash
# Set your project ID
export GOOGLE_CLOUD_PROJECT=your-project-id

# Authenticate
gcloud auth login --update-adc --enable-gdrive-access --project ${GOOGLE_CLOUD_PROJECT} --billing-project ${GOOGLE_CLOUD_PROJECT}
gcloud auth application-default set-quota-project ${GOOGLE_CLOUD_PROJECT}
gcloud config set project ${GOOGLE_CLOUD_PROJECT}
```

### Azure (Optional)

```bash
# Install Azure CLI if not already installed
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Authenticate
az login
```

## System Dependencies (Linux/Debian)

If you're running on Debian/Ubuntu, you may need additional system packages:

```bash
# Set timezone (optional)
sudo ln -sf /usr/share/zoneinfo/Australia/Brisbane /etc/localtime
echo "Australia/Brisbane" | sudo tee /etc/timezone

# Install base packages
sudo apt update && sudo apt install -y --no-install-recommends \
    python3-pip \
    build-essential \
    git \
    curl \
    jq \
    less \
    htop

# Clean up
sudo apt-get autoremove -y && sudo apt-get clean
```

## Development Setup

### Additional Tools for Development

```bash
# Install Node.js for frontend development
export NODE_MAJOR=20
curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | sudo tee /etc/apt/sources.list.d/nodesource.list
sudo apt update && sudo apt install -y nodejs
```

### Install Google Cloud SDK

```bash
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
sudo apt-get update && sudo apt-get install -y google-cloud-cli
```

## Configuration

### Basic Configuration

Buttermilk uses Hydra for configuration management. Basic configurations are provided in the `conf/` directory:

```bash
# View available configurations
uv run python -m buttermilk.runner.cli --help

# Check default configuration
uv run python -m buttermilk.runner.cli -c job
```

### Local Configuration

Create a `conf/local.yaml` file for local overrides:

```yaml
# Example local configuration
defaults:
  - _self_

# Local overrides
storage:
  local_path: /tmp/buttermilk-data

logging:
  level: DEBUG
```

## Verification

### Test Basic Functionality

```bash
# Test console mode
uv run python -m buttermilk.runner.cli run=console +flow=trans_clean +prompt="Hello, world!"

# Test API mode (in background)
uv run python -m buttermilk.runner.cli run=api &
curl -X GET http://localhost:8000/
```

### Run Tests

```bash
# Run unit tests
uv run python -m pytest

# Run with coverage
uv run python -m pytest --cov
```

## Troubleshooting

### Common Issues

**uv command not found**
```bash
pip install uv
```

**Python version issues**
```bash
# Check Python version
python3 --version

# Install Python 3.12 if needed (Ubuntu/Debian)
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.12
```

**Authentication errors**
```bash
# Re-authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login
```

**Permission errors**
```bash
# Use user installation
pip install --user uv
```

### Getting Help

- Check the [Troubleshooting Guide](../reference/troubleshooting.md)
- Review [Common Issues](../reference/troubleshooting.md#common-issues)
- Contact the team at [nic@suzor.com](mailto:nic@suzor.com)

## Next Steps

Once installation is complete, proceed to the [Quick Start Guide](quickstart.md) to run your first flow.