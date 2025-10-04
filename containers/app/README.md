# Deploying FastMCP Apps on Buttermilk Base

This directory contains reference templates for deploying FastMCP applications that build on the buttermilk base image.

## Overview

The buttermilk base image (`us-central1-docker.pkg.dev/prosocial-443205/reg/buttermilk:latest`) provides:
- Python 3.11 with uv package manager
- Node.js and npm
- Google Cloud tools (gcsfuse, gcloud)
- Full buttermilk library with all dependencies pre-installed

Applications built on this base inherit all buttermilk functionality with minimal additional configuration.

## Template Files

- **`Dockerfile.template`**: Minimal Dockerfile pattern for MCP servers
- **`entrypoint.sh`**: Generic entrypoint supporting stdio and HTTP modes

## Usage Pattern

### 1. Project Structure

Your MCP server project should have:
```
your-mcp-server/
├── src/
│   └── main.py          # FastMCP server
├── deploy/
│   ├── Dockerfile       # Copy from template
│   ├── entrypoint.sh    # Copy from template
│   └── build.sh         # Build script
└── pyproject.toml       # Must include buttermilk dependency
```

### 2. pyproject.toml Configuration

Include buttermilk as a git dependency:

```toml
[project]
name = "your-mcp-server"
dependencies = [
    "fastmcp>=2.12.3",
    "buttermilk",
]

[tool.uv.sources]
buttermilk = { git = "https://github.com/qut-dmrc/buttermilk.git", rev = "dev" }
```

### 3. Dockerfile Customization

Copy `Dockerfile.template` to your `deploy/Dockerfile` and customize:

```dockerfile
FROM us-central1-docker.pkg.dev/prosocial-443205/reg/buttermilk:latest

ARG PROJECT_NAME=your-mcp-server
WORKDIR /app/${PROJECT_NAME}

# Add project-specific resources (optional)
COPY .cache/your-data .cache/your-data

# Standard pattern (usually no changes needed)
COPY pyproject.toml ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv uv sync

ENV PYTHONPATH="/app/${PROJECT_NAME}/src:/src/buttermilk:${PYTHONPATH}"
ENV PATH="/src/buttermilk/.venv/bin:/app/${PROJECT_NAME}/.venv/bin:$PATH"
ENV MODE=stdio
EXPOSE 8024

COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
```

### 4. Building and Running

Build:
```bash
docker build -f deploy/Dockerfile -t your-image:latest .
```

Run in stdio mode (for MCP clients):
```bash
docker run --rm -i your-image:latest
```

Run in HTTP mode:
```bash
docker run --rm -e MODE=http -p 8024:8024 your-image:latest
```

## Examples

See these projects for working implementations:
- `projects/osbchatmcp/` - OSB chat with ChromaDB cache
- `projects/zotmcp/` - Zotero search with ChromaDB cache
- `projects/omcp/` - Outlook MCP proxy (no additional resources)

## Key Principles

1. **Minimal Dockerfiles**: Only customize PROJECT_NAME and resource COPY lines
2. **No relative paths**: All paths are absolute or relative to build context
3. **Shared base**: Inherits all buttermilk updates automatically
4. **Consistent pattern**: Same structure across all MCP servers
