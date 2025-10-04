#!/bin/bash
# Generic entrypoint for FastMCP applications
#
# Supports two modes:
# - stdio (default): For MCP client integration
# - http: Run as HTTP server on port 8024
#
# Set MODE environment variable to switch modes

if [ "$MODE" = "http" ]; then
  echo "Starting MCP server in HTTP mode on port 8024..." >&2
  exec uv run python src/main.py
else
  echo "Starting MCP server in stdio mode for MCP clients..." >&2
  exec uv run fastmcp run src/main.py
fi
