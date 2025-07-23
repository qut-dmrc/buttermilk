#!/bin/bash
# Wrapper script for buttermilk_server.py
# This script delegates to the Python implementation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/mcp_debug/buttermilk_server.py"

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Pass all arguments to the Python script
exec python "$PYTHON_SCRIPT" "$@"