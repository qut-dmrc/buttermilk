#!/bin/bash
# Wrapper script for test_websocket_flow.py
# This script delegates to the Python implementation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Determine which script to use based on mode
if [ "$3" = "websocket" ] || [ -z "$3" ]; then
    PYTHON_SCRIPT="$PROJECT_ROOT/scripts/mcp_debug/test_websocket_flow.py"
else
    # For HTTP mode, we'll need to create a simple HTTP test script
    # For now, just use the websocket script which supports both
    PYTHON_SCRIPT="$PROJECT_ROOT/scripts/mcp_debug/test_websocket_flow.py"
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Pass all arguments to the Python script
exec python "$PYTHON_SCRIPT" "$@"