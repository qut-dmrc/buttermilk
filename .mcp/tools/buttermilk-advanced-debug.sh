#!/bin/bash
# Advanced WebSocket Debug CLI wrapper for MCP
# 
# This script provides MCP access to the full buttermilk/debug/ws_debug_cli.py functionality,
# including debug server management, session persistence, log analysis, and flow-specific debugging.
#
# Requirements:
# - Full Buttermilk installation with dependencies
# - uv package manager
# - Must be run from Buttermilk project root
#
# Usage: Called by MCP server with command and parameters
# See buttermilk-server.json for parameter schema

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Check if we're in the right directory
if [ ! -f "buttermilk/debug/ws_debug_cli.py" ]; then
    echo "Error: buttermilk/debug/ws_debug_cli.py not found"
    echo "Make sure you're running from the buttermilk project root"
    exit 1
fi

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' command not found"
    echo "Please install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Parse MCP parameters and convert to CLI arguments
COMMAND="$1"
shift

# Collect global options first
GLOBAL_ARGS=()
JSON_OUTPUT="false"

case "$COMMAND" in
    "start")
        FLOW_NAME="$1"
        QUERY="$2"
        WAIT_TIME="${3:-5}"
        RECORD="$4"
        CRITERIA="$5"
        JSON_OUTPUT="$6"
        
        ARGS=("start" "$FLOW_NAME")
        if [ -n "$QUERY" ]; then
            ARGS+=("$QUERY")
        fi
        if [ -n "$WAIT_TIME" ] && [ "$WAIT_TIME" != "5" ]; then
            ARGS+=("--wait" "$WAIT_TIME")
        fi
        if [ -n "$RECORD" ]; then
            ARGS+=("--record" "$RECORD")
        fi
        if [ -n "$CRITERIA" ]; then
            ARGS+=("--criteria" "$CRITERIA")
        fi
        ;;
        
    "start-debug")
        FLOW_NAME="$1"
        QUERY="$2"
        WAIT_TIME="${3:-5}"
        RECORD="$4"
        CRITERIA="${5:-hrc}"
        JSON_OUTPUT="$6"
        
        ARGS=("start-debug" "$FLOW_NAME")
        if [ -n "$QUERY" ]; then
            ARGS+=("$QUERY")
        fi
        if [ -n "$WAIT_TIME" ] && [ "$WAIT_TIME" != "5" ]; then
            ARGS+=("--wait" "$WAIT_TIME")
        fi
        if [ -n "$RECORD" ]; then
            ARGS+=("--record" "$RECORD")
        fi
        if [ -n "$CRITERIA" ] && [ "$CRITERIA" != "hrc" ]; then
            ARGS+=("--criteria" "$CRITERIA")
        fi
        ;;
        
    "start-server")
        FLOW_NAME="${1:-trans}"
        CRITERIA="${2:-hrc}"
        HOST="${3:-localhost}"
        PORT="${4:-8000}"
        
        ARGS=("start-server" "$FLOW_NAME" "--criteria" "$CRITERIA" "--host" "$HOST" "--port" "$PORT")
        ;;
        
    "send")
        CONTENT="$1"
        MESSAGE_TYPE="${2:-response}"
        WAIT_TIME="${3:-5}"
        SESSION_ID="$4"
        JSON_OUTPUT="$5"
        
        ARGS=("send" "$CONTENT")
        if [ -n "$MESSAGE_TYPE" ] && [ "$MESSAGE_TYPE" != "response" ]; then
            ARGS+=("--type" "$MESSAGE_TYPE")
        fi
        if [ -n "$WAIT_TIME" ] && [ "$WAIT_TIME" != "5" ]; then
            ARGS+=("--wait" "$WAIT_TIME")
        fi
        if [ -n "$SESSION_ID" ]; then
            ARGS+=("--session" "$SESSION_ID")
        fi
        ;;
        
    "wait")
        WAIT_TIME="${1:-5}"
        PATTERN="$2"
        MESSAGE_TYPE="$3"
        SESSION_ID="$4"
        JSON_OUTPUT="$5"
        
        ARGS=("wait")
        if [ -n "$WAIT_TIME" ] && [ "$WAIT_TIME" != "5" ]; then
            ARGS+=("--wait" "$WAIT_TIME")
        fi
        if [ -n "$PATTERN" ]; then
            ARGS+=("--pattern" "$PATTERN")
        fi
        if [ -n "$MESSAGE_TYPE" ]; then
            ARGS+=("--type" "$MESSAGE_TYPE")
        fi
        if [ -n "$SESSION_ID" ]; then
            ARGS+=("--session" "$SESSION_ID")
        fi
        ;;
        
    "session")
        JSON_OUTPUT="$1"
        ARGS=("session")
        ;;
        
    "clear-session")
        JSON_OUTPUT="$1"
        ARGS=("clear-session")
        ;;
        
    "logs")
        LINES="${1:-50}"
        JSON_OUTPUT="$2"
        ARGS=("logs" "--lines" "$LINES")
        ;;
        
    "list-flows")
        JSON_OUTPUT="$1"
        ARGS=("list-flows")
        ;;
        
    "test-connection")
        JSON_OUTPUT="$1"
        ARGS=("test-connection")
        ;;
        
    *)
        echo "Error: Unknown command '$COMMAND'"
        echo "Available commands: start, start-debug, start-server, send, wait, session, clear-session, logs, list-flows, test-connection"
        exit 1
        ;;
esac

# Add global flags if needed
if [ "$JSON_OUTPUT" = "true" ]; then
    GLOBAL_ARGS+=("--json-output")
fi

# Execute the command with global args first, then command and its args
exec uv run python buttermilk/debug/ws_debug_cli.py "${GLOBAL_ARGS[@]}" "${ARGS[@]}"