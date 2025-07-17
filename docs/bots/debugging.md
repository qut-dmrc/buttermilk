# Buttermilk Debugging Guide

## Overview
Buttermilk provides comprehensive debugging tools for LLM-driven development. This guide covers tools, strategies, and best practices for debugging flows and agents.

## Available Debugging Tools

### Core Commands

#### 1. API Server Management
```bash
# Start API server with flows
uv run python -m buttermilk.runner.cli "+flows=[zot,osb,trans]" +run=api llms=full

# Check server health
curl -s http://localhost:8000/health | jq .

# Stop API server
pkill -f "uv run.*buttermilk.runner"
```

#### 2. WebSocket Debug CLI
```bash
# Test WebSocket connection
uv run python -m buttermilk.debug.ws_debug_cli test-connection

# Start a flow session (returns session_id)
uv run python -m buttermilk.debug.ws_debug_cli --json-output start <flow_name> --wait 20

# Send message to active flow
uv run python -m buttermilk.debug.ws_debug_cli --json-output send "message" --session <session_id> --wait 10

# Monitor session activity
uv run python -m buttermilk.debug.ws_debug_cli --json-output wait --session <session_id> --wait 30
```

#### Log Analysis

In debug environments API logs are saved to the most recent file matching `/tmp/buttermilk*.log`

### Debug Agent

The DebugAgent (`buttermilk.debug.debug_agent`) provides LLM-accessible tools for debugging:

```python
# Available tools:
- read_log_file(): Read buttermilk log files
- list_log_files(): List available logs
- search_logs(): Search logs with grep
- test_websocket(): Test WebSocket connection
- start_flow_session(): Start a flow for testing
- send_flow_message(): Send messages to flows
- monitor_session(): Watch for new messages
```

## Debugging Checklist

### Before Debugging
- [ ] Can reproduce issue consistently
- [ ] Have minimal test case
- [ ] Checked recent commits
- [ ] Read relevant GitHub issues

### During Debugging
- [ ] Following systematic approach
- [ ] Taking notes on findings
- [ ] Testing hypotheses individually
- [ ] Not making assumptions

### After Debugging
- [ ] Root cause identified
- [ ] Fix tested thoroughly
- [ ] Regression tests added
- [ ] Documentation updated

## Log Analysis Tips

### Useful Grep Patterns
```bash
# Find errors
grep -E "ERROR|CRITICAL|Exception" /tmp/buttermilk*.log

# Track specific request
grep "session_id.*abc123" /tmp/buttermilk*.log

# Find slow operations
grep -E "took [0-9]{4,}" /tmp/buttermilk*.log

# Agent lifecycle
grep -E "Initializing|Registered|Cleanup" /tmp/buttermilk*.log
```

### Log Levels
- **DEBUG**: Detailed diagnostic info
- **INFO**: General informational messages
- **WARNING**: Warning messages
- **ERROR**: Error messages
- **CRITICAL**: Critical problems


## Testing Strategies

### Unit Test Debugging
```bash
# Run single test with output
uv run pytest -s tests/test_agent.py::test_specific

# Run with debugger
uv run pytest --pdb tests/test_agent.py

# Run with verbose output
uv run pytest -vvv tests/
```


## Emergency Procedures

### Server Won't Stop
```bash
# Force kill all Python processes
pkill -9 -f python

# Clear ports
lsof -ti:8000 | xargs kill -9
```
