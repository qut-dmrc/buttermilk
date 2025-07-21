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

# Start debug session with specific configuration
uv run python -m buttermilk.debug.ws_debug_cli start-debug trans --record "record_id" --criteria "hrc"

# Start debug server with specific llms config
uv run python -m buttermilk.debug.ws_debug_cli start-server trans --criteria "hrc"

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


## Common Issue Patterns

### Frontend-Backend Data Flow Issues
**Symptom**: Frontend components not updating despite successful API calls

**Root Cause Patterns**:
1. **Reactive Statement Issues**: Check `$store` vs `$store.data` in Svelte components
2. **API Response Structure Mismatch**: Backend returns `{records: [...]}` but frontend expects direct array

**Debugging Steps**:
```bash
# Check API response structure
curl -s http://localhost:8000/api/records | jq .

# Check browser console for store updates
# Look for reactive statement patterns in .svelte files
```

**Common Fix Pattern**:
```javascript
// In apiStore.ts - handle both array and object responses
const data: RecordItem[] = Array.isArray(responseData) ? responseData : (responseData.records || []);

// In Component.svelte - check reactive statements
$: records = $recordsStore.data || []; // Not just $recordsStore
```

### Agent Registration Issues  
**Symptom**: `ValueError` during orchestrator creation, missing agent registration

**Root Cause Patterns**:
1. **Missing Module Path**: Agent class not fully qualified in YAML config
2. **Missing Register Method**: RoutedAgent subclasses need explicit `register` classmethod
3. **Type Checking Logic**: `isinstance(agent_cls, type(Agent))` should be `issubclass(agent_cls, Agent)`

**Debugging Steps**:
```bash
# Check agent registration in logs
grep -E "Registering|register.*agent" /tmp/buttermilk*.log

# Check agent class hierarchy
uv run python -c "from buttermilk.agents.spy import SpyAgent; print(SpyAgent.__mro__)"
```

**Common Fix Patterns**:
```python
# Add register method to RoutedAgent subclasses
@classmethod
async def register(cls, runtime: "AgentRuntime", type: str, factory: Callable[[], Any], ...):
    return await RoutedAgent.register(runtime=runtime, type=type, factory=factory, ...)

# Fix type checking logic
if issubclass(agent_cls, Agent):  # Not isinstance
```

### Unhandled Exception Spillage
**Symptom**: Exceptions appear in console instead of logs, breaking error handling

**Root Cause Pattern**: Async task exceptions not caught by main exception handlers

**Debugging Steps**:
```bash
# Look for uncaught exceptions
grep -E "Traceback|Exception.*not.*handled" /tmp/buttermilk*.log
```

**Common Fix Pattern**:
```python
# Add task completion callbacks
def handle_task_exception(task_future):
    if task_future.exception() is not None:
        exc = task_future.exception()
        logger.error(f"🚨 FATAL: Unhandled exception in task: {exc}", exc_info=exc)

task = asyncio.create_task(flow_execution())
task.add_done_callback(handle_task_exception)
```

### WebSocket Protocol Issues
**Symptom**: Messages not reaching intended handlers, parameter not passed

**Root Cause Pattern**: Message format not matching expected protocol

**Expected WebSocket Message Format**:
```json
{
  "type": "run_flow",
  "flow": "trans", 
  "prompt": "optional query",
  "record": "optional_record_id",
  "criteria": "optional_criteria"
}
```

**Debugging Steps**:
```bash
# Monitor WebSocket messages
uv run python -m buttermilk.debug.ws_debug_cli --json-output start trans --record "test" --criteria "hrc"
```

## Emergency Procedures

### Server Won't Stop
```bash
# Force kill all Python processes
pkill -9 -f python

# Clear ports
lsof -ti:8000 | xargs kill -9
```
