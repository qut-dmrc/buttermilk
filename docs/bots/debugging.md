# Buttermilk Debugging Guide

## Overview
Buttermilk provides comprehensive debugging tools for LLM-driven development. This guide covers tools, strategies, and best practices for debugging flows and agents.

## Available Debugging Tools

### MCP Debugging Scripts

Buttermilk provides Python-based debugging scripts in `/scripts/mcp_debug/` that can be used directly or through MCP tools:

#### Server Management
```bash
# Start server
python scripts/mcp_debug/buttermilk_server.py start
python scripts/mcp_debug/buttermilk_server.py start true  # debug mode
python scripts/mcp_debug/buttermilk_server.py start true trans,zot  # specific flows

# Server control
python scripts/mcp_debug/buttermilk_server.py stop
python scripts/mcp_debug/buttermilk_server.py status
python scripts/mcp_debug/buttermilk_server.py health
python scripts/mcp_debug/buttermilk_server.py flows
```

#### Log Analysis
```bash
# View logs
python scripts/mcp_debug/buttermilk_logs.py tail 100
python scripts/mcp_debug/buttermilk_logs.py errors
python scripts/mcp_debug/buttermilk_logs.py warnings
python scripts/mcp_debug/buttermilk_logs.py search "pattern" 50
python scripts/mcp_debug/buttermilk_logs.py websocket
python scripts/mcp_debug/buttermilk_logs.py follow
python scripts/mcp_debug/buttermilk_logs.py list
```

#### WebSocket Debugging
```bash
# Test WebSocket flows
python scripts/mcp_debug/websocket_debug.py test
python scripts/mcp_debug/websocket_debug.py start osb "What is AI?"
python scripts/mcp_debug/websocket_debug.py send "Tell me more"
python scripts/mcp_debug/websocket_debug.py wait "task_complete" agent_message 30
python scripts/mcp_debug/websocket_debug.py session
python scripts/mcp_debug/websocket_debug.py clear
```

#### Workflow Validation
```bash
# Check workflow compliance
python scripts/mcp_debug/workflow_check.py STOP
python scripts/mcp_debug/workflow_check.py ANALYZE "Fix WebSocket issue"
```

#### Configuration Validation
```bash
# Validate YAML configs
python scripts/mcp_debug/validate_config.py conf/flows/zot.yaml
python scripts/mcp_debug/validate_config.py conf/flows/trans.yaml false
```

#### GitHub Issue Management
```bash
# Work with GitHub issues
python scripts/mcp_debug/github_issue.py search "workflow validation"
python scripts/mcp_debug/github_issue.py create "Add feature" "Description"
python scripts/mcp_debug/github_issue.py link 123
```

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

### MCP Tool Integration

For LLM agents, the MCP tools in `.mcp/tools/` provide wrapper scripts that call the Python debugging scripts:

- `buttermilk-server.sh` - Server management
- `buttermilk-logs.sh` - Log viewing and analysis  
- `buttermilk-ws-debug.sh` - WebSocket debugging
- `buttermilk-workflow-check.sh` - Workflow validation
- `buttermilk-config-validate.sh` - Configuration validation
- `buttermilk-github-issue.sh` - GitHub issue management
- `buttermilk-test-flow.sh` - Flow testing

### Debug Infrastructure

The debug infrastructure is now organized as:
- **MCP tool definitions**: `.mcp/buttermilk-server.json`
- **Shell wrappers**: `.mcp/tools/*.sh` (call Python scripts)
- **Python implementations**: `scripts/mcp_debug/*.py` (actual logic)
- **Standalone tools**: Don't depend on buttermilk imports

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

## Debug Infrastructure Notes

### API Endpoints

The following API endpoints may be useful for debugging:
- `/api/session` - Create/validate sessions
- `/api/sessions` - List all active sessions  
- `/api/session/{session_id}/status` - Get session status
- `/monitoring/health` - Basic health check
- `/monitoring/fatal-errors` - Check for fatal errors
- `/monitoring/metrics/basic` - Basic system metrics

### Design Philosophy

The debugging tools follow the principle of "Simple tools, smart LLM":
- No intelligence in tools - just raw capabilities
- LLM reads files/logs directly when possible
- Focus on MCP tools for LLM integration
- No reports, suggestions, or pattern matching - just data access

## Emergency Procedures

### Server Won't Stop
```bash
# Use the Python script
python scripts/mcp_debug/buttermilk_server.py stop

# If that fails, force kill
pkill -9 -f python

# Clear ports
lsof -ti:8000 | xargs kill -9
```
