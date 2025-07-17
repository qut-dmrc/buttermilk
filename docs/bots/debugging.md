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

#### 3. Log Analysis
```bash
# Find latest log file
ls -la /tmp/buttermilk*.log

# Tail logs in real-time
tail -f /tmp/buttermilk*.log

# Search for errors
grep -i error /tmp/buttermilk*.log

# Filter by component
grep "MyAgent" /tmp/buttermilk*.log
```

### Debug Agent (`buttermilk.debug.debug_agent`)

The DebugAgent provides LLM-accessible tools for debugging:

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

### MCP Debug Server
```bash
# Start debug MCP server (port 8090)
uv run python scripts/run_debug_mcp_server.py

# Access debug tools via REST API
curl http://localhost:8090/tools
```

## Expected Flow Behavior

### 1. Initialization Phase
✅ **Success Indicators**:
- API server starts on port 8000
- Health check returns `{"status": "healthy"}`
- WebSocket connection establishes
- Logs show "Starting API server"

❌ **Common Issues**:
- Port already in use
- Missing authentication
- Configuration errors

### 2. Agent Registration
✅ **Success Indicators**:
- Agents announce with configurations
- Tool definitions registered
- HOST receives agent manifests
- Logs show "Agent X registered"

❌ **Common Issues**:
- Agent initialization failures
- Missing dependencies
- Configuration mismatches

### 3. Message Flow
✅ **Success Indicators**:
- USER messages received by HOST
- HOST calls appropriate agents
- Agents return results
- Results displayed to user

❌ **Common Issues**:
- Message routing failures
- Timeout errors
- Serialization problems

### 4. Tool Execution
✅ **Success Indicators**:
- Tools discovered from agents
- LLM receives tool schemas
- Tool calls execute successfully
- Results returned properly

❌ **Common Issues**:
- Schema validation errors
- Missing tool implementations
- Parameter mismatches

## Debugging Strategies

### 1. Configuration Issues

#### Check Composed Configuration
```bash
# View full configuration
uv run python -m buttermilk.runner.cli -c job

# Check specific values
uv run python -m buttermilk.runner.cli -c job | grep -A 10 "agents:"

# Enable Hydra debug
export HYDRA_FULL_ERROR=1
```

#### Common Config Problems
- Missing interpolation sources
- Type mismatches
- Circular dependencies
- Override conflicts

### 2. Agent Issues

#### Agent Not Working
```python
# 1. Check initialization
grep "Initializing.*YourAgent" /tmp/buttermilk*.log

# 2. Verify configuration
uv run python -m buttermilk.runner.cli -c job | grep -A 20 "your_agent:"

# 3. Test in isolation
# Create minimal test script
```

#### Agent Not Receiving Messages
```python
# 1. Check topic subscription
grep "Subscribing.*YOUR_ROLE" /tmp/buttermilk*.log

# 2. Verify message publishing
grep "Publishing.*YOUR_ROLE" /tmp/buttermilk*.log

# 3. Check role matching
# Ensure agent role matches flow configuration
```

### 3. Flow Issues

#### Flow Not Starting
```bash
# 1. Check orchestrator initialization
grep "orchestrator" /tmp/buttermilk*.log

# 2. Verify flow configuration exists
uv run python -m buttermilk.runner.cli -c job | grep -A 50 "flow:"

# 3. Check for missing agents
# Ensure all referenced agents exist
```

#### Flow Hanging
```bash
# 1. Check for deadlocks
grep -i "timeout\|waiting" /tmp/buttermilk*.log

# 2. Monitor message queue
uv run python -m buttermilk.debug.ws_debug_cli wait --session <id>

# 3. Check agent state
# Look for agents stuck in processing
```

### 4. WebSocket Issues

#### Connection Failures
```python
# 1. Test basic connection
uv run python -m buttermilk.debug.ws_debug_cli test-connection

# 2. Check server logs
grep "WebSocket" /tmp/buttermilk*.log

# 3. Verify URL and port
# Default: ws://localhost:8000/ws
```

#### Message Not Received
```python
# 1. Check message format
# Must be valid JSON with required fields

# 2. Verify session active
# Sessions timeout after inactivity

# 3. Check authorization
# Some flows require auth tokens
```

## Debugging Workflow

### Step 1: Reproduce the Issue
```bash
# 1. Clear logs
rm /tmp/buttermilk*.log

# 2. Start fresh server
uv run python -m buttermilk.runner.cli "+flows=[your_flow]" +run=api

# 3. Reproduce exact steps
# Document each action taken
```

### Step 2: Gather Information
```bash
# 1. Collect logs
cp /tmp/buttermilk*.log ./debug_logs/

# 2. Save configuration
uv run python -m buttermilk.runner.cli -c job > debug_config.yaml

# 3. Note environment
env | grep -E "GOOGLE|OPENAI|ANTHROPIC" > debug_env.txt
```

### Step 3: Isolate the Problem
```python
# 1. Test components individually
# - Test agents in isolation
# - Test configuration loading
# - Test message passing

# 2. Create minimal reproduction
# Remove unnecessary components

# 3. Binary search
# Disable half of agents/features
```

### Step 4: Apply Fix
```python
# 1. Write failing test first
def test_reproduction():
    # Should fail before fix
    pass

# 2. Apply minimal fix
# Change only what's necessary

# 3. Verify fix works
# Run test suite
```

## Common Issues & Solutions

### 1. "Agent not found" Errors
**Symptoms**: `KeyError: 'agent_name'`

**Solutions**:
- Check agent is in flow config
- Verify agent_obj path correct
- Ensure agent class imported

### 2. Timeout Errors
**Symptoms**: Operations timing out

**Solutions**:
- Increase wait times
- Check for blocking operations
- Verify async/await usage

### 3. Serialization Errors
**Symptoms**: JSON encoding failures

**Solutions**:
- Check for non-serializable objects
- Use Pydantic models
- Implement custom encoders

### 4. Configuration Errors
**Symptoms**: Hydra/OmegaConf errors

**Solutions**:
- Check YAML syntax
- Verify interpolations exist
- Use `+` for new keys

## Advanced Debugging

### 1. Performance Profiling
```python
import cProfile
import pstats

# Profile specific function
cProfile.run('your_function()', 'profile_stats')

# Analyze results
p = pstats.Stats('profile_stats')
p.sort_stats('cumulative').print_stats(20)
```

### 2. Memory Debugging
```python
import tracemalloc

# Start tracing
tracemalloc.start()

# Your code here

# Get top memory users
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

### 3. Async Debugging
```python
import asyncio

# Enable debug mode
asyncio.get_event_loop().set_debug(True)

# Log all tasks
for task in asyncio.all_tasks():
    print(f"Task: {task.get_name()}, State: {task._state}")
```

### 4. LLM Call Debugging
```python
# Enable weave tracing
import weave
weave.init('buttermilk-debug')

# Calls will be traced automatically
# View at: https://wandb.ai/
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

Set level with:
```bash
export BUTTERMILK_LOG_LEVEL=DEBUG
```

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

### Integration Test Debugging
```bash
# Run with real services
export USE_REAL_LLM=true
uv run pytest tests/integration/

# Run specific flow test
uv run pytest tests/integration/test_flows.py::test_osb_flow
```

## Emergency Procedures

### Server Won't Stop
```bash
# Force kill all Python processes
pkill -9 -f python

# Clear ports
lsof -ti:8000 | xargs kill -9
```

### Logs Growing Too Large
```bash
# Rotate logs
mv /tmp/buttermilk*.log /tmp/buttermilk_old.log

# Clear logs
> /tmp/buttermilk.log
```

### Configuration Corrupted
```bash
# Reset to defaults
git checkout conf/

# Use minimal config
uv run python -m buttermilk.runner.cli run=console
```

## Remember

1. **Systematic Approach**: Don't guess, investigate
2. **One Change at a Time**: Isolate variables
3. **Document Everything**: Future you will thank you
4. **Ask for Help**: Check issues, ask team
5. **Take Breaks**: Fresh eyes see more

When stuck: Step back, summarize what you know, identify what you don't know, then systematically fill the gaps.