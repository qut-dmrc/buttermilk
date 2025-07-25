# Buttermilk Debugging Guide

## Overview
Buttermilk provides comprehensive debugging tools for LLM-driven development. This guide covers tools, strategies, and best practices for debugging flows and agents.

## Frontend Access Points

### Development Server URLs
- **Main landing page**: http://localhost:5173 (marketing/info page)
- **Terminal interface**: http://localhost:5173/terminal (interactive flow execution)
- **API backend**: http://localhost:8000 (FastAPI server)

### Prerequisites
1. **Frontend dev server must be running**: `npm run dev` in `/buttermilk/frontend/chat`
2. **Backend API server must be running**: `uv run python -m buttermilk.runner.cli "+flows=[zot,osb,trans]" +run=api llms=full`
3. **Playwright browser must be installed**: Use `mcp__playwright__browser_install` if you get Chrome not found errors

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
# Get latest log filename:
python scripts/mcp_debug/buttermilk_logs.py

# View logs - IMPORTANT: Always specify --level to get desired log level
python scripts/mcp_debug/buttermilk_logs.py tail 100 --level INFO   # INFO and above
python scripts/mcp_debug/buttermilk_logs.py tail 100 --level DEBUG  # DEBUG and above
python scripts/mcp_debug/buttermilk_logs.py tail 100               # Without --level shows ALL logs including DEBUG

# Other log commands
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

#### Frontend Browser Automation with Playwright MCP

The official Playwright MCP tool (`@playwright/mcp`) provides browser automation capabilities for frontend debugging. 

**Installation**:
```bash
# Install the official Playwright MCP tool
npx @playwright/mcp@latest
```

**Usage**: The Playwright MCP tool provides standard browser automation commands like:
- `navigate` - Navigate to URLs
- `screenshot` - Capture screenshots
- `click` - Click elements
- `fill` - Fill form fields
- `evaluate` - Execute JavaScript in browser context

For detailed usage, refer to the [Playwright MCP documentation](https://github.com/microsoft/playwright/tree/main/packages/playwright-mcp).

### Core Commands

#### 1. API Server Management
```bash
# Start API server with flows
uv run python -m buttermilk.runner.cli "+flows=[zot,osb,trans]" +run=api llms=full

# Check server health
curl -s http://localhost:8000/health | jq .

# Stop API server
make kill_api
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

# View logs (defaults to INFO level)
uv run python -m buttermilk.debug.ws_debug_cli logs

# View logs with a specific level and line count
uv run python -m buttermilk.debug.ws_debug_cli logs --level DEBUG --lines 100
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

For browser automation, use the official Playwright MCP tool (`npx @playwright/mcp@latest`)

### Debug Infrastructure

The debug infrastructure is now organized as:
- **MCP tool definitions**: `.mcp/buttermilk-server.json`
- **Shell wrappers**: `.mcp/tools/*.sh` (call Python scripts)
- **Python implementations**: `scripts/mcp_debug/*.py` (actual logic)
- **Standalone tools**: Don't depend on buttermilk imports

## CRITICAL: Debugging Workflow Example

### ❌ WRONG Approach (What NOT to do):
```
User: "Debug why assessments aren't coming through to the UI"
Agent: *Immediately uses grep to search for 'assessment' in source files*
Agent: *Reads multiple source files trying to understand data flow*
Agent: *Makes assumptions about the problem based on code inspection*
```

### ✅ CORRECT Approach (Follow this pattern):
```bash
# 1. STOP - Check debugging documentation first
cat docs/bots/debugging.md

# 2. Use WebSocket debug tools to reproduce and monitor
uv run python -m buttermilk.debug.ws_debug_cli start trans --record "test" --criteria "hrc"
# Monitor the actual message flow and agent outputs

# 3. Check logs for error patterns
python scripts/mcp_debug/buttermilk_logs.py search "assessment" 100
python scripts/mcp_debug/buttermilk_logs.py errors

# 4. ONLY after understanding the actual data flow, then investigate source code
```

## Debugging Checklist

### Before Debugging
- [ ] Read `docs/bots/debugging.md` completely
- [ ] Can reproduce issue consistently
- [ ] Have minimal test case
- [ ] Checked recent commits
- [ ] Read relevant GitHub issues

### During Debugging
- [ ] Using documented debugging tools FIRST
- [ ] Following systematic approach
- [ ] Taking notes on findings
- [ ] Testing hypotheses individually
- [ ] Not making assumptions

### After Debugging
- [ ] Root cause identified
- [ ] Fix tested thoroughly
- [ ] Regression tests added
- [ ] Documentation updated

## Output Management for Agents

**CRITICAL**: When debugging, focus outputs on the problem at hand:
- Use `head_limit` parameter in grep commands
- Extract only relevant JSON fields, not entire objects
- Summarize patterns rather than showing all occurrences
- For WebSocket debugging, show only messages related to the issue

**Example of Good vs Bad Output**:
```bash
# ❌ BAD: Dumps entire session data
uv run python -m buttermilk.debug.ws_debug_cli --json-output wait --session xyz

# ✅ GOOD: Focuses on specific message types
uv run python -m buttermilk.debug.ws_debug_cli wait --session xyz | jq '.messages[] | select(.type == "agent_message") | {agent: .agent, type: .data.type}'
```

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

### Frontend Debugging with Playwright MCP

**Use Cases**: Debug Svelte component state, validate UI behavior, monitor API integration from browser perspective

**Key Capabilities**:
1. **Browser Automation**: Navigate, click, fill forms, take screenshots
2. **JavaScript Evaluation**: Execute code in browser context to inspect state
3. **Visual Debugging**: Screenshots for layout/styling issues  
4. **Network Monitoring**: Track API calls from browser perspective
5. **Interactive Testing**: Simulate complex user interactions

**Setup**:
```bash
# Install Playwright MCP tool
npx @playwright/mcp@latest

# The tool will be available to LLM agents through MCP protocol
```

**Example Debugging Workflow with Playwright MCP**:
1. Navigate to the frontend URL (http://localhost:5173)
2. Take screenshots to verify UI state
3. Click buttons and fill forms to trigger flows
4. Evaluate JavaScript to inspect Svelte stores:
   ```javascript
   // Example: Get Svelte store values
   window.__svelte_stores?.recordsStore?.subscribe(v => console.log(v))
   ```
5. Monitor network activity during flow execution

**Executing Trans Flow via Frontend**:
1. Navigate to http://localhost:5173/terminal
2. Select flow parameters:
   - Flow: TRANS (or ZOT, OSB depending on available flows)
   - Dataset: TJA (auto-selected when flow is chosen)
   - Record: Select from dropdown (e.g., "kansas_highway")
   - Criteria: Select from dropdown (e.g., "hrc")
3. Click "Run Flow" to commence
4. Monitor judge results in the terminal interface:
   - Judge agents will display assessment scores
   - Detailed reasons available via "[+] reasons" button
   - Flow progresses through FETCH → JUDGE → SYNTHESISER steps

**Frontend Flow Selection Behavior**:
- Datasets auto-populate based on selected flow
- Records load asynchronously after dataset selection
- Models may show "No models available" (this is normal)
- WebSocket connection status shown in top bar
- "human in loop" button controls auto-approval mode

**Common Frontend Issues Debuggable with Playwright**:
- Reactive statements not updating (`$:` syntax issues)
- Store subscriptions not working (evaluate JS to check store values)
- API response not reflected in UI (monitor network + inspect DOM)
- WebSocket disconnections (check console logs)
- CSS/layout problems (screenshot for visual comparison)
- Flow execution monitoring (capture judge results and agent messages)

For detailed Playwright MCP commands and options, consult the official documentation.

### Frontend Troubleshooting

**Common Issues and Solutions**:

1. **"No flows available" in dropdown**:
   - Verify backend API is running with flows enabled
   - Check browser console for API errors
   - Ensure correct proxy configuration in vite.config.ts
   - **Be patient**: Frontend elements often need time to populate. Wait 5-10 seconds before assuming something is broken.

2. **WebSocket connection fails**:
   - Check that both frontend and backend servers are running
   - Verify WebSocket proxy configuration in vite.config.ts
   - Look for CORS errors in browser console

3. **Playwright "Chrome not found" error**:
   - Run `mcp__playwright__browser_install` before using browser automation
   - This installs the required Chrome/Chromium browser

4. **Flow execution hangs**:
   - Check backend logs for agent initialization errors
   - Verify LLM configuration (llms=full) when starting API server
   - Monitor WebSocket messages in browser DevTools

5. **Judge results not appearing**:
   - Ensure proper criteria selection (e.g., "hrc" not "HRC")
   - Check that record exists in selected dataset
   - **Be patient**: Judge results may take 10-30 seconds to appear
   - Known issue: Assessment counts may show "(0)" even when judges are producing results

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

## Common Anti-Patterns to Avoid

### 1. Source Code First Approach
**❌ WRONG**: Immediately grep/read source files when debugging
**✅ RIGHT**: Use WebSocket CLI and log analyzers to understand actual behavior first

### 2. Assuming Instead of Verifying
**❌ WRONG**: "The agent probably publishes this way..."
**✅ RIGHT**: Use `ws_debug_cli` to see exact message format and flow

### 3. Large Unfocused Outputs
**❌ WRONG**: Dumping entire JSON responses or full log files
**✅ RIGHT**: Extract only relevant fields using `jq` or focused grep patterns

### 4. Ignoring Documentation Structure
**❌ WRONG**: Not checking debugging.md when debugging
**✅ RIGHT**: Always start with documented procedures and tools

### 5. Tool Substitution
**❌ WRONG**: Using logs when asked to use WebSocket CLI
**✅ RIGHT**: Use the exact tool requested - each has specific capabilities

## Emergency Procedures

### Server Won't Stop
```bash
# Use the Python script
python scripts/mcp_debug/buttermilk_server.py stop

# If that fails, force kill
make kill_api 

# or `make kill_chat` for the frontend, `make kill` for both.
```
