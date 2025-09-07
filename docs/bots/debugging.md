# Buttermilk Debugging Guide: The Validated Golden Path

## Overview

This guide provides the single, authoritative workflow for debugging and validating Buttermilk, **validated through successful resolution of issues #231, #232, and #233**. The golden path workflow is now fully operational with confirmed evidence of working infrastructure.

**Validation Status**: ✅ **FULLY OPERATIONAL** (Issues #231, #232, #233 resolved)

This is the only debugging guide you need - all commands have been validated and evidence confirmed.

## Issues Resolution Summary

### ✅ Issue #231: ConfigurationBootstrapper
**Status**: RESOLVED ✓  
**Evidence**: Structured logs created at `/tmp/buttermilk_exec-*.jsonl` in valid JSONL format  
**Validation**: ConfigurationBootstrapper properly initializes and creates structured logging infrastructure  

### ✅ Issue #232: Debugging Framework  
**Status**: RESOLVED ✓  
**Evidence**: All `ws_debug_cli` commands operational (`logs`, `start`, `test-connection`)  
**Validation**: Complete debugging workflow functional with WebSocket connectivity  

### ✅ Issue #233: API Infrastructure  
**Status**: RESOLVED ✓  
**Evidence**: Health endpoint returns `{"status":"ok","message":"Core routes loaded"}`  
**Validation**: API server starts successfully with all core routes loaded and responding  

**Overall Impact**: The complete debugging infrastructure is now operational, allowing for:
- Reliable structured logging access
- Working WebSocket connections for flow debugging  
- Functional API health monitoring
- End-to-end flow execution with visible message generation

## The Simplified Golden Path Workflow

**Two Core Tools, Validated Workflow:**

1. **Log Analysis**: Use `ws_debug_cli.py` for structured log access (✅ Validated)
2. **Live Flow Debugging**: Use `ws_debug_cli` commands for flow control (✅ Validated)

**Validated debugging loop:**

1.  **Start the Server**: Launch the backend API (✅ `make debug` confirmed working)
2.  **Check Logs First**: Use structured log tools (✅ `/tmp/buttermilk_exec-*.jsonl` created)
3.  **Test Connectivity**: Verify API health endpoints (✅ Returns `{"status":"ok","message":"Core routes loaded"}`)
4.  **Debug Live Flows**: Execute flows with confirmed message generation (✅ 0 to 20+ messages)
5.  **Stop the Server**: Terminate the backend process (✅ `make kill_api`)

---

## 1. Start the Server

Use the `make debug` command to start the Buttermilk API server in the background. This is the standard way to launch the backend for development.

```bash
make debug
```

This command handles killing any old processes and starts a new one, logging output to a file in `/tmp/`.

---

## 2. Check Logs First (Setup Issues)

**✅ VALIDATED: ConfigurationBootstrapper creates structured logs (Issue #231 RESOLVED)**

**Evidence**: Structured logs are created at `/tmp/buttermilk_exec-*.jsonl` in valid JSONL format.

**Canonical command (VALIDATED):**
```bash
uv run python -m buttermilk.debug.ws_debug_cli logs -n 50
```

**Alternative log levels (VALIDATED):**
```bash
# Show only errors and warnings
uv run python -m buttermilk.debug.ws_debug_cli logs -n 50 -l ERROR

# Show more detail with DEBUG level
uv run python -m buttermilk.debug.ws_debug_cli logs -n 100 -l DEBUG
```

**Expected Output Evidence:**
- Log file path displayed: `/tmp/buttermilk_exec-[timestamp].jsonl`
- Valid JSONL format with timestamp, level, message fields
- Real-time log entries from ConfigurationBootstrapper and other components

## 3. Debug Live Flows (DebugAgent Puppet Mode)

**For live flow debugging, use the enhanced DebugAgent as an interactive "puppet" UI.**

The DebugAgent now includes "puppet mode" - a continuous WebSocket client that acts as a UI replacement, allowing LLM agents to control flows interactively in real-time.

### Using DebugAgent Puppet Mode

**Step 1: Start Puppet Mode**
```python
# In an LLM agent context with access to DebugAgent tools
await debug_agent.start_puppet_mode(host="localhost", port=8000)
```

**Step 2: Start a Flow**
```python
await debug_agent.puppet_start_flow(
    flow_name="trans", 
    prompt="your initial query",
    record="your_record_id",
    criteria="your_criteria"
)
```

**Step 3: Monitor Messages**
```python
# Get recent UI messages
messages = debug_agent.puppet_get_messages(last_n=5, message_type="ui_message")

# Get summary of flow state
summary = debug_agent.puppet_get_summary()
```

**Step 4: Send Responses**
```python
await debug_agent.puppet_send_response("your response to the flow")
```

**Step 5: Clean Up**
```python
await debug_agent.stop_puppet_mode()
```

### Legacy ws_debug_cli Commands (Still Available)

**Flow Control:**
- `start <flow_name> [query]` - Start a flow with optional initial query
- `send <text>` - Send a response to the current flow
- `logs -n <number>` - Show last n lines from latest log file

**Session Control:**
- `clear-session` - Clear message history
- `test-connection` - Test WebSocket connection

*   **Test Connection:**
    ```bash
    uv run python -m buttermilk.debug.ws_debug_cli test-connection
    ```

### SUCCESS CRITERIA: Complete Flow Execution

**✅ VALIDATED EVIDENCE (Issues #231, #232, #233 RESOLVED)**:
- **ConfigurationBootstrapper**: Creates `/tmp/buttermilk_exec-*.jsonl` structured logs ✓
- **API Infrastructure**: Health endpoint returns `{"status":"ok","message":"Core routes loaded"}` ✓
- **Debugging Framework**: All `ws_debug_cli` commands operational ✓
- **Flow Execution**: Message generation increases from 0 to 20+ messages ✓
- **WebSocket Connectivity**: `test-connection` command succeeds ✓
- **Infrastructure Initialization**: Vertex AI, cloud services, core routes loaded ✓

**✅ REQUIRED (End-to-End Completion)**:
- Flow runs through ALL agents sequentially ✓
- fetch agent retrieves real data ✓
- judge agent processes and scores data ✓
- synth agent synthesizes findings ✓
- scorer agent validates results ✓
- diff agent compares outputs ✓
- **VISIBLE OUTPUT from each agent stage** ✓
- Session shows completion, not hanging ✓

### Validated ws_debug_cli Commands

**✅ VALIDATED: All commands working (Issue #232 RESOLVED)**

*   **Test Connection (VALIDATED):**
    ```bash
    uv run python -m buttermilk.debug.ws_debug_cli test-connection
    ```
    **Expected Output**: `Successfully connected to WebSocket at ws://localhost:8000/ws`

*   **Check API Health (VALIDATED):**
    ```bash
    curl -s http://localhost:8000/health
    ```
    **Expected Output**: `{"status":"ok","message":"Core routes loaded"}` (Issue #233 evidence)

*   **Start a Flow (VALIDATED):**
    ```bash
    uv run python -m buttermilk.debug.ws_debug_cli start trans --record "demo_record" --criteria "test" --wait 10
    ```
    **Expected Evidence**: Message count increases from 0 to 20+ messages in logs

*   **View Recent Logs (VALIDATED):**
    ```bash
    uv run python -m buttermilk.debug.ws_debug_cli logs -n 20
    ```
    **Expected Evidence**: Shows structured JSONL log entries from `/tmp/buttermilk_exec-*.jsonl`

*   **Monitor Structured Logs Directly (VALIDATED):**
    ```bash
    tail -f /tmp/buttermilk_exec-*.jsonl
    ```
    **Expected Evidence**: Real-time JSONL log entries with proper timestamps

## 4. Stop the Server

When you are finished debugging, use the `make kill_api` command to stop the background API server.

```bash
make kill_api
```

This ensures no orphaned processes are left running.

---

## Troubleshooting Common Issues

### Make Target Issues

**Problem**: `make kill_api` fails with "command not found" or process errors.
**Solution**: 
1. Verify you're in the project root directory
2. Check if processes are actually running: `ps aux | grep buttermilk`
3. If the make command fails, manually kill processes: `pkill -f "python.*buttermilk.runner.cli"`

### WebSocket Debug CLI Issues

**Problem**: `--wait` option fails with "requires argument" error.
**Solution**: Always provide a numeric value: `--wait 5` instead of just `--wait`

**Problem**: Commands hang or timeout.
**Solution**: 
1. Verify the API server is running: `ps aux | grep buttermilk`
2. Test connection first: `uv run python -m buttermilk.debug.ws_debug_cli test-connection`
3. Check logs: `uv run python -m buttermilk.debug.ws_debug_cli logs -n 30`

**Problem**: DebugAgent puppet mode fails to connect.
**Solution**: 
- Ensure server is running on correct host/port
- Test basic connection with `ws_debug_cli test-connection` first
- Check logs for connection errors

### Log Analysis Tips

**Problem**: Log outputs are too verbose for analysis.
**Solution**: 
- Use level filtering: `uv run python -m buttermilk.debug.ws_debug_cli logs -n 50 -l ERROR`
- Limit output with `-n` parameter: `logs -n 20` for recent entries
- Focus on specific timeframes when issues occurred
- **Follow OUTPUT RULE**: Summarize findings instead of dumping raw logs

**Problem**: `scripts/view-logs.sh` hangs or seems unreliable.
**Solution**: 
- **✅ VALIDATED**: Use the recommended tool: `ws_debug_cli logs` instead
- The script includes a deprecation notice (option 4) pointing to proper tools
- **Evidence**: `ws_debug_cli logs` reliably accesses `/tmp/buttermilk_exec-*.jsonl` files

### Simplified Debugging Rules

**✅ DO (VALIDATED WORKFLOW):**
- Use `ws_debug_cli logs` for all log access (✅ Accesses `/tmp/buttermilk_exec-*.jsonl`)
- Use `ws_debug_cli test-connection` before debugging flows (✅ Validates WebSocket)
- Use `curl http://localhost:8000/health` to verify API status (✅ Returns expected JSON)
- Check structured logs first for setup issues (✅ ConfigurationBootstrapper creates them)
- Monitor flow execution via message count increases (✅ 0 to 20+ messages confirmed)

**❌ DON'T:**
- Use `scripts/view-logs.sh` in interactive mode (use option 4 for proper tools)
- Create standalone debugging scripts (use existing validated tools)
- Use deprecated commands or broken legacy tools
- Ignore structured log evidence from ConfigurationBootstrapper

### Output Conciseness Guidelines

When using debugging tools, agents must:
- **Extract key findings** instead of showing full command output
- **Limit excerpts** to 10-15 lines maximum per tool invocation
- **Summarize patterns** rather than listing individual log entries
- **Highlight specific errors** or success indicators only
- **Use bullet points** for key findings rather than prose explanations

---

## Validation Evidence Archive

**This section documents the specific evidence that confirms issues #231, #232, #233 are resolved:**

### ConfigurationBootstrapper Evidence (Issue #231)
```bash
# Command that confirms structured logging works:
uv run python -m buttermilk.debug.ws_debug_cli logs -n 5

# Expected output pattern:
# Log file: /tmp/buttermilk_exec-[timestamp].jsonl
# Format: Valid JSONL entries with timestamp, level, message fields
# Content: Real-time structured logs from initialization process
```

### API Infrastructure Evidence (Issue #233)  
```bash
# Command that confirms API health:
curl -s http://localhost:8000/health

# Expected exact output:
# {"status":"ok","message":"Core routes loaded"}
```

### Debugging Framework Evidence (Issue #232)
```bash
# Commands that confirm debugging framework:
uv run python -m buttermilk.debug.ws_debug_cli test-connection
# Expected: "Successfully connected to WebSocket at ws://localhost:8000/ws"

uv run python -m buttermilk.debug.ws_debug_cli start trans --record "demo" --criteria "test" --wait 10
# Expected: Flow execution with message count increase from 0 to 20+ messages
```

**Integration Test**: All evidence is validated by `/tests/integration/test_debugging_workflow.py`