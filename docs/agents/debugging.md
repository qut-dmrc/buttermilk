# Buttermilk Debugging Guide: The Validated Golden Path

## Overview

This guide provides the single, authoritative workflow for debugging and validating Buttermilk. The golden path workflow is now fully operational with confirmed evidence of working infrastructure.

**Validation Status**: ✅ **FULLY OPERATIONAL**

This is the only debugging guide you need - all commands have been validated and evidence confirmed.

## Valid System Configuration Parameters

**⚠️ CRITICAL: Use Only Valid Parameters**

**Available Flows**: 
- `trans` - Transgender journalist ethics research flow

**Valid Criteria Templates**:
- `tja` - Trans Journalists Association stylebook criteria
- `glaad` - GLAAD media reference criteria
- Other criteria from `/buttermilk/conf/flows/criteria/` and template files

**Record ID Requirements**:
- **MUST** use actual record IDs from your data sources
- **NEVER** use placeholder values like 'demo_record', 'demo', 'test_record'
- **Recommended valid records**: 'betoota_snape_trans'
- Check your data files or storage configurations for valid record IDs

**❌ CRITICAL**: Flows WILL NOT run with arbitrary parameters. All parameters (record_id, criteria, flow) MUST match live data configuration.

**❌ INVALID EXAMPLES** (DO NOT USE):
- Flows: 'simple', 'test hashing', 'demo_flow' 
- Records: 'demo_record', 'demo', 'test_record'
- Criteria: 'test', 'demo_criteria'

**✅ VALID EXAMPLES** (CONFIRMED WORKING):
- Flow: 'trans'
- Records:  'betoota_snape_trans'
- Criteria: 'tja', 'glaad'

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

### For Standard Debugging (Most Agents)
Use the `make debug` command to start the Buttermilk API server in the background:

```bash
make debug
```

This command handles killing any old processes and starts a new one, logging output to a file in `/tmp/`.

### For Advanced Debugging (Agents with Background Process Capability)
Agents capable of running background processes (like Claude Code) should launch the server directly to monitor stdio in real-time:

```bash
uv run python -m buttermilk.runner.cli "+flows=[trans,zot,osb]" run=api llms=debug verbose=true
```

**⚠️ WARNING**: This command does not time out. Only use if your agent can manage background processes. Other agents should use `make debug` instead.

---

## 2. Check Logs First

**✅ VALIDATED: ConfigurationBootstrapper creates structured logs**

**Canonical command (VALIDATED):**
```bash
uv run python -m buttermilk.debug.ws_debug_cli logs -n 20
```

**Alternative log levels (VALIDATED):**
```bash
# Show only errors and warnings
uv run python -m buttermilk.debug.ws_debug_cli logs -n 20 -l ERROR

# Show more detail with DEBUG level
uv run python -m buttermilk.debug.ws_debug_cli logs -n 50 -l DEBUG
```

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

**✅ VALIDATED EVIDENCE**:
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
- **VISIBLE and VALID output from each agent stage** ✓
- Session shows completion, not hanging ✓

### Validated ws_debug_cli Commands

**✅ VALIDATED: All commands working**

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
    # CRITICAL: Use ONLY validated record IDs and criteria combinations
    # These examples use confirmed working live data:
    uv run python -m buttermilk.debug.ws_debug_cli start trans --record "kerri_colby_children_transitioning" --criteria "tja" --wait 10
    uv run python -m buttermilk.debug.ws_debug_cli start trans --record "betoota_snape_trans" --criteria "glaad" --wait 10
    ```
    **Expected Evidence**: Message count increases from 0 to 20+ messages in logs
    
    **❌ CRITICAL**: Flows WILL FAIL with arbitrary parameters. Do NOT use placeholder values - parameters must match existing live data configuration.

*   **List Recent Log Files (NEW):**
    ```bash
    uv run python -m buttermilk.debug.ws_debug_cli list-logs -n 5
    ```
    **Expected Evidence**: Shows the 5 most recent Buttermilk log files with timestamps and sizes

*   **View Recent Logs (VALIDATED):**
    ```bash
    # Default: reads from most recent bm_*.jsonl file
    uv run python -m buttermilk.debug.ws_debug_cli logs -n 20

    # Specify a specific log file
    uv run python -m buttermilk.debug.ws_debug_cli logs -n 20 --file /tmp/bm_project_session-id.jsonl
    ```
    **Expected Evidence**: Shows structured JSONL log entries from `/tmp/bm_*.jsonl` files

    **Note**: Log files now use the prefix `bm_` and follow the format: `bm_{project_name}_{execution_context_id}.jsonl`

*   **Monitor Structured Logs Directly (VALIDATED):**
    ```bash
    tail -f /tmp/bm_*.jsonl
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

### Log Analysis Tips

**Problem**: Log outputs are too verbose for analysis.
**Solution**: 
- Use level filtering: `uv run python -m buttermilk.debug.ws_debug_cli logs -n 50 -l ERROR`
- Limit output with `-n` parameter: `logs -n 20` for recent entries
- Focus on specific timeframes when issues occurred
- **Follow OUTPUT RULE**: Summarize findings instead of dumping raw logs


### Simplified Debugging Rules

**✅ DO (VALIDATED WORKFLOW):**
- Use `ws_debug_cli list-logs` to see recent log files (✅ Shows 5 most recent `bm_*.jsonl` files)
- Use `ws_debug_cli logs` for log access (✅ Accesses `/tmp/bm_*.jsonl` files with `bm_` prefix)
- Use `ws_debug_cli logs --file <path>` to read specific log files (✅ Accepts file parameter)
- Use `ws_debug_cli test-connection` before debugging flows (✅ Validates WebSocket)
- Use `curl http://localhost:8000/health` to verify API status (✅ Returns expected JSON)
- Check structured logs first for setup issues (✅ ConfigurationBootstrapper creates them)
- Monitor flow execution via message count increases (✅ 0 to 20+ messages confirmed)

**❌ DON'T:**
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

