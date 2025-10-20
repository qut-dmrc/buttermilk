# Buttermilk Debugging Guide: The Validated Golden Path

## Overview

This guide provides the single, authoritative workflow for debugging and validating Buttermilk. The golden path workflow is now fully operational with confirmed evidence of working infrastructure.

**Validation Status**: ✅ **FULLY OPERATIONAL**

This is the only debugging guide you need - all commands have been validated and evidence confirmed.

## Valid System Configuration Parameters

**⚠️ CRITICAL: Use Only Valid Parameters**

**Available Flows**:
- `trans` - Transgender journalist ethics research flow
- Other flows from `/buttermilk/conf/flows/`

**Valid Criteria Templates**:
- `tja` - Trans Journalists Association stylebook criteria
- `glaad` - GLAAD media reference criteria
- Other criteria from `/buttermilk/conf/flows/criteria/` and template files

**Record ID Requirements**:
- **MUST** use actual record IDs from your data sources
- **NEVER** use placeholder values like 'demo_record', 'demo', 'test_record'
- **Recommended valid records**: 'betoota_snape_trans', 'kerri_colby_children_transitioning'
- Check your data files or storage configurations for valid record IDs

**❌ CRITICAL**: Flows WILL NOT run with arbitrary parameters. All parameters (record_id, criteria, flow) MUST match live data configuration. This applies to both puppet mode and NonInteractiveDebugClient usage.

**❌ INVALID EXAMPLES** (DO NOT USE):
- Flows: 'simple', 'test hashing', 'demo_flow'
- Records: 'demo_record', 'demo', 'test_record'
- Criteria: 'test', 'demo_criteria'

**✅ VALID EXAMPLES** (CONFIRMED WORKING):
- Flow: 'trans'
- Records: 'betoota_snape_trans', 'kerri_colby_children_transitioning'
- Criteria: 'tja', 'glaad'

## The Simplified Golden Path Workflow

**Two Core Approaches, Validated Workflow:**

1. **Log Analysis**: Use `ws_debug_cli` infrastructure commands for structured log access (✅ Validated)
2. **Live Flow Debugging**: Use DebugAgent puppet mode or NonInteractiveDebugClient (✅ Validated)

**Validated debugging loop:**

1.  **Start the Server**: Launch the backend API (✅ `make debug` confirmed working)
2.  **Check Logs First**: Use structured log tools (✅ `/tmp/bm_*.jsonl` created)
3.  **Test Connectivity**: Verify API health endpoints (✅ Returns `{"status":"ok","message":"Core routes loaded"}`)
4.  **Debug Live Flows**: Execute flows using puppet mode with full message capture (✅ 0 to 20+ messages)
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

## 3. Debug Live Flows

**Two primary approaches for interactive flow debugging:**

1. **CLI Commands** (Simple, JSON-based, LLM-optimized) - **RECOMMENDED for most cases**
2. **DebugAgent Puppet Mode** (Advanced programmatic control)

### 3a. CLI Commands (Recommended for LLMs)

**✅ VALIDATED: Simple JSON in/out interface perfect for LLM control**

All commands output JSON by default for easy parsing. Use `--pretty` flag for human-readable output.

**Start a flow and capture all messages:**
```bash
# JSON output (default - optimized for LLMs)
uv run python -m buttermilk.debug.ws_debug_cli start trans \
  --record betoota_snape_trans \
  --criteria tja > flow_output.json

# Human-readable output
uv run python -m buttermilk.debug.ws_debug_cli --pretty start trans \
  --record betoota_snape_trans \
  --criteria tja
```

**Send a message to the current session:**
```bash
# Send manager response (JSON output)
uv run python -m buttermilk.debug.ws_debug_cli send "approved"

# Send to specific session
uv run python -m buttermilk.debug.ws_debug_cli send "approved" \
  --session abc123-def456-789
```

**Wait for and collect messages:**
```bash
# Wait 30 seconds for messages containing "conclusion"
uv run python -m buttermilk.debug.ws_debug_cli wait \
  --wait 30 \
  --pattern "conclusion"

# Get only ui_message type messages
uv run python -m buttermilk.debug.ws_debug_cli wait \
  --type ui_message \
  --wait 10
```

**Session management:**
```bash
# Show current session info (JSON)
uv run python -m buttermilk.debug.ws_debug_cli session

# Clear saved session
uv run python -m buttermilk.debug.ws_debug_cli clear-session
```

**Key Features:**
- **JSON by default**: All output is structured JSON unless `--pretty` is used
- **Session persistence**: Sessions are automatically saved and reused
- **Complete message capture**: No truncation in JSON output
- **Filtering**: Pattern matching and type filtering for messages
- **Simple interface**: Perfect for LLM tool calling

### 3b. DebugAgent Puppet Mode (Advanced Programmatic Control)

The DebugAgent provides "puppet mode" - a continuous WebSocket client that acts as a UI replacement, allowing programmatic control of flows in real-time with full message capture.

**When to use puppet mode:**
- Need continuous connection across multiple operations
- Building automated testing workflows
- Require event-driven message handling
- Complex multi-step debugging scenarios

### Using DebugAgent Puppet Mode

**Step 1: Start Puppet Mode**
```python
# In an LLM agent context with access to DebugAgent tools
await debug_agent.start_puppet_mode(host="localhost", port=8000)
```

**Step 2: Start a Flow**
```python
# CRITICAL: Use ONLY validated record IDs and criteria combinations
# Example with confirmed working live data:
await debug_agent.puppet_start_flow(
    flow_name="trans",
    prompt="Analyze this article for trans representation",
    record="betoota_snape_trans",  # Must be a valid record ID
    criteria="tja"  # Must be a valid criteria template
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
# Send manager responses to the flow
await debug_agent.puppet_send_response("your response to the flow")
```

**Step 5: Clean Up**
```python
await debug_agent.stop_puppet_mode()
```

### Programmatic Usage via NonInteractiveDebugClient

For Python scripts and automated testing, use the `NonInteractiveDebugClient` class directly:

```python
from buttermilk.debug.ws_debug_cli import NonInteractiveDebugClient

client = NonInteractiveDebugClient(host="localhost", port=8000)

# Start a flow and wait for completion
result = await client.start_flow(
    flow_name="trans",
    query="Analyze this article",
    record="betoota_snape_trans",
    criteria="tja",
    wait_time=60
)

# Result contains full message history
print(f"Flow completed: {result['completed']}")
print(f"Total messages: {len(result['messages'])}")
```

### SUCCESS CRITERIA: Complete Flow Execution

**✅ VALIDATED EVIDENCE**:
- **ConfigurationBootstrapper**: Creates `/tmp/bm_*.jsonl` structured logs ✓
- **API Infrastructure**: Health endpoint returns `{"status":"ok","message":"Core routes loaded"}` ✓
- **Debugging Framework**: Infrastructure commands (`logs`, `list-logs`, `test-connection`) operational ✓
- **Puppet Mode**: DebugAgent provides full message capture and flow control ✓
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
- **Puppet mode captures all messages** without truncation ✓

### Complete ws_debug_cli Command Reference

**✅ ALL COMMANDS AVAILABLE** (Flow commands restored in Issue #274 resolution)

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

*   **List Recent Log Files (VALIDATED):**
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

    # Filter by log level
    uv run python -m buttermilk.debug.ws_debug_cli logs -n 50 -l ERROR
    ```
    **Expected Evidence**: Shows structured JSONL log entries from `/tmp/bm_*.jsonl` files

    **Note**: Log files use the prefix `bm_` and follow the format: `bm_{project_name}_{execution_context_id}.jsonl`

*   **Monitor Structured Logs Directly (VALIDATED):**
    ```bash
    tail -f /tmp/bm_*.jsonl
    ```
    **Expected Evidence**: Real-time JSONL log entries with proper timestamps

**Flow Control Commands** (Restored after Issue #274):

*   **Start Flow (VALIDATED):**
    ```bash
    # Start a flow with JSON output (default)
    uv run python -m buttermilk.debug.ws_debug_cli start trans \
      --record betoota_snape_trans --criteria tja

    # With human-readable output
    uv run python -m buttermilk.debug.ws_debug_cli --pretty start trans \
      --record betoota_snape_trans --criteria tja
    ```
    **Expected Output**: Complete JSON object with session_id, messages array, flow completion status

*   **Send Message (VALIDATED):**
    ```bash
    # Send manager response to current session
    uv run python -m buttermilk.debug.ws_debug_cli send "approved"

    # Send to specific session
    uv run python -m buttermilk.debug.ws_debug_cli send "approved" --session abc-123
    ```
    **Expected Output**: JSON with session_id and all new messages

*   **Wait for Messages (VALIDATED):**
    ```bash
    # Wait and collect all messages
    uv run python -m buttermilk.debug.ws_debug_cli wait --wait 30

    # Filter by pattern
    uv run python -m buttermilk.debug.ws_debug_cli wait --pattern "conclusion" --wait 10
    ```
    **Expected Output**: JSON with filtered messages array

*   **Session Info (VALIDATED):**
    ```bash
    uv run python -m buttermilk.debug.ws_debug_cli session
    ```
    **Expected Output**: JSON with session_id, host, port, timestamp

*   **Clear Session (VALIDATED):**
    ```bash
    uv run python -m buttermilk.debug.ws_debug_cli clear-session
    ```
    **Expected Output**: JSON with status message

**For Advanced Programmatic Control**: Use DebugAgent puppet mode (see section above) or the NonInteractiveDebugClient class.

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
- **Use CLI commands for simple flow debugging** (✅ JSON in/out, perfect for LLMs)
- **Use DebugAgent puppet mode for advanced programmatic control** (✅ Continuous connection)
- **Use NonInteractiveDebugClient for Python automation** (✅ Complete message history)
- Monitor flow execution via message count increases (✅ 0 to 20+ messages confirmed)

**❌ DON'T:**
- Create standalone debugging scripts (use existing validated tools)
- Ignore structured log evidence from ConfigurationBootstrapper
- Use CLI commands or puppet mode without valid record IDs and criteria (flows will fail)
- Use `--json-output` flag (JSON is now the default; use `--pretty` for console output instead)

### Output Conciseness Guidelines

When using debugging tools, agents must:
- **Extract key findings** instead of showing full command output
- **Limit excerpts** to 10-15 lines maximum per tool invocation
- **Summarize patterns** rather than listing individual log entries
- **Highlight specific errors** or success indicators only
- **Use bullet points** for key findings rather than prose explanations

