# Buttermilk Debugging Guide: The Golden Path

## Overview

This guide provides the single, authoritative workflow for debugging and validating Buttermilk. It follows the "golden path" principle: a simple, clear, and powerful set of tools for the most common development tasks. This is the only debugging guide you need.

## The Simplified Golden Path Workflow

**Two Core Tools, One Simple Workflow:**

1. **Log Analysis**: Use `ws_debug_cli.py` for structured log access
2. **Live Flow Debugging**: Use enhanced `DebugAgent` as interactive "puppet" UI

**Standard debugging loop:**

1.  **Start the Server**: Launch the backend API.
2.  **Check Logs First**: Use structured log tools to diagnose setup issues.
3.  **Debug Live Flows**: Use DebugAgent puppet mode for interactive flow debugging.
4.  **Stop the Server**: Terminate the backend process.

---

## 1. Start the Server

Use the `make debug` command to start the Buttermilk API server in the background. This is the standard way to launch the backend for development.

```bash
make debug
```

This command handles killing any old processes and starts a new one, logging output to a file in `/tmp/`.

---

## 2. Check Logs First (Setup Issues)

**For environment and setup issues, ALWAYS check logs first.**

Use the structured log tool for reliable log access:

**Canonical command:**
```bash
uv run python -m buttermilk.debug.ws_debug_cli logs -n 50
```

**Alternative log levels:**
```bash
# Show only errors and warnings
uv run python -m buttermilk.debug.ws_debug_cli logs -n 50 -l ERROR

# Show more detail with DEBUG level
uv run python -m buttermilk.debug.ws_debug_cli logs -n 100 -l DEBUG
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

**✅ REQUIRED (End-to-End Completion)**:
- Flow runs through ALL agents sequentially ✓
- fetch agent retrieves real data ✓
- judge agent processes and scores data ✓
- synth agent synthesizes findings ✓
- scorer agent validates results ✓
- diff agent compares outputs ✓
- **VISIBLE OUTPUT from each agent stage** ✓
- Session shows completion, not hanging ✓

### Quick ws_debug_cli Examples

*   **Test Connection:**
    ```bash
    uv run python -m buttermilk.debug.ws_debug_cli test-connection
    ```

*   **Start a Flow:**
    ```bash
    uv run python -m buttermilk.debug.ws_debug_cli start trans --record "your_record" --criteria "cte" --wait 60
    ```

*   **Send a Response:**
    ```bash
    uv run python -m buttermilk.debug.ws_debug_cli send "your response text" --session <session_id>
    ```

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
- Use the recommended tool: `ws_debug_cli logs` instead
- The script now includes a deprecation notice pointing to proper tools

### Simplified Debugging Rules

**✅ DO:**
- Use `ws_debug_cli logs` for all log access
- Use DebugAgent puppet mode for interactive flow debugging  
- Test connections before debugging flows
- Check logs first for setup issues

**❌ DON'T:**
- Use `scripts/view-logs.sh` in interactive mode (use option 4 for proper tools)
- Create standalone debugging scripts (use existing tools)
- Use deprecated commands or broken legacy tools

### Output Conciseness Guidelines

When using debugging tools, agents must:
- **Extract key findings** instead of showing full command output
- **Limit excerpts** to 10-15 lines maximum per tool invocation
- **Summarize patterns** rather than listing individual log entries
- **Highlight specific errors** or success indicators only
- **Use bullet points** for key findings rather than prose explanations