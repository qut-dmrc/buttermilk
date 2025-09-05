# Buttermilk Debugging Guide: The Golden Path

## Overview

This guide provides the single, authoritative workflow for debugging and validating Buttermilk. It follows the "golden path" principle: a simple, clear, and powerful set of tools for the most common development tasks. This is the only debugging guide you need.

## The Golden Path Workflow

The standard debugging loop consists of five steps, each with a single, recommended tool.

1.  **Start the Server**: Launch the backend API.
2.  **Validate the Backend**: Run a flow and observe its live output.
3.  **Analyze the Logs**: Inspect the logs for errors and details.
4.  **Validate the Frontend**: Interact with the UI to confirm visual and functional correctness.
5.  **Stop the Server**: Terminate the backend process.

---

## 1. Start the Server

Use the `make debug` command to start the Buttermilk API server in the background. This is the standard way to launch the backend for development.

```bash
make debug
```

This command handles killing any old processes and starts a new one, logging output to a file in `/tmp/`.

---

## 2. Validate the Backend (Live Run)

To interact with flows, use the primary WebSocket debug client: `ws_debug_cli.py`. This is the most powerful tool for observing the system's live behavior.

**Canonical command:**
```bash
uv run python -m buttermilk.debug.ws_debug_cli <command>
```

### Available Commands

**Flow Control:**
- `start <flow_name> [query]` - Start a flow with optional initial query
- `send <text>` - Send a response to the current flow
- `logs -n <number>` - Show last n lines from latest log file

**Session Control:**
- `clear-session` - Clear message history
- `list-flows` - Get available flows
- `help` - Show available commands

**❌ DEPRECATED COMMANDS:**
- `export <file>` - Command removed, use other export methods
- `logs <number>` - Wrong syntax, use `logs -n <number>` instead

*   **Test Connection:**
    ```bash
    uv run python -m buttermilk.debug.ws_debug_cli test-connection
    ```

*   **Start a Flow:**
    ```bash
    # Usage: uv run python -m buttermilk.debug.ws_debug_cli start <flow_name> --record <record_id> --criteria <criteria>
    uv run python -m buttermilk.debug.ws_debug_cli start trans --record "snape_betoota_trans" --criteria "cte" --wait 60
    ```
    This will return a `session_id` for use in other commands.
    
    **Note**: The `--wait` option requires a numeric value (seconds). Default is 60 seconds if omitted.

*   **Send a Message to a Flow:**
    ```bash
    uv run python -m buttermilk.debug.ws_debug_cli send "what is digital constitutionalism?" --session <session_id> [--wait <seconds>]
    ```

*   **Wait for/Monitor Messages:**
    ```bash
    uv run python -m buttermilk.debug.ws_debug_cli wait --session <session_id>
    ```

## 3. Analyze the Logs

**⚠️ CRITICAL WARNING: Do not use `scripts/view-logs.sh` directly - it hangs indefinitely**

**Preferred method**: Use ws_debug_cli for log access:
```bash
uv run python -m buttermilk.debug.ws_debug_cli logs -n 30
```

**Alternative**: Direct log file access (if ws_debug_cli fails):
```bash
# Find latest log file first
ls -la /tmp/buttermilk_*.log | tail -1
# Then view specific lines
tail -n 50 /path/to/latest/log/file
```

**❌ NEVER USE**: `scripts/view-logs.sh` - This command hangs and violates debugging workflow


## 4. Validate the Frontend

To validate the web interface, use the official Playwright MCP tool. This allows you to automate browser actions and inspect the UI.

The Playwright tool provides commands like `navigate`, `screenshot`, `click`, and `fill`. You must use these commands to interact with the frontend at `http://localhost:5173`.

**Example Workflow:**

1.  **Navigate to the page:** Use `navigate` to go to `http://localhost:5173/terminal`.
2.  **Take a screenshot:** Use `screenshot` to capture the initial state.
3.  **Interact with elements:** Use `click` and `fill` to select a flow, record, and criteria.
4.  **Run the flow:** Use `click` on the "Run Flow" button.
5.  **Observe results:** Use `screenshot` and `evaluate` to check if the output appears correctly in the UI.

## 5. Stop the Server

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
3. Check logs with correct syntax: `uv run python -m buttermilk.debug.ws_debug_cli logs -n 30`

**Problem**: Command syntax errors or "command not found".
**Solution**: 
- Use `logs -n <number>` not `logs <number>`
- Check available commands with `help`
- Verify command exists before using (some commands have been deprecated)

**Problem**: Stale log data returned.
**Solution**: 
- Check if log files are being created for current date
- Verify session is actually running and generating logs
- May indicate infrastructure regression (see GitHub issues #226, #227)

### Playwright Browser Issues

**Problem**: Browser installation warnings or "browser not found" errors.
**Solution**: 
- Warnings about browser downloads are usually non-critical - the MCP tool often works despite warnings
- If screenshots fail, verify the server is accessible at `http://localhost:5173`
- The Playwright MCP handles browser installation automatically

### Log Analysis Tips

**Problem**: Log outputs are too verbose for analysis.
**Solution**: 
- Use focused searches with ws_debug_cli: `uv run python -m buttermilk.debug.ws_debug_cli logs -n 100 | grep "ERROR\|WebSocket"`
- Limit output with `-n` parameter: `logs -n 20` for recent entries
- Focus on specific timeframes when the issue occurred
- **Follow OUTPUT RULE**: Summarize findings instead of dumping raw logs

**❌ AVOID**: Any commands that pipe from `scripts/view-logs.sh` - use ws_debug_cli instead

### Hanging Commands Prevention

**🚨 CRITICAL: Commands That Will Hang Your Session**

These commands will hang indefinitely and violate debugging workflow:
- `scripts/view-logs.sh` - Hangs indefinitely, use `ws_debug_cli logs -n X` instead
- `tail -f /path/to/log` - Follow mode hangs, use `tail -n X` for specific line count
- Any command with continuous monitoring without timeout

**✅ Safe Alternatives:**
- Instead of `scripts/view-logs.sh`: Use `ws_debug_cli logs -n 30`
- Instead of `tail -f logfile`: Use `tail -n 50 logfile` for snapshot
- Always use commands with explicit limits and timeouts

**🛑 If a Command Hangs:**
1. Stop immediately - don't wait to see if it completes
2. Use the timeout mechanisms in the environment
3. Switch to the safe alternative documented above
4. Never proceed with hanging commands "just to see what happens"

### Output Conciseness Guidelines

When using debugging tools, agents must:
- **Extract key findings** instead of showing full command output
- **Limit excerpts** to 10-15 lines maximum per tool invocation
- **Summarize patterns** rather than listing individual log entries
- **Highlight specific errors** or success indicators only
- **Use bullet points** for key findings rather than prose explanations