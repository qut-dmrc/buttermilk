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
- `logs [n]` - Show last n lines from latest log file

**Session Control:**
- `clear-session` - Clear message history
- `list-flows` - Get available flows
- `export <file>` - Export messages to JSON file
- `help` - Show available commands

*   **Test Connection:**
    ```bash
    uv run python -m buttermilk.debug.ws_debug_cli test-connection
    ```

*   **Start a Flow:**
    ```bash
    # Usage: uv run python -m buttermilk.debug.ws_debug_cli start <flow_name> --record <record_id> --criteria <criteria>
    uv run python -m buttermilk.debug.ws_debug_cli start trans --record "snape_betoota_trans" --criteria "cte"
    ```
    This will return a `session_id` for use in other commands.

*   **Send a Message to a Flow:**
    ```bash
    uv run python -m buttermilk.debug.ws_debug_cli send "Your message here" --session <session_id>
    ```

*   **Wait for/Monitor Messages:**
    ```bash
    uv run python -m buttermilk.debug.ws_debug_cli wait --session <session_id>
    ```

## 3. Analyze the Logs

To inspect the detailed logs from the running server, use the `buttermilk_logs.py` script directly.

**Canonical command:**
```bash
scripts/view-logs.sh <command>
```


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