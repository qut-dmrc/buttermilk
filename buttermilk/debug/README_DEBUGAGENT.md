# DebugAgent - LLM-Driven Debugging Tools

The DebugAgent provides MCP-exposed tools for debugging Buttermilk flows. It's designed to be used by LLMs through the MCP protocol to investigate and debug issues in running flows.

## Available Tools

### Log Reading Tools

1. **get_latest_buttermilk_logs(lines: int = 100)**
   - Returns the last N lines from the most recent buttermilk log file
   - Useful for quickly checking recent errors or activity

2. **list_log_files()**
   - Lists all buttermilk log files with metadata (size, modification time)
   - Helps identify which log files to examine

### WebSocket Testing Tools

3. **start_websocket_client(flow_id: str, host: str = "localhost", port: int = 8000)**
   - Starts a WebSocket client connected to the Buttermilk API
   - Returns session ID and connection status
   - Use unique flow_id to manage multiple clients

4. **send_websocket_message(flow_id: str, message_type: str, content: str = None, flow_name: str = None)**
   - Sends messages through an active WebSocket client
   - message_type can be "run_flow" or "manager_response"
   - For "run_flow": provide flow_name
   - For "manager_response": provide content

5. **get_websocket_messages(flow_id: str, last_n: int = None, message_type: str = None)**
   - Retrieves messages from a WebSocket client
   - Can filter by message type: "ui_message", "agent_announcement", "agent_trace", "error", "flow_event"
   - Returns messages with type, timestamp, content, and data

6. **get_websocket_summary(flow_id: str)**
   - Returns summary of message counts and active agents
   - Useful for understanding flow state at a glance

7. **stop_websocket_client(flow_id: str)**
   - Stops and cleans up a WebSocket client

8. **list_active_clients()**
   - Lists all active WebSocket client IDs

## Usage Example

```python
# Start the MCP server with DebugAgent
from buttermilk.mcp.server import MCPServer
from buttermilk.debug.debug_agent import DebugAgent

# Create and register the debug agent
debug_agent = DebugAgent(agent_name="debug_agent", role="debugger")

# Start MCP server (it will discover the agent's tools)
server = MCPServer()
await server.start()
```

## LLM Usage Pattern

When debugging with an LLM:

1. **Check logs first**:
   ```
   logs = get_latest_buttermilk_logs(200)
   # LLM analyzes logs for errors
   ```

2. **Start a test client**:
   ```
   start_websocket_client("test_flow_1")
   ```

3. **Run a flow**:
   ```
   send_websocket_message("test_flow_1", "run_flow", flow_name="example_flow")
   ```

4. **Monitor messages**:
   ```
   messages = get_websocket_messages("test_flow_1", message_type="error")
   summary = get_websocket_summary("test_flow_1")
   ```

5. **Interact with the flow**:
   ```
   send_websocket_message("test_flow_1", "manager_response", content="Yes, continue")
   ```

6. **Clean up**:
   ```
   stop_websocket_client("test_flow_1")
   ```

## Integration with Existing Infrastructure

The DebugAgent builds on:
- Existing WebSocket test client from integration tests
- Buttermilk's MCP infrastructure for tool discovery and invocation
- Standard agent patterns for easy integration

## Starting the MCP Server

Since Buttermilk doesn't have a standalone MCP server yet, we've created a simple one:

1. **Start the Debug MCP Server**:
   ```bash
   python scripts/run_debug_mcp_server.py
   ```
   
   This starts a server on http://localhost:8090 with:
   - `GET /tools` - List all available debugging tools
   - `POST /tools/{tool_name}` - Execute a specific tool

2. **Test the server**:
   ```bash
   python scripts/test_debug_mcp.py
   ```

3. **Use with an LLM**:
   Configure your LLM client to access the server at http://localhost:8090

## HTTP API Examples

List available tools:
```bash
curl http://localhost:8090/tools
```

Get latest logs:
```bash
curl -X POST http://localhost:8090/tools/get_latest_buttermilk_logs \
  -H "Content-Type: application/json" \
  -d '{"params": {"lines": 50}}'
```

Start a WebSocket client:
```bash
curl -X POST http://localhost:8090/tools/start_websocket_client \
  -H "Content-Type: application/json" \
  -d '{"params": {"flow_id": "test_flow", "host": "localhost", "port": 8000}}'
```

## Next Steps

To use this in practice:

1. Ensure the Buttermilk API server is running (`make debug`)
2. Start the Debug MCP server (`python scripts/run_debug_mcp_server.py`)
3. Configure your LLM client to use the endpoints at http://localhost:8090
4. Start debugging flows interactively through the LLM interface