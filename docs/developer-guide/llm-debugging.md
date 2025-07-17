# LLM-Driven Debugging Guide

This guide explains how AI assistants can debug Buttermilk flows using the available debugging tools.


## Available Tools

- API Server Health Check: `curl -s http://localhost:8000/health | jq .`
- WebSocket Connection Test: `uv run python -m buttermilk.debug.ws_debug_cli test-connection`
- Start Flow Session: `uv run python -m buttermilk.debug.ws_debug_cli --json-output start <flow_name> --wait <seconds>`. Use `flow_name`: `zot`, `osb`, or `trans`. Wait at least 20 seconds for flow to start and agents to initialize. Returns session_id and initial messages.
- Send Interactive Messages on behalf of user to active flow: `uv run python -m buttermilk.debug.ws_debug_cli --json-output send "<message>" --session <session_id> --wait <seconds>`
- Monitor Session Activity: `uv run python -m buttermilk.debug.ws_debug_cli --json-output wait --session <session_id> --wait <seconds>`. Checks for new messages without sending input.
- Start the API server: `uv run python -m buttermilk.runner.cli "+flows=[zot,osb,trans]" +run=api llms=full`
- Stop the API server: `pkill -f "uv run.*buttermilk.runner"`
- Log Analysis: use bash tools to read the most recent log file in `/tmp/buttermilk*.log`

## Expected Working Flow Behavior

1. **Initialization**: Agents announce with configurations and tool definitions
2. **Registration**: HOST registers agents and creates callable tools
3. **Interaction**: USER sends request; HOST responds to user messages by calling appropriate agents
4. **Results**: Agents execute searches/analysis and return results
5. **Coordination**: HOST synthesizes responses and continues conversation

## Debugging Success Criteria

✅ **Infrastructure Working**: Health check passes, WebSocket connects  
✅ **Agents Initialize**: All expected agents announce themselves  
✅ **Tools Available**: HOST has callable tools for coordinating agents  
✅ **Interactive Response**: HOST processes user messages and calls agents  
✅ **Results Generated**: Agents return meaningful research/analysis results
