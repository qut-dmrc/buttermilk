# WebSocket Integration Documentation

## Overview

This document explains how the WebSocket integration works in Buttermilk for interactive flows, particularly with the AutogenOrchestrator.

## Architecture

### 1. Session Management

Sessions are managed by the `SessionManager` class and must be pre-created before WebSocket connection:

```python
# In /api/session endpoint
await flows.session_manager.get_or_create_session(session_id, websocket=None)
```

### 2. WebSocket Flow

1. Client requests session ID from `/api/session`
2. Client connects to WebSocket at `/ws/{session_id}`
3. Client sends `run_flow` message to start a flow
4. Orchestrator initializes and sets up agents
5. Client can send `manager_response` messages for interaction

### 3. Message Types

#### From Client to Server:
- `run_flow`: Start a new flow with parameters
- `manager_response`: User response in groupchat
- `ui_message`: General UI message

#### From Server to Client:
- `agent_announcement`: Agent joining/leaving notifications
- `ui_message`: Messages for display
- `agent_trace`: Agent execution results
- `flow_complete`: Flow completion notification

## Known Issues

### Race Condition with AutogenOrchestrator

**Issue**: When sending a `manager_response` too quickly after starting a flow, you may get:
```
'AutogenOrchestrator' object has no attribute '_runtime'
```

**Cause**: The AutogenOrchestrator creates its `_runtime` in the `_setup` method, but the callback 
(`make_publish_callback`) is created before `_setup` completes. If a message arrives during this 
initialization window, it fails.

**Workaround**: Wait for the orchestrator to send an initial UI message before sending user responses.
The orchestrator will typically send a prompt or confirmation request when ready.

### MANAGER Agent

The AutogenOrchestrator automatically creates a MANAGER agent using a ClosureAgent. You don't need to 
configure one explicitly. This agent handles UI communication through the registered callback.

## Example WebSocket Client

```python
import asyncio
import aiohttp
import json

async def interact_with_flow():
    async with aiohttp.ClientSession() as session:
        # Get session
        resp = await session.get("http://localhost:8000/api/session")
        session_id = (await resp.json())["sessionId"]
        
        # Connect WebSocket
        async with session.ws_connect(f"ws://localhost:8000/ws/{session_id}") as ws:
            # Start flow
            await ws.send_json({
                "type": "run_flow",
                "flow": "osb",
                "prompt": "Your question here"
            })
            
            # Wait for orchestrator to be ready
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    
                    # Look for UI messages asking for input
                    if data.get("type") == "ui_message":
                        content = data.get("data", {}).get("content", "")
                        if "confirm" in content.lower():
                            # Orchestrator is ready, send response
                            await ws.send_json({
                                "type": "manager_response",
                                "content": "yes"
                            })
```

## Best Practices

1. **Always wait for orchestrator readiness** before sending user messages
2. **Handle WebSocket disconnections** gracefully
3. **Monitor for agent announcements** to understand flow state
4. **Use proper message types** for different interactions
5. **NEVER provide multiple case formats in API responses** - This violates the principle of having one clear way to do things. Choose either camelCase or snake_case and stick with it consistently:
   ```python
   # DON'T DO THIS - it creates confusion and ambiguity
   return JSONResponse({
       "sessionId": new_session_id,
       "session_id": new_session_id,  # Also include snake_case for compatibility
   })
   
   # DO THIS - pick one format and use it consistently
   return JSONResponse({
       "sessionId": new_session_id,
   })
   ```

## Future Improvements

1. The `make_publish_callback` should handle the case where `_runtime` doesn't exist yet
2. The orchestrator should send an explicit "ready" message when initialization is complete
3. Consider queueing messages that arrive during initialization