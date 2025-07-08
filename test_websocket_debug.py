#!/usr/bin/env python
"""Debug WebSocket connection issue"""

import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/ws/debug_session"
    print(f"Connecting to {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            
            # Send a test message
            test_msg = {
                "flow": "osb",
                "prompt": "test",
                "session_id": "debug_session"
            }
            print(f"Sending: {json.dumps(test_msg)}")
            await websocket.send(json.dumps(test_msg))
            
            # Wait for response
            print("Waiting for response...")
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            print(f"Received: {response}")
            
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())