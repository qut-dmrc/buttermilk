#!/usr/bin/env python
"""Debug script to test WebSocket message flow"""

import asyncio
import websockets
import json
from buttermilk._core.types import RunRequest

async def test_websocket():
    uri = "ws://localhost:8000/ws/debug_test_session"
    
    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket")
        
        # Send RunRequest
        run_request = RunRequest(
            flow="osb",
            prompt="Test message",
            ui_type="test",
        )
        
        print(f"Sending RunRequest: {run_request.model_dump_json()}")
        await websocket.send(run_request.model_dump_json())
        
        # Try to receive messages for 10 seconds
        print("\nWaiting for messages...")
        try:
            while True:
                message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                print(f"\nReceived message: {message}")
                msg_json = json.loads(message)
                print(f"Message type: {msg_json.get('type')}")
                print(f"Message content: {json.dumps(msg_json, indent=2)}")
        except asyncio.TimeoutError:
            print("\nNo more messages received (timeout)")
        except websockets.exceptions.ConnectionClosed:
            print("\nConnection closed")

if __name__ == "__main__":
    # Make sure the server is running first
    print("Make sure the server is running: uv run python -m buttermilk.runner.cli +flows=[trans,zot,osb] +run=api +llms=full")
    input("Press Enter when server is ready...")
    
    asyncio.run(test_websocket())