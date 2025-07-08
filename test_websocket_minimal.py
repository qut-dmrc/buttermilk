#!/usr/bin/env python
"""Minimal WebSocket test to debug connection issues"""

import asyncio
import websockets
import json

async def test():
    uri = "ws://localhost:8000/ws/minimal_test"
    
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            
            # Send a simple message
            data = {"flow": "osb", "prompt": "test", "ui_type": "test"}
            print(f"Sending: {json.dumps(data)}")
            await websocket.send(json.dumps(data))
            
            # Try to receive something
            print("Waiting for response...")
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"Received: {response}")
            except asyncio.TimeoutError:
                print("No response received within 5 seconds")
                
    except Exception as e:
        print(f"Error: {e}")

# First start the server manually:
print("Make sure the server is running:")
print("uv run python -m buttermilk.runner.cli +flows=[trans,zot,osb] +run=api +llms=full")
input("\nPress Enter when ready...")

asyncio.run(test())