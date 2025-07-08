#!/usr/bin/env python
"""Simple test to debug WebSocket connection and message flow."""

import asyncio
import json
import websockets
from buttermilk._core.types import RunRequest

async def test_websocket():
    uri = "ws://localhost:8000/ws/test_session"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"[CLIENT] Connected to {uri}")
            
            # Send RunRequest
            run_request = RunRequest(
                flow="osb",
                prompt="Tell me about the latest news.",
                ui_type="test",
            )
            request_json = run_request.model_dump_json()
            print(f"[CLIENT] Sending RunRequest: {request_json}")
            await websocket.send(request_json)
            
            # Wait for messages
            print("[CLIENT] Waiting for messages...")
            for i in range(10):  # Try to receive up to 10 messages
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    msg_data = json.loads(message)
                    print(f"[CLIENT] Message {i+1}: type='{msg_data.get('type')}', preview='{msg_data.get('preview', '')[:50]}'")
                    if msg_data.get('outputs'):
                        print(f"[CLIENT]   outputs content: {str(msg_data['outputs'].get('content', ''))[:100]}")
                except asyncio.TimeoutError:
                    print(f"[CLIENT] No message received after 2 seconds")
                    break
                except Exception as e:
                    print(f"[CLIENT] Error receiving message: {e}")
                    break
            
            print("[CLIENT] Test complete")
            
    except Exception as e:
        print(f"[CLIENT] Connection error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())