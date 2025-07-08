#!/usr/bin/env python
"""Debug WebSocket flow startup."""

import asyncio
import json
import websockets

async def test_ws():
    uri = "ws://localhost:8000/ws/debug_session"
    
    async with websockets.connect(uri) as ws:
        print("[CLIENT] Connected")
        
        # Send a RunRequest
        request = {
            "flow": "osb",
            "prompt": "test",
            "record_id": "",
            "session_id": "debug_session",
            "uri": "",
            "records": [],
            "parameters": {},
            "batch_id": None,
            "source": [],
            "mime_type": None,
            "data": None,
            "name": "osb"
        }
        
        print("[CLIENT] Sending request...")
        await ws.send(json.dumps(request))
        
        # Wait for messages
        print("[CLIENT] Waiting for messages...")
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(msg)
                print(f"[CLIENT] Got: type={data.get('type')}, content={data.get('outputs', {}).get('content', '')[:100] if data.get('outputs') else ''}")
                
        except asyncio.TimeoutError:
            print("[CLIENT] Timeout - no more messages")

if __name__ == "__main__":
    print("Make sure the backend is running with:")
    print('  uv run python -m buttermilk.runner.cli "+flows=[osb]" "+run=api" "+llms=full"')
    print("\nPress Enter to continue...")
    input()
    
    asyncio.run(test_ws())