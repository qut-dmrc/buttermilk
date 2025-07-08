#!/usr/bin/env python
"""Simple test to debug WebSocket message flow"""

import asyncio
import sys

# Start the backend server in a subprocess
async def start_backend():
    process = await asyncio.create_subprocess_exec(
        "uv", "run", "python", "-m", "buttermilk.runner.cli",
        "+flows=[trans,zot,osb]", "+run=api", "+llms=full",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    
    # Wait for server to start
    print("Waiting for server to start...")
    while True:
        line = await process.stderr.readline()
        if not line:
            break
        line = line.decode("utf-8").strip()
        print(f"[backend] {line}")
        if "Uvicorn running on" in line:
            print("Server started!")
            await asyncio.sleep(2)
            break
    
    return process

async def test_client():
    import websockets
    import json
    from buttermilk._core.types import RunRequest
    
    uri = "ws://localhost:8000/ws/simple_test"
    
    async with websockets.connect(uri) as websocket:
        print(f"\n[client] Connected to {uri}")
        
        # Send RunRequest
        run_request = RunRequest(
            flow="osb",
            prompt="Test message",
            ui_type="test",
        )
        
        print(f"[client] Sending RunRequest...")
        await websocket.send(run_request.model_dump_json())
        
        # Wait for messages
        print("[client] Waiting for messages...\n")
        
        timeout_count = 0
        while timeout_count < 3:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"[client] Received message: {message}")
                msg_json = json.loads(message)
                print(f"[client] Message type: {msg_json.get('type')}")
                print(f"[client] Message preview: {msg_json.get('preview', 'No preview')}")
                if msg_json.get('type') == 'system_message':
                    print(f"[client] System message content: {msg_json.get('outputs', {}).get('content')}")
                print("")
            except asyncio.TimeoutError:
                timeout_count += 1
                print(f"[client] Timeout {timeout_count}/3 - no message received")
        
        print("[client] Test complete")

async def main():
    # Start backend
    backend = await start_backend()
    
    try:
        # Run test
        await test_client()
    finally:
        # Cleanup
        print("\nTerminating backend...")
        backend.terminate()
        await backend.wait()

if __name__ == "__main__":
    asyncio.run(main())