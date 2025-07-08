#!/usr/bin/env python
"""Minimal test to debug flow startup"""

import asyncio
import websockets
import json

async def test_minimal():
    # Start backend first
    print("Starting backend...")
    backend = await asyncio.create_subprocess_exec(
        "uv", "run", "python", "-m", "buttermilk.runner.cli",
        "+flows=[trans,zot,osb]", "+run=api", "+llms=full",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    
    # Monitor backend output
    async def monitor_backend():
        while True:
            line = await backend.stderr.readline()
            if not line:
                break
            decoded = line.decode("utf-8").strip()
            if decoded:
                print(f"[backend] {decoded}")
            if "Uvicorn running on" in decoded:
                return True
        return False
    
    # Wait for server
    print("Waiting for server...")
    ready = await asyncio.wait_for(monitor_backend(), timeout=30)
    if not ready:
        print("Server failed to start!")
        backend.terminate()
        return
    
    await asyncio.sleep(2)  # Give it time to fully initialize
    
    # Test WebSocket
    print("\nTesting WebSocket connection...")
    uri = "ws://localhost:8000/ws/test_minimal"
    
    try:
        async with websockets.connect(uri) as ws:
            print(f"Connected to {uri}")
            
            # Send run_flow message (like web UI)
            msg = {
                "type": "run_flow",
                "flow": "osb",
                "record_id": "",
                "prompt": "test message"
            }
            print(f"\nSending: {json.dumps(msg)}")
            await ws.send(json.dumps(msg))
            
            # Listen for messages
            print("\nListening for messages...")
            timeout_count = 0
            while timeout_count < 3:
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    print(f"\n✓ Received: {response[:200]}...")
                    data = json.loads(response)
                    print(f"  Type: {data.get('type')}")
                    print(f"  Preview: {data.get('preview', 'N/A')}")
                    if data.get('type') == 'system_message':
                        outputs = data.get('outputs', {})
                        content = outputs.get('content', '')
                        print(f"  System message content: {content}")
                except asyncio.TimeoutError:
                    timeout_count += 1
                    print(f"\n✗ Timeout {timeout_count}/3")
                except Exception as e:
                    print(f"\n✗ Error: {e}")
                    break
                    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("\nTerminating backend...")
        backend.terminate()
        await backend.wait()

if __name__ == "__main__":
    asyncio.run(test_minimal())