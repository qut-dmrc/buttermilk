#!/usr/bin/env python
"""Test with very long wait to see if messages eventually arrive"""

import asyncio
import websockets
import json
import time

async def test_long_wait():
    uri = "ws://localhost:8000/ws/long_wait_test"
    
    print(f"Connecting to {uri}")
    async with websockets.connect(uri) as ws:
        print("Connected!")
        
        # Send run_flow message
        msg = {
            "type": "run_flow",
            "flow": "osb",
            "record_id": "",
            "prompt": "test message"
        }
        print(f"\nSending: {json.dumps(msg)}")
        await ws.send(json.dumps(msg))
        
        # Monitor for a long time
        print("\nMonitoring for messages (will wait up to 60 seconds)...")
        start_time = time.time()
        message_count = 0
        
        while time.time() - start_time < 60:
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                message_count += 1
                elapsed = int(time.time() - start_time)
                print(f"\n[{elapsed}s] Message #{message_count} received!")
                
                try:
                    data = json.loads(response)
                    print(f"  Type: {data.get('type')}")
                    print(f"  Preview: {data.get('preview', 'N/A')[:100]}")
                    
                    if data.get('type') == 'system_message':
                        content = data.get('outputs', {}).get('content', '')
                        print(f"  ✓ System message: {content}")
                        if "Setting up AutogenOrchestrator" in content:
                            print("\n🎉 SUCCESS! Found the setup message!")
                            print(f"   It took {elapsed} seconds from connection")
                            return True
                            
                except json.JSONDecodeError:
                    print(f"  Raw: {response[:100]}")
                    
            except asyncio.TimeoutError:
                # Print a dot every 2 seconds to show we're still waiting
                print(".", end="", flush=True)
                
        print(f"\n\nTest complete. Received {message_count} messages in 60 seconds")
        return False

# Ensure server is running
print("Make sure the server is running:")
print("uv run python -m buttermilk.runner.cli +flows=[trans,zot,osb] +run=api +llms=full")
input("\nPress Enter when ready...")

result = asyncio.run(test_long_wait())
print(f"\nTest {'PASSED' if result else 'FAILED'}")