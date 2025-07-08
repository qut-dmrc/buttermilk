#!/usr/bin/env python3
"""Simple test to verify full flow execution."""

import asyncio
import json
import websockets
import sys

async def test_flow():
    uri = "ws://localhost:8000/ws/test_simple_flow"
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as websocket:
        print("Connected!")
        
        # Start flow with a simple prompt
        # Start with empty prompt
        start_msg = {
            "type": "run_flow",
            "flow": "osb",
            "record_id": "",
            "prompt": ""
        }
        await websocket.send(json.dumps(start_msg))
        print("Sent empty RunRequest to initialize flow")
        
        # Wait for initialization
        initialized = False
        for _ in range(10):
            try:
                msg_str = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                msg = json.loads(msg_str)
                if msg.get("type") == "system_message":
                    content = msg.get("outputs", {}).get("content", "")
                    if "Setting up AutogenOrchestrator" in content:
                        print(f"✓ Flow initialized: {content}")
                        initialized = True
                        break
            except asyncio.TimeoutError:
                continue
        
        if not initialized:
            print("ERROR: Flow did not initialize properly")
            return
        
        # Now send the actual user message
        await asyncio.sleep(2)  # Brief pause
        print("\nSending user message...")
        user_msg = {
            "type": "manager_response",
            "content": "What is digital constitutionalism?",
            "confirm": False,
            "halt": False,
            "interrupt": False,
            "human_in_loop": None,
            "selection": None,
            "error": [],
            "metadata": {},
            "params": None
        }
        await websocket.send(json.dumps(user_msg))
        print("Sent user message")
        
        # Collect messages for 60 seconds
        messages = []
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < 60:
            try:
                msg_str = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                msg = json.loads(msg_str)
                messages.append(msg)
                
                msg_type = msg.get("type", "unknown")
                preview = msg.get("preview", "")
                
                if msg_type == "system_message":
                    content = msg.get("outputs", {}).get("content", "")
                    print(f"[SYSTEM] {content}")
                elif msg_type == "agent_output":
                    content = msg.get("outputs", {}).get("content", "")
                    print(f"[AGENT] {content[:100]}...")
                elif msg_type == "ui_message":
                    content = msg.get("outputs", {}).get("content", "")
                    print(f"[UI] {content[:100]}...")
                elif msg_type == "research_result":
                    print(f"[RESULT] Research result received!")
                else:
                    print(f"[{msg_type.upper()}] {preview[:100] if preview else 'No preview'}")
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error: {e}")
                break
        
        print(f"\n--- Summary ---")
        print(f"Total messages received: {len(messages)}")
        for msg in messages:
            print(f"- {msg.get('type', 'unknown')}")
        
        # Check if we got meaningful responses
        agent_outputs = [m for m in messages if m.get("type") == "agent_output"]
        if agent_outputs:
            print(f"\nFound {len(agent_outputs)} agent outputs!")
            for output in agent_outputs[:2]:  # Show first 2
                content = output.get("outputs", {}).get("content", "")
                print(f"Content preview: {content[:200]}...")

if __name__ == "__main__":
    asyncio.run(test_flow())