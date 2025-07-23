#!/usr/bin/env python3
"""Buttermilk WebSocket Flow Tester

Test specific Buttermilk flows via WebSocket.
This script is used by MCP tools to test flows programmatically.
"""

import asyncio
import websockets
import json
import sys
import time
from typing import Optional, Dict, Any, List


class WebSocketFlowTester:
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.messages: List[Dict[str, Any]] = []
        
    async def test_flow(self, session_id: str, flow: str, prompt: str, timeout: int = 30) -> int:
        """Test a flow via WebSocket connection."""
        uri = f"ws://{self.host}:{self.port}/ws/{session_id}"
        
        try:
            async with websockets.connect(uri) as websocket:
                print(f"Connected to WebSocket at {uri}")
                
                # Send flow start message
                start_msg = {
                    "type": "start_flow",
                    "flow": flow,
                    "inputs": {
                        "prompt": prompt
                    }
                }
                
                await websocket.send(json.dumps(start_msg))
                print(f"Sent: {start_msg}")
                print("")
                
                # Receive messages
                print("Receiving messages:")
                start_time = asyncio.get_event_loop().time()
                
                while True:
                    try:
                        if asyncio.get_event_loop().time() - start_time > timeout:
                            print("\nTimeout reached")
                            break
                            
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        self.messages.append(data)
                        
                        msg_type = data.get("type", "unknown")
                        print(f"[{msg_type}] {json.dumps(data, indent=2)}")
                        
                        if msg_type == "flow_complete" or msg_type == "error":
                            break
                            
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        print("\nConnection closed")
                        break
                        
        except Exception as e:
            print(f"Error: {e}")
            return 1
        
        return 0
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the test results."""
        summary = {
            "total_messages": len(self.messages),
            "message_types": {},
            "errors": [],
            "completion_status": None
        }
        
        for msg in self.messages:
            msg_type = msg.get("type", "unknown")
            summary["message_types"][msg_type] = summary["message_types"].get(msg_type, 0) + 1
            
            if msg_type == "error":
                summary["errors"].append(msg.get("message", "Unknown error"))
            elif msg_type == "flow_complete":
                summary["completion_status"] = msg.get("status", "unknown")
                
        return summary


async def get_session(host: str = "localhost", port: int = 8000) -> Optional[str]:
    """Get a new session ID from the API."""
    import aiohttp
    
    url = f"http://{host}:{port}/api/session"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("session_id")
                else:
                    print(f"Failed to get session: {response.status}")
                    return None
    except Exception as e:
        print(f"Error getting session: {e}")
        return None


async def main():
    """Main entry point for the WebSocket flow tester."""
    if len(sys.argv) < 3:
        print("Usage: test_websocket_flow.py <flow> <prompt> [host] [port] [timeout]")
        print("  flow: Flow name to test (e.g., zot, trans, osb)")
        print("  prompt: Test prompt/query")
        print("  host: Server host (default: localhost)")
        print("  port: Server port (default: 8000)")
        print("  timeout: Timeout in seconds (default: 30)")
        sys.exit(1)
        
    flow = sys.argv[1]
    prompt = sys.argv[2]
    host = sys.argv[3] if len(sys.argv) > 3 else "localhost"
    port = int(sys.argv[4]) if len(sys.argv) > 4 else 8000
    timeout = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    
    print(f"🧪 Buttermilk WebSocket Flow Tester")
    print(f"==================================")
    print(f"Flow: {flow}")
    print(f"Prompt: {prompt}")
    print(f"Server: {host}:{port}")
    print(f"Timeout: {timeout}s")
    print("")
    
    # Get session ID
    print("Getting session ID...")
    session_id = await get_session(host, port)
    
    if not session_id:
        print("❌ Failed to get session ID")
        sys.exit(1)
        
    print(f"Session ID: {session_id}")
    print("")
    
    # Test the flow
    tester = WebSocketFlowTester(host, port)
    exit_code = await tester.test_flow(session_id, flow, prompt, timeout)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    summary = tester.get_summary()
    print(json.dumps(summary, indent=2))
    
    if summary["errors"]:
        print(f"\n❌ Test failed with {len(summary['errors'])} error(s)")
        exit_code = 1
    elif summary["completion_status"] == "success":
        print(f"\n✅ Test completed successfully")
        exit_code = 0
    else:
        print(f"\n⚠️ Test completed with status: {summary['completion_status']}")
        exit_code = 1
        
    sys.exit(exit_code)


if __name__ == "__main__":
    # Handle asyncio running
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)