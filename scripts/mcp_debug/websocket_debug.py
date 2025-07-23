#!/usr/bin/env python3
"""Standalone WebSocket Debug Client for Buttermilk

A WebSocket client for debugging Buttermilk flows without dependencies on Buttermilk code.
This script is used by MCP tools to interact with running flows.
"""

import asyncio
import json
import sys
import time
import aiohttp
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

try:
    import websockets
except ImportError:
    print("ERROR: websockets library required. Install with: pip install websockets")
    sys.exit(1)


class WebSocketDebugClient:
    """Standalone WebSocket client for debugging Buttermilk flows."""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.session_file = Path(tempfile.gettempdir()) / "buttermilk_debug_session.json"
        self.messages: List[Dict[str, Any]] = []
        
    async def get_session(self) -> Optional[str]:
        """Get a new session ID from the API."""
        url = f"{self.base_url}/api/session"
        
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
            
    def save_session(self, session_id: str):
        """Save session ID to file for reuse."""
        session_data = {
            "session_id": session_id,
            "host": self.host,
            "port": self.port,
            "timestamp": datetime.now().isoformat()
        }
        self.session_file.write_text(json.dumps(session_data, indent=2))
        
    def load_session(self) -> Optional[str]:
        """Load session ID from file if it exists."""
        if self.session_file.exists():
            try:
                data = json.loads(self.session_file.read_text())
                # Check if session is from same server
                if data.get("host") == self.host and data.get("port") == self.port:
                    return data.get("session_id")
            except Exception:
                pass
        return None
        
    def clear_session(self):
        """Clear saved session."""
        if self.session_file.exists():
            self.session_file.unlink()
            
    async def start_flow(self, flow_name: str, query: str, wait_time: int = 10) -> Dict[str, Any]:
        """Start a flow and wait for initial messages."""
        # Get or create session
        session_id = await self.get_session()
        if not session_id:
            return {"error": "Failed to get session ID"}
            
        self.save_session(session_id)
        
        # Connect and start flow
        uri = f"{self.ws_url}/{session_id}"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Send flow start message
                start_msg = {
                    "type": "run_flow",
                    "flow_name": flow_name,
                    "content": query
                }
                
                await websocket.send(json.dumps(start_msg))
                self.messages = []
                
                # Collect messages for specified time
                start_time = asyncio.get_event_loop().time()
                
                while asyncio.get_event_loop().time() - start_time < wait_time:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        self.messages.append(data)
                        
                        # Check for completion
                        if data.get("type") in ["flow_complete", "error"]:
                            break
                            
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        break
                        
                return {
                    "session_id": session_id,
                    "flow_name": flow_name,
                    "messages_count": len(self.messages),
                    "messages": self.messages
                }
                
        except Exception as e:
            return {"error": str(e)}
            
    async def send_message(self, content: str, message_type: str = "manager_response") -> Dict[str, Any]:
        """Send a message to the current session."""
        session_id = self.load_session()
        if not session_id:
            return {"error": "No active session. Start a flow first."}
            
        uri = f"{self.ws_url}/{session_id}"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Send message
                msg = {
                    "type": message_type,
                    "content": content
                }
                
                await websocket.send(json.dumps(msg))
                
                # Wait briefly for any immediate responses
                messages = []
                start_time = asyncio.get_event_loop().time()
                
                while asyncio.get_event_loop().time() - start_time < 2:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                        data = json.loads(message)
                        messages.append(data)
                    except asyncio.TimeoutError:
                        break
                    except websockets.exceptions.ConnectionClosed:
                        break
                        
                return {
                    "session_id": session_id,
                    "sent": msg,
                    "responses": messages
                }
                
        except Exception as e:
            return {"error": str(e)}
            
    async def wait_for_messages(self, pattern: Optional[str] = None, 
                              message_type: Optional[str] = None, 
                              wait_time: int = 30) -> Dict[str, Any]:
        """Wait for messages matching criteria."""
        session_id = self.load_session()
        if not session_id:
            return {"error": "No active session. Start a flow first."}
            
        uri = f"{self.ws_url}/{session_id}"
        matching_messages = []
        
        try:
            async with websockets.connect(uri) as websocket:
                start_time = asyncio.get_event_loop().time()
                
                while asyncio.get_event_loop().time() - start_time < wait_time:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        
                        # Check if message matches criteria
                        matches = True
                        
                        if message_type and data.get("type") != message_type:
                            matches = False
                            
                        if pattern and pattern not in json.dumps(data):
                            matches = False
                            
                        if matches:
                            matching_messages.append(data)
                            
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        break
                        
                return {
                    "session_id": session_id,
                    "wait_time": wait_time,
                    "pattern": pattern,
                    "message_type": message_type,
                    "matching_messages": matching_messages
                }
                
        except Exception as e:
            return {"error": str(e)}
            
    async def test_connection(self) -> Dict[str, Any]:
        """Test WebSocket connection to server."""
        try:
            # Test HTTP endpoint
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    http_ok = response.status == 200
                    
            # Test WebSocket
            session_id = await self.get_session()
            ws_ok = session_id is not None
            
            return {
                "host": self.host,
                "port": self.port,
                "http_endpoint": http_ok,
                "websocket_endpoint": ws_ok,
                "status": "connected" if (http_ok and ws_ok) else "failed"
            }
            
        except Exception as e:
            return {
                "host": self.host,
                "port": self.port,
                "error": str(e),
                "status": "failed"
            }


async def main():
    """Main entry point for the WebSocket debug client."""
    if len(sys.argv) < 2:
        print("Usage: websocket_debug.py <command> [args...]")
        print("Commands:")
        print("  start <flow_name> <query> [wait_time] - Start a flow")
        print("  send <content> [type] - Send a message to active session")
        print("  wait [pattern] [type] [wait_time] - Wait for messages")
        print("  test - Test connection to server")
        print("  session - Show current session")
        print("  clear - Clear saved session")
        print("")
        print("Examples:")
        print('  websocket_debug.py start osb "What is AI?"')
        print('  websocket_debug.py send "Tell me more"')
        print('  websocket_debug.py wait "task_complete" agent_message 30')
        sys.exit(1)
        
    command = sys.argv[1]
    client = WebSocketDebugClient()
    
    if command == "start":
        if len(sys.argv) < 4:
            print("Usage: websocket_debug.py start <flow_name> <query> [wait_time]")
            sys.exit(1)
            
        flow_name = sys.argv[2]
        query = sys.argv[3]
        wait_time = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        
        result = await client.start_flow(flow_name, query, wait_time)
        print(json.dumps(result, indent=2))
        
    elif command == "send":
        if len(sys.argv) < 3:
            print("Usage: websocket_debug.py send <content> [type]")
            sys.exit(1)
            
        content = sys.argv[2]
        msg_type = sys.argv[3] if len(sys.argv) > 3 else "manager_response"
        
        result = await client.send_message(content, msg_type)
        print(json.dumps(result, indent=2))
        
    elif command == "wait":
        pattern = sys.argv[2] if len(sys.argv) > 2 else None
        msg_type = sys.argv[3] if len(sys.argv) > 3 else None
        wait_time = int(sys.argv[4]) if len(sys.argv) > 4 else 30
        
        result = await client.wait_for_messages(pattern, msg_type, wait_time)
        print(json.dumps(result, indent=2))
        
    elif command == "test":
        result = await client.test_connection()
        print(json.dumps(result, indent=2))
        
    elif command == "session":
        session_id = client.load_session()
        if session_id:
            print(json.dumps({
                "session_id": session_id,
                "host": client.host,
                "port": client.port
            }, indent=2))
        else:
            print(json.dumps({"status": "No active session"}, indent=2))
            
    elif command == "clear":
        client.clear_session()
        print(json.dumps({"status": "Session cleared"}, indent=2))
        
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)