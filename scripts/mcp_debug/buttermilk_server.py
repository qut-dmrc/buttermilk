#!/usr/bin/env python3
"""Buttermilk Server Management Tool

Start, stop, and manage the Buttermilk API server.
This script is used by MCP tools to control the server.
"""

import sys
import subprocess
import time
import os
import signal
import json
import psutil
import requests
from typing import Optional, List


class ButtermilkServer:
    def __init__(self):
        self.process_name = "buttermilk.runner.cli"
        
    def find_process(self) -> Optional[int]:
        """Find the Buttermilk server process."""
        for proc in psutil.process_iter(['pid', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and self.process_name in ' '.join(cmdline):
                    return proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
        
    def start(self, debug: bool = False, flows: str = "trans,zot,osb") -> bool:
        """Start the Buttermilk server."""
        if self.find_process():
            print("⚠️ Server already running")
            return True
            
        # Convert flows to array format
        flows_list = flows.split(",")
        flows_json = json.dumps(flows_list)
        
        # Build command
        cmd = [
            "uv", "run", "python", "-m", "buttermilk.runner.cli",
            f"+flows={flows_json}",
            "+run=api",
            "llms=debug"
        ]
        
        if debug:
            cmd.append("verbose=true")
            print(f"Starting server in debug mode...")
            print(f"Flows: {flows}")
            print(f"Log location: /tmp/buttermilk_*.log")
        else:
            print(f"Starting server...")
            print(f"Flows: {flows}")
            
        # Start the process
        try:
            # Start in background
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            
            # Wait for startup
            time.sleep(3)
            
            # Check if started
            if self.find_process():
                print("✅ Server started successfully")
                return True
            else:
                print("❌ Failed to start server")
                return False
                
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            return False
            
    def stop(self) -> bool:
        """Stop the Buttermilk server."""
        pid = self.find_process()
        if not pid:
            print("⚠️ Server not running")
            return True
            
        try:
            # Try graceful shutdown
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
            
            # Check if stopped
            if not self.find_process():
                print("✅ Server stopped")
                return True
                
            # Force kill if needed
            print("Forcing shutdown...")
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
            
            if not self.find_process():
                print("✅ Server stopped (forced)")
                return True
            else:
                print("❌ Failed to stop server")
                return False
                
        except Exception as e:
            print(f"❌ Error stopping server: {e}")
            return False
            
    def status(self) -> bool:
        """Check server status."""
        pid = self.find_process()
        
        if pid:
            print(f"Status: Running")
            print(f"PID: {pid}")
            
            # Check if listening
            try:
                response = requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    print("Port 8000: Listening")
                    return True
            except:
                print("Port 8000: Not responding")
                
            return True
        else:
            print("Status: Not running")
            return False
            
    def health(self) -> bool:
        """Check server health."""
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("API Health: OK")
                data = response.json()
                print(json.dumps(data, indent=2))
                return True
            else:
                print(f"API Health: Failed (status {response.status_code})")
                return False
        except requests.exceptions.ConnectionError:
            print("API Health: Failed (connection error)")
            print("Server may not be running")
            return False
        except Exception as e:
            print(f"API Health: Failed ({e})")
            return False
            
    def flows(self) -> bool:
        """List available flows."""
        try:
            response = requests.get("http://localhost:8000/api/flows", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print("Available flows:")
                print(json.dumps(data, indent=2))
                return True
            else:
                print(f"❌ Failed to get flows (status {response.status_code})")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ Cannot retrieve flows (connection error)")
            print("Server may not be running")
            return False
        except Exception as e:
            print(f"❌ Failed to get flows: {e}")
            return False


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: buttermilk_server.py <action> [debug] [flows]")
        print("Actions: start, stop, status, health, flows")
        print("Examples:")
        print("  buttermilk_server.py start")
        print("  buttermilk_server.py start true")
        print("  buttermilk_server.py start true trans,zot")
        sys.exit(1)
        
    action = sys.argv[1]
    server = ButtermilkServer()
    
    if action == "start":
        debug = sys.argv[2].lower() == "true" if len(sys.argv) > 2 else False
        flows = sys.argv[3] if len(sys.argv) > 3 else "trans,zot,osb"
        success = server.start(debug, flows)
        sys.exit(0 if success else 1)
        
    elif action == "stop":
        success = server.stop()
        sys.exit(0 if success else 1)
        
    elif action == "status":
        success = server.status()
        sys.exit(0 if success else 1)
        
    elif action == "health":
        success = server.health()
        sys.exit(0 if success else 1)
        
    elif action == "flows":
        success = server.flows()
        sys.exit(0 if success else 1)
        
    else:
        print(f"❌ Invalid action: {action}")
        print("Valid actions: start, stop, status, health, flows")
        sys.exit(1)


if __name__ == "__main__":
    main()