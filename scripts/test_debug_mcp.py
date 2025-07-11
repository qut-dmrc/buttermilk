#!/usr/bin/env python3
"""Test script for the Debug MCP Server.

This demonstrates how an LLM or client would interact with the debug tools.

Usage:
    1. Start the debug MCP server: python scripts/run_debug_mcp_server.py
    2. Run this test: python scripts/test_debug_mcp.py
"""

import requests
import json
import sys
from pathlib import Path

# Server URL
BASE_URL = "http://localhost:8090"


def test_list_tools():
    """Test listing available tools."""
    print("📋 Listing available tools...")
    response = requests.get(f"{BASE_URL}/tools")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Found {len(data['tools'])} tools:")
        for tool in data['tools']:
            print(f"   - {tool['name']}: {tool['description'].split('.')[0]}")
        return True
    else:
        print(f"❌ Failed to list tools: {response.status_code}")
        return False


def test_log_operations():
    """Test log reading operations."""
    print("\n🗄️  Testing log operations...")
    
    # List log files
    print("   Listing log files...")
    response = requests.post(f"{BASE_URL}/tools/list_log_files", json={"params": {}})
    
    if response.status_code == 200:
        data = response.json()
        if data['success']:
            log_files = data['result']
            print(f"   ✅ Found {len(log_files)} log files")
            if log_files:
                print(f"      Most recent: {log_files[0]['path']}")
        else:
            print(f"   ❌ Error: {data['error']}")
            return False
    
    # Get latest logs
    print("   Getting latest logs...")
    response = requests.post(f"{BASE_URL}/tools/get_latest_buttermilk_logs", 
                           json={"params": {"lines": 10}})
    
    if response.status_code == 200:
        data = response.json()
        if data['success']:
            logs = data['result']
            print(f"   ✅ Retrieved {len(logs.splitlines())} lines of logs")
            print(f"      Preview: {logs[:100]}..." if logs else "      (No logs)")
        else:
            print(f"   ❌ Error: {data['error']}")
    
    return True


def test_websocket_operations():
    """Test WebSocket client operations."""
    print("\n🔌 Testing WebSocket operations...")
    
    # Start a client (will fail if no server running, but tests the API)
    print("   Starting WebSocket client...")
    response = requests.post(f"{BASE_URL}/tools/start_websocket_client", 
                           json={"params": {"flow_id": "test_flow", "use_direct_ws": True}})
    
    if response.status_code == 200:
        data = response.json()
        if data['success']:
            result = data['result']
            if result['status'] == 'success':
                print(f"   ✅ Client started: {result['message']}")
                
                # List active clients
                print("   Listing active clients...")
                response = requests.post(f"{BASE_URL}/tools/list_active_clients", 
                                       json={"params": {}})
                if response.status_code == 200:
                    data = response.json()
                    if data['success']:
                        clients = data['result']
                        print(f"   ✅ Active clients: {clients}")
                
                # Stop the client
                print("   Stopping client...")
                response = requests.post(f"{BASE_URL}/tools/stop_websocket_client",
                                       json={"params": {"flow_id": "test_flow"}})
                if response.status_code == 200:
                    data = response.json()
                    if data['success']:
                        print(f"   ✅ Client stopped: {data['result']['message']}")
            else:
                print(f"   ⚠️  Expected failure (no server): {result['message']}")
        else:
            print(f"   ❌ Error: {data['error']}")
    
    return True


def main():
    """Run all tests."""
    print("🧪 Testing Debug MCP Server")
    print(f"📍 Server URL: {BASE_URL}")
    print()
    
    try:
        # Check if server is running
        response = requests.get(BASE_URL, timeout=2)
        if response.status_code != 200:
            print("❌ Server is not responding correctly")
            return 1
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to Debug MCP Server")
        print("💡 Make sure to run: python scripts/run_debug_mcp_server.py")
        return 1
    
    # Run tests
    all_passed = True
    all_passed &= test_list_tools()
    all_passed &= test_log_operations()
    all_passed &= test_websocket_operations()
    
    print("\n" + "="*50)
    if all_passed:
        print("✅ All tests completed successfully!")
    else:
        print("❌ Some tests failed")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())