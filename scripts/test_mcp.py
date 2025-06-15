#!/usr/bin/env python3
"""
Test helper script for MCP endpoints.

This script provides easy commands to:
1. Run unit tests
2. Run integration tests  
3. Start a test server
4. Run manual endpoint tests
"""

import asyncio
import argparse
import subprocess
import sys
import time
from pathlib import Path

# Add buttermilk to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_unit_tests():
    """Run unit tests for MCP endpoints."""
    print("🧪 Running MCP unit tests...")
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/api/test_mcp_endpoints.py",
        "-v", "--tb=short"
    ]
    return subprocess.run(cmd).returncode


def run_integration_tests():
    """Run integration tests for MCP endpoints."""
    print("🔗 Running MCP integration tests...")
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/api/test_mcp_integration.py", 
        "-v", "--tb=short", "-m", "not slow"
    ]
    return subprocess.run(cmd).returncode


def run_all_tests():
    """Run all MCP tests."""
    print("🚀 Running all MCP tests...")
    unit_result = run_unit_tests()
    if unit_result != 0:
        print("❌ Unit tests failed")
        return unit_result
    
    integration_result = run_integration_tests()
    if integration_result != 0:
        print("❌ Integration tests failed")
        return integration_result
    
    print("✅ All MCP tests passed!")
    return 0


def start_test_server():
    """Start a test server for manual testing."""
    print("🖥️  Starting test server for MCP endpoints...")
    
    # Create a simple test server
    server_code = '''
import uvicorn
from fastapi import FastAPI
from unittest.mock import MagicMock
from buttermilk.api.mcp import mcp_router

# Create mock app state
app = FastAPI(title="MCP Test Server")
mock_flow_runner = MagicMock()
mock_flow_runner.flows = {"tox": MagicMock(), "trans": MagicMock()}
app.state.flow_runner = mock_flow_runner

app.include_router(mcp_router)

@app.get("/")
async def root():
    return {
        "message": "MCP Test Server", 
        "endpoints": {
            "tools": "/mcp/tools",
            "judge": "/mcp/tools/judge",
            "synthesize": "/mcp/tools/synthesize", 
            "differences": "/mcp/tools/find_differences"
        }
    }

if __name__ == "__main__":
    print("🚀 MCP Test Server starting on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
'''
    
    # Write temp server file and run it
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(server_code)
        temp_file = f.name
    
    try:
        subprocess.run([sys.executable, temp_file])
    finally:
        Path(temp_file).unlink()


def test_endpoints_manually():
    """Run manual tests against a running server."""
    import requests
    import json
    
    base_url = "http://localhost:8001"
    
    print("🔧 Testing MCP endpoints manually...")
    
    try:
        # Test 1: List tools
        print("📋 Testing tools listing...")
        response = requests.get(f"{base_url}/mcp/tools", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {len(data['result']['tools'])} tools")
        else:
            print(f"❌ Tools listing failed: {response.status_code}")
            return False
        
        # Test 2: Judge endpoint (will fail without real setup, but tests the structure)
        print("⚖️  Testing judge endpoint structure...")
        response = requests.post(f"{base_url}/mcp/tools/judge", 
            json={
                "text": "Test message",
                "criteria": "toxicity",
                "model": "gpt4o",
                "flow": "tox"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                print("✅ Judge endpoint responded successfully")
            else:
                print(f"⚠️  Judge endpoint responded with error (expected): {data['error'][:100]}...")
        else:
            print(f"❌ Judge endpoint failed: {response.status_code}")
        
        print("✅ Manual endpoint testing complete")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to test server: {e}")
        print("💡 Make sure to run 'python scripts/test_mcp.py server' first")
        return False


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="MCP testing helper")
    parser.add_argument("command", choices=[
        "unit", "integration", "all", "server", "manual"
    ], help="Test command to run")
    
    args = parser.parse_args()
    
    if args.command == "unit":
        return run_unit_tests()
    elif args.command == "integration": 
        return run_integration_tests()
    elif args.command == "all":
        return run_all_tests()
    elif args.command == "server":
        start_test_server()
        return 0
    elif args.command == "manual":
        return 0 if test_endpoints_manually() else 1
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())