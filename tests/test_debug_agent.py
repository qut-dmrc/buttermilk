"""Test the DebugAgent MCP tools."""

import asyncio
import pytest
from buttermilk.debug.debug_agent import DebugAgent


@pytest.mark.anyio
async def test_debug_agent():
    """Test the debug agent tools."""
    print("Creating DebugAgent...")
    agent = DebugAgent(
        agent_name="debug_agent",
        role="debugger"
    )
    
    # Test getting tool definitions
    print("\nGetting tool definitions...")
    # We can't use get_tool_definitions() without binding the agent,
    # but we can test the actual tool methods directly
    
    # Test log reading
    print("\nTesting log reading...")
    log_files = agent.list_log_files()
    print(f"Found {len(log_files)} log files")
    
    # Assert that log_files is a list
    assert isinstance(log_files, list)
    
    if log_files:
        print(f"Most recent: {log_files[0]['path']}")
        logs = agent.get_latest_buttermilk_logs(20)
        print(f"Last 20 lines preview:\n{logs[:200]}...")
        
        # Assert that logs is a string
        assert isinstance(logs, str)
    
    # Test WebSocket client (if server is running)
    print("\nTesting WebSocket client...")
    try:
        result = await agent.start_websocket_client("test_flow", use_direct_ws=True)
        print(f"Start result: {result}")
        
        if result["status"] == "success":
            # List active clients
            clients = agent.list_active_clients()
            print(f"Active clients: {clients}")
            
            # Stop the client
            stop_result = await agent.stop_websocket_client("test_flow")
            print(f"Stop result: {stop_result}")
    except Exception as e:
        print(f"WebSocket test failed (server may not be running): {e}")
    
    print("\nDebugAgent test complete!")
    
    # Cleanup the agent
    await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(test_debug_agent())