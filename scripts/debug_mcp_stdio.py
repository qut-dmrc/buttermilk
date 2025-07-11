#!/usr/bin/env python3
"""MCP server for DebugAgent tools using stdio transport.

This creates an MCP server that uses stdio (standard input/output) transport,
which is compatible with Claude Desktop/Code.

Usage:
    python scripts/debug_mcp_stdio.py
"""

import asyncio
import sys
from pathlib import Path

# Add buttermilk to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import MCP SDK
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
except ImportError:
    print("Error: MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

from buttermilk.debug.debug_agent import DebugAgent


async def main():
    """Run the MCP server with DebugAgent tools."""
    # Create the server
    server = Server("buttermilk-debug")
    
    # Create debug agent instance
    debug_agent = DebugAgent(agent_name="debug_agent", role="debugger")
    
    # Register tools from DebugAgent
    # Note: We need to register each tool method with the MCP server
    
    @server.tool()
    async def list_log_files(max_files: int = 10):
        """List the most recent buttermilk log files.
        
        Args:
            max_files: Maximum number of log files to return
        """
        return debug_agent.list_log_files(max_files=max_files)
    
    @server.tool()
    async def get_latest_buttermilk_logs(lines: int = 100):
        """Get the latest logs from the most recent buttermilk log file.
        
        Args:
            lines: Number of lines to return from the log file
        """
        return debug_agent.get_latest_buttermilk_logs(lines=lines)
    
    @server.tool()
    async def search_logs(pattern: str, files: int = 5, context: int = 2):
        """Search for a pattern in recent log files.
        
        Args:
            pattern: Regex pattern to search for
            files: Number of recent log files to search
            context: Number of context lines before/after matches
        """
        return debug_agent.search_logs(pattern=pattern, files=files, context=context)
    
    @server.tool()
    async def get_flow_status(flow_id: str):
        """Get the current status of a flow by analyzing logs.
        
        Args:
            flow_id: The flow ID to check status for
        """
        return debug_agent.get_flow_status(flow_id=flow_id)
    
    @server.tool()
    async def start_websocket_client(flow_id: str, use_direct_ws: bool = False):
        """Start a WebSocket client for debugging a flow.
        
        Args:
            flow_id: The flow ID to monitor
            use_direct_ws: If True, use direct WebSocket (ws://); if False, use proxied (http://)
        """
        return debug_agent.start_websocket_client(flow_id=flow_id, use_direct_ws=use_direct_ws)
    
    @server.tool()
    async def stop_websocket_client(flow_id: str):
        """Stop the WebSocket client for a flow.
        
        Args:
            flow_id: The flow ID to stop monitoring
        """
        return debug_agent.stop_websocket_client(flow_id=flow_id)
    
    @server.tool()
    async def list_active_clients():
        """List all active WebSocket clients."""
        return debug_agent.list_active_clients()
    
    @server.tool()
    async def get_client_messages(flow_id: str, limit: int = 50):
        """Get recent messages received by a WebSocket client.
        
        Args:
            flow_id: The flow ID to get messages for
            limit: Maximum number of messages to return
        """
        return debug_agent.get_client_messages(flow_id=flow_id, limit=limit)
    
    @server.tool()
    async def analyze_exception(error_text: str):
        """Analyze an exception or error message from logs.
        
        Args:
            error_text: The error or exception text to analyze
        """
        return debug_agent.analyze_exception(error_text=error_text)
    
    @server.tool()
    async def summarize_flow_execution(flow_id: str):
        """Summarize the execution of a flow from logs.
        
        Args:
            flow_id: The flow ID to summarize
        """
        return debug_agent.summarize_flow_execution(flow_id=flow_id)
    
    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    asyncio.run(main())