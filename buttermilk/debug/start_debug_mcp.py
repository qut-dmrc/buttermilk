#!/usr/bin/env python3
"""Start an MCP server with the DebugAgent tools.

This script starts a standalone MCP server that exposes the DebugAgent's
debugging tools via HTTP endpoints for LLM access.
"""

import asyncio
import logging
from buttermilk.mcp.server import MCPServer, MCPServerMode
from buttermilk.mcp.tool_registry import ToolDiscoveryService
from buttermilk._core.tool_definition import MCPServerConfig
from buttermilk.debug.debug_agent import DebugAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Start the MCP server with DebugAgent."""
    
    # Create the debug agent
    logger.info("Creating DebugAgent...")
    debug_agent = DebugAgent(
        agent_name="debug_agent",
        role="debugger"
    )
    
    # Create tool discovery service
    logger.info("Setting up tool discovery...")
    discovery = ToolDiscoveryService()
    discovery.register_agent(debug_agent)
    
    # Create MCP server config
    config = MCPServerConfig(
        mode=MCPServerMode.DAEMON,
        port=8090,  # Different port from main API
        allowed_origins=["*"]  # Allow all origins for debugging
    )
    
    # Create and configure server
    logger.info("Creating MCP server...")
    server = MCPServer(config=config)
    
    # Register all discovered tools
    for tool_key, tool_info in discovery.tool_registry.items():
        tool_def = tool_info["definition"]
        logger.info(f"Registering tool: {tool_def.name}")
        
        # For DebugAgent tools, we need to create a handler that calls the tool
        async def create_handler(tool_def, agent):
            async def handler(request: dict) -> dict:
                # Get the tool method
                method = getattr(agent, tool_def.name)
                
                # Call it with the request parameters
                if asyncio.iscoroutinefunction(method):
                    result = await method(**request)
                else:
                    result = method(**request)
                
                return {"result": result}
            return handler
        
        handler = await create_handler(tool_def, tool_info["agent"])
        server.register_route(tool_def, handler)
    
    # Start the server
    logger.info(f"Starting MCP server on http://localhost:{config.port}")
    logger.info("Available endpoints:")
    logger.info("  - http://localhost:8090/mcp/tools - List all tools")
    logger.info("  - http://localhost:8090/mcp/health - Health check")
    
    await server.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)