#!/usr/bin/env python3
"""Simple MCP server for DebugAgent tools.

This creates a minimal HTTP server that exposes the DebugAgent's tools
for LLM access. It's designed to run alongside the main Buttermilk API.

Usage:
    python scripts/run_debug_mcp_server.py
    
The server will start on http://localhost:8090 with endpoints:
    - GET  /tools - List available tools
    - POST /tools/{tool_name} - Execute a tool
"""

import asyncio
import json
import logging
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add buttermilk to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from buttermilk.debug.debug_agent import DebugAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Buttermilk Debug MCP Server")

# Global debug agent instance
debug_agent = None


class ToolRequest(BaseModel):
    """Generic request for tool execution."""
    params: Dict[str, Any] = {}


class ToolResponse(BaseModel):
    """Response from tool execution."""
    success: bool
    result: Any = None
    error: str = None


@app.on_event("startup")
async def startup():
    """Initialize the debug agent on startup."""
    global debug_agent
    logger.info("Initializing DebugAgent...")
    debug_agent = DebugAgent(agent_name="debug_agent", role="debugger")
    logger.info("DebugAgent ready")


@app.get("/")
async def root():
    """Root endpoint with server info."""
    return {
        "service": "Buttermilk Debug MCP Server",
        "description": "Provides MCP tools for debugging Buttermilk flows",
        "endpoints": {
            "list_tools": "GET /tools",
            "execute_tool": "POST /tools/{tool_name}"
        }
    }


@app.get("/tools")
async def list_tools():
    """List all available debugging tools."""
    tools = []
    
    # Get all methods with @tool decorator
    for attr_name in dir(debug_agent):
        if attr_name.startswith('_'):
            continue
        
        # Skip properties that might fail without runtime binding
        if attr_name in ['id', 'metadata']:
            continue
            
        try:
            attr = getattr(debug_agent, attr_name)
        except RuntimeError:
            # Skip attributes that require runtime binding
            continue
        if callable(attr) and hasattr(attr, '_tool_metadata'):
            tool_info = {
                "name": attr_name,
                "description": attr.__doc__.strip() if attr.__doc__ else "No description",
                "async": asyncio.iscoroutinefunction(attr)
            }
            
            # Try to extract parameters from docstring
            if attr.__doc__:
                lines = attr.__doc__.strip().split('\n')
                params = []
                in_args = False
                for line in lines:
                    if line.strip().startswith('Args:'):
                        in_args = True
                        continue
                    if in_args and line.strip().startswith('Returns:'):
                        break
                    if in_args and ':' in line:
                        param_line = line.strip()
                        if param_line:
                            param_name = param_line.split(':')[0].strip()
                            param_desc = param_line.split(':', 1)[1].strip()
                            params.append({
                                "name": param_name,
                                "description": param_desc
                            })
                tool_info["parameters"] = params
            
            tools.append(tool_info)
    
    return {"tools": tools}


@app.post("/tools/{tool_name}")
async def execute_tool(tool_name: str, request: ToolRequest):
    """Execute a specific debugging tool."""
    # Check if tool exists
    if not hasattr(debug_agent, tool_name):
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    
    tool_method = getattr(debug_agent, tool_name)
    
    # Check if it's actually a tool
    if not callable(tool_method) or not hasattr(tool_method, '_tool_metadata'):
        raise HTTPException(status_code=400, detail=f"'{tool_name}' is not a valid tool")
    
    try:
        # Execute the tool
        logger.info(f"Executing tool: {tool_name} with params: {request.params}")
        
        if asyncio.iscoroutinefunction(tool_method):
            result = await tool_method(**request.params)
        else:
            result = tool_method(**request.params)
        
        return ToolResponse(success=True, result=result)
        
    except TypeError as e:
        # Usually means wrong parameters
        return ToolResponse(
            success=False,
            error=f"Invalid parameters: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Tool execution error: {e}", exc_info=True)
        return ToolResponse(
            success=False,
            error=f"Tool execution failed: {str(e)}"
        )


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    if debug_agent:
        await debug_agent.cleanup()


if __name__ == "__main__":
    print("🚀 Starting Buttermilk Debug MCP Server")
    print("📍 Server will be available at: http://localhost:8090")
    print("📋 List tools: GET http://localhost:8090/tools")
    print("🔧 Execute tool: POST http://localhost:8090/tools/{tool_name}")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8090, log_level="info")