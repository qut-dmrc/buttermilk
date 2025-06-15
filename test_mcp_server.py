#!/usr/bin/env python3
"""Minimal test server for MCP endpoints."""

from fastapi import FastAPI
import uvicorn
from buttermilk.api.mcp import mcp_router
from unittest.mock import MagicMock

# Create a mock FlowRunner for testing
mock_flow_runner = MagicMock()
mock_flow_runner.flows = {"tox": MagicMock(), "trans": MagicMock()}

app = FastAPI(title="Test MCP Server")

# Set up mock app state
app.state.flow_runner = mock_flow_runner

app.include_router(mcp_router)

@app.get("/")
async def root():
    return {"message": "Test MCP Server", "mcp_endpoints": "/mcp/tools"}

if __name__ == "__main__":
    print("🚀 Starting test MCP server...")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")