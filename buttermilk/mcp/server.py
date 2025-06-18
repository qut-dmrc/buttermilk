"""MCP Server implementation with dual-mode support.

This module provides the MCP server that can run in either embedded or daemon mode,
exposing agent tools as HTTP/WebSocket endpoints.
"""

import asyncio
from enum import Enum
from typing import Any, Callable

from autogen_core import Component
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from buttermilk._core import AgentInput, logger
from buttermilk._core.tool_definition import AgentToolDefinition, MCPServerConfig, UnifiedRequest
from buttermilk._core.schema_validation import validate_tool_input, validate_tool_output
from buttermilk.orchestrators import GroupChatOrchestrator


class MCPServerMode(str, Enum):
    """Server operation modes."""
    EMBEDDED = "embedded"
    DAEMON = "daemon"


class MCPServer(Component):
    """MCP Server for exposing agent tools as HTTP endpoints.
    
    Can run in two modes:
    - Embedded: Runs within the main application for local agent communication
    - Daemon: Separate server process for MCP routes only
    """
    
    def __init__(
        self,
        config: MCPServerConfig | None = None,
        orchestrator: GroupChatOrchestrator | None = None
    ):
        """Initialize MCP server.
        
        Args:
            config: Server configuration. Uses defaults if not provided.
            orchestrator: Orchestrator for embedded mode. Required for embedded mode.
        """
        self.config = config or MCPServerConfig()
        self.orchestrator = orchestrator
        self.app = FastAPI(title="Buttermilk MCP Server")
        self._routes: dict[str, dict[str, Any]] = {}
        self._setup_middleware()
        
        if self.config.mode == MCPServerMode.EMBEDDED and not orchestrator:
            raise ValueError("Orchestrator required for embedded mode")
    
    def _setup_middleware(self) -> None:
        """Configure CORS and other middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def register_route(self, tool_def: AgentToolDefinition, handler: Callable | None = None) -> None:
        """Register a tool definition as an MCP route.
        
        Args:
            tool_def: Tool definition to register
            handler: Optional custom handler. If not provided, uses default handler.
        """
        if not tool_def.mcp_route:
            return
        
        route_info = {
            "tool_def": tool_def,
            "handler": handler or self._create_default_handler(tool_def)
        }
        
        self._routes[tool_def.mcp_route] = route_info
        
        # Register with FastAPI
        self.app.post(
            tool_def.mcp_route,
            summary=tool_def.description,
            response_model=None  # Dynamic based on output schema
        )(route_info["handler"])
        
        logger.info(f"Registered MCP route: {tool_def.mcp_route} for tool: {tool_def.name}")
    
    def _create_default_handler(self, tool_def: AgentToolDefinition) -> Callable:
        """Create a default handler for a tool definition.
        
        Args:
            tool_def: Tool definition to create handler for
            
        Returns:
            Async handler function
        """
        async def handler(request: Request) -> Any:
            # Parse request body
            body = await request.json()
            
            # Validate input against schema
            if tool_def.input_schema:
                try:
                    validated_input = validate_tool_input(tool_def.input_schema, body)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Input validation failed: {e}")
            else:
                validated_input = body
            
            # Check permissions if auth is required
            if self.config.auth_required and tool_def.permissions:
                # TODO: Implement actual permission checking
                # For now, just log the requirement
                logger.debug(f"Permissions required for {tool_def.name}: {tool_def.permissions}")
            
            # In embedded mode, route to orchestrator
            if self.config.mode == MCPServerMode.EMBEDDED:
                if not self.orchestrator:
                    raise HTTPException(status_code=500, detail="No orchestrator available")
                
                # Create unified request
                unified_req = UnifiedRequest(
                    target=f"{tool_def.name}",  # Will be resolved to agent.tool
                    inputs=validated_input,
                    metadata={"mcp_route": tool_def.mcp_route}
                )
                
                # Execute via orchestrator
                # This is a simplified version - actual implementation would need
                # proper agent resolution and execution
                result = await self._execute_via_orchestrator(unified_req)
            else:
                # In daemon mode, would need different execution strategy
                raise HTTPException(status_code=501, detail="Daemon mode execution not yet implemented")
            
            # Validate output if schema provided
            if tool_def.output_schema:
                try:
                    validated_output = validate_tool_output(tool_def.output_schema, result)
                except Exception as e:
                    logger.error(f"Output validation failed for {tool_def.name}: {e}")
                    # Log but don't fail - return result anyway
                    validated_output = result
            else:
                validated_output = result
            
            return validated_output
        
        return handler
    
    async def _execute_via_orchestrator(self, request: UnifiedRequest) -> Any:
        """Execute a request via the orchestrator in embedded mode.
        
        Args:
            request: Unified request to execute
            
        Returns:
            Execution result
        """
        # This is a placeholder - actual implementation would need to:
        # 1. Resolve the target to specific agent and tool
        # 2. Create appropriate AgentInput
        # 3. Execute via orchestrator
        # 4. Extract and return result
        
        logger.warning("Orchestrator execution not yet implemented")
        return {"status": "not_implemented", "request": request.model_dump()}
    
    def register_discovery_endpoints(self) -> None:
        """Register endpoints for tool discovery."""
        
        @self.app.get("/mcp/tools")
        async def list_tools():
            """List all available tools."""
            tools = []
            for route_path, route_info in self._routes.items():
                tool_def = route_info["tool_def"]
                tools.append({
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "route": route_path,
                    "input_schema": tool_def.input_schema,
                    "output_schema": tool_def.output_schema,
                    "permissions": tool_def.permissions
                })
            return {"tools": tools}
        
        @self.app.get("/mcp/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "mode": self.config.mode,
                "routes_registered": len(self._routes)
            }
    
    async def start(self) -> None:
        """Start the MCP server."""
        self.register_discovery_endpoints()
        
        if self.config.mode == MCPServerMode.DAEMON:
            # Run as standalone server
            logger.info(f"Starting MCP server in daemon mode on port {self.config.port}")
            config = uvicorn.Config(
                app=self.app,
                host="0.0.0.0",
                port=self.config.port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
        else:
            # In embedded mode, just register the app
            logger.info("MCP server initialized in embedded mode")
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI app instance for embedded mode."""
        return self.app