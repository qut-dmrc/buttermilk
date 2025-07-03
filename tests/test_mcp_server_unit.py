"""Unit tests for MCP server implementation."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
import asyncio

from buttermilk._core.tool_definition import AgentToolDefinition, MCPServerConfig
from buttermilk.mcp.server import MCPServer, MCPServerMode
from buttermilk._core.schema_validation import SchemaValidationError


class TestMCPServerConfig:
    """Test MCP server configuration."""
    
    def test_default_config(self):
        """Test default server configuration."""
        server = MCPServer()
        assert server.config.mode == "embedded"
        assert server.config.port == 8787
        assert server.config.auth_required is True
        assert server.config.allowed_origins == ["*"]
    
    def test_custom_config(self):
        """Test custom server configuration."""
        config = MCPServerConfig(
            mode="daemon",
            port=9000,
            auth_required=False,
            allowed_origins=["http://localhost:3000"]
        )
        server = MCPServer(config=config)
        assert server.config.mode == "daemon"
        assert server.config.port == 9000
        assert server.config.auth_required is False
    
    def test_embedded_requires_orchestrator(self):
        """Test that embedded mode requires orchestrator."""
        config = MCPServerConfig(mode="embedded")
        with pytest.raises(ValueError, match="Orchestrator required for embedded mode"):
            MCPServer(config=config, orchestrator=None)


class TestMCPServerRoutes:
    """Test MCP server route registration and handling."""
    
    @pytest.fixture
    def server(self):
        """Create test server with mock orchestrator."""
        mock_orchestrator = Mock()
        return MCPServer(orchestrator=mock_orchestrator)
    
    @pytest.fixture
    def client(self, server):
        """Create test client."""
        return TestClient(server.app)
    
    def test_register_route(self, server):
        """Test route registration."""
        tool_def = AgentToolDefinition(
            name="test_tool",
            description="Test tool",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"]
            },
            output_schema={"type": "string"},
            mcp_route="/test",
            permissions=["read"]
        )
        
        server.register_route(tool_def)
        
        assert "/test" in server._routes
        assert server._routes["/test"]["tool_def"] == tool_def
        assert server._routes["/test"]["handler"] is not None
    
    def test_route_without_mcp_path(self, server):
        """Test that tools without MCP route are not registered."""
        tool_def = AgentToolDefinition(
            name="test_tool",
            description="Test tool",
            input_schema={},
            output_schema={},
            mcp_route=None  # No MCP route
        )
        
        server.register_route(tool_def)
        assert len(server._routes) == 0
    
    @pytest.mark.asyncio
    async def test_discovery_endpoints(self, server, client):
        """Test tool discovery endpoints."""
        # Register some tools
        tool1 = AgentToolDefinition(
            name="tool1",
            description="First tool",
            input_schema={"type": "object"},
            output_schema={"type": "string"},
            mcp_route="/tool1",
            permissions=["read"]
        )
        tool2 = AgentToolDefinition(
            name="tool2",
            description="Second tool",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            mcp_route="/tool2",
            permissions=["write"]
        )
        
        server.register_route(tool1)
        server.register_route(tool2)
        server.register_discovery_endpoints()
        
        # Test /mcp/tools endpoint
        response = client.get("/mcp/tools")
        assert response.status_code == 200
        data = response.json()
        assert len(data["tools"]) == 2
        assert data["tools"][0]["name"] == "tool1"
        assert data["tools"][1]["name"] == "tool2"
        
        # Test /mcp/health endpoint
        response = client.get("/mcp/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["mode"] == "embedded"
        assert data["routes_registered"] == 2
    
    @pytest.mark.asyncio
    async def test_route_handler_validation(self, server, client):
        """Test route handler with input validation."""
        tool_def = AgentToolDefinition(
            name="validated_tool",
            description="Tool with validation",
            input_schema={
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "message": {"type": "string"}
                },
                "required": ["count"]
            },
            output_schema={"type": "object"},
            mcp_route="/validated"
        )
        
        server.register_route(tool_def)
        server.register_discovery_endpoints()
        
        # Valid input
        response = client.post("/validated", json={"count": 5, "message": "test"})
        assert response.status_code == 200
        
        # Invalid input (wrong type)
        response = client.post("/validated", json={"count": "not a number"})
        assert response.status_code == 400
        assert "Input validation failed" in response.json()["detail"]
        
        # Missing required field
        response = client.post("/validated", json={"message": "test"})
        assert response.status_code == 400
    
    def test_custom_handler(self, server):
        """Test registering route with custom handler."""
        tool_def = AgentToolDefinition(
            name="custom",
            description="Custom handler",
            input_schema={},
            output_schema={},
            mcp_route="/custom"
        )
        
        async def custom_handler(request):
            return {"custom": "response"}
        
        server.register_route(tool_def, handler=custom_handler)
        
        assert server._routes["/custom"]["handler"] == custom_handler


class TestMCPServerModes:
    """Test different server operation modes."""
    
    @pytest.mark.asyncio
    async def test_embedded_mode_execution(self):
        """Test execution in embedded mode."""
        mock_orchestrator = AsyncMock()
        server = MCPServer(orchestrator=mock_orchestrator)
        
        tool_def = AgentToolDefinition(
            name="test",
            description="Test",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            mcp_route="/test"
        )
        
        server.register_route(tool_def)
        
        # The default handler should create UnifiedRequest
        handler = server._routes["/test"]["handler"]
        
        # Mock request
        mock_request = AsyncMock()
        mock_request.json = AsyncMock(return_value={"key": "value"})
        
        # Execute handler
        result = await handler(mock_request)
        
        # Should return not_implemented for now
        assert result["status"] == "not_implemented"
        assert result["request"]["target"] == "test"
        assert result["request"]["inputs"] == {"key": "value"}
    
    @pytest.mark.asyncio
    async def test_daemon_mode_start(self):
        """Test daemon mode server start (mocked)."""
        config = MCPServerConfig(mode="daemon", port=9999)
        server = MCPServer(config=config)
        
        # Mock uvicorn to avoid actual server start
        with patch("buttermilk.mcp.server.uvicorn") as mock_uvicorn:
            mock_uvicorn.Config = Mock
            mock_uvicorn.Server = Mock
            mock_uvicorn.Server.return_value.serve = AsyncMock()
            
            await server.start()
            
            # Verify uvicorn was configured correctly
            mock_uvicorn.Config.assert_called_once()
            config_call = mock_uvicorn.Config.call_args
            assert config_call[1]["port"] == 9999
            assert config_call[1]["host"] == "0.0.0.0"


class TestMCPServerIntegration:
    """Test MCP server integration scenarios."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent with tool definitions."""
        agent = Mock()
        
        tool_def = AgentToolDefinition(
            name="analyze",
            description="Analyze data",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            },
            output_schema={"type": "object"},
            mcp_route="/analyze"
        )
        
        agent.get_tool_definitions.return_value = [tool_def]
        agent.agent_name = "test_agent"
        return agent
    
    def test_register_agent_tools(self, mock_agent):
        """Test registering all tools from an agent."""
        mock_orchestrator = Mock()
        server = MCPServer(orchestrator=mock_orchestrator)
        
        # Register all tools from agent
        for tool_def in mock_agent.get_tool_definitions():
            server.register_route(tool_def)
        
        assert len(server._routes) == 1
        assert "/analyze" in server._routes
    
    @pytest.mark.asyncio
    async def test_cors_middleware(self):
        """Test CORS middleware configuration."""
        config = MCPServerConfig(
            allowed_origins=["http://example.com", "https://app.example.com"]
        )
        server = MCPServer(config=config, orchestrator=Mock())
        client = TestClient(server.app)
        
        server.register_discovery_endpoints()
        
        # Test CORS headers
        response = client.options(
            "/mcp/health",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET"
            }
        )
        
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://example.com"
    
    def test_get_app(self):
        """Test getting FastAPI app instance."""
        server = MCPServer(orchestrator=Mock())
        app = server.get_app()
        
        assert app == server.app
        assert app.title == "Buttermilk MCP Server"