"""
Tests for MCP endpoints.

This module provides both unit tests (with mocked dependencies) and integration tests
(with real Buttermilk components) for the MCP API endpoints.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from buttermilk.api.mcp import mcp_router, MCPToolResponse, run_single_agent
from buttermilk._core.agent import AgentInput, AgentTrace
from buttermilk._core.types import RunRequest

# Mark all tests in this module as async
pytestmark = pytest.mark.anyio


class TestMCPEndpoints:
    """Unit tests for MCP endpoints with mocked dependencies."""

    @pytest.fixture
    def mock_flow_runner(self):
        """Create a mock FlowRunner for testing."""
        mock_runner = MagicMock()
        mock_runner.flows = {
            "tox": MagicMock(),
            "trans": MagicMock()
        }
        
        # Mock the flow config structure
        mock_flow_config = MagicMock()
        mock_flow_config.agents = {
            "judge": MagicMock(),
            "synth": MagicMock(),
            "differences": MagicMock()
        }
        mock_flow_config.parameters = {}
        
        mock_runner.flows["tox"] = mock_flow_config
        return mock_runner

    @pytest.fixture
    def test_app(self, mock_flow_runner):
        """Create a test FastAPI app with mocked dependencies."""
        app = FastAPI()
        app.state.flow_runner = mock_flow_runner
        
        # Override the get_flows dependency for testing
        from buttermilk.api.routes import get_flows
        
        async def override_get_flows():
            return mock_flow_runner
        
        app.dependency_overrides[get_flows] = override_get_flows
        app.include_router(mcp_router)
        
        return TestClient(app)

    def test_list_tools(self, test_app):
        """Test the tools listing endpoint."""
        response = test_app.get("/mcp/tools")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "tools" in data["result"]
        assert len(data["result"]["tools"]) == 3
        
        tool_names = [tool["name"] for tool in data["result"]["tools"]]
        assert "judge" in tool_names
        assert "synthesize" in tool_names
        assert "find_differences" in tool_names

    async def test_judge_endpoint_success(self, test_app, mock_flow_runner):
        """Test successful judge tool call."""
        # Mock the agent execution
        mock_agent_instance = AsyncMock()
        mock_agent_trace = MagicMock()
        mock_agent_trace.outputs = {"toxicity_score": 0.2, "reasoning": "Test reasoning"}
        mock_agent_instance.invoke.return_value = mock_agent_trace
        
        with patch('buttermilk.api.mcp.run_single_agent') as mock_run_agent:
            mock_run_agent.return_value = mock_agent_trace.outputs
            
            response = test_app.post("/mcp/tools/judge", json={
                "text": "This is a test message.",
                "criteria": "toxicity",
                "model": "gpt4o",
                "flow": "tox"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "result" in data
            assert data["trace_id"] is not None

    async def test_judge_endpoint_flow_not_found(self, test_app):
        """Test judge tool with invalid flow."""
        response = test_app.post("/mcp/tools/judge", json={
            "text": "Test message",
            "criteria": "toxicity", 
            "model": "gpt4o",
            "flow": "nonexistent_flow"
        })
        
        assert response.status_code == 200  # MCP returns 200 with error in body
        data = response.json()
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    async def test_synthesize_endpoint(self, test_app):
        """Test synthesize tool endpoint.""" 
        with patch('buttermilk.api.mcp.run_single_agent') as mock_run_agent:
            mock_result = {"synthesized_text": "Test synthesis result"}
            mock_run_agent.return_value = mock_result
            
            response = test_app.post("/mcp/tools/synthesize", json={
                "text": "Original text",
                "criteria": "clarity",
                "model": "gpt4o",
                "flow": "tox"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    async def test_differences_endpoint(self, test_app):
        """Test differences tool endpoint."""
        with patch('buttermilk.api.mcp.run_single_agent') as mock_run_agent:
            mock_result = {"differences": ["diff1", "diff2"]}
            mock_run_agent.return_value = mock_result
            
            response = test_app.post("/mcp/tools/find_differences", json={
                "text1": "First text",
                "text2": "Second text", 
                "criteria": "semantic",
                "model": "gpt4o",
                "flow": "tox"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True


class TestMCPHelperFunctions:
    """Unit tests for MCP helper functions."""

    async def test_run_single_agent_success(self):
        """Test successful single agent execution."""
        mock_flow_runner = MagicMock()
        mock_flow_config = MagicMock()
        mock_agent_config = MagicMock()
        
        # Setup mock structure
        mock_flow_runner.flows = {"tox": mock_flow_config}
        mock_flow_config.agents = {"judge": mock_agent_config}
        mock_flow_config.parameters = {}
        
        # Mock agent class and instance
        mock_agent_class = MagicMock()
        mock_agent_instance = AsyncMock()
        mock_agent_trace = MagicMock()
        mock_agent_trace.outputs = {"result": "success"}
        
        mock_agent_instance.invoke.return_value = mock_agent_trace
        mock_agent_class.return_value = mock_agent_instance
        
        # Mock the get_configs method
        mock_variant_config = MagicMock()
        mock_variant_config.model_dump.return_value = {}
        mock_agent_config.get_configs.return_value = [(mock_agent_class, mock_variant_config)]
        
        result = await run_single_agent("judge", "tox", {"text": "test"}, mock_flow_runner)
        
        assert result == {"result": "success"}
        mock_agent_instance.invoke.assert_called_once()

    async def test_run_single_agent_flow_not_found(self):
        """Test single agent execution with invalid flow."""
        mock_flow_runner = MagicMock()
        mock_flow_runner.flows = {}
        
        from buttermilk.api.mcp import MCPError
        
        with pytest.raises(MCPError) as exc_info:
            await run_single_agent("judge", "invalid_flow", {}, mock_flow_runner)
        
        assert "not found" in str(exc_info.value)

    async def test_run_single_agent_agent_not_found(self):
        """Test single agent execution with invalid agent role."""
        mock_flow_runner = MagicMock()
        mock_flow_config = MagicMock()
        mock_flow_config.agents = {}
        mock_flow_runner.flows = {"tox": mock_flow_config}
        
        from buttermilk.api.mcp import MCPError
        
        with pytest.raises(MCPError) as exc_info:
            await run_single_agent("invalid_agent", "tox", {}, mock_flow_runner)
        
        assert "not found" in str(exc_info.value)


class TestMCPModels:
    """Test MCP data models."""

    def test_mcp_tool_response_success(self):
        """Test MCPToolResponse for successful call."""
        response = MCPToolResponse(
            success=True,
            result={"data": "test"},
            trace_id="test-123",
            execution_time_ms=150.5
        )
        
        assert response.success is True
        assert response.result == {"data": "test"}
        assert response.error is None
        assert response.trace_id == "test-123"
        assert response.execution_time_ms == 150.5

    def test_mcp_tool_response_error(self):
        """Test MCPToolResponse for failed call."""
        response = MCPToolResponse(
            success=False,
            error="Test error message",
            trace_id="test-456"
        )
        
        assert response.success is False
        assert response.result is None
        assert response.error == "Test error message"
        assert response.trace_id == "test-456"