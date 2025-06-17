"""
Comprehensive tests for Flow-Agnostic MCP Agent endpoints.

This test suite validates all generic MCP endpoints for component testing:
- Vector store query testing for any flow configuration
- Individual agent invocation testing for any agent type
- Multi-agent workflow testing for any agent combination
- Session state management testing across flows
- Generic message validation testing

Tests use pytest-asyncio with FastAPI TestClient for comprehensive validation
of agent and flow functionality in isolation, supporting any YAML configuration.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
import json

from buttermilk.api.mcp_agents import agent_mcp_router, MCPToolResponse
from buttermilk.runner.flowrunner import FlowRunner


@pytest.fixture
def mock_flow_runner():
    """Mock FlowRunner for testing MCP endpoints with multiple flows."""
    mock_runner = MagicMock(spec=FlowRunner)
    
    # Mock multiple flow configurations
    mock_runner.flows = {
        "osb": {
            "name": "OSB Interactive Flow",
            "agents": {
                "researcher": {"type": "research_agent"},
                "policy_analyst": {"type": "policy_agent"},
                "fact_checker": {"type": "verification_agent"},
                "explorer": {"type": "exploration_agent"}
            }
        },
        "content_moderation": {
            "name": "Content Moderation Flow",
            "agents": {
                "classifier": {"type": "classification_agent"},
                "reviewer": {"type": "review_agent"}
            }
        },
        "research": {
            "name": "Research Flow",
            "agents": {
                "researcher": {"type": "research_agent"},
                "analyst": {"type": "analysis_agent"},
                "synthesizer": {"type": "synthesis_agent"}
            }
        }
    }
    return mock_runner


@pytest.fixture
def test_app(mock_flow_runner):
    """Create test FastAPI app with generic Agent MCP router."""
    app = FastAPI()
    app.include_router(agent_mcp_router)
    
    # Mock the dependency
    app.dependency_overrides = {
        "buttermilk.api.routes.get_flows": lambda: mock_flow_runner
    }
    
    return app


@pytest.fixture
def client(test_app):
    """Create test client for Agent MCP endpoints."""
    return TestClient(test_app)


class TestVectorQueryEndpoint:
    """Test flow-agnostic vector store query MCP endpoint."""

    @pytest.mark.parametrize("flow_name,agent_name", [
        ("osb", "researcher"),
        ("osb", "policy_analyst"),
        ("content_moderation", "classifier"),
        ("research", "analyst"),
        ("osb", None)  # No specific agent
    ])
    def test_vector_query_multiple_flows(self, client, flow_name, agent_name):
        """Test vector query across different flow configurations."""
        request_data = {
            "query": f"Test query for {flow_name} flow",
            "flow": flow_name,
            "agent_name": agent_name,
            "max_results": 5,
            "confidence_threshold": 0.5,
            "include_metadata": True
        }
        
        response = client.post("/mcp/agents/vector-query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "result" in data
        assert data["result"]["query"] == request_data["query"]
        assert data["result"]["flow"] == flow_name
        assert data["result"]["agent_name"] == agent_name
        assert "results" in data["result"]
        assert "performance" in data["result"]
        assert data["trace_id"] is not None

    def test_vector_query_invalid_flow(self, client):
        """Test vector query with non-existent flow."""
        request_data = {
            "query": "Test query",
            "flow": "nonexistent_flow",
            "agent_name": "some_agent"
        }
        
        response = client.post("/mcp/agents/vector-query", json=request_data)
        
        assert response.status_code == 404
        data = response.json()
        assert "Flow 'nonexistent_flow' not found" in data["detail"]

    def test_vector_query_invalid_agent_for_flow(self, client):
        """Test vector query with agent not available in specified flow."""
        request_data = {
            "query": "Test query",
            "flow": "osb",
            "agent_name": "nonexistent_agent"
        }
        
        response = client.post("/mcp/agents/vector-query", json=request_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "Agent 'nonexistent_agent' not found in flow 'osb'" in data["detail"]
        assert "Available agents:" in data["detail"]

    def test_vector_query_empty_query(self, client):
        """Test vector query with empty query string."""
        request_data = {
            "query": "",
            "flow": "osb"
        }
        
        response = client.post("/mcp/agents/vector-query", json=request_data)
        
        assert response.status_code == 422  # Validation error

    def test_vector_query_too_long(self, client):
        """Test vector query with query exceeding length limit."""
        request_data = {
            "query": "x" * 2001,  # Exceeds 2000 character limit
            "flow": "osb"
        }
        
        response = client.post("/mcp/agents/vector-query", json=request_data)
        
        assert response.status_code == 422  # Validation error


class TestAgentInvokeEndpoint:
    """Test flow-agnostic individual agent invocation MCP endpoint."""

    @pytest.mark.parametrize("flow_name,agent_name", [
        ("osb", "researcher"),
        ("osb", "policy_analyst"),
        ("osb", "fact_checker"),
        ("osb", "explorer"),
        ("content_moderation", "classifier"),
        ("content_moderation", "reviewer"),
        ("research", "researcher"),
        ("research", "analyst"),
        ("research", "synthesizer")
    ])
    def test_agent_invoke_all_flows_and_agents(self, client, flow_name, agent_name):
        """Test agent invocation across all flow and agent combinations."""
        request_data = {
            "query": f"Test query for {agent_name} in {flow_name}",
            "agent_name": agent_name,
            "flow": flow_name,
            "max_processing_time": 30,
            "parameters": {
                "test_parameter": "test_value",
                "flow_specific_param": f"{flow_name}_param"
            }
        }
        
        response = client.post("/mcp/agents/agent-invoke", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        result = data["result"]
        assert result["agent_name"] == agent_name
        assert result["flow"] == flow_name
        assert result["query"] == request_data["query"]
        assert "response" in result
        assert "confidence" in result
        assert "sources" in result
        assert "performance" in result

    def test_agent_invoke_invalid_flow(self, client):
        """Test agent invocation with non-existent flow."""
        request_data = {
            "query": "Test query",
            "agent_name": "researcher",
            "flow": "invalid_flow"
        }
        
        response = client.post("/mcp/agents/agent-invoke", json=request_data)
        
        assert response.status_code == 404

    def test_agent_invoke_invalid_agent_for_flow(self, client):
        """Test agent invocation with agent not in specified flow."""
        request_data = {
            "query": "Test query",
            "agent_name": "nonexistent_agent",
            "flow": "osb"
        }
        
        response = client.post("/mcp/agents/agent-invoke", json=request_data)
        
        assert response.status_code == 400

    def test_agent_invoke_with_custom_parameters(self, client):
        """Test agent invocation with custom parameters."""
        request_data = {
            "query": "Custom parameter test",
            "agent_name": "researcher",
            "flow": "osb",
            "max_processing_time": 45,
            "parameters": {
                "enable_deep_analysis": True,
                "analysis_depth": "comprehensive",
                "priority": "high",
                "custom_config": {"setting1": "value1", "setting2": "value2"}
            }
        }
        
        response = client.post("/mcp/agents/agent-invoke", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestMultiAgentWorkflowEndpoint:
    """Test flow-agnostic multi-agent workflow MCP endpoint."""

    @pytest.mark.parametrize("flow_name", ["osb", "content_moderation", "research"])
    def test_multi_agent_workflow_all_flows(self, client, flow_name):
        """Test multi-agent workflow across different flows."""
        request_data = {
            "query": f"Multi-agent test for {flow_name} flow",
            "flow": flow_name,
            "enable_synthesis": True,
            "parallel_execution": True,
            "timeout_seconds": 60
        }
        
        response = client.post("/mcp/agents/multi-agent-workflow", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        result = data["result"]
        assert result["flow"] == flow_name
        assert result["query"] == request_data["query"]
        assert "agents_executed" in result
        assert "agent_responses" in result
        assert "synthesis_result" in result
        assert "performance" in result

    def test_multi_agent_workflow_specific_agents(self, client):
        """Test multi-agent workflow with specific agent selection."""
        request_data = {
            "query": "Specific agents test",
            "flow": "osb",
            "agent_names": ["researcher", "policy_analyst"],
            "enable_synthesis": True,
            "parallel_execution": False,
            "timeout_seconds": 30
        }
        
        response = client.post("/mcp/agents/multi-agent-workflow", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        result = data["result"]
        assert set(result["agents_executed"]) == {"researcher", "policy_analyst"}
        assert len(result["agent_responses"]) == 2

    def test_multi_agent_workflow_invalid_agent(self, client):
        """Test multi-agent workflow with invalid agent name."""
        request_data = {
            "query": "Invalid agent test",
            "flow": "osb",
            "agent_names": ["researcher", "invalid_agent"],
            "enable_synthesis": True
        }
        
        response = client.post("/mcp/agents/multi-agent-workflow", json=request_data)
        
        assert response.status_code == 400

    def test_multi_agent_workflow_no_synthesis(self, client):
        """Test multi-agent workflow with synthesis disabled."""
        request_data = {
            "query": "No synthesis test",
            "flow": "research",
            "enable_synthesis": False,
            "parallel_execution": True
        }
        
        response = client.post("/mcp/agents/multi-agent-workflow", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        result = data["result"]
        # Should have agent responses but no synthesis
        assert "agent_responses" in result
        assert result["synthesis_result"] == {}


class TestSessionStateEndpoint:
    """Test flow-agnostic session state management MCP endpoint."""

    @pytest.mark.parametrize("operation", ["create", "get_status", "update_metadata", "record_metrics", "cleanup", "list_sessions"])
    def test_session_operations_all_types(self, client, operation):
        """Test all session operations across flows."""
        request_data = {
            "session_id": f"test-session-{operation}",
            "flow": "osb",
            "operation": operation,
            "test_data": {"test_key": "test_value"} if operation == "update_metadata" else None
        }
        
        response = client.post("/mcp/agents/session-state", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        result = data["result"]
        assert result["session_id"] == request_data["session_id"]
        assert result["flow"] == "osb"
        assert result["operation"] == operation
        assert "operation_result" in result

    @pytest.mark.parametrize("flow_name", ["osb", "content_moderation", "research"])
    def test_session_operations_multiple_flows(self, client, flow_name):
        """Test session operations across different flows."""
        request_data = {
            "session_id": f"test-session-{flow_name}",
            "flow": flow_name,
            "operation": "create"
        }
        
        response = client.post("/mcp/agents/session-state", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["flow"] == flow_name

    def test_session_invalid_operation(self, client):
        """Test session state with invalid operation."""
        request_data = {
            "session_id": "test-session",
            "flow": "osb",
            "operation": "invalid_operation"
        }
        
        response = client.post("/mcp/agents/session-state", json=request_data)
        
        assert response.status_code == 422  # Validation error


class TestMessageValidationEndpoint:
    """Test flow-agnostic message validation MCP endpoint."""

    @pytest.mark.parametrize("message_type,flow,message_data", [
        ("run_flow", "osb", {"query": "Test query", "flow": "osb"}),
        ("run_flow", "research", {"query": "Research query", "flow": "research"}),
        ("status_update", None, {"status": "processing", "timestamp": "2025-01-17T10:30:00Z"}),
        ("agent_response", "osb", {"agent": "researcher", "response": "analysis complete"}),
        ("workflow_complete", "research", {"summary": "Research complete", "results": []})
    ])
    def test_message_validation_various_types(self, client, message_type, flow, message_data):
        """Test message validation for various message types and flows."""
        request_data = {
            "message_type": message_type,
            "message_data": message_data,
            "flow": flow,
            "strict_validation": True
        }
        
        response = client.post("/mcp/agents/message-validation", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        result = data["result"]
        assert result["message_type"] == message_type
        assert result["flow"] == flow
        assert "is_valid" in result
        assert "validation_details" in result

    def test_message_validation_strict_vs_lenient(self, client):
        """Test message validation with strict vs lenient modes."""
        incomplete_data = {"query": "incomplete"}  # Missing flow field
        
        # Test strict validation
        strict_request = {
            "message_type": "run_flow",
            "message_data": incomplete_data,
            "flow": "osb",
            "strict_validation": True
        }
        
        response = client.post("/mcp/agents/message-validation", json=strict_request)
        assert response.status_code == 200
        
        # Test lenient validation
        lenient_request = {
            "message_type": "run_flow",
            "message_data": incomplete_data,
            "flow": "osb",
            "strict_validation": False
        }
        
        response = client.post("/mcp/agents/message-validation", json=lenient_request)
        assert response.status_code == 200

    def test_message_validation_no_flow_context(self, client):
        """Test message validation without flow context."""
        request_data = {
            "message_type": "status_update",
            "message_data": {"status": "processing"},
            "strict_validation": False
        }
        
        response = client.post("/mcp/agents/message-validation", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["flow"] is None


class TestMCPIntegration:
    """Integration tests for generic Agent MCP endpoints."""

    def test_mcp_response_format_consistency(self, client):
        """Test that all MCP endpoints return consistent response format."""
        endpoints_and_data = [
            ("/mcp/agents/vector-query", {
                "query": "Test query",
                "flow": "osb"
            }),
            ("/mcp/agents/agent-invoke", {
                "query": "Test query",
                "agent_name": "researcher",
                "flow": "osb"
            }),
            ("/mcp/agents/multi-agent-workflow", {
                "query": "Test query",
                "flow": "osb"
            }),
            ("/mcp/agents/session-state", {
                "session_id": "test-session",
                "flow": "osb",
                "operation": "create"
            }),
            ("/mcp/agents/message-validation", {
                "message_type": "run_flow",
                "message_data": {"query": "test", "flow": "osb"}
            })
        ]
        
        for endpoint, request_data in endpoints_and_data:
            response = client.post(endpoint, json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify consistent MCP response format
            assert "success" in data
            assert "result" in data
            assert "trace_id" in data
            assert "execution_time_ms" in data
            assert data["success"] is True
            assert data["trace_id"] is not None
            assert isinstance(data["execution_time_ms"], (int, float))

    def test_mcp_error_handling_consistency(self, client):
        """Test consistent error handling across Agent MCP endpoints."""
        # Test with invalid flows
        endpoints = [
            "/mcp/agents/vector-query",
            "/mcp/agents/agent-invoke",
            "/mcp/agents/multi-agent-workflow",
            "/mcp/agents/session-state"
        ]
        
        for endpoint in endpoints:
            if "session-state" in endpoint:
                request_data = {
                    "session_id": "test",
                    "flow": "invalid_flow",
                    "operation": "create"
                }
            else:
                request_data = {
                    "query": "test" if "message-validation" not in endpoint else None,
                    "flow": "invalid_flow",
                    "agent_name": "test_agent" if "agent-invoke" in endpoint else None
                }
                if "agent-invoke" in endpoint:
                    request_data["agent_name"] = "test_agent"
            
            response = client.post(endpoint, json=request_data)
            
            # Should return consistent error response
            assert response.status_code in [404, 422]

    def test_mcp_flow_validation_across_endpoints(self, client):
        """Test that flow validation is consistent across all endpoints."""
        valid_flows = ["osb", "content_moderation", "research"]
        invalid_flows = ["nonexistent", "invalid_flow", ""]
        
        for flow in valid_flows:
            # Test vector query
            response = client.post("/mcp/agents/vector-query", json={
                "query": "Test query",
                "flow": flow
            })
            assert response.status_code == 200
            
        for flow in invalid_flows:
            if flow:  # Skip empty string as it may be handled differently
                response = client.post("/mcp/agents/vector-query", json={
                    "query": "Test query",
                    "flow": flow
                })
                assert response.status_code == 404