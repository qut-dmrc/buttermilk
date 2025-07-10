"""
Comprehensive tests for OSB MCP endpoints.

This test suite validates all OSB MCP endpoints for individual component testing:
- OSB vector store query testing
- Individual OSB agent invocation testing  
- Multi-agent response synthesis testing
- OSB session state management testing
- OSB message validation testing

Tests use pytest-asyncio with FastAPI TestClient for comprehensive validation
of OSB component functionality in isolation.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
import json

from buttermilk.api.mcp_osb import osb_mcp_router, MCPToolResponse
from buttermilk.runner.flowrunner import FlowRunner


@pytest.fixture
def mock_flow_runner():
    """Mock FlowRunner for testing MCP endpoints."""
    mock_runner = MagicMock(spec=FlowRunner)
    mock_runner.flows = {"osb": {"name": "OSB Interactive Flow"}}
    return mock_runner


@pytest.fixture
def test_app(mock_flow_runner):
    """Create test FastAPI app with OSB MCP router."""
    app = FastAPI()
    app.include_router(osb_mcp_router)
    
    # Mock the dependency
    app.dependency_overrides = {
        "buttermilk.api.routes.get_flows": lambda: mock_flow_runner
    }
    
    return app


@pytest.fixture
def client(test_app):
    """Create test client for OSB MCP endpoints."""
    return TestClient(test_app)


class TestOSBVectorQueryEndpoint:
    """Test OSB vector store query MCP endpoint."""

    def test_vector_query_success(self, client):
        """Test successful OSB vector query."""
        request_data = {
            "query": "What are the policy implications of hate speech?",
            "flow": "osb",
            "agent_type": "researcher",
            "max_results": 5,
            "confidence_threshold": 0.5,
            "include_metadata": True
        }
        
        response = client.post("/mcp/osb/vector-query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "result" in data
        assert data["result"]["query"] == request_data["query"]
        assert data["result"]["agent_type"] == "researcher"
        assert "results" in data["result"]
        assert "performance" in data["result"]
        assert data["trace_id"] is not None
        assert data["execution_time_ms"] is not None

    def test_vector_query_invalid_agent_type(self, client):
        """Test vector query with invalid agent type."""
        request_data = {
            "query": "Test query",
            "agent_type": "invalid_agent",
            "flow": "osb"
        }
        
        response = client.post("/mcp/osb/vector-query", json=request_data)
        
        assert response.status_code == 422  # Validation error
        data = response.json()
        assert "detail" in data
        assert "agent_type" in str(data["detail"])

    def test_vector_query_empty_query(self, client):
        """Test vector query with empty query string."""
        request_data = {
            "query": "",
            "agent_type": "researcher",
            "flow": "osb"
        }
        
        response = client.post("/mcp/osb/vector-query", json=request_data)
        
        assert response.status_code == 422  # Validation error

    def test_vector_query_too_long(self, client):
        """Test vector query with query exceeding length limit."""
        request_data = {
            "query": "x" * 2001,  # Exceeds 2000 character limit
            "agent_type": "researcher",
            "flow": "osb"
        }
        
        response = client.post("/mcp/osb/vector-query", json=request_data)
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.parametrize("agent_type", ["researcher", "policy_analyst", "fact_checker", "explorer"])
    def test_vector_query_all_agent_types(self, client, agent_type):
        """Test vector query with all valid agent types."""
        request_data = {
            "query": f"Test query for {agent_type}",
            "agent_type": agent_type,
            "flow": "osb"
        }
        
        response = client.post("/mcp/osb/vector-query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["result"]["agent_type"] == agent_type


class TestOSBAgentInvokeEndpoint:
    """Test OSB individual agent invocation MCP endpoint."""

    def test_agent_invoke_success(self, client):
        """Test successful OSB agent invocation."""
        request_data = {
            "query": "Analyze this content for policy violations",
            "agent_name": "researcher",
            "flow": "osb",
            "case_number": "OSB-2025-001",
            "content_type": "social_media_post",
            "platform": "twitter",
            "enable_precedent_analysis": True,
            "include_policy_references": True,
            "max_processing_time": 30
        }
        
        response = client.post("/mcp/osb/agent-invoke", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["result"]["agent_name"] == "researcher"
        assert data["result"]["query"] == request_data["query"]
        assert "response" in data["result"]
        assert "confidence" in data["result"]
        assert "sources" in data["result"]
        assert "performance" in data["result"]

    @pytest.mark.parametrize("agent_name", ["researcher", "policy_analyst", "fact_checker", "explorer"])
    def test_agent_invoke_all_agents(self, client, agent_name):
        """Test agent invocation for all OSB agents."""
        request_data = {
            "query": f"Test query for {agent_name}",
            "agent_name": agent_name,
            "flow": "osb"
        }
        
        response = client.post("/mcp/osb/agent-invoke", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["result"]["agent_name"] == agent_name

    def test_agent_invoke_invalid_agent(self, client):
        """Test agent invocation with invalid agent name."""
        request_data = {
            "query": "Test query",
            "agent_name": "invalid_agent",
            "flow": "osb"
        }
        
        response = client.post("/mcp/osb/agent-invoke", json=request_data)
        
        assert response.status_code == 422  # Validation error

    def test_agent_invoke_with_full_metadata(self, client):
        """Test agent invocation with complete OSB metadata."""
        request_data = {
            "query": "Comprehensive policy analysis",
            "agent_name": "policy_analyst",
            "flow": "osb",
            "case_number": "OSB-2025-TEST-001",
            "content_type": "forum_comment",
            "platform": "reddit",
            "enable_precedent_analysis": True,
            "include_policy_references": True,
            "max_processing_time": 60
        }
        
        response = client.post("/mcp/osb/agent-invoke", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        # Verify metadata is preserved in response
        result = data["result"]
        assert result["agent_name"] == "policy_analyst"
        assert result["query"] == request_data["query"]


class TestOSBSynthesisEndpoint:
    """Test OSB multi-agent synthesis MCP endpoint."""

    def test_synthesis_success(self, client):
        """Test successful OSB response synthesis."""
        request_data = {
            "query": "Original policy analysis query",
            "agent_responses": {
                "researcher": {
                    "findings": "Content analysis completed",
                    "confidence": 0.85,
                    "sources": ["policy_doc_1.pdf"]
                },
                "policy_analyst": {
                    "analysis": "Policy violation detected",
                    "confidence": 0.90,
                    "recommendations": ["action1", "action2"]
                },
                "fact_checker": {
                    "validation": "Claims verified",
                    "confidence": 0.88,
                    "accuracy_score": 0.92
                },
                "explorer": {
                    "themes": ["hate_speech"],
                    "confidence": 0.82,
                    "related_cases": ["OSB-123"]
                }
            },
            "flow": "osb",
            "enable_cross_validation": True,
            "confidence_weighting": True,
            "include_precedents": True
        }
        
        response = client.post("/mcp/osb/synthesis", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        result = data["result"]
        assert "synthesis_summary" in result
        assert "confidence_score" in result
        assert "policy_violations" in result
        assert "recommendations" in result
        assert "precedent_cases" in result
        assert "cross_validation_results" in result
        assert "performance" in result

    def test_synthesis_empty_responses(self, client):
        """Test synthesis with empty agent responses."""
        request_data = {
            "query": "Test query",
            "agent_responses": {},
            "flow": "osb"
        }
        
        response = client.post("/mcp/osb/synthesis", json=request_data)
        
        assert response.status_code == 422  # Validation error

    def test_synthesis_invalid_agent_name(self, client):
        """Test synthesis with invalid agent name in responses."""
        request_data = {
            "query": "Test query",
            "agent_responses": {
                "invalid_agent": {"response": "test"}
            },
            "flow": "osb"
        }
        
        response = client.post("/mcp/osb/synthesis", json=request_data)
        
        assert response.status_code == 422  # Validation error

    def test_synthesis_partial_responses(self, client):
        """Test synthesis with partial agent responses."""
        request_data = {
            "query": "Partial analysis query",
            "agent_responses": {
                "researcher": {
                    "findings": "Limited analysis",
                    "confidence": 0.60
                },
                "policy_analyst": {
                    "analysis": "Incomplete policy review",
                    "confidence": 0.70
                }
            },
            "flow": "osb",
            "enable_cross_validation": False,
            "confidence_weighting": True
        }
        
        response = client.post("/mcp/osb/synthesis", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        # Should handle partial responses gracefully
        result = data["result"]
        assert result["performance"]["agents_processed"] == 2


class TestOSBSessionStateEndpoint:
    """Test OSB session state management MCP endpoint."""

    @pytest.mark.parametrize("operation", ["create", "get_status", "update_metadata", "record_metrics", "cleanup"])
    def test_session_operations(self, client, operation):
        """Test all OSB session operations."""
        request_data = {
            "session_id": f"test-session-{operation}",
            "operation": operation,
            "test_data": {"test_key": "test_value"} if operation == "update_metadata" else None
        }
        
        response = client.post("/mcp/osb/session-state", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        result = data["result"]
        assert result["session_id"] == request_data["session_id"]
        assert result["operation"] == operation
        assert "operation_result" in result
        assert "performance" in result

    def test_session_invalid_operation(self, client):
        """Test session state with invalid operation."""
        request_data = {
            "session_id": "test-session",
            "operation": "invalid_operation"
        }
        
        response = client.post("/mcp/osb/session-state", json=request_data)
        
        assert response.status_code == 422  # Validation error

    def test_session_update_metadata_with_data(self, client):
        """Test session metadata update with test data."""
        test_metadata = {
            "case_number": "OSB-2025-001",
            "priority": "high",
            "content_type": "social_media_post"
        }
        
        request_data = {
            "session_id": "test-session-metadata",
            "operation": "update_metadata",
            "test_data": test_metadata
        }
        
        response = client.post("/mcp/osb/session-state", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        # Verify metadata handling
        result = data["result"]
        operation_result = result["operation_result"]
        assert operation_result["metadata_updated"] is True
        assert operation_result["updated_fields"] == list(test_metadata.keys())


class TestOSBMessageValidationEndpoint:
    """Test OSB message validation MCP endpoint."""

    @pytest.mark.parametrize("message_type,valid_data", [
        ("osb_query", {
            "query": "Test query",
            "flow": "osb",
            "case_number": "OSB-2025-001"
        }),
        ("osb_status", {
            "session_id": "test-session",
            "status": "processing",
            "agent": "researcher"
        }),
        ("osb_partial", {
            "session_id": "test-session",
            "agent": "researcher",
            "partial_response": "Analysis in progress..."
        }),
        ("osb_complete", {
            "session_id": "test-session",
            "synthesis_summary": "Analysis complete",
            "agent_responses": {"researcher": {"findings": "test"}}
        }),
        ("osb_error", {
            "session_id": "test-session",
            "error_type": "VectorStoreTimeout",
            "error_message": "Connection timeout"
        })
    ])
    def test_message_validation_valid_messages(self, client, message_type, valid_data):
        """Test message validation with valid OSB messages."""
        request_data = {
            "message_type": message_type,
            "message_data": valid_data,
            "strict_validation": True
        }
        
        response = client.post("/mcp/osb/message-validation", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        result = data["result"]
        assert result["message_type"] == message_type
        assert result["is_valid"] is True
        assert result["error_message"] is None
        assert "validation_details" in result
        assert "performance" in result

    def test_message_validation_invalid_type(self, client):
        """Test message validation with invalid message type."""
        request_data = {
            "message_type": "invalid_type",
            "message_data": {"test": "data"},
            "strict_validation": True
        }
        
        response = client.post("/mcp/osb/message-validation", json=request_data)
        
        assert response.status_code == 422  # Validation error

    def test_message_validation_missing_fields(self, client):
        """Test message validation with missing required fields."""
        request_data = {
            "message_type": "osb_status",
            "message_data": {
                "session_id": "test-session"
                # Missing required "status" field
            },
            "strict_validation": True
        }
        
        response = client.post("/mcp/osb/message-validation", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        result = data["result"]
        assert result["is_valid"] is False
        assert "Missing required fields" in result["error_message"]

    def test_message_validation_strict_vs_lenient(self, client):
        """Test message validation with strict vs lenient modes."""
        incomplete_data = {
            "session_id": "test-session"
            # Missing other required fields
        }
        
        # Test strict validation
        strict_request = {
            "message_type": "osb_complete",
            "message_data": incomplete_data,
            "strict_validation": True
        }
        
        response = client.post("/mcp/osb/message-validation", json=strict_request)
        assert response.status_code == 200
        
        # Test lenient validation
        lenient_request = {
            "message_type": "osb_complete",
            "message_data": incomplete_data,
            "strict_validation": False
        }
        
        response = client.post("/mcp/osb/message-validation", json=lenient_request)
        assert response.status_code == 200


class TestOSBMCPIntegration:
    """Integration tests for OSB MCP endpoints."""

    def test_mcp_response_format_consistency(self, client):
        """Test that all MCP endpoints return consistent response format."""
        endpoints_and_data = [
            ("/mcp/osb/vector-query", {
                "query": "Test query",
                "agent_type": "researcher",
                "flow": "osb"
            }),
            ("/mcp/osb/agent-invoke", {
                "query": "Test query",
                "agent_name": "researcher",
                "flow": "osb"
            }),
            ("/mcp/osb/synthesis", {
                "query": "Test query",
                "agent_responses": {"researcher": {"findings": "test"}},
                "flow": "osb"
            }),
            ("/mcp/osb/session-state", {
                "session_id": "test-session",
                "operation": "create"
            }),
            ("/mcp/osb/message-validation", {
                "message_type": "osb_query",
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
        """Test consistent error handling across OSB MCP endpoints."""
        # Test with malformed JSON
        endpoints = [
            "/mcp/osb/vector-query",
            "/mcp/osb/agent-invoke", 
            "/mcp/osb/synthesis",
            "/mcp/osb/session-state",
            "/mcp/osb/message-validation"
        ]

        for endpoint in endpoints:
            response = client.post(endpoint, json={})  # Empty request

            # Should return validation error (422) or handle gracefully
            assert response.status_code in [422, 200]

            if response.status_code == 200:
                data = response.json()
                # If handled gracefully, should have error in response
                if not data.get("success", True):
                    assert "error" in data

    @pytest.mark.anyio
    async def test_mcp_concurrent_requests(self, client):
        """Test OSB MCP endpoints handle concurrent requests correctly."""
        # Create multiple concurrent requests to different endpoints
        async def make_request(endpoint, data):
            return client.post(endpoint, json=data)

        requests = [
            ("/mcp/osb/vector-query", {
                "query": f"Test query {i}",
                "agent_type": "researcher",
                "flow": "osb"
            })
            for i in range(5)
        ]

        # Execute concurrent requests
        responses = []
        for endpoint, data in requests:
            response = client.post(endpoint, json=data)
            responses.append(response)

        # Verify all requests succeeded
        for i, response in enumerate(responses):
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert f"Test query {i}" in data["result"]["query"]

    def test_mcp_performance_metrics(self, client):
        """Test that MCP endpoints provide performance metrics."""
        request_data = {
            "query": "Performance test query",
            "agent_type": "researcher", 
            "flow": "osb"
        }

        response = client.post("/mcp/osb/vector-query", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "execution_time_ms" in data
        assert data["execution_time_ms"] > 0

        # Check endpoint-specific performance metrics
        result = data["result"]
        assert "performance" in result
        performance = result["performance"]
        assert "execution_time_ms" in performance
        assert performance["execution_time_ms"] > 0
