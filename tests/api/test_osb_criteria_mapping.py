"""Test OSB criteria to query parameter mapping."""

import pytest
from buttermilk._core.config import RunRequest
from buttermilk.api.services.message_service import MessageService
from buttermilk.api.osb_message_enhancements import enhance_run_request_if_osb


@pytest.mark.anyio
async def test_osb_criteria_mapped_to_query():
    """Test that 'criteria' is mapped to 'query' for OSB flows."""
    # Simulate the message from frontend
    frontend_message = {
        "type": "run_flow",
        "flow": "osb",
        "record_id": "test-record-123",
        "criteria": "What are the policy implications of this content?"
    }
    
    # Process through MessageService
    result = await MessageService.process_message_from_ui(frontend_message.copy())
    
    # Verify it's a RunRequest
    assert isinstance(result, RunRequest)
    assert result.flow == "osb"
    assert result.record_id == "test-record-123"
    
    # Check that criteria was mapped to query and osb_query
    assert "query" in result.parameters
    assert "osb_query" in result.parameters
    assert result.parameters["query"] == "What are the policy implications of this content?"
    assert result.parameters["osb_query"] == "What are the policy implications of this content?"
    
    # Original criteria should still be there
    assert result.parameters["criteria"] == "What are the policy implications of this content?"


@pytest.mark.anyio
async def test_non_osb_flow_criteria_unchanged():
    """Test that non-OSB flows don't have criteria mapped to query."""
    # Simulate a non-OSB flow message
    frontend_message = {
        "type": "run_flow",
        "flow": "some_other_flow",
        "record_id": "test-record-456",
        "criteria": "Some criteria text"
    }
    
    # Process through MessageService
    result = await MessageService.process_message_from_ui(frontend_message.copy())
    
    # Verify it's a RunRequest
    assert isinstance(result, RunRequest)
    assert result.flow == "some_other_flow"
    
    # Check that criteria remains as criteria
    assert "criteria" in result.parameters
    assert result.parameters["criteria"] == "Some criteria text"
    
    # Query should not be added
    assert "query" not in result.parameters
    assert "osb_query" not in result.parameters


@pytest.mark.anyio
async def test_osb_flow_with_empty_criteria():
    """Test OSB flow with empty criteria."""
    frontend_message = {
        "type": "run_flow",  
        "flow": "osb",
        "record_id": "test-record-789",
        "criteria": ""
    }
    
    # Process through MessageService
    result = await MessageService.process_message_from_ui(frontend_message.copy())
    
    # Should still process but with empty query
    assert isinstance(result, RunRequest)
    assert result.flow == "osb"
    
    # Empty criteria should map to empty query
    assert result.parameters.get("query") == ""
    assert result.parameters.get("osb_query") == ""


@pytest.mark.anyio
async def test_osb_flow_with_direct_query():
    """Test OSB flow when query is provided directly (not via criteria)."""
    # If someone sends query directly instead of criteria
    frontend_message = {
        "type": "run_flow",
        "flow": "osb",
        "record_id": "test-record-999",
        "query": "Direct query text"
    }
    
    # Process through MessageService
    result = await MessageService.process_message_from_ui(frontend_message.copy())
    
    # Should use the direct query
    assert isinstance(result, RunRequest)
    assert result.flow == "osb"
    assert result.parameters.get("query") == "Direct query text"
    assert result.parameters.get("osb_query") == "Direct query text"