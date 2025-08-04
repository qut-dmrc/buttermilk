"""Test AgentTrace serialization with new run_info structure."""

import json
from datetime import datetime

import pytest

from buttermilk._core.bm_init import BM, SessionInfo
from buttermilk._core.config import AgentConfig
from buttermilk._core.contract import AgentInput, AgentTrace


@pytest.fixture
def mock_session_info():
    """Create a mock SessionInfo for testing."""
    return SessionInfo(
        name="test_project",
        job="test_job",
        platform="local",
        run_id="test-run-123",
        save_dir="/tmp/test",
        flow_api="http://localhost:8000/flow/",
    )


@pytest.fixture
def mock_bm(mock_session_info, monkeypatch):
    """Create a mock BM instance with test session info."""
    bm_config = {
        "run_info": mock_session_info,
        "save_dir_base": "/tmp",
        "connections": [],
    }
    bm = BM(**bm_config)
    
    # Monkeypatch the global bm instance
    import buttermilk
    monkeypatch.setattr(buttermilk, "buttermilk", bm)
    
    return bm


@pytest.fixture
def agent_config():
    """Create a mock agent configuration."""
    return AgentConfig(
        name="test_agent",
        role="TESTER", 
        instructions="Test instructions",
    )


@pytest.fixture
def agent_input():
    """Create a mock agent input."""
    return AgentInput(
        inputs={"prompt": "test prompt"},
        parameters={"model": "test-model"},
    )


def test_agenttrace_serializes_runinfo_correctly(mock_bm, agent_config, agent_input):
    """Test that AgentTrace correctly serializes run_info from BM instance."""
    # Create AgentTrace which should pick up run_info from bm
    trace = AgentTrace(
        call_id="test-call-123",
        agent_id="test-agent",
        agent_info=agent_config,
        inputs=agent_input,
        outputs={"result": "test output"},
        timestamp=datetime.utcnow(),
    )
    
    # Test serialization
    serialized = trace.model_dump(mode="json")
    
    # Verify the structure matches what BigQuery expects
    assert "run_info" in serialized
    assert isinstance(serialized["run_info"], dict)
    
    # Check all required fields are present
    run_info = serialized["run_info"]
    assert "name" in run_info
    assert "job" in run_info
    assert "run_id" in run_info
    assert "platform" in run_info
    assert "save_dir" in run_info
    assert "flow_api" in run_info
    
    # Verify values match
    assert run_info["name"] == "test_project"
    assert run_info["job"] == "test_job"
    assert run_info["run_id"] == "test-run-123"
    assert run_info["platform"] == "local"
    # save_dir gets modified during BM initialization to include full path
    assert run_info["save_dir"] == "/tmp/test_project/test_job/test-run-123"
    assert run_info["flow_api"] == "http://localhost:8000/flow/"


def test_agenttrace_runinfo_is_json_serializable(mock_bm, agent_config, agent_input):
    """Test that AgentTrace run_info can be serialized to JSON for BigQuery."""
    trace = AgentTrace(
        call_id="test-call-456",
        agent_id="test-agent-2",
        agent_info=agent_config,
        inputs=agent_input,
        outputs={"status": "success"},
        timestamp=datetime.utcnow(),
    )
    
    serialized = trace.model_dump(mode="json")
    
    # Ensure run_info can be JSON serialized (required for BigQuery JSON field)
    run_info_json = json.dumps(serialized["run_info"])
    assert isinstance(run_info_json, str)
    
    # Verify it can be deserialized back
    deserialized = json.loads(run_info_json)
    assert deserialized == serialized["run_info"]


def test_agenttrace_handles_missing_bm_gracefully(monkeypatch, agent_config, agent_input):
    """Test that AgentTrace handles missing BM instance gracefully."""
    # Remove the global bm instance
    import buttermilk
    monkeypatch.delattr(buttermilk, "buttermilk", raising=False)
    
    # Create AgentTrace without bm available
    trace = AgentTrace(
        call_id="test-call-789",
        agent_id="test-agent-3",
        agent_info=agent_config,
        inputs=agent_input,
        outputs={"error": "no bm"},
        timestamp=datetime.utcnow(),
    )
    
    serialized = trace.model_dump(mode="json")
    
    # When run_info is None, it's excluded from serialization due to exclude_none=True
    # Note: If the BigQuery schema marks run_info as REQUIRED, it must always be present and non-null.
    # This test checks behavior when run_info is missing, which is only valid if the field is NULLABLE.
    assert "run_info" not in serialized or serialized.get("run_info") is None