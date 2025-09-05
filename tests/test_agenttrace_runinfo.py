"""Test AgentTrace serialization with new session_info structure."""

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
        "session_info": mock_session_info,
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
    """Test that AgentTrace correctly serializes session_info from BM instance."""
    # Create AgentTrace which should pick up session_info from bm
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
    assert "session_info" in serialized
    assert isinstance(serialized["session_info"], dict)
    
    # Check all required fields are present
    session_info = serialized["session_info"]
    assert "name" in session_info
    assert "job" in session_info
    assert "run_id" in session_info
    assert "platform" in session_info
    assert "save_dir" in session_info
    assert "flow_api" in session_info
    
    # Verify values match
    assert session_info["name"] == "test_project"
    assert session_info["job"] == "test_job"
    assert session_info["run_id"] == "test-run-123"
    assert session_info["platform"] == "local"
    # save_dir gets modified during BM initialization to include full path
    assert session_info["save_dir"] == "/tmp/test_project/test_job/test-run-123"
    assert session_info["flow_api"] == "http://localhost:8000/flow/"


def test_agenttrace_runinfo_is_json_serializable(mock_bm, agent_config, agent_input):
    """Test that AgentTrace session_info can be serialized to JSON for BigQuery."""
    trace = AgentTrace(
        call_id="test-call-456",
        agent_id="test-agent-2",
        agent_info=agent_config,
        inputs=agent_input,
        outputs={"status": "success"},
        timestamp=datetime.utcnow(),
    )
    
    serialized = trace.model_dump(mode="json")

    # Ensure session_info can be JSON serialized (required for BigQuery JSON field)
    session_info_json = json.dumps(serialized["session_info"])
    assert isinstance(session_info_json, str)
    
    # Verify it can be deserialized back
    deserialized = json.loads(session_info_json)
    assert deserialized == serialized["session_info"]


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

    # When session_info is None, it's excluded from serialization due to exclude_none=True
    # Note: If the BigQuery schema marks session_info as REQUIRED, it must always be present and non-null.
    # This test checks behavior when session_info is missing, which is only valid if the field is NULLABLE.
    assert "session_info" not in serialized or serialized.get("session_info") is None
