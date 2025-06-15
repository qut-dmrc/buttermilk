

import pytest

from buttermilk._core.agent import Agent


@pytest.mark.anyio
async def test_agent_name_generation():
    """Test that agent_name is generated correctly based on name_components."""
    # Sample data
    inputs_data = {"records": "FETCH.outputs[]||*.records[]", "template": "judge", "model": "gemini25pro", "criteria": "trans_factored"}
    name_components = ["⚖️", "role", "model", "criteria", "unique_identifier"]

    # Mock AgentConfig and Agent
    class MockAgent(Agent):
        async def _process(self, *args, **kwargs):
            return None  # Dummy implementation

    # Instantiate Agent with mocked data
    agent = MockAgent(
        unique_identifier="test_agent_id",
        role="TestRole",
        name_components=name_components,
        inputs=inputs_data,
        parameters=inputs_data,
        session_id="test_session_id",
    )

    # Expected agent name
    expected_name = "⚖️ TESTROLE gemini25pro trans_factored"

    # Assert that the generated agent name matches the expected name
    assert agent.agent_name == expected_name


@pytest.mark.anyio
async def test_agent_name_generation_empty_components():
    """Test that agent_name falls back to agent_id when name_components resolve to empty strings."""
    class MockAgent(Agent):
        async def _process(self, *args, **kwargs):
            return None  # Dummy implementation

    agent = MockAgent(
        agent_id="test_agent_id",
        unique_identifier="test_agent_id",
        role="",
        name_components=["role"],  # Empty role
        inputs={},
        parameters={},
        session_id="test_session_id",
    )

    assert agent.agent_name == "test_agent_id"


@pytest.mark.anyio
async def test_agent_name_generation_jmespath_failure():
    """Test that agent_name handles JMESPath expression failures gracefully."""
    class MockAgent(Agent):
        async def _process(self, *args, **kwargs):
            return None  # Dummy implementation

    agent = MockAgent(
        agent_id="test_agent_id",
        unique_identifier="test_agent_id",
        role="TestRole",
        name_components=["nonexistent.field"],  # Invalid JMESPath expression
        inputs={},
        parameters={},
        session_id="test_session_id",
    )

    assert agent.agent_name == "TESTROLE-test_agent_id"


@pytest.mark.anyio
async def test_agent_name_generation_short_string_component():
    """Test that agent_name includes short string components directly."""
    class MockAgent(Agent):
        async def _process(self, *args, **kwargs):
            return None  # Dummy implementation

    agent = MockAgent(
        agent_id="test_agent_id",
        unique_identifier="test_agent_id",
        role="TestRole",
        name_components=["OK"],  # Short string component
        inputs={},
        parameters={},
        session_id="test_session_id",
    )

    assert agent.agent_name == "OK"
