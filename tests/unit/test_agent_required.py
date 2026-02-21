"""Test AgentConfig required field for input whitelisting."""

import pytest
<<<<<<< HEAD

=======
>>>>>>> origin/stable
from buttermilk._core.config import AgentConfig


class TestAgentConfigRequiredField:
    """Test required field on AgentConfig."""

    def test_agentconfig_has_required_field_in_schema(self):
        """Test that AgentConfig has required field defined in model schema."""
        # This should fail because 'required' is not in AgentConfig.model_fields
<<<<<<< HEAD
        assert "required" in AgentConfig.model_fields, "AgentConfig must have 'required' field defined in schema, not just accepted via extra='allow'"
=======
        assert "required" in AgentConfig.model_fields, (
            "AgentConfig must have 'required' field defined in schema, "
            "not just accepted via extra='allow'"
        )
>>>>>>> origin/stable


class TestAgentInputFiltering:
    """Test that Agent filters inputs based on required field."""

    def test_agent_has_required_inputs_property(self):
        """Test that Agent exposes required_inputs property from config."""
        from buttermilk._core.agent import Agent

        # Check the Agent class has required_inputs property
<<<<<<< HEAD
        assert hasattr(Agent, "required_inputs"), "Agent must have 'required_inputs' property to expose config.required"
=======
        assert hasattr(Agent, "required_inputs"), (
            "Agent must have 'required_inputs' property to expose config.required"
        )
>>>>>>> origin/stable

    @pytest.mark.anyio
    async def test_inputs_filtered_to_required_only(self):
        """Test that only required inputs are kept, others filtered out."""
<<<<<<< HEAD
        from unittest.mock import MagicMock

        from buttermilk._core.agent import Agent
        from buttermilk._core.config import AgentConfig
        from buttermilk._core.contract import AgentInput
=======
        from buttermilk._core.config import AgentConfig
        from buttermilk._core.contract import AgentInput
        from buttermilk._core.agent import Agent
        from unittest.mock import MagicMock, AsyncMock
>>>>>>> origin/stable

        # Create config with required=["criteria"]
        config = AgentConfig(
            role="judge",
            description="Test agent",
            required=["criteria"],
            inputs={},
        )

        # Create a mock agent with this config
        # We need to test _add_state_to_input behavior
        agent = MagicMock(spec=Agent)
        agent._config = config
        agent.required_inputs = config.required
        agent._data = {"criteria": "test_value", "extra": "should_be_filtered"}
        agent.inputs = {}
        agent.record_mapping = None
        agent.context_mapping = None
        agent.agent_id = "TEST-001"

        # Create input with extra keys that should be filtered
        agent_input = AgentInput(
            inputs={"criteria": "test_value", "extra_field": "should_be_filtered", "another": "also_filtered"},
        )

        # Call the real _add_state_to_input method
        result = await Agent._add_state_to_input(agent, agent_input)

        # Should only have "criteria", not "extra_field" or "another"
        assert "criteria" in result.inputs
        assert "extra_field" not in result.inputs
        assert "another" not in result.inputs
        assert result.inputs["criteria"] == "test_value"

    @pytest.mark.anyio
    async def test_missing_required_input_raises_fatal_error(self):
        """Test that FatalError is raised when a required input is missing."""
<<<<<<< HEAD
        from unittest.mock import MagicMock

        from buttermilk._core.agent import Agent
        from buttermilk._core.config import AgentConfig
        from buttermilk._core.contract import AgentInput
        from buttermilk._core.exceptions import FatalError
=======
        from buttermilk._core.config import AgentConfig
        from buttermilk._core.contract import AgentInput
        from buttermilk._core.agent import Agent
        from buttermilk._core.exceptions import FatalError
        from unittest.mock import MagicMock
>>>>>>> origin/stable

        # Create config with required=["criteria", "model"]
        config = AgentConfig(
            role="judge",
            description="Test agent",
            required=["criteria", "model"],  # Both required
            inputs={},
        )

        # Create a mock agent with this config
        agent = MagicMock(spec=Agent)
        agent._config = config
        agent.required_inputs = config.required
        agent._data = {}
        agent.inputs = {}
        agent.record_mapping = None
        agent.context_mapping = None
        agent.agent_id = "TEST-001"

        # Create input with only "criteria" - missing "model"
        agent_input = AgentInput(
            inputs={"criteria": "test_value"},  # "model" is missing!
        )

        # Should raise FatalError because "model" is required but missing
        with pytest.raises(FatalError, match="model"):
            await Agent._add_state_to_input(agent, agent_input)
