"""Test AgentConfig required field for input whitelisting."""

import pytest
from buttermilk._core.config import AgentConfig


class TestAgentConfigRequiredField:
    """Test required field on AgentConfig."""

    def test_agentconfig_has_required_field_in_schema(self):
        """Test that AgentConfig has required field defined in model schema."""
        # This should fail because 'required' is not in AgentConfig.model_fields
        assert "required" in AgentConfig.model_fields, (
            "AgentConfig must have 'required' field defined in schema, "
            "not just accepted via extra='allow'"
        )
