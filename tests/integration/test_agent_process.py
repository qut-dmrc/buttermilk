"""Integration tests for individual agent processes.

This module provides utilities and test cases to run full examples of individual agent processes
in isolation, ensuring they work as expected and identifying common failure patterns.
"""

from unittest.mock import MagicMock

import pytest


pytestmark = pytest.mark.anyio
# Try to import LLMJudge, but don't fail if it doesn't exist
try:
    from buttermilk.agents.judge import Reasons
except ImportError:
    # Create placeholder for tests if not available
    Reasons = MagicMock()
    Judge = MagicMock()


# Deleted mock_weave fixture and TestScorerAgent class - tests were mocking internal
# implementation details instead of testing real agent behavior. To test scorer, create
# integration tests that use real configurations and verify outputs, not mock assertions.


@pytest.mark.anyio
class TestDifferentiatorAgent:
    """Integration tests for the differentiator agent.

    This specifically tests the agent with regard to the BaseModel __private_attributes__
    error seen in the logs.
    """

    async def test_pydantic_model_compatibility(self):
        """Test compatibility of Pydantic models to catch attribute errors."""
        # This tests the compatibility of Pydantic models that could have issues with
        # __private_attributes__ which appeared in the error logs

        from pydantic import BaseModel, Field, PrivateAttr

        # Create a model that uses private attributes and verify compatibility
        class TestModel(BaseModel):
            field1: str = Field(default="test")
            _private: str = PrivateAttr(default="private")

        model = TestModel()

        # With Pydantic v1, we should have __fields__
        # With Pydantic v2, we should have __pydantic_fields__
        assert hasattr(model, "__pydantic_fields__") or hasattr(model, "__fields__")

        # Verify we can access private attributes correctly
        assert model._private == "private"

        data = model.model_dump()

        assert "field1" in data
        assert "_private" not in data  # Private attributes should not be in the output dict
