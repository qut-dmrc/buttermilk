"""Test the RAG agent."""

from unittest.mock import MagicMock

# Patch bm before importing any buttermilk modules that use it
from buttermilk.agents.rag.simple_rag_agent import RagAgent


# Mock tool call for testing
class MockToolCall:
    def __init__(self, id, function):
        self.id = id
        self.function = MagicMock()
        self.function.name = function["name"]
        self.function.arguments = function["arguments"]


def test_iterative_rag_agent_uses_correct_template():
    """Test that RagAgent uses the iterative_rag template."""
    agent = RagAgent(
        name="test_agent",
        parameters={"model": "fake-model"},
    )
    assert agent.template == "iterative_rag"

# should also test that the agent returns ResearchResult
