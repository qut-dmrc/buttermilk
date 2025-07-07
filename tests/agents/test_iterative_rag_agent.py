"""Tests for IterativeRagAgent."""

import pytest
from unittest.mock import MagicMock, AsyncMock

# Patch bm before importing any buttermilk modules that use it
import buttermilk.agents.llm
mock_bm_instance = MagicMock()
buttermilk.agents.llm.get_bm = MagicMock(return_value=mock_bm_instance)

from buttermilk.agents.rag.iterative_rag_agent import IterativeRagAgent
from buttermilk._core.contract import ToolOutput
from autogen_core.models import CreateResult, RequestUsage


# Mock tool call for testing
class MockToolCall:
    def __init__(self, id, function):
        self.id = id
        self.function = MagicMock()
        self.function.name = function['name']
        self.function.arguments = function['arguments']


@pytest.mark.asyncio
async def test_iterative_rag_agent_can_call_tool_multiple_times(mocker):
    """Test that the IterativeRagAgent can call tools multiple times in one session."""
    # 1. Mock a search tool
    mock_search_tool = MagicMock()
    mock_search_tool.name = "search_tool"
    mock_search_tool.run = AsyncMock(return_value=ToolOutput(content="some result", call_id="mock_call_id_1", function_name="search_tool", name="search_tool", results=["some result"]))

    # 2. Mock the LLM to simulate iterative calls
    mock_llm_client = MagicMock()
    mock_llm_client.call_chat = AsyncMock(side_effect=[
        # First call: LLM decides to use the search tool
        CreateResult(content="", thought="Thinking about initial search", finish_reason="function_calls", usage=RequestUsage(prompt_tokens=10, completion_tokens=10), cached=False, tool_calls=[MockToolCall(id="call_1", function={'name': 'search_tool', 'arguments': '{"query": "initial query"}'})]),
        # Second call: LLM decides to use the search tool again
        CreateResult(content="", thought="Refining search", finish_reason="function_calls", usage=RequestUsage(prompt_tokens=10, completion_tokens=10), cached=False, tool_calls=[MockToolCall(id="call_2", function={'name': 'search_tool', 'arguments': '{"query": "follow-up query"}'})]),
        # Final call: LLM synthesizes the answer
        CreateResult(content="Final answer based on two searches", thought="Synthesizing results", finish_reason="stop", usage=RequestUsage(prompt_tokens=10, completion_tokens=10), cached=False)
    ])

    mock_bm_instance.llms.get_autogen_chat_client.return_value = mock_llm_client

    # 3. Create agent instance with mocked tools
    agent = IterativeRagAgent(
        name="test_agent",
        parameters={"model": "fake-model", "max_iterations": 3},
        tools={"search_tool": mock_search_tool},
        _tools_list=[mock_search_tool],
    )
    agent._fill_template = AsyncMock(return_value=[])

    # 4. Create test input message
    from buttermilk._core.contract import AgentInput
    test_message = AgentInput(
        inputs={"prompt": "test question"},
        context=[],
        parameters={"model": "fake-model"},
        records=[]
    )

    # 5. Call the agent
    result = await agent._process(message=test_message)

    # 6. Verify that the LLM was called multiple times and tools were executed
    assert mock_llm_client.call_chat.call_count == 3
    assert mock_search_tool.run.call_count == 2
    assert "Final answer based on two searches" in str(result.outputs)


def test_iterative_rag_agent_uses_correct_template():
    """Test that IterativeRagAgent uses the iterative_rag template."""
    agent = IterativeRagAgent(
        name="test_agent",
        parameters={"model": "fake-model"},
    )
    assert agent.template == "iterative_rag"