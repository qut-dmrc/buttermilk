"""Test the IterativeRagAgent."""

import pytest
from unittest.mock import MagicMock, AsyncMock
import json

# Patch bm before importing any buttermilk modules that use it
import buttermilk.agents.llm
from autogen_core.models import CreateResult, RequestUsage


# Define a simple MockToolCall class to match the expected structure
class MockToolCall:
    def __init__(self, id: str, function: dict):
        self.id = id
        self.function = function


@pytest.mark.asyncio
async def test_iterative_rag_agent_can_call_tool_multiple_times(mocker):
    """Verify that the agent can call a tool multiple times in a single turn."""
    # Mock buttermilk.agents.llm.bm and its llms attribute at the very beginning
    mock_bm_instance = MagicMock()
    mock_bm_instance.llms = MagicMock()
    mocker.patch('buttermilk.agents.llm.bm', new=mock_bm_instance)

    # Import IterativeRagAgent and Agent after patching bm
    from buttermilk.agents.rag.iterative_rag_agent import IterativeRagAgent
    from buttermilk._core.agent import Agent
    from buttermilk._core.contract import AgentInput, ToolOutput

    # 1. Mock the search tool
    mock_search_tool = MagicMock(spec=Agent)
    mock_search_tool.name = "search_tool"
    mock_search_tool.run = AsyncMock(return_value=ToolOutput(content="some result", call_id="mock_call_id_1", function_name="search_tool", name="search_tool", results=["some result"]))

    # 2. Mock the LLM to simulate iterative calls and tool execution
    mock_llm_client = MagicMock()

    async def call_chat_side_effect(*args, **kwargs):
        messages = kwargs.get('messages', [])
        last_message = messages[-1] if messages else None

        # Simulate LLM deciding to call a tool
        if "tool_code" in last_message.content:
            # Extract tool call from the content (simplified for this mock)
            # In a real scenario, you'd parse the tool_calls from the LLM's response
            tool_call_str = last_message.content
            if "search_tool" in tool_call_str:
                # Simulate tool execution
                tool_output = await mock_search_tool.run(query="mock query")
                return CreateResult(content=tool_output.content, thought="Tool executed", finish_reason="tool_calls", usage=RequestUsage(prompt_tokens=10, completion_tokens=10), cached=False)
        
        # Simulate LLM synthesizing final answer
        return CreateResult(content="Final answer based on two searches", thought="Synthesizing results", finish_reason="stop", usage=RequestUsage(prompt_tokens=10, completion_tokens=10), cached=False)

    mock_llm_client.call_chat = AsyncMock(side_effect=call_chat_side_effect)

    mock_bm_instance.llms.get_autogen_chat_client.return_value = mock_llm_client

    # 3. Configure and run the agent
    agent = IterativeRagAgent(
        name="test_agent",
        parameters={"model": "fake-model", "template": "iterative_rag"},
        tools={"search_tool": mock_search_tool}
    )

    chat_history = AgentInput(context=[])
    await agent.invoke(chat_history, public_callback=AsyncMock())

    # 4. Assert that the search tool was called twice
    assert mock_search_tool.run.call_count == 2

def test_iterative_rag_agent_uses_correct_template():
    """Test that IterativeRagAgent uses the iterative_rag template by default."""
    from buttermilk.agents.rag.iterative_rag_agent import IterativeRagAgent
    agent = IterativeRagAgent(
        name="test_agent",
        parameters={"model": "fake-model"},
    )
    assert agent.template == "iterative_rag"