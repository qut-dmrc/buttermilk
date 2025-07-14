"""Integration tests for OSB flow with new tool definition system."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio
from typing import Any

from buttermilk._core import AgentInput
from buttermilk._core.agent import Agent
from buttermilk._core.contract import AgentOutput
from buttermilk._core.mcp_decorators import tool, MCPRoute
from buttermilk.agents.flowcontrol.structured_llmhost import StructuredLLMHostAgent


class MockRAGAgent(Agent):
    """Mock RAG agent for OSB testing."""
    
    def __init__(self, agent_name: str, **kwargs):
        super().__init__(agent_name=agent_name, **kwargs)
        self.search_called = False
        self.analyze_called = False
    
    async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput:
        """Process RAG requests."""
        query = message.inputs.get("query", "")
        return AgentOutput(
            source=self.agent_name,
            role=self.role,
            outputs={
                "query": query,
                "results": [f"Result from {self.agent_name}"],
                "agent": self.agent_name
            }
        )
    
    @tool
    @MCPRoute("/search", permissions=["read:osb"])
    async def search_osb(self, query: str, limit: int = 10) -> dict[str, Any]:
        """Search OSB vector store."""
        self.search_called = True
        return {
            "query": query,
            "results": [
                {
                    "id": f"osb_{i}",
                    "content": f"OSB result {i} for {query}",
                    "score": 0.9 - (i * 0.1)
                }
                for i in range(min(limit, 3))
            ],
            "total": min(limit, 3)
        }
    
    @tool
    async def analyze_findings(self, findings: list[dict]) -> dict[str, Any]:
        """Analyze OSB search findings."""
        self.analyze_called = True
        return {
            "summary": f"Analyzed {len(findings)} findings",
            "key_insights": ["Insight 1", "Insight 2"],
            "confidence": 0.85
        }


class TestOSBFlowIntegration:
    """Test OSB flow with new tool definition system."""

    @pytest.fixture
    def mock_osb_agents(self):
        """Create mock OSB agents."""
        return {
            "RESEARCHER": MockRAGAgent(
                agent_name="researcher",
                model_name="test",
                role="RESEARCHER"
            ),
            "POLICY_ANALYST": MockRAGAgent(
                agent_name="policy_analyst",
                model_name="test",
                role="POLICY_ANALYST"
            ),
            "FACT_CHECKER": MockRAGAgent(
                agent_name="fact_checker",
                model_name="test",
                role="FACT_CHECKER"
            ),
            "EXPLORER": MockRAGAgent(
                agent_name="explorer",
                model_name="test",
                role="EXPLORER"
            )
        }

    @pytest.mark.anyio
    async def test_osb_host_initialization(self, mock_osb_agents):
        """Test OSB host initializes with agent tools."""
        host = StructuredLLMHostAgent(
            agent_name="assistant",
            model_name="test-model",
            role="ASSISTANT"
        )

        # Setup host
        host._participants = mock_osb_agents
        host.tools = {}
        host.parameters = {
            "model": "test-model",
            "template": "host_structured_tools"
        }
        host.callback_to_groupchat = AsyncMock()

        # Initialize
        await host._initialize(callback_to_groupchat=host.callback_to_groupchat)

        # Verify tools were registered
        tool_names = [tool.name for tool in host._tools_list]

        # Each agent should have 2 tools (search_osb, analyze_findings)
        # Total: 4 agents * 2 tools = 8 tools
        assert len(host._tools_list) == 8

        # Check specific tools
        assert "researcher.search_osb" in tool_names
        assert "researcher.analyze_findings" in tool_names
        assert "policy_analyst.search_osb" in tool_names
        assert "fact_checker.search_osb" in tool_names
        assert "explorer.search_osb" in tool_names

    @pytest.mark.anyio
    async def test_osb_tool_invocation(self, mock_osb_agents):
        """Test invoking OSB agent tools through host."""
        host = StructuredLLMHostAgent(
            agent_name="assistant",
            model_name="test-model",
            role="ASSISTANT"
        )

        # Setup
        host._participants = mock_osb_agents
        host.tools = {}
        host.parameters = {"model": "test-model"}
        host.callback_to_groupchat = AsyncMock()

        await host._initialize(callback_to_groupchat=host.callback_to_groupchat)

        # Find the researcher.search_osb tool
        search_tool = next(
            tool for tool in host._tools_list 
            if tool.name == "researcher.search_osb"
        )

        # Invoke the tool
        await search_tool._func(query="test query", limit=5)

        # Verify callback was called with StepRequest
        host.callback_to_groupchat.assert_called_once()
        step_request = host.callback_to_groupchat.call_args[0][0]

        assert step_request.role == "RESEARCHER"
        assert step_request.inputs["tool"] == "search_osb"
        assert step_request.inputs["tool_inputs"]["query"] == "test query"
        assert step_request.inputs["tool_inputs"]["limit"] == 5

    @pytest.mark.anyio
    async def test_osb_multi_agent_coordination(self, mock_osb_agents):
        """Test coordinating multiple OSB agents."""
        host = StructuredLLMHostAgent(
            agent_name="assistant",
            model_name="test-model",
            role="ASSISTANT"
        )

        # Setup
        host._participants = mock_osb_agents
        host.tools = {}
        host.parameters = {"model": "test-model", "template": "host_structured_tools"}
        host.callback_to_groupchat = AsyncMock()

        await host._initialize(callback_to_groupchat=host.callback_to_groupchat)

        # Simulate calling multiple agents
        tools_to_call = [
            ("researcher.search_osb", {"query": "OSB case 123"}),
            ("policy_analyst.search_osb", {"query": "policy implications"}),
            ("fact_checker.analyze_findings", {"findings": [{"id": "1"}]})
        ]

        for tool_name, inputs in tools_to_call:
            tool = next(t for t in host._tools_list if t.name == tool_name)
            await tool._func(**inputs)

        # Verify all callbacks
        assert host.callback_to_groupchat.call_count == 3

        # Check each call
        calls = host.callback_to_groupchat.call_args_list

        # Researcher call
        assert calls[0][0][0].role == "RESEARCHER"
        assert calls[0][0][0].inputs["tool"] == "search_osb"

        # Policy analyst call
        assert calls[1][0][0].role == "POLICY_ANALYST"
        assert calls[1][0][0].inputs["tool"] == "search_osb"

        # Fact checker call
        assert calls[2][0][0].role == "FACT_CHECKER"
        assert calls[2][0][0].inputs["tool"] == "analyze_findings"



    @pytest.mark.anyio
    async def test_osb_backward_compatibility(self, mock_osb_agents):
        """Test OSB agents work with traditional AgentInput."""
        researcher = mock_osb_agents["RESEARCHER"]

        # Traditional AgentInput
        agent_input = AgentInput(
            inputs={"query": "traditional OSB query"},
            context=[],
            parameters={},
            records=[]
        )

        result = await researcher._process(message=agent_input)

        assert result.outputs["query"] == "traditional OSB query"
        assert result.outputs["agent"] == "researcher"
        assert len(result.outputs["results"]) == 1

    def test_osb_tool_permissions(self, mock_osb_agents):
        """Test OSB tool permissions are properly set."""
        researcher = mock_osb_agents["RESEARCHER"]
        tools = researcher.get_tool_definitions()

        # Find search_osb tool
        search_tool = next(t for t in tools if t.name == "search_osb")

        # Verify permissions
        assert search_tool.permissions == ["read:osb"]
        assert search_tool.mcp_route == "/search"

        # Analyze tool should have no special permissions
        analyze_tool = next(t for t in tools if t.name == "analyze_findings")
        assert analyze_tool.permissions == []
