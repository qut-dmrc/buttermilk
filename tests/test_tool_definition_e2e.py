"""End-to-end tests for tool definition system with complete flows."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Any

from buttermilk._core import AgentInput, StepRequest
from buttermilk._core.agent import Agent, ManagerMessage
from buttermilk._core.contract import AgentOutput, AgentTrace
from buttermilk._core.constants import END, MANAGER
from buttermilk._core.tool_definition import AgentToolDefinition, UnifiedRequest
from buttermilk._core.mcp_decorators import tool, MCPRoute
from buttermilk.agents.flowcontrol.structured_llmhost import StructuredLLMHostAgent
from buttermilk.agents.llm import LLMAgent
from buttermilk.mcp.server import MCPServer
from buttermilk._core.tool_definition import MCPServerConfig

# Use anyio for async tests
pytestmark = pytest.mark.anyio


class ResearchAgent(LLMAgent):
    """Mock research agent for E2E testing."""
    
    @tool
    @MCPRoute("/search", permissions=["read:data"])
    async def search_documents(self, query: str, max_results: int = 10) -> dict[str, Any]:
        """Search for relevant documents."""
        await asyncio.sleep(0.1)  # Simulate async work
        return {
            "query": query,
            "results": [
                {"id": f"doc_{i}", "title": f"Document {i}", "relevance": 0.9 - i*0.1}
                for i in range(min(max_results, 3))
            ],
            "total_found": min(max_results, 3)
        }
    
    @tool
    async def analyze_document(self, doc_id: str) -> dict[str, Any]:
        """Analyze a specific document."""
        await asyncio.sleep(0.05)  # Simulate analysis
        return {
            "doc_id": doc_id,
            "summary": f"Summary of {doc_id}",
            "key_points": ["Point 1", "Point 2", "Point 3"],
            "sentiment": "neutral"
        }


class WriterAgent(Agent):
    """Mock writer agent for E2E testing."""
    
    async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput:
        """Process writing requests."""
        content = message.inputs.get("content", "")
        style = message.inputs.get("style", "formal")
        
        return AgentOutput(
            agent_id=self.agent_id,
            outputs={
                "written_content": f"[{style.upper()}] {content}",
                "word_count": len(content.split()),
                "style": style
            }
        )
    
    @tool
    def generate_summary(self, points: list[str], max_words: int = 100) -> str:
        """Generate a summary from key points."""
        summary = f"Summary of {len(points)} points (max {max_words} words): "
        summary += "; ".join(points[:3])
        return summary
    
    @tool
    @MCPRoute("/format", permissions=["write"])
    def format_content(self, text: str, format_type: str = "markdown") -> dict[str, str]:
        """Format content in specified format."""
        formatted = {
            "markdown": f"# {text}\n\nFormatted as markdown.",
            "html": f"<h1>{text}</h1><p>Formatted as HTML.</p>",
            "plain": text
        }
        return {
            "original": text,
            "formatted": formatted.get(format_type, text),
            "format": format_type
        }


class TestEndToEndFlows:
    """Test complete end-to-end flows."""
    
    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        research = ResearchAgent(
            agent_name="researcher",
            model_name="test-model",
            role="RESEARCHER",
            parameters={"model": "test-model", "template": "agent_tool_description"}
        )
        research._model = "test-model"
        
        writer = WriterAgent(
            agent_name="writer",
            model_name="test-model",
            role="WRITER"
        )
        
        return {
            "RESEARCHER": research,
            "WRITER": writer
        }
    
    async def test_complete_research_flow(self, mock_agents):
        """Test a complete research and writing flow."""
        # Create structured host
        host = StructuredLLMHostAgent(
            agent_name="host",
            model_name="test-model",
            role="HOST",
            parameters={"model": "test-model", "template": "host_structured_tools"}
        )
        host._model = "test-model"
        
        # Setup host with agents
        host._participants = mock_agents
        host.tools = []  # tools should be a list
        host.callback_to_groupchat = AsyncMock()
        
        # Track all StepRequests
        step_requests = []
        
        async def capture_steps(step_request):
            step_requests.append(step_request)
            return None
        
        host.callback_to_groupchat.side_effect = capture_steps
        
        # Initialize host
        await host._initialize(callback_to_groupchat=host.callback_to_groupchat)
        
        # Verify tools were registered
        tool_names = [t.name for t in host._tools_list]
        assert "researcher.search_documents" in tool_names
        assert "researcher.analyze_document" in tool_names
        assert "writer.generate_summary" in tool_names
        assert "writer.format_content" in tool_names
        
        # Simulate flow: Search -> Analyze -> Summarize -> Format
        
        # 1. Search documents
        search_tool = next(t for t in host._tools_list if t.name == "researcher.search_documents")
        await search_tool._func(query="AI safety", max_results=5)
        
        assert len(step_requests) == 1
        assert step_requests[0].role == "RESEARCHER"
        assert step_requests[0].inputs["tool"] == "search_documents"
        assert step_requests[0].inputs["tool_inputs"]["query"] == "AI safety"
        
        # 2. Analyze a document
        analyze_tool = next(t for t in host._tools_list if t.name == "researcher.analyze_document")
        await analyze_tool._func(doc_id="doc_0")
        
        assert len(step_requests) == 2
        assert step_requests[1].role == "RESEARCHER"
        assert step_requests[1].inputs["tool"] == "analyze_document"
        
        # 3. Generate summary
        summary_tool = next(t for t in host._tools_list if t.name == "writer.generate_summary")
        await summary_tool._func(points=["Point 1", "Point 2"], max_words=50)
        
        assert len(step_requests) == 3
        assert step_requests[2].role == "WRITER"
        assert step_requests[2].inputs["tool"] == "generate_summary"
        
        # 4. Format content
        format_tool = next(t for t in host._tools_list if t.name == "writer.format_content")
        await format_tool._func(text="Final summary", format_type="markdown")
        
        assert len(step_requests) == 4
        assert step_requests[3].role == "WRITER"
        assert step_requests[3].inputs["tool"] == "format_content"
    
    async def test_mcp_server_integration(self, mock_agents):
        """Test MCP server integration with agents."""
        # Create MCP server
        config = MCPServerConfig(mode="embedded", auth_required=False)
        mock_orchestrator = Mock()
        server = MCPServer(config=config, orchestrator=mock_orchestrator)
        
        # Register agent tools with MCP server
        for agent in mock_agents.values():
            for tool_def in agent.get_tool_definitions():
                server.register_route(tool_def)
        
        # Check registered routes
        assert "/search" in server._routes
        assert "/format" in server._routes
        
        # Test discovery endpoint
        from fastapi.testclient import TestClient
        server.register_discovery_endpoints()
        client = TestClient(server.app)
        
        response = client.get("/mcp/tools")
        assert response.status_code == 200
        
        tools = response.json()["tools"]
        tool_names = [t["name"] for t in tools]
        assert "search_documents" in tool_names
        assert "format_content" in tool_names
        
        # Check permissions
        search_tool = next(t for t in tools if t["name"] == "search_documents")
        assert search_tool["permissions"] == ["read:data"]
        
        format_tool = next(t for t in tools if t["name"] == "format_content")
        assert format_tool["permissions"] == ["write"]
    
    async def test_unified_request_flow(self, mock_agents):
        """Test flow using UnifiedRequests."""
        researcher = mock_agents["RESEARCHER"]
        writer = mock_agents["WRITER"]
        
        # Execute research flow via UnifiedRequests
        
        # 1. Search
        search_req = UnifiedRequest(
            target="researcher.search_documents",
            inputs={"query": "climate change", "max_results": 3},
            metadata={"request_id": "test_001"}
        )
        
        search_result = await researcher.handle_unified_request(search_req)
        assert search_result["query"] == "climate change"
        assert len(search_result["results"]) == 3
        
        # 2. Analyze first result
        analyze_req = UnifiedRequest(
            target="researcher.analyze_document",
            inputs={"doc_id": search_result["results"][0]["id"]},
            context={"search_results": search_result}
        )
        
        analysis = await researcher.handle_unified_request(analyze_req)
        assert analysis["doc_id"] == "doc_0"
        assert len(analysis["key_points"]) == 3
        
        # 3. Generate summary from analysis
        summary_req = UnifiedRequest(
            target="writer.generate_summary",
            inputs={
                "points": analysis["key_points"],
                "max_words": 75
            }
        )
        
        summary = await writer.handle_unified_request(summary_req)
        assert "Summary of 3 points" in summary
        
        # 4. Format the summary
        format_req = UnifiedRequest(
            target="writer.format_content",
            inputs={
                "text": summary,
                "format_type": "html"
            }
        )
        
        formatted = await writer.handle_unified_request(format_req)
        assert formatted["format"] == "html"
        assert "<h1>" in formatted["formatted"]
    
    async def test_error_handling_flow(self, mock_agents):
        """Test error handling in the flow."""
        researcher = mock_agents["RESEARCHER"]
        
        # Test invalid tool
        invalid_tool_req = UnifiedRequest(
            target="researcher.nonexistent_tool",
            inputs={}
        )
        
        with pytest.raises(ValueError, match="Tool nonexistent_tool not found"):
            await researcher.handle_unified_request(invalid_tool_req)
        
        # Test missing required parameter
        missing_param_req = UnifiedRequest(
            target="researcher.search_documents",
            inputs={}  # Missing required 'query'
        )
        
        # This should work as the tool has default handling
        # but let's test with a tool that strictly requires params
        
        class StrictAgent(Agent):
            async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
                return AgentOutput(agent_id=self.agent_id, outputs={})
            
            @tool
            def strict_tool(self, required_param: str) -> str:
                """Tool with required parameter."""
                return required_param
        
        strict_agent = StrictAgent(agent_name="strict", model_name="test", role="STRICT")
        
        strict_req = UnifiedRequest(
            target="strict.strict_tool",
            inputs={}  # Missing required_param
        )
        
        with pytest.raises(TypeError, match="required_param"):
            await strict_agent.handle_unified_request(strict_req)
    
    async def test_concurrent_tool_execution(self, mock_agents):
        """Test concurrent execution of multiple tools."""
        researcher = mock_agents["RESEARCHER"]
        
        # Create multiple search requests
        queries = ["AI safety", "climate change", "quantum computing"]
        
        # Execute searches concurrently
        tasks = []
        for query in queries:
            req = UnifiedRequest(
                target="researcher.search_documents",
                inputs={"query": query, "max_results": 2}
            )
            task = researcher.handle_unified_request(req)
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        
        # Verify all completed successfully
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["query"] == queries[i]
            assert len(result["results"]) == 2
    
    async def test_host_message_handling(self, mock_agents):
        """Test host handling of manager messages."""
        host = StructuredLLMHostAgent(
            agent_name="host",
            model_name="test-model",
            role="HOST",
            parameters={"model": "test-model"}
        )
        host._model = "test-model"
        host._participants = mock_agents
        host.callback_to_groupchat = AsyncMock()
        
        await host._initialize(callback_to_groupchat=host.callback_to_groupchat)
        
        # Mock LLM response
        mock_trace = Mock(spec=AgentTrace)
        mock_trace.outputs = "Analysis complete"
        host.invoke = AsyncMock(return_value=mock_trace)
        
        # Send manager message
        manager_msg = ManagerMessage(
            content="Please search for information about renewable energy",
            source="user"
        )
        
        await host._listen(
            message=manager_msg,
            cancellation_token=None,
            source="test",
            public_callback=AsyncMock(),
            message_callback=AsyncMock()
        )
        
        # Verify invoke was called
        host.invoke.assert_called_once()
        call_args = host.invoke.call_args[1]["message"]
        assert call_args.inputs["prompt"] == "Please search for information about renewable energy"
        assert set(call_args.inputs["participants"]) == {"RESEARCHER", "WRITER"}