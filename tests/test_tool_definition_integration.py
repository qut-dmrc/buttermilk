"""Integration tests for tool definition system with existing flows."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Any

# Use anyio for async tests
pytestmark = pytest.mark.anyio

from buttermilk._core import AgentInput
from buttermilk._core.agent import Agent
from buttermilk._core.contract import AgentOutput
from buttermilk._core.tool_definition import AgentToolDefinition
from buttermilk._core.mcp_decorators import tool, MCPRoute
from buttermilk.agents.flowcontrol.structured_llmhost import StructuredLLMHostAgent
from buttermilk.agents.llm import LLMAgent


class TestAgentWithTools(Agent):
    """Test agent with tool definitions."""
    
    async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput:
        """Process requests."""
        operation = message.inputs.get("operation", "default")
        
        if operation == "analyze":
            # Don't call the tool method here - that's for direct tool invocation
            # Just return a result for the operation
            result = {
                "analysis": f"Analyzed {len(message.inputs.get('data', ''))} characters",
                "format": message.inputs.get("format", "json"),
                "insights": ["insight1", "insight2"]
            }
        elif operation == "search":
            # Don't call the tool method - just simulate the operation
            query = message.inputs.get("query", "")
            limit = message.inputs.get("limit", 10)
            result = {
                "query": query,
                "results": [f"result_{i}" for i in range(min(limit, 5))],
                "total": min(limit, 5)
            }
        else:
            result = {"message": "No specific operation requested"}
        
        return AgentOutput(
            agent_id=self.agent_id,
            outputs=result
        )
    
    @tool
    @MCPRoute("/analyze", permissions=["read"])
    async def analyze_data(self, data: str, format: str = "json") -> dict[str, Any]:
        """Analyze provided data."""
        return {
            "analysis": f"Analyzed {len(data)} characters",
            "format": format,
            "insights": ["insight1", "insight2"]
        }
    
    @tool(name="search_documents")
    def search(self, query: str, limit: int = 10) -> dict[str, Any]:
        """Search for documents."""
        return {
            "query": query,
            "results": [f"result_{i}" for i in range(min(limit, 5))],
            "total": min(limit, 5)
        }


class TestLLMAgentWithTools(LLMAgent):
    """Test LLM agent with tools."""
    
    @tool
    async def summarize(self, text: str, max_length: int = 100) -> str:
        """Summarize text."""
        return text[:max_length] + "..." if len(text) > max_length else text
    
    @MCPRoute("/classify")
    def classify_text(self, text: str) -> dict[str, Any]:
        """Classify text content."""
        return {
            "text": text,
            "category": "general",
            "confidence": 0.95
        }


class TestToolDefinitionIntegration:
    """Test integration of tool definition system."""
    
    def test_agent_tool_extraction(self):
        """Test extracting tools from an agent."""
        agent = TestAgentWithTools(
            agent_name="test_agent",
            model_name="test",
            role="analyzer"
        )
        
        tools = agent.get_tool_definitions()
        
        assert len(tools) == 2
        
        # Check analyze_data tool
        analyze_tool = next(t for t in tools if t.name == "analyze_data")
        assert analyze_tool.description == "Analyze provided data."
        assert analyze_tool.mcp_route == "/analyze"
        assert analyze_tool.permissions == ["read"]
        assert "data" in analyze_tool.input_schema["properties"]
        assert "format" in analyze_tool.input_schema["properties"]
        
        # Check search tool
        search_tool = next(t for t in tools if t.name == "search_documents")
        assert search_tool.description == "Search for documents."
        assert "query" in search_tool.input_schema["properties"]
        assert "limit" in search_tool.input_schema["properties"]
    
    def test_llm_agent_tool_extraction(self):
        """Test extracting tools from LLM agent."""
        # LLMAgent requires parameters with model
        agent = TestLLMAgentWithTools(
            agent_name="llm_agent",
            model_name="test-model",
            role="assistant",
            parameters={"model": "test-model"}
        )
        
        tools = agent.get_tool_definitions()
        
        assert len(tools) == 2
        
        # Check summarize tool
        summarize_tool = next(t for t in tools if t.name == "summarize")
        assert summarize_tool.mcp_route == "/summarize"  # Default route
        
        # Check classify tool
        classify_tool = next(t for t in tools if t.name == "classify_text")
        assert classify_tool.mcp_route == "/classify"
    

    
    async def test_structured_host_with_agent_tools(self):
        """Test StructuredLLMHostAgent with agent tools."""
        # Create test agents
        agent1 = TestAgentWithTools(
            agent_name="analyzer",
            model_name="test",
            role="ANALYZER"
        )
        
        agent2 = TestLLMAgentWithTools(
            agent_name="assistant",
            model_name="test-model",
            role="ASSISTANT",
            parameters={"model": "test-model"}
        )
        
        # Create host
        host = StructuredLLMHostAgent(
            agent_name="host",
            model_name="test-model",
            role="HOST",
            parameters={"model": "test-model"}
        )
        
        # Mock initialization
        host._participants = {
            "ANALYZER": agent1,
            "ASSISTANT": agent2
        }
        host.tools = []  # tools should be a list
        host.parameters = {"model": "test-model"}
        host.callback_to_groupchat = AsyncMock()
        
        # Initialize host using the public method
        await host.initialize(callback_to_groupchat=host.callback_to_groupchat)
        
        # Check tools were registered
        tool_names = [tool.name for tool in host._tools_list]
        assert "analyzer.analyze_data" in tool_names
        assert "analyzer.search_documents" in tool_names
        assert "assistant.summarize" in tool_names
        assert "assistant.classify_text" in tool_names
        
        # Total: 4 agent tools
        assert len(host._tools_list) == 4
    
    async def test_backward_compatibility(self):
        """Test that existing agents without tools still work."""
        
        class LegacyAgent(Agent):
            """Agent without tool decorators."""
            
            async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput:
                return AgentOutput(
                    agent_id=self.agent_id,
                    outputs={"message": "Legacy response"}
                )
        
        agent = LegacyAgent(
            agent_name="legacy",
            model_name="test",
            role="LEGACY"
        )
        
        # Should return empty tool list
        tools = agent.get_tool_definitions()
        assert tools == []
    
    async def test_mixed_sync_async_tools(self):
        """Test handling both sync and async tools."""
        agent = TestAgentWithTools(
            agent_name="test",
            model_name="test",
            role="test"
        )
        
        # Test that both sync and async tools are properly decorated
        tools = agent.get_tool_definitions()
        assert len(tools) == 2
        
        tool_names = [tool.name for tool in tools]
        assert "analyze_data" in tool_names
        assert "search_documents" in tool_names