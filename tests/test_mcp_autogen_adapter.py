"""Tests for the AutoGen MCP adapter functionality.

This test suite verifies the MCP adapter system that allows Buttermilk agents
to be exposed as MCP tools while maintaining groupchat compatibility.
"""

import pytest
import asyncio
from typing import Any
from unittest.mock import Mock, AsyncMock

from buttermilk._core import Agent, AgentInput, AgentOutput
from buttermilk._core.mcp_decorators import tool, MCPRoute
from buttermilk._core.tool_definition import UnifiedRequest
from buttermilk.mcp.autogen_adapter import (
    MCPToolAdapter, 
    AutoGenMCPAdapter, 
    MCPHostProvider
)
from buttermilk.mcp.tool_registry import ToolDiscoveryService


class TestAgent(Agent):
    """Test agent with tools for MCP adapter testing."""
    
    async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput:
        """Simple process method for testing."""
        operation = message.inputs.get("operation", "echo")
        
        if operation == "echo":
            result = self.echo_message(message.inputs.get("message", ""))
        elif operation == "add":
            result = self.add_numbers(
                message.inputs.get("a", 0),
                message.inputs.get("b", 0)
            )
        else:
            result = {"error": f"Unknown operation: {operation}"}
        
        return AgentOutput(
            source=self.agent_name,
            role=self.role,
            outputs={"result": result}
        )
    
    @tool
    def echo_message(self, message: str) -> str:
        """Echo the provided message.
        
        Args:
            message: Message to echo
            
        Returns:
            The same message
        """
        return message
    
    @tool
    @MCPRoute("/test/add", permissions=["math:basic"])
    def add_numbers(self, a: int, b: int) -> int:
        """Add two numbers.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Sum of the numbers
        """
        return a + b
    
    @tool
    async def async_multiply(self, x: float, y: float) -> float:
        """Multiply two numbers asynchronously.
        
        Args:
            x: First number
            y: Second number
            
        Returns:
            Product of the numbers
        """
        # Simulate async work
        await asyncio.sleep(0.01)
        return x * y


@pytest.fixture
def test_agent():
    """Create a test agent instance."""
    return TestAgent(
        agent_name="test_agent",
        role="tester",
        session_id="test-session"
    )


@pytest.fixture
def discovery_service():
    """Create a fresh tool discovery service."""
    return ToolDiscoveryService()


class TestToolDiscoveryService:
    """Test the tool discovery service."""
    
    def test_register_agent(self, discovery_service, test_agent):
        """Test agent registration and tool discovery."""
        discovery_service.register_agent(test_agent)
        
        # Check agent is registered
        agent_key = f"{test_agent.agent_name}_{test_agent.role}"
        assert agent_key in discovery_service.agent_registry
        
        # Check tools are discovered
        tools = discovery_service.get_all_tools()
        assert len(tools) == 3  # echo_message, add_numbers, async_multiply
        
        tool_names = [tool["name"] for tool in tools]
        assert "echo_message" in tool_names
        assert "add_numbers" in tool_names
        assert "async_multiply" in tool_names
    
    def test_get_tools_by_agent(self, discovery_service, test_agent):
        """Test getting tools for a specific agent."""
        discovery_service.register_agent(test_agent)
        
        agent_tools = discovery_service.get_tools_by_agent(
            test_agent.agent_name, 
            test_agent.role
        )
        
        assert len(agent_tools) == 3
        for tool in agent_tools:
            assert tool["agent_name"] == test_agent.agent_name
            assert tool["agent_role"] == test_agent.role
    
    def test_get_tool_by_name(self, discovery_service, test_agent):
        """Test finding a tool by name."""
        discovery_service.register_agent(test_agent)
        
        tool = discovery_service.get_tool_by_name("add_numbers")
        assert tool is not None
        assert tool["name"] == "add_numbers"
        assert tool["agent"] == test_agent
        
        # Test non-existent tool
        assert discovery_service.get_tool_by_name("nonexistent") is None
    
    @pytest.mark.asyncio
    async def test_invoke_tool(self, discovery_service, test_agent):
        """Test tool invocation via discovery service."""
        discovery_service.register_agent(test_agent)
        
        # Test sync tool
        result = await discovery_service.invoke_tool(
            f"{test_agent.agent_name}_{test_agent.role}.add_numbers",
            {"a": 5, "b": 3}
        )
        assert result == 8
        
        # Test async tool
        result = await discovery_service.invoke_tool(
            f"{test_agent.agent_name}_{test_agent.role}.async_multiply",
            {"x": 4.0, "y": 2.5}
        )
        assert result == 10.0
    
    def test_unregister_agent(self, discovery_service, test_agent):
        """Test agent unregistration."""
        discovery_service.register_agent(test_agent)
        assert len(discovery_service.get_all_tools()) == 3
        
        discovery_service.unregister_agent(test_agent.agent_name, test_agent.role)
        assert len(discovery_service.get_all_tools()) == 0
        
        agent_key = f"{test_agent.agent_name}_{test_agent.role}"
        assert agent_key not in discovery_service.agent_registry


class TestMCPToolAdapter:
    """Test the MCP tool adapter."""
    
    def test_adapter_initialization(self, discovery_service, test_agent):
        """Test MCP tool adapter initialization."""
        discovery_service.register_agent(test_agent)
        tool_info = discovery_service.get_tool_by_name("add_numbers")
        
        adapter = MCPToolAdapter(tool_info)
        
        assert adapter.name == "add_numbers"
        assert adapter.tool_key == tool_info["key"]
        assert adapter.agent == test_agent
        assert adapter.tool_definition == tool_info["tool_definition"]
    
    @pytest.mark.asyncio
    async def test_run_json_sync_tool(self, discovery_service, test_agent):
        """Test running a synchronous tool via MCP adapter."""
        discovery_service.register_agent(test_agent)
        tool_info = discovery_service.get_tool_by_name("add_numbers")
        adapter = MCPToolAdapter(tool_info)
        
        result = await adapter.run_json({"a": 10, "b": 15})
        assert result == 25
    
    @pytest.mark.asyncio
    async def test_run_json_async_tool(self, discovery_service, test_agent):
        """Test running an asynchronous tool via MCP adapter."""
        discovery_service.register_agent(test_agent)
        tool_info = discovery_service.get_tool_by_name("async_multiply")
        adapter = MCPToolAdapter(tool_info)
        
        result = await adapter.run_json({"x": 3.0, "y": 7.0})
        assert result == 21.0
    
    @pytest.mark.asyncio
    async def test_run_json_error_handling(self, discovery_service, test_agent):
        """Test error handling in MCP adapter."""
        discovery_service.register_agent(test_agent)
        tool_info = discovery_service.get_tool_by_name("add_numbers")
        adapter = MCPToolAdapter(tool_info)
        
        # Test with missing parameters
        result = await adapter.run_json({})
        assert isinstance(result, dict)
        assert "error" in result
        assert result["success"] is False


class TestAutoGenMCPAdapter:
    """Test the main AutoGen MCP adapter."""
    
    def test_adapter_initialization(self):
        """Test adapter initialization."""
        adapter = AutoGenMCPAdapter()
        
        assert adapter.discovery_service is not None
        assert isinstance(adapter.mcp_adapters, dict)
        assert len(adapter.mcp_adapters) == 0
    
    def test_register_agent(self, test_agent):
        """Test agent registration with adapter."""
        adapter = AutoGenMCPAdapter()
        adapter.register_agent(test_agent)
        
        # Check that MCP adapters were created
        assert len(adapter.mcp_adapters) == 3
        
        # Check adapter keys
        expected_keys = [
            f"{test_agent.agent_name}_{test_agent.role}.echo_message",
            f"{test_agent.agent_name}_{test_agent.role}.add_numbers",
            f"{test_agent.agent_name}_{test_agent.role}.async_multiply"
        ]
        
        for key in expected_keys:
            assert key in adapter.mcp_adapters
    
    def test_unregister_agent(self, test_agent):
        """Test agent unregistration."""
        adapter = AutoGenMCPAdapter()
        adapter.register_agent(test_agent)
        assert len(adapter.mcp_adapters) == 3
        
        adapter.unregister_agent(test_agent.agent_name, test_agent.role)
        assert len(adapter.mcp_adapters) == 0
    
    @pytest.mark.asyncio
    async def test_invoke_tool(self, test_agent):
        """Test tool invocation via adapter."""
        adapter = AutoGenMCPAdapter()
        adapter.register_agent(test_agent)
        
        # Test by full key
        result = await adapter.invoke_tool(
            f"{test_agent.agent_name}_{test_agent.role}.add_numbers",
            {"a": 7, "b": 8}
        )
        assert result == 15
        
        # Test by tool name
        result = await adapter.invoke_tool("echo_message", {"message": "hello"})
        assert result == "hello"
    
    def test_get_available_tools(self, test_agent):
        """Test getting available tool schemas."""
        adapter = AutoGenMCPAdapter()
        adapter.register_agent(test_agent)
        
        tools = adapter.get_available_tools()
        assert len(tools) == 3
        
        tool_names = [tool.name for tool in tools]
        assert "echo_message" in tool_names
        assert "add_numbers" in tool_names
        assert "async_multiply" in tool_names
    
    def test_list_tools(self, test_agent):
        """Test listing tools with metadata."""
        adapter = AutoGenMCPAdapter()
        adapter.register_agent(test_agent)
        
        tools_info = adapter.list_tools()
        assert len(tools_info) == 3
        
        # Check tool info structure
        for tool_key, tool_info in tools_info.items():
            assert "name" in tool_info
            assert "description" in tool_info
            assert "agent_name" in tool_info
            assert "agent_role" in tool_info
            assert "input_schema" in tool_info
            assert "output_schema" in tool_info


class TestMCPHostProvider:
    """Test the MCP host provider."""
    
    def test_provider_initialization(self):
        """Test provider initialization."""
        provider = MCPHostProvider(port=8790)
        
        assert provider.port == 8790
        assert provider.discovery_service is not None
        assert provider.adapter is not None
        assert provider.running is False
    
    def test_register_agent(self, test_agent):
        """Test agent registration with provider."""
        provider = MCPHostProvider()
        provider.register_agent(test_agent)
        
        # Check that tools are available
        tools = provider.list_available_tools()
        assert len(tools) == 3
    
    def test_unregister_agent(self, test_agent):
        """Test agent unregistration."""
        provider = MCPHostProvider()
        provider.register_agent(test_agent)
        assert len(provider.list_available_tools()) == 3
        
        provider.unregister_agent(test_agent.agent_name, test_agent.role)
        assert len(provider.list_available_tools()) == 0
    
    @pytest.mark.asyncio
    async def test_invoke_tool(self, test_agent):
        """Test tool invocation via provider."""
        provider = MCPHostProvider()
        provider.register_agent(test_agent)
        
        result = await provider.invoke_tool(
            f"{test_agent.agent_name}_{test_agent.role}.add_numbers",
            {"a": 20, "b": 22}
        )
        assert result == 42


class TestUnifiedRequestIntegration:
    """Test unified request handling with MCP context."""
    
    @pytest.mark.asyncio
    async def test_mcp_request_handling(self, test_agent):
        """Test handling MCP-style unified requests."""
        request = UnifiedRequest.from_mcp_call(
            tool_name="add_numbers",
            parameters={"a": 100, "b": 200},
            agent_name=f"{test_agent.agent_name}_{test_agent.role}"
        )
        
        assert request.is_mcp_request
        assert not request.is_groupchat_request
        assert request.tool_name == "add_numbers"
        
        result = await test_agent.handle_unified_request(request)
        assert result == 300
    
    @pytest.mark.asyncio
    async def test_groupchat_request_handling(self, test_agent):
        """Test handling groupchat-style unified requests."""
        request = UnifiedRequest.from_groupchat_step(
            agent_role=test_agent.role,
            inputs={"a": 50, "b": 75},
            tool_name="add_numbers"
        )
        
        assert request.is_groupchat_request
        assert not request.is_mcp_request
        assert request.tool_name == "add_numbers"
        
        result = await test_agent.handle_unified_request(request)
        assert result == 125


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""
    
    @pytest.mark.asyncio
    async def test_dual_mode_agent_usage(self, test_agent):
        """Test using the same agent in both MCP and groupchat modes."""
        # Set up MCP exposure
        provider = MCPHostProvider()
        provider.register_agent(test_agent)
        
        # Test MCP invocation
        mcp_result = await provider.invoke_tool(
            f"{test_agent.agent_name}_{test_agent.role}.echo_message",
            {"message": "MCP mode"}
        )
        assert mcp_result == "MCP mode"
        
        # Test groupchat invocation (direct agent call)
        groupchat_input = AgentInput(inputs={
            "operation": "echo",
            "message": "Groupchat mode"
        })
        groupchat_result = await test_agent._process(message=groupchat_input)
        assert groupchat_result.outputs["result"] == "Groupchat mode"
        
        # Test unified request (MCP context)
        unified_mcp = UnifiedRequest.from_mcp_call(
            tool_name="echo_message",
            parameters={"message": "Unified MCP"},
            agent_name=f"{test_agent.agent_name}_{test_agent.role}"
        )
        unified_result = await test_agent.handle_unified_request(unified_mcp)
        assert unified_result == "Unified MCP"
    
    def test_tool_discovery_callback_integration(self, test_agent):
        """Test integration between discovery service and MCP adapter via callbacks."""
        adapter = AutoGenMCPAdapter()
        
        # Verify callback is set up
        assert len(adapter.discovery_service._discovery_callbacks) > 0
        
        # Register agent - should trigger callback
        adapter.register_agent(test_agent)
        
        # Verify MCP adapters were created via callback
        assert len(adapter.mcp_adapters) == 3
        
        # Verify tools are available
        tools = adapter.list_tools()
        assert len(tools) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])