#!/usr/bin/env python3
"""
Test script for the new agent-centric tool calling system.

This script tests:
1. Agent tool definition creation
2. Agent announcements with tool definitions
3. StructuredLLMHostAgent tool collection from announcements
4. Tool calling flow

Run with: uv run python test_agent_centric_tools.py
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from buttermilk._core.agent import Agent
from buttermilk._core.config import AgentConfig
from buttermilk._core.contract import AgentAnnouncement, AgentInput, AgentOutput
from buttermilk.agents.flowcontrol.structured_llmhost import StructuredLLMHostAgent


class MockSimpleAgent(Agent):
    """Mock agent for testing tool definitions."""
    
    def __init__(self, role: str, description: str = "Test agent"):
        # Create minimal config
        config_data = {
            "agent_id": f"test_{role.lower()}",
            "agent_name": f"Test {role}",
            "role": role,
            "description": description,
            "parameters": {"model": "test-model"},
            "inputs": {},
            "tools": []
        }
        config = AgentConfig(**config_data)
        # Initialize with config values
        super().__init__(**config.model_dump())
    
    async def _process(self, *, message: AgentInput, **kwargs) -> AgentOutput:
        """Mock process method."""
        return AgentOutput(
            agent_id=self.agent_id,
            outputs=f"Processed: {message.inputs}",
            metadata={"test": True}
        )


class MockStructuredHost(StructuredLLMHostAgent):
    """Mock host for testing without LLM dependencies."""
    
    def __init__(self):
        config_data = {
            "agent_id": "test_host",
            "agent_name": "Test Host",
            "role": "HOST",
            "description": "Test host agent",
            "parameters": {"model": "test-model", "template": "test"},
            "inputs": {},
            "tools": []
        }
        config = AgentConfig(**config_data)
        super().__init__(**config.model_dump())
    
    async def invoke(self, *args, **kwargs):
        """Mock invoke method."""
        return MagicMock(outputs="Mock response")
    
    async def callback_to_groupchat(self, *args, **kwargs):
        """Mock callback method."""
        pass


async def test_agent_tool_definition():
    """Test that agents can generate tool definitions."""
    print("🧪 Testing agent tool definition generation...")
    
    # Create a mock agent
    agent = MockSimpleAgent("RESEARCHER", "Researches topics")
    
    # Test get_autogen_tool_definition
    tool_def = agent.get_autogen_tool_definition()
    
    print(f"✅ Tool definition: {tool_def}")
    
    # Validate structure
    assert tool_def['name'] == "call_researcher"
    assert "Use this tool when you need to" in tool_def['description']  # Enhanced description
    assert "researches topics" in tool_def['description'].lower()  # Contains original description
    assert 'input_schema' in tool_def
    assert tool_def['input_schema']['type'] == "object"
    assert 'prompt' in tool_def['input_schema']['properties']
    
    print("✅ Agent tool definition generation works!")


async def test_agent_announcement():
    """Test that agent announcements include tool definitions."""
    print("\n🧪 Testing agent announcements with tool definitions...")
    
    # Create a mock agent
    agent = MockSimpleAgent("ANALYZER", "Analyzes data")
    
    # Create announcement
    announcement = agent.create_announcement(
        announcement_type="initial",
        status="joining"
    )
    
    print(f"✅ Announcement: {announcement}")
    
    # Validate announcement has tool definition
    assert isinstance(announcement, AgentAnnouncement)
    assert announcement.tool_definition
    assert announcement.tool_definition['name'] == "call_analyzer"
    assert announcement.agent_config.role == "ANALYZER"
    
    print("✅ Agent announcements include tool definitions!")


async def test_host_tool_collection():
    """Test that host collects tools from agent announcements."""
    print("\n🧪 Testing host tool collection from announcements...")
    
    # Create mock host
    host = MockStructuredHost()
    
    # Create mock agents
    agent1 = MockSimpleAgent("RESEARCHER", "Researches topics")
    agent2 = MockSimpleAgent("WRITER", "Writes content")
    
    # Create announcements
    ann1 = agent1.create_announcement("initial", "joining")
    ann2 = agent2.create_announcement("initial", "joining") 
    
    # Add to host registry
    host._agent_registry[agent1.agent_id] = ann1
    host._agent_registry[agent2.agent_id] = ann2
    
    # Build tools
    await host._build_agent_tools()
    
    print(f"✅ Host built {len(host._tools_list)} tools")
    
    # Validate tools were created
    assert len(host._tools_list) >= 2  # At least 2 agent tools
    
    tool_names = [tool.name for tool in host._tools_list]
    print(f"✅ Tool names: {tool_names}")
    
    assert "call_researcher" in tool_names
    assert "call_writer" in tool_names
    
    print("✅ Host tool collection from announcements works!")


async def test_tool_calling_flow():
    """Test the end-to-end tool calling flow."""
    print("\n🧪 Testing end-to-end tool calling flow...")
    
    # Create host
    host = MockStructuredHost()
    
    # Create and register an agent
    agent = MockSimpleAgent("TESTER", "Tests things")
    announcement = agent.create_announcement("initial", "joining")
    
    # Simulate announcement handling
    await host.update_agent_registry(announcement)
    
    # Verify tool was registered
    assert len(host._tools_list) >= 1
    
    # Find the agent tool
    agent_tool = None
    for tool in host._tools_list:
        if tool.name == "call_tester":
            agent_tool = tool
            break
    
    assert agent_tool is not None, "Agent tool not found"
    
    # Test calling the tool (FunctionTool uses _func attribute)
    result = await agent_tool._func(prompt="Test prompt")
    
    print(f"✅ Tool call result: {result}")
    
    # Verify step request was queued
    assert not host._proposed_step.empty(), "No step request was queued"
    
    step_request = await host._proposed_step.get()
    print(f"✅ Step request: {step_request}")
    
    assert step_request.role == "TESTER"
    assert step_request.inputs["prompt"] == "Test prompt"
    
    print("✅ End-to-end tool calling flow works!")


async def test_registry_update_rebuilds_tools():
    """Test that registry updates trigger tool rebuilding."""
    print("\n🧪 Testing automatic tool rebuilding on registry updates...")
    
    # Create host
    host = MockStructuredHost()
    
    # Initially no tools
    await host._build_agent_tools()
    initial_count = len(host._tools_list)
    
    # Add agent via announcement
    agent = MockSimpleAgent("DYNAMIC", "Dynamic agent")
    announcement = agent.create_announcement("initial", "joining")
    
    await host.update_agent_registry(announcement)
    
    # Should have more tools now
    final_count = len(host._tools_list)
    
    print(f"✅ Tools before: {initial_count}, after: {final_count}")
    assert final_count > initial_count, "Tools not rebuilt on registry update"
    
    # Verify the new tool exists
    tool_names = [tool.name for tool in host._tools_list]
    assert "call_dynamic" in tool_names
    
    print("✅ Automatic tool rebuilding works!")


async def main():
    """Run all tests."""
    print("🚀 Testing Agent-Centric Tool Calling System")
    print("=" * 50)
    
    try:
        await test_agent_tool_definition()
        await test_agent_announcement()
        await test_host_tool_collection()
        await test_tool_calling_flow()
        await test_registry_update_rebuilds_tools()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("The agent-centric tool calling system is working correctly.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)