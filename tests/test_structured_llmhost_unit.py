"""Unit tests for structured LLMHost agent."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Any

from autogen_core.tools import ToolSchema
from buttermilk._core import AgentInput, StepRequest
from buttermilk._core.agent import ManagerMessage
from buttermilk._core.config import AgentConfig
from buttermilk._core.constants import END, MANAGER
from buttermilk._core.contract import AgentAnnouncement
from buttermilk.agents.flowcontrol.structured_llmhost import StructuredLLMHostAgent


class TestStructuredLLMHostInitialization:
    """Test initialization of structured LLMHost."""
    
    def create_tool_schema(self, name: str, description: str) -> ToolSchema:
        """Create a mock ToolSchema."""
        return {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query to process"}
                },
                "required": ["query"]
            }
        }
    
    def create_agent_announcement(self, agent_id: str, role: str, tool_def: ToolSchema | None = None) -> AgentAnnouncement:
        """Create a mock AgentAnnouncement."""
        config = AgentConfig(
            agent_id=agent_id,
            role=role,
            agent_name=agent_id.lower(),
            description=f"{role} agent"
        )
        return AgentAnnouncement(
            agent_config=config,
            available_tools=[tool_def["name"]] if tool_def else [],
            tool_definition=tool_def,
            status="active",
            announcement_type="initial",
            content=f"{role} agent joining the group"
        )
    
    @pytest.mark.asyncio
    async def test_agent_registry_with_tools(self):
        """Test that agent announcements with tools are properly registered."""
        host = StructuredLLMHostAgent(
            agent_name="host",
            role="host",
            parameters={"model": "test-model", "human_in_loop": False}
        )
        
        # Initialize minimal state
        host.callback_to_groupchat = AsyncMock()
        await host.initialize(callback_to_groupchat=host.callback_to_groupchat)
        
        # Create agent announcements
        search_tool = self.create_tool_schema("search_cases", "Search for oversight board cases")
        analyze_tool = self.create_tool_schema("analyze_data", "Analyze collected data")
        
        search_announcement = self.create_agent_announcement("search-001", "SEARCH_AGENT", search_tool)
        analyze_announcement = self.create_agent_announcement("analyze-001", "ANALYST", analyze_tool)
        
        # Update registry with announcements
        await host.update_agent_registry(search_announcement)
        await host.update_agent_registry(analyze_announcement)
        
        # Check that tools were collected
        assert len(host._tool_schemas) == 2
        tool_names = [schema["name"] for schema in host._tool_schemas]
        assert "search_cases" in tool_names
        assert "analyze_data" in tool_names
    
    @pytest.mark.asyncio
    async def test_agent_leaving_removes_tools(self):
        """Test that agent leaving removes their tools."""
        host = StructuredLLMHostAgent(
            agent_name="host",
            role="host",
            parameters={"model": "test-model", "human_in_loop": False}
        )
        
        host.callback_to_groupchat = AsyncMock()
        await host.initialize(callback_to_groupchat=host.callback_to_groupchat)
        
        # Add agent with tool
        tool = self.create_tool_schema("process", "Process data")
        announcement = self.create_agent_announcement("proc-001", "PROCESSOR", tool)
        await host.update_agent_registry(announcement)
        
        assert len(host._tool_schemas) == 1
        
        # Agent leaves
        leaving_announcement = AgentAnnouncement(
            agent_config=announcement.agent_config,
            available_tools=[],
            status="leaving",
            announcement_type="update",
            content="Agent leaving"
        )
        await host.update_agent_registry(leaving_announcement)
        
        # Tool should be removed
        assert len(host._tool_schemas) == 0


class TestStructuredLLMHostSequence:
    """Test sequence generation."""
    
    @pytest.mark.asyncio
    async def test_sequence_generation(self):
        """Test the sequence generator."""
        host = StructuredLLMHostAgent(
            agent_name="host",
            role="host",
            parameters={"model": "test-model", "human_in_loop": False}
        )
        
        # Add test steps to queue
        step1 = StepRequest(role="ANALYST", inputs={"test": "data"})
        step2 = StepRequest(role=END)
        
        async def add_steps():
            await asyncio.sleep(0.1)  # Let sequence start
            await host._proposed_step.put(step1)
            await host._proposed_step.put(step2)
        
        # Start adding steps
        asyncio.create_task(add_steps())
        
        # Collect sequence
        steps = []
        async for step in host._sequence():
            steps.append(step)
            if step.role == END:
                break
        
        # Should have initial greeting + 2 steps
        assert len(steps) == 3
        assert steps[0].role == MANAGER
        assert "What would you like to do?" in steps[0].content
        assert steps[1] == step1
        assert steps[2] == step2


class TestStructuredLLMHostListen:
    """Test message listening and processing."""
    
    @pytest.fixture
    def mock_host(self):
        """Create a mock host with initialized state."""
        host = StructuredLLMHostAgent(
            agent_name="host",
            role="host",
            parameters={"model": "test-model", "human_in_loop": False}
        )
        
        # Mock dependencies
        host._tools_list = []
        host._tool_schemas = []
        host._agent_registry = {}
        host.callback_to_groupchat = AsyncMock()
        
        return host
    
    @pytest.mark.asyncio
    async def test_listen_manager_message(self, mock_host):
        """Test processing manager messages."""
        message = ManagerMessage(content="Analyze this data")
        
        # Mock invoke to return a trace
        mock_trace = Mock()
        mock_trace.outputs = "Analysis complete"
        
        # Use patch to mock the invoke method at the class level
        from unittest.mock import patch
        with patch.object(StructuredLLMHostAgent, 'invoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_trace
            
            await mock_host._listen(
                message=message,
                cancellation_token=None,
                source="test",
                public_callback=AsyncMock()
            )
            
            # Should have called invoke
            mock_invoke.assert_called_once()
            call_args = mock_invoke.call_args[1]["message"]
            assert isinstance(call_args, AgentInput)
            assert call_args.inputs["prompt"] == "Analyze this data"
            # No longer passing participants list
            assert "user_feedback" in call_args.inputs
    
    @pytest.mark.asyncio
    async def test_listen_skip_command_messages(self, mock_host):
        """Test that command messages are skipped."""
        message = ManagerMessage(content="/command test")
        
        # Add a step to the queue before the command
        await mock_host._proposed_step.put(StepRequest(role="TEST"))
        
        await mock_host._listen(
            message=message,
            cancellation_token=None,
            source="test",
            public_callback=AsyncMock()
        )
        
        # Queue should still have the original step (not cleared)
        assert not mock_host._proposed_step.empty()
        step = await mock_host._proposed_step.get()
        assert step.role == "TEST"
    
    @pytest.mark.asyncio
    async def test_listen_clear_pending_steps(self, mock_host):
        """Test that pending steps are cleared on new message."""
        # Add some pending steps
        await mock_host._proposed_step.put(StepRequest(role="OLD"))
        await mock_host._proposed_step.put(StepRequest(role="OLD2"))
        
        message = ManagerMessage(content="New request")
        
        # Use patch to mock the invoke method at the class level
        from unittest.mock import patch
        with patch.object(StructuredLLMHostAgent, 'invoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = Mock(outputs="Done")
            
            await mock_host._listen(
                message=message,
                cancellation_token=None,
                source="test",
                public_callback=AsyncMock()
            )
            
            # Queue should be empty
            assert mock_host._proposed_step.empty()
    
    @pytest.mark.asyncio
    async def test_listen_handle_end_response(self, mock_host):
        """Test handling of END responses from LLM."""
        message = ManagerMessage(content="I'm done")
        
        # Mock invoke to return END indication
        mock_trace = Mock()
        mock_trace.outputs = "I think we're DONE here"
        
        # Use patch to mock the invoke method at the class level
        from unittest.mock import patch
        with patch.object(StructuredLLMHostAgent, 'invoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_trace
            
            await mock_host._listen(
                message=message,
                cancellation_token=None,
                source="test",
                public_callback=AsyncMock()
            )
        
        # Should have added END step to queue
        step = await mock_host._proposed_step.get()
        assert step.role == END
    
    @pytest.mark.asyncio
    async def test_process_routes_tool_calls(self, mock_host):
        """Test that _process routes tool calls to agents."""
        # Setup agent registry with tool
        from buttermilk._core.config import AgentConfig
        from buttermilk._core.contract import AgentAnnouncement
        
        tool_schema: ToolSchema = {
            "name": "search_cases",
            "description": "Search oversight board cases",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
        
        agent_config = AgentConfig(
            agent_id="search-001",
            role="SEARCH_AGENT",
            agent_name="search",
            description="Search agent"
        )
        
        announcement = AgentAnnouncement(
            agent_config=agent_config,
            available_tools=["search_cases"],
            tool_definition=tool_schema,
            status="active",
            announcement_type="initial",
            content="Search agent joining"
        )
        
        mock_host._agent_registry = {"search-001": announcement}
        mock_host._tool_schemas = [tool_schema]
        
        # Mock the LLM to return tool calls
        from autogen_core import FunctionCall
        from autogen_core.models import CreateResult
        
        tool_call = FunctionCall(
            id="call-123",
            name="search_cases",
            arguments='{"query": "ICCPR Article 20"}'
        )
        
        # Patch the necessary methods
        from unittest.mock import patch
        with patch.object(mock_host, '_fill_template', new_callable=AsyncMock) as mock_fill:
            mock_fill.return_value = [Mock()]  # Return mock messages
            
            with patch('buttermilk.buttermilk.get_bm') as mock_get_bm:
                mock_bm = Mock()
                mock_llms = Mock()
                mock_bm.llms = mock_llms
                mock_get_bm.return_value = mock_bm
                
                mock_llms.get_autogen_chat_client.return_value = mock_client
                mock_client = Mock()
                mock_client.create = AsyncMock(return_value=CreateResult(
                    content=[tool_call],
                    finish_reason="tool_calls",
                    usage=None,
                    cached=False
                ))
                mock_get_client.return_value = mock_client
                
                # Call _process
                result = await mock_host._process(
                    message=AgentInput(inputs={"prompt": "Search for cases"}),
                    cancellation_token=None
                )
                
                # Should have routed the tool call
                assert "Routing 1 tool calls" in result.outputs
                
                # Check that step was queued
                step = await mock_host._proposed_step.get()
                assert step.role == "SEARCH_AGENT"
                assert step.inputs["query"] == "ICCPR Article 20"
    
    @pytest.mark.asyncio
    async def test_no_matching_agent_for_tool(self, mock_host):
        """Test handling when no agent owns the requested tool."""
        # Empty agent registry
        mock_host._agent_registry = {}
        mock_host._tool_schemas = []
        
        # Mock the LLM to return a tool call for non-existent tool
        from autogen_core import FunctionCall
        from autogen_core.models import CreateResult
        
        tool_call = FunctionCall(
            id="call-456",
            name="NonExistentTool",
            arguments='{"data": "test"}'
        )
        
        from unittest.mock import patch
        with patch.object(mock_host, '_fill_template', new_callable=AsyncMock) as mock_fill:
            mock_fill.return_value = [Mock()]
            
            with patch('buttermilk.buttermilk.get_bm') as mock_get_bm:
                mock_bm = Mock()
                mock_llms = Mock()
                mock_bm.llms = mock_llms
                mock_get_bm.return_value = mock_bm
                
                mock_llms.get_autogen_chat_client.return_value = mock_client
                mock_client = Mock()
                mock_client.create = AsyncMock(return_value=CreateResult(
                    content=[tool_call],
                    finish_reason="tool_calls",
                    usage=None,
                    cached=False
                ))
                mock_get_client.return_value = mock_client
                
                # Call _process
                result = await mock_host._process(
                    message=AgentInput(inputs={"prompt": "Do something"}),
                    cancellation_token=None
                )
                
                # Should still return success but queue should be empty
                assert "Routing 1 tool calls" in result.outputs
                assert mock_host._proposed_step.empty()


class TestStructuredLLMHostTools:
    """Test tool handling in structured LLMHost."""
    
    @pytest.mark.asyncio
    async def test_tool_routing_to_agents(self):
        """Test that tool calls are routed to correct agents."""
        host = StructuredLLMHostAgent(
            agent_name="host",
            role="host",
            parameters={"model": "test-model", "human_in_loop": False}
        )
        
        host.callback_to_groupchat = AsyncMock()
        await host.initialize(callback_to_groupchat=host.callback_to_groupchat)
        
        # Create multiple agents with tools
        from buttermilk._core.config import AgentConfig
        from buttermilk._core.contract import AgentAnnouncement
        
        # Search agent
        search_tool: ToolSchema = {
            "name": "search",
            "description": "Search for information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
        search_config = AgentConfig(
            agent_id="search-001",
            role="SEARCH",
            agent_name="search",
            description="Search agent"
        )
        search_announcement = AgentAnnouncement(
            agent_config=search_config,
            available_tools=["search"],
            tool_definition=search_tool,
            status="active",
            announcement_type="initial",
            content="Search agent joining"
        )
        
        # Analyze agent
        analyze_tool: ToolSchema = {
            "name": "analyze",
            "description": "Analyze data",
            "parameters": {
                "type": "object",
                "properties": {"data": {"type": "string"}},
                "required": ["data"]
            }
        }
        analyze_config = AgentConfig(
            agent_id="analyze-001",
            role="ANALYZER",
            agent_name="analyzer",
            description="Analyzer agent"
        )
        analyze_announcement = AgentAnnouncement(
            agent_config=analyze_config,
            available_tools=["analyze"],
            tool_definition=analyze_tool,
            status="active",
            announcement_type="initial",
            content="Analyzer agent joining"
        )
        
        # Register agents
        await host.update_agent_registry(search_announcement)
        await host.update_agent_registry(analyze_announcement)
        
        # Test routing tool calls
        from autogen_core import FunctionCall
        
        search_call = FunctionCall(
            id="call-1",
            name="search",
            arguments='{"query": "test search"}'
        )
        analyze_call = FunctionCall(
            id="call-2",
            name="analyze",
            arguments='{"data": "test data"}'
        )
        
        # Route the calls
        await host._route_tool_calls_to_agents([search_call, analyze_call])
        
        # Should have called callback twice with StepRequests
        assert host.callback_to_groupchat.call_count == 2
        
        # Check first call (search)
        first_call = host.callback_to_groupchat.call_args_list[0][0][0]
        assert isinstance(first_call, StepRequest)
        assert first_call.role == "SEARCH"
        assert first_call.inputs["query"] == "test search"
        assert first_call.metadata["tool_name"] == "search"
        
        # Check second call (analyze)
        second_call = host.callback_to_groupchat.call_args_list[1][0][0]
        assert isinstance(second_call, StepRequest)
        assert second_call.role == "ANALYZER"
        assert second_call.inputs["data"] == "test data"
        assert second_call.metadata["tool_name"] == "analyze"
