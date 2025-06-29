"""Unit tests for structured LLMHost agent."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Any

from buttermilk._core import AgentInput, StepRequest
from buttermilk._core.agent import ManagerMessage
from buttermilk._core.constants import END, MANAGER
from buttermilk._core.tool_definition import AgentToolDefinition
from buttermilk.agents.flowcontrol.structured_llmhost import StructuredLLMHostAgent


class MockAgent:
    """Mock agent with tool definitions."""
    
    def __init__(self, name: str, tools: list[AgentToolDefinition] | None = None):
        self.agent_name = name
        self._tools = tools or []
    
    def get_tool_definitions(self) -> list[AgentToolDefinition]:
        """Return tool definitions."""
        return self._tools


class TestStructuredLLMHostInitialization:
    """Test initialization of structured LLMHost."""
    
    @pytest.fixture
    def mock_participants(self):
        """Create mock participants with tool definitions."""
        # Agent with explicit tools
        agent1_tools = [
            AgentToolDefinition(
                name="analyze",
                description="Analyze data",
                input_schema={
                    "type": "object",
                    "properties": {"data": {"type": "string"}},
                    "required": ["data"]
                },
                output_schema={"type": "object"}
            ),
            AgentToolDefinition(
                name="summarize",
                description="Summarize text",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]
                },
                output_schema={"type": "string"}
            )
        ]
        
        # Agent without tools (should get default)
        agent2_tools = []
        
        return {
            "ANALYST": MockAgent("analyst", agent1_tools),
            "WRITER": MockAgent("writer", agent2_tools),
            "MANAGER": Mock(spec=[])  # Non-agent participant
        }
    
    @pytest.mark.asyncio
    async def test_initialize_with_agent_tools(self, mock_participants):
        """Test initialization collects tools from agents."""
        host = StructuredLLMHostAgent(
            agent_name="host",
            role="host",
            parameters={"model": "test-model"}
        )
        
        # Mock the parent class initialization
        host._participants = mock_participants
        host.tools = {}
        host.parameters = {"model": "test-model"}
        host.callback_to_groupchat = AsyncMock()
        
        # Initialize
        await host._initialize(callback_to_groupchat=host.callback_to_groupchat)
        
        # Should have tools from agents
        # 2 tools from ANALYST + 1 default tool for WRITER = 3 total
        assert len(host._tools_list) == 3
        
        # Check tool names
        tool_names = [tool.name for tool in host._tools_list]
        assert "analyst.analyze" in tool_names
        assert "analyst.summarize" in tool_names
        assert "call_writer" in tool_names
    
    @pytest.mark.asyncio
    async def test_default_tool_creation(self, mock_participants):
        """Test default tool creation for agents without tools."""
        host = StructuredLLMHostAgent(
            agent_name="host",
            role="host",
            parameters={"model": "test-model"}
        )
        
        # Only agent without tools
        host._participants = {"WRITER": mock_participants["WRITER"]}
        host.tools = {}
        host.parameters = {"model": "test-model"}
        host.callback_to_groupchat = AsyncMock()
        
        await host._initialize(callback_to_groupchat=host.callback_to_groupchat)
        
        # Should have one default tool
        assert len(host._tools_list) == 1
        assert host._tools_list[0].name == "call_writer"
        assert "Send a request to the WRITER agent" in host._tools_list[0].description


class TestStructuredLLMHostSequence:
    """Test sequence generation."""
    
    @pytest.mark.asyncio
    async def test_sequence_generation(self):
        """Test the sequence generator."""
        host = StructuredLLMHostAgent(
            agent_name="host",
            role="host",
            parameters={"model": "test-model"}
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
            model_name="test-model",
            role="host",
            parameters={"model": "test-model"}  # Add required parameters
        )
        
        # Mock dependencies
        host._model = "test-model"
        host._tools_list = []
        host._participants = {"ANALYST": Mock()}
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
            assert call_args.inputs["participants"] == ["ANALYST"]
    
    @pytest.mark.asyncio
    async def test_listen_skip_command_messages(self, mock_host):
        """Test that command messages are skipped."""
        message = ManagerMessage(content="/command test")
        
        # Use patch to mock the invoke method and verify it's not called
        from unittest.mock import patch
        with patch.object(mock_host, 'invoke', new_callable=AsyncMock) as mock_invoke:
            await mock_host._listen(
                message=message,
                cancellation_token=None,
                source="test",
                public_callback=AsyncMock()
            )
            
            # Should not invoke
            mock_invoke.assert_not_called()
    
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
    async def test_tool_code_handling(self, mock_host):
        """Test handling of tool_code dict format from LLM."""
        # Setup participants with tools
        tool_def = AgentToolDefinition(
            name="oversight_board_case_search",
            description="Search oversight board cases",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            },
            output_schema={"type": "object"}
        )
        search_agent = MockAgent("search_agent", [tool_def])
        mock_host._participants = {"SEARCH_AGENT": search_agent}
        
        message = ManagerMessage(content="Search for ICCPR Article 20")
        
        # Mock invoke to return tool_code format
        from buttermilk._core.contract import AgentTrace
        mock_trace = Mock(spec=AgentTrace)
        mock_trace.outputs = {
            'tool_code': 'oversight_board_case_search',
            'tool_name': 'oversight_board_case_search',
            'parameters': {'query': 'ICCPR Article 20 and incitement to discrimination'}
        }
        
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
        
        # Should have added step to queue
        step = await mock_host._proposed_step.get()
        assert step.role == "SEARCH_AGENT"
        assert step.inputs["tool"] == "oversight_board_case_search"
        assert step.inputs["tool_inputs"] == {'query': 'ICCPR Article 20 and incitement to discrimination'}
    
    @pytest.mark.asyncio
    async def test_tool_code_no_matching_agent(self, mock_host):
        """Test handling when no agent owns the requested tool."""
        mock_host._participants = {"OTHER_AGENT": MockAgent("other", [])}
        
        message = ManagerMessage(content="Search for something")
        
        # Mock invoke to return tool_code for non-existent tool
        from buttermilk._core.contract import AgentTrace
        mock_trace = Mock(spec=AgentTrace)
        mock_trace.outputs = {
            'tool_code': 'NonExistentTool',
            'parameters': {'data': 'test'}
        }
        
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
        
        # Should pass response to groupchat instead
        mock_host.callback_to_groupchat.assert_called_with(mock_trace)
        # Queue should be empty
        assert mock_host._proposed_step.empty()


class TestStructuredLLMHostTools:
    """Test tool handling in structured LLMHost."""
    
    @pytest.mark.asyncio
    async def test_tool_function_creation(self):
        """Test that tool functions are created correctly."""
        host = StructuredLLMHostAgent(
            agent_name="host",
            role="host",
            parameters={"model": "test-model"}
        )
        
        # Create a mock agent with a tool
        tool_def = AgentToolDefinition(
            name="process",
            description="Process data",
            input_schema={
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"]
            },
            output_schema={"type": "object"}
        )
        
        agent = MockAgent("processor", [tool_def])
        host._participants = {"PROCESSOR": agent}
        host.tools = {}
        host.parameters = {"model": "test-model"}
        host.callback_to_groupchat = AsyncMock()
        
        await host._initialize(callback_to_groupchat=host.callback_to_groupchat)
        
        # Should have created function tool
        assert len(host._tools_list) == 1
        tool = host._tools_list[0]
        assert tool.name == "processor.process"
        assert tool.description == "Process data"
        
        # Test calling the tool function
        # The function should create a StepRequest
        await tool._func(input="test data")
        
        # Verify callback was called with StepRequest
        host.callback_to_groupchat.assert_called_once()
        step_request = host.callback_to_groupchat.call_args[0][0]
        assert isinstance(step_request, StepRequest)
        assert step_request.role == "PROCESSOR"
        assert step_request.inputs["tool"] == "process"
        assert step_request.inputs["tool_inputs"] == {"input": "test data"}
