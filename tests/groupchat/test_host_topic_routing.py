"""Tests for HostAgent topic routing functionality."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from autogen_core import DefaultTopicId

from buttermilk._core.contract import (
    FlowEvent,
    StepRequest,
    SystemPromptMessage,
)
from buttermilk.agents.flowcontrol.host import HostAgent

pytestmark = pytest.mark.anyio


@pytest.fixture
def mock_host_agent(monkeypatch):
    """Create a mocked HostAgent for testing."""
    from autogen_core import AgentId
    from autogen_core import RoutedAgent

    # Create an AsyncMock for tracking publish_message calls
    publish_mock = AsyncMock()

    # Patch RoutedAgent.publish_message before creating the agent
    monkeypatch.setattr(RoutedAgent, 'publish_message', publish_mock)

    agent = HostAgent(
        agent_id="test-host",
        agent_name="TestHost",
        role="HOST",
        description="Test host agent",
        parameters={"human_in_loop": False}
    )

    # Store the mock on the agent for test access
    agent.publish_message = publish_mock

    # Set _runtime to non-None so _publish doesn't early-return
    agent._runtime = MagicMock()
    # Set _id to prevent AttributeError when accessing agent metadata
    agent._id = AgentId(key="test-host", type="host")
    agent._topic_id = DefaultTopicId(type="main-topic")
    agent._participants = {
        "RESEARCHER": "Research agent",
        "WRITER": "Writing agent",
        # Don't include MANAGER here - it should be handled specially by the elif block
    }

    # Set up tool routing attributes for _route_tool_calls_to_agents
    agent._tool_to_agent_map = {"researcher_call": "researcher-agent-id"}
    agent._agent_registry = {
        "researcher-agent-id": MagicMock(agent_config=MagicMock(role="researcher"))
    }
    agent.human_in_loop = False

    return agent


class TestHostTopicRouting:
    """Test cases for host agent topic routing."""

    @pytest.mark.anyio
    async def test_step_request_routes_to_role_topic(self, mock_host_agent):
        """Test that StepRequest messages are routed to role-specific topics."""
        # Create a StepRequest
        step = StepRequest(
            role="RESEARCHER",
            content="Execute researcher step",
            inputs={"query": "test query"}
        )
        
        # Execute the step
        await mock_host_agent._execute_step(step)
        
        # Verify two publish calls were made
        assert mock_host_agent.publish_message.call_count == 2
        
        # First call should be FlowEvent to main topic
        first_call = mock_host_agent.publish_message.call_args_list[0]
        assert isinstance(first_call[0][0], FlowEvent)
        assert first_call[1]["topic_id"] == mock_host_agent._topic_id
        
        # Second call should be StepRequest to role-specific topic
        second_call = mock_host_agent.publish_message.call_args_list[1]
        assert isinstance(second_call[0][0], StepRequest)
        assert second_call[1]["topic_id"] == DefaultTopicId(type="RESEARCHER")

    @pytest.mark.anyio
    async def test_flow_event_sent_before_step_request(self, mock_host_agent):
        """Test that FlowEvent is sent to main topic before StepRequest."""
        step = StepRequest(
            role="WRITER",
            content="Execute writer step"
        )
        
        await mock_host_agent._execute_step(step)
        
        # Verify FlowEvent content
        flow_event = mock_host_agent.publish_message.call_args_list[0][0][0]
        assert isinstance(flow_event, FlowEvent)
        assert flow_event.source == mock_host_agent.agent_id
        assert "Starting WRITER step" in flow_event.content
        assert "Writing agent" in flow_event.content

    @pytest.mark.anyio
    async def test_manager_step_sends_ui_message_only(self, mock_host_agent):
        """Test that MANAGER steps only send SystemPromptMessage to main topic."""
        step = StepRequest(
            role="MANAGER",
            content="What would you like to do next?"
        )
        
        await mock_host_agent._execute_step(step)
        
        # Should only send SystemPromptMessage, not StepRequest
        assert mock_host_agent.publish_message.call_count == 1
        call_args = mock_host_agent.publish_message.call_args_list[0]
        
        message = call_args[0][0]
        assert isinstance(message, SystemPromptMessage)
        assert message.content == "What would you like to do next?"
        assert call_args[1]["topic_id"] == mock_host_agent._topic_id

    @pytest.mark.anyio
    async def test_end_step_routes_to_main_topic(self, mock_host_agent):
        """Test that END steps are sent to main topic."""
        from buttermilk._core.constants import END
        
        step = StepRequest(
            role=END,
            content="Flow completed"
        )
        
        await mock_host_agent._execute_step(step)
        
        # END step should go to main topic
        assert mock_host_agent.publish_message.call_count == 1
        call_args = mock_host_agent.publish_message.call_args_list[0]
        assert isinstance(call_args[0][0], StepRequest)
        assert call_args[1]["topic_id"] == mock_host_agent._topic_id

    @pytest.mark.anyio
    async def test_unknown_role_still_routes_to_role_topic(self, mock_host_agent):
        """Test that unknown roles still route to role-specific topics."""
        step = StepRequest(
            role="UNKNOWN_ROLE",
            content="Execute unknown step"
        )
        
        await mock_host_agent._execute_step(step)
        
        # Should only send StepRequest (no FlowEvent for unknown roles)
        assert mock_host_agent.publish_message.call_count == 1
        call_args = mock_host_agent.publish_message.call_args_list[0]
        assert isinstance(call_args[0][0], StepRequest)
        assert call_args[1]["topic_id"] == DefaultTopicId(type="UNKNOWN_ROLE")

    @pytest.mark.anyio
    async def test_route_tool_calls_uses_role_topics(self, mock_host_agent):
        """Test that _route_tool_calls_to_agents routes to role-specific topics."""
        # Mock tool call
        mock_call = MagicMock()
        mock_call.name = "researcher_call"
        mock_call.arguments = '{"query": "test query"}'
        mock_call.id = "call-123"
        
        await mock_host_agent._route_tool_calls_to_agents([mock_call])
        
        # Verify StepRequest was sent to RESEARCHER topic
        assert mock_host_agent.publish_message.call_count == 1
        call_args = mock_host_agent.publish_message.call_args_list[0]
        
        message = call_args[0][0]
        assert isinstance(message, StepRequest)
        assert message.role == "RESEARCHER"
        assert call_args[1]["topic_id"] == DefaultTopicId(type="RESEARCHER")

    @pytest.mark.anyio
    async def test_base_agent_publish_with_topic_parameter(self, mock_host_agent):
        """Test that base Agent._publish method accepts topic_id parameter."""
        
        # Create a test message
        test_message = FlowEvent(content="Test message", source="test")
        test_topic = DefaultTopicId(type="custom-topic")
        
        # Call _publish with custom topic
        await mock_host_agent._publish(test_message, topic_id=test_topic)
        
        # Verify message was published to custom topic
        mock_host_agent.publish_message.assert_called_once_with(
            test_message,
            topic_id=test_topic,
            cancellation_token=None
        )

    @pytest.mark.anyio
    async def test_base_agent_publish_defaults_to_agent_topic(self, mock_host_agent):
        """Test that base Agent._publish defaults to agent's topic when no topic specified."""
        test_message = FlowEvent(content="Test message", source="test")
        
        # Call _publish without topic parameter
        await mock_host_agent._publish(test_message)
        
        # Verify message was published to agent's default topic
        mock_host_agent.publish_message.assert_called_once_with(
            test_message,
            topic_id=mock_host_agent._topic_id,
            cancellation_token=None
        )
