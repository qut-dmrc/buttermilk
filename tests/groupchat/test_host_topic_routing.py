"""Tests for HostAgent topic routing functionality."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from buttermilk._core.runtime_types import DefaultTopicId
from buttermilk._core.contract import (
    FlowEvent,
    StepRequest,
    SystemPromptMessage,
)
from buttermilk.agents.flowcontrol.host import HostAgent

pytestmark = pytest.mark.anyio


@pytest.fixture
def mock_host_agent():
    """Create a mocked HostAgent for testing."""
    publish_mock = AsyncMock()

    agent = HostAgent(
        agent_id="test-host",
        agent_name="TestHost",
        role="HOST",
        description="Test host agent",
        parameters={"human_in_loop": False},
        publish_fn=publish_mock,
    )

    agent._topic_id = DefaultTopicId(type="main-topic")
    agent._participants = {
        "RESEARCHER": "Research agent",
        "WRITER": "Writing agent",
    }

    agent._tool_to_agent_map = {"researcher_call": "researcher-agent-id"}
    agent._agent_registry = {"researcher-agent-id": MagicMock(agent_config=MagicMock(role="researcher"))}
    agent.human_in_loop = False

    return agent, publish_mock


class TestHostTopicRouting:
    """Test cases for host agent topic routing."""

    @pytest.mark.anyio
    async def test_step_request_routes_to_role_topic(self, mock_host_agent):
        """Test that StepRequest messages are routed to role-specific topics."""
        agent, publish_mock = mock_host_agent

        step = StepRequest(
            role="RESEARCHER",
            content="Execute researcher step",
            inputs={"query": "test query"},
        )

        await agent._execute_step(step)

        assert publish_mock.call_count == 2

        # First call should be FlowEvent to main topic
        first_call = publish_mock.call_args_list[0]
        assert isinstance(first_call[0][0], FlowEvent)
        assert first_call[0][1] == agent._topic_id

        # Second call should be StepRequest to role-specific topic
        second_call = publish_mock.call_args_list[1]
        assert isinstance(second_call[0][0], StepRequest)
        assert second_call[0][1] == DefaultTopicId(type="RESEARCHER")

    @pytest.mark.anyio
    async def test_flow_event_sent_before_step_request(self, mock_host_agent):
        """Test that FlowEvent is sent to main topic before StepRequest."""
        agent, publish_mock = mock_host_agent

        step = StepRequest(role="WRITER", content="Execute writer step")

        await agent._execute_step(step)

        flow_event = publish_mock.call_args_list[0][0][0]
        assert isinstance(flow_event, FlowEvent)
        assert flow_event.source == agent.agent_id
        assert "Starting WRITER step" in flow_event.content
        assert "Writing agent" in flow_event.content

    @pytest.mark.anyio
    async def test_manager_step_sends_ui_message_only(self, mock_host_agent):
        """Test that MANAGER steps only send SystemPromptMessage to main topic."""
        agent, publish_mock = mock_host_agent

        step = StepRequest(role="MANAGER", content="What would you like to do next?")

        await agent._execute_step(step)

        assert publish_mock.call_count == 1
        call_args = publish_mock.call_args_list[0]

        message = call_args[0][0]
        assert isinstance(message, SystemPromptMessage)
        assert message.content == "What would you like to do next?"
        assert call_args[0][1] == agent._topic_id

    @pytest.mark.anyio
    async def test_end_step_routes_to_main_topic(self, mock_host_agent):
        """Test that END steps are sent to main topic."""
        from buttermilk._core.constants import END

        agent, publish_mock = mock_host_agent

        step = StepRequest(role=END, content="Flow completed")

        await agent._execute_step(step)

        assert publish_mock.call_count == 1
        call_args = publish_mock.call_args_list[0]
        assert isinstance(call_args[0][0], StepRequest)
        assert call_args[0][1] == agent._topic_id

    @pytest.mark.anyio
    async def test_unknown_role_still_routes_to_role_topic(self, mock_host_agent):
        """Test that unknown roles still route to role-specific topics."""
        agent, publish_mock = mock_host_agent

        step = StepRequest(role="UNKNOWN_ROLE", content="Execute unknown step")

        await agent._execute_step(step)

        assert publish_mock.call_count == 1
        call_args = publish_mock.call_args_list[0]
        assert isinstance(call_args[0][0], StepRequest)
        assert call_args[0][1] == DefaultTopicId(type="UNKNOWN_ROLE")

    @pytest.mark.anyio
    async def test_route_tool_calls_uses_role_topics(self, mock_host_agent):
        """Test that _route_tool_calls_to_agents routes to role-specific topics."""
        agent, publish_mock = mock_host_agent

        mock_call = MagicMock()
        mock_call.name = "researcher_call"
        mock_call.arguments = '{"query": "test query"}'
        mock_call.id = "call-123"

        await agent._route_tool_calls_to_agents([mock_call])

        assert publish_mock.call_count == 1
        call_args = publish_mock.call_args_list[0]

        message = call_args[0][0]
        assert isinstance(message, StepRequest)
        assert message.role == "RESEARCHER"
        assert call_args[0][1] == DefaultTopicId(type="RESEARCHER")

    @pytest.mark.anyio
    async def test_base_agent_publish_with_topic_parameter(self, mock_host_agent):
        """Test that base Agent._publish method accepts topic_id parameter."""
        agent, publish_mock = mock_host_agent

        test_message = FlowEvent(content="Test message", source="test")
        test_topic = DefaultTopicId(type="custom-topic")

        await agent._publish(test_message, topic_id=test_topic)

        publish_mock.assert_called_once_with(test_message, test_topic)

    @pytest.mark.anyio
    async def test_base_agent_publish_defaults_to_agent_topic(self, mock_host_agent):
        """Test that base Agent._publish defaults to agent's topic when no topic specified."""
        agent, publish_mock = mock_host_agent

        test_message = FlowEvent(content="Test message", source="test")

        await agent._publish(test_message)

        publish_mock.assert_called_once_with(test_message, agent._topic_id)
