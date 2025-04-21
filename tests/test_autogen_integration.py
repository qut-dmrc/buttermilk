import asyncio
import pytest
from typing import List, Optional

from autogen_core import MessageContext, DefaultTopicId
from autogen_core import SingleThreadedAgentRuntime

from buttermilk._core.agent import AgentConfig, AgentInput, AgentOutput, buttermilk_handler
from buttermilk._core.contract import UserInstructions
from buttermilk.agents.judge import Judge
from buttermilk.libs.autogen import AutogenAgentAdapter


@pytest.mark.anyio
async def test_buttermilk_handler_adapter():
    """Test that the Buttermilk handler adaptation works correctly."""
    # Create an agent configuration
    agent_config = AgentConfig(
        role="judge",
        name="test-judge",
        description="A test judge agent",
        parameters={"model": "gpt-4"},
    )

    # Create adapter
    adapter = AutogenAgentAdapter(
        agent_cfg=agent_config,
        wrapped_agent_cls=Judge,
    )

    # Verify adapter setup correctly
    assert adapter.wrapped_agent.name == "test-judge"
    assert adapter.wrapped_agent.role == "judge"

    # Create a test message
    test_input = AgentInput(
        prompt="Test prompt",
        inputs={"prompt": "Is this content harmful?"},
        parameters={},
    )

    # Create a simple runtime for testing
    runtime = SingleThreadedAgentRuntime()
    topic_id = DefaultTopicId(type="test-topic")

    # Mock message context
    context = MessageContext(
        runtime=runtime,
        topic_id=topic_id,
        message_id="test-message-id",
        sender=None,
        sender_name=None,
    )

    # Simulate Autogen calling the handler with the context
    # This would happen when using the adapter in the group chat
    for msg_type, handler in adapter.message_handlers.items():
        if msg_type == AgentInput:
            result = await handler(test_input, context)
            # In a real scenario, this would then be published by the Autogen runtime
            # In this test, we're just verifying the handler is set up correctly
            assert result is not None
            break


@pytest.mark.anyio
async def test_autogen_registration():
    """Test that the Autogen registration process works correctly."""
    # Create a runtime
    runtime = SingleThreadedAgentRuntime()

    # Register an adapter
    agent_type = await AutogenAgentAdapter.register(
        runtime=runtime,
        type="test-judge",
        factory=lambda: AutogenAgentAdapter(
            agent_cfg=AgentConfig(
                role="judge",
                name="test-judge",
                description="A test judge agent",
                parameters={"model": "gemini2flashlite"},
            ),
            wrapped_agent_cls=Judge,
        ),
    )

    # Verify registration
    assert agent_type.name == "test-judge"

    # Get an instance of the registered agent
    agent_id = await runtime.get(agent_type)
    assert agent_id is not None

    # Verify the adapter is accessible in the runtime
    # This is a bit of a hack to access the internal runtime agent store
    # In a real scenario, you would use the runtime APIs
    agent = runtime._agent_store.get_entry(agent_id)
    assert agent is not None
    assert isinstance(agent.value, AutogenAgentAdapter)
    assert agent.value.wrapped_agent.name == "test-judge"
