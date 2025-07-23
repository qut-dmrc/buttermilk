"""Simple validation script to verify ErrorEvent fix works correctly."""

import sys
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Add the project root to Python path
sys.path.insert(0, '/home/runner/work/buttermilk/buttermilk')

try:
    from buttermilk.agents.flowcontrol.structured_llmhost import StructuredLLMHostAgent
    from buttermilk._core.contract import ErrorEvent, AgentOutput
    from buttermilk._core.agent import ManagerMessage
    from autogen_core import MessageContext
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


async def test_error_event_publishing():
    """Test that ErrorEvent is published when no tools are available."""
    
    # Create a mock host agent with minimal setup
    host = StructuredLLMHostAgent(
        agent_id="test_host",
        agent_name="TestHost",
        role="HOST",
        description="Test host agent",
        parameters={"model": "gpt-4", "template": "test", "human_in_loop": False}
    )
    
    # Mock the _publish method to capture messages
    published_messages = []
    
    async def mock_publish(message, **kwargs):
        published_messages.append(message)
        print(f"Mock published: {type(message).__name__} - {message}")
    
    host._publish = mock_publish
    
    # Ensure no tools are available (should be default)
    assert not host._tools, "Expected no tools to be available for test"
    
    # Create a mock message and context
    test_message = ManagerMessage(content="Test message")
    mock_context = MagicMock()
    mock_context.cancellation_token = None
    
    # Call the method that should publish an ErrorEvent
    await host._receive_instructions(test_message, mock_context)
    
    # Verify that an ErrorEvent was published
    assert len(published_messages) == 1, f"Expected 1 message, got {len(published_messages)}"
    
    published_msg = published_messages[0]
    assert isinstance(published_msg, ErrorEvent), f"Expected ErrorEvent, got {type(published_msg)}"
    assert published_msg.source == host.agent_id, "ErrorEvent source should be the host agent ID"
    assert "no tools available" in published_msg.content, "ErrorEvent content should mention no tools"
    assert published_msg.is_error == True, "ErrorEvent should have is_error=True"
    
    print("✅ ErrorEvent publishing test passed")


async def test_message_type_properties():
    """Test that ErrorEvent and AgentOutput have expected properties."""
    
    # Test ErrorEvent properties
    error = ErrorEvent(source="test_agent", content="Test error message")
    assert hasattr(error, 'source'), "ErrorEvent should have source attribute"
    assert hasattr(error, 'content'), "ErrorEvent should have content attribute"
    assert hasattr(error, 'is_error'), "ErrorEvent should have is_error property"
    assert error.is_error == True, "ErrorEvent.is_error should be True"
    assert str(error).startswith("ERROR:"), "ErrorEvent string should start with 'ERROR:'"
    
    # Test AgentOutput properties
    output = AgentOutput(agent_id="test_agent", outputs="Test output")
    assert hasattr(output, 'agent_id'), "AgentOutput should have agent_id attribute"
    assert hasattr(output, 'outputs'), "AgentOutput should have outputs attribute"
    assert hasattr(output, 'content'), "AgentOutput should have content property"
    assert output.content == "Test output", "AgentOutput.content should return outputs as string"
    
    print("✅ Message type properties test passed")


async def main():
    """Run all validation tests."""
    print("🔍 Running ErrorEvent fix validation...")
    
    try:
        await test_error_event_publishing()
        await test_message_type_properties()
        print("\n🎉 All validation tests passed! ErrorEvent fix is working correctly.")
        return True
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)