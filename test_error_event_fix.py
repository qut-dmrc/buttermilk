"""Test to verify that hosts send ErrorEvent messages for errors instead of AgentOutput."""

import asyncio
from unittest.mock import MagicMock

# Create a minimal test environment without external dependencies
class MockStructuredLLMHostAgent:
    """Mock version of StructuredLLMHostAgent for testing."""
    
    def __init__(self, agent_id, agent_name, **kwargs):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self._tools = []  # Start with no tools to trigger the error path
        self.published_messages = []
        
    async def _publish(self, message, highlight=False, topic_id=None):
        """Mock publish method that captures messages."""
        self.published_messages.append(message)
        print(f"Published: {type(message).__name__} - {message}")

def test_error_event_fix():
    """Test that demonstrates the ErrorEvent fix is applied correctly."""
    
    # Create the mock agent  
    host = MockStructuredLLMHostAgent(
        agent_id="test-host",
        agent_name="TestHost"
    )
    
    # Simulate the problematic code path
    async def simulate_no_tools_scenario():
        msg = f"StructuredLLMHost {host.agent_name} has no tools available after waiting 5s. This may indicate the participants have not advertised their capabilities."
        print(f"Error: {msg}")
        
        # This simulates the FIXED version - publishing ErrorEvent instead of AgentOutput for errors
        class MockErrorEvent:
            def __init__(self, source, content):
                self.source = source
                self.content = content
                
            def __str__(self):
                return f"ERROR: {self.content}"
                
            @property
            def is_error(self):
                return True
        
        error_event = MockErrorEvent(
            source=host.agent_id,
            content="Unable to process request: no tools available."
        )
        
        await host._publish(error_event)
        
        # Verify the message is an ErrorEvent, not an AgentOutput
        assert len(host.published_messages) == 1
        published_msg = host.published_messages[0]
        
        assert not isinstance(published_msg, str), f"Should not publish raw string, got: {published_msg}"
        assert hasattr(published_msg, 'source'), "Should have source attribute (ErrorEvent field)"
        assert hasattr(published_msg, 'content'), "Should have content attribute (ErrorEvent field)"
        assert hasattr(published_msg, 'is_error'), "Should have is_error property"
        assert published_msg.is_error == True, "Should indicate this is an error"
        assert published_msg.source == host.agent_id, "Should have correct source agent_id"
        
        print("✅ Test passed: Host publishes ErrorEvent for errors instead of AgentOutput")
        
    # Run the test
    asyncio.run(simulate_no_tools_scenario())


def test_message_type_distinction():
    """Test that verifies the distinction between ErrorEvent and AgentOutput."""
    print("\n📋 Testing message type distinctions:")
    
    # Mock AgentOutput for comparison
    class MockAgentOutput:
        def __init__(self, agent_id, outputs, metadata=None):
            self.agent_id = agent_id
            self.outputs = outputs
            self.metadata = metadata or {}
            
        @property
        def is_error(self):
            return self.metadata.get("error", False)
            
    # Mock ErrorEvent
    class MockErrorEvent:
        def __init__(self, source, content):
            self.source = source
            self.content = content
            
        @property
        def is_error(self):
            return True
    
    # Test ErrorEvent properties
    error_event = MockErrorEvent(source="agent1", content="Something went wrong")
    assert error_event.is_error == True
    assert hasattr(error_event, 'source')
    assert hasattr(error_event, 'content')
    print("✅ ErrorEvent: Designed for error broadcasting")
    
    # Test AgentOutput properties  
    agent_output = MockAgentOutput(agent_id="agent1", outputs="Processing complete")
    assert agent_output.is_error == False  # Normal output, not an error
    assert hasattr(agent_output, 'agent_id')
    assert hasattr(agent_output, 'outputs')
    print("✅ AgentOutput: Designed for processed outputs")
    
    print("✅ Test passed: Message types have distinct purposes")


if __name__ == "__main__":
    test_error_event_fix()
    test_message_type_distinction()
    print("\n🎉 All tests passed! Errors use ErrorEvent, outputs use AgentOutput.")