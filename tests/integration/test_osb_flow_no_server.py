"""Integration tests for the OSB flow without starting a test server.

These tests assume the API server is already running on localhost:8000.
"""

import asyncio
import logging
import pytest
import pytest_asyncio
import time

from tests.integration.flow_test_client import FlowTestClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_osb_hate_speech_query():
    """Test OSB flow with a hate speech policy query."""
    
    async with FlowTestClient.create() as client:
        # Start the OSB flow
        await client.start_flow("osb", "What is Meta's hate speech policy?")
        
        # Wait for the host's greeting and initial prompt
        # The host typically asks for confirmation to proceed
        logger.info("Waiting for initial prompt...")
        prompt = await client.wait_for_prompt(timeout=60)
        logger.info(f"Got prompt: {prompt}")
        
        # Confirm to proceed
        await client.send_manager_response("Yes, please proceed")
        
        # Wait for agents to work
        # OSB flow typically uses researcher and policy_analyst agents
        logger.info("Waiting for agent results...")
        results = await client.wait_for_agent_results(
            expected_agents=["researcher", "policy_analyst"],
            timeout=180  # 3 minutes for agents to work
        )
        
        # Verify we got meaningful results
        assert len(results) > 0, "No agent results received"
        
        # Check that at least one agent mentioned hate speech
        found_hate_speech = False
        for result in results:
            if "hate speech" in result.content.lower():
                found_hate_speech = True
                break
        
        assert found_hate_speech, "No agent mentioned hate speech in their results"
        
        # Wait for flow completion
        logger.info("Waiting for flow completion...")
        all_messages = await client.wait_for_completion(timeout=300)
        
        # Verify we got a reasonable number of messages
        assert len(all_messages) > 10, f"Too few messages: {len(all_messages)}"
        
        # Log summary
        logger.info(f"Flow completed successfully with {len(all_messages)} messages")
        logger.info(f"Agents involved: {client.collector.get_agents_announced()}")
        logger.info(f"UI messages: {len(client.collector.ui_messages)}")
        logger.info(f"Agent traces: {len(client.collector.agent_traces)}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_osb_simple_connection():
    """Test simple connection and message flow."""
    
    async with FlowTestClient.create() as client:
        # Start the flow
        await client.start_flow("osb", "test query")
        
        # Wait briefly for any response
        await asyncio.sleep(5)
        
        # Check if we received any messages
        logger.info(f"Total messages received: {len(client.collector.all_messages)}")
        for msg in client.collector.all_messages:
            logger.info(f"Message type: {msg.type}, content: {msg.content[:100] if msg.content else 'N/A'}")
        
        assert len(client.collector.all_messages) > 0, "Should have received some messages"


if __name__ == "__main__":
    # Run the simple test directly
    asyncio.run(test_osb_simple_connection())