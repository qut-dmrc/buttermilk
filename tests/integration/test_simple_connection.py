"""Simple test to verify WebSocket connection works."""

import asyncio
import logging
import pytest
import pytest_asyncio
from tests.integration.flow_test_client import FlowTestClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.anyio
async def test_websocket_connection():
    """Test basic WebSocket connection."""
    
    try:
        async with FlowTestClient.create() as client:
            logger.info("Connected with session ID", session_id=client.session_id)
            
            # Send a test flow request
            await client.start_flow("osb", "test query")
            
            # Wait briefly for any response
            await asyncio.sleep(2)
            
            # Check if we received any messages
            logger.info("Total messages received", num_messages=len(client.collector.all_messages))
            for msg in client.collector.all_messages:
                logger.info("Message received", message_type=msg.type, content=msg.content[:100] if msg.content else 'N/A')
            
            assert client.session_id is not None, "Should have a session ID"
            
    except Exception as e:
        logger.error("Connection test failed", error=e)
        raise


@pytest.mark.integration
@pytest.mark.anyio
async def test_api_session_endpoint():
    """Test that API session endpoint works."""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get("http://localhost:8000/api/session") as resp:
                logger.info("Session endpoint status", status=resp.status)
                if resp.status == 200:
                    data = await resp.json()
                    logger.info("Session data", data=data)
                    assert "session_id" in data
                else:
                    text = await resp.text()
                    logger.error("Session endpoint error", error=text)
                    
        except Exception as e:
            logger.error("Session endpoint test failed", error=e)
            raise


if __name__ == "__main__":
    # Run the simple connection test
    asyncio.run(test_websocket_connection())
