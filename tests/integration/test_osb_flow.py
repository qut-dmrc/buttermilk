"""Integration tests for the OSB (Online Safety Benchmarks) flow.

These tests run the actual OSB flow with real orchestrator and agents,
using the FlowTestClient to simulate user interaction.
"""

import asyncio
import logging
import pytest
import pytest_asyncio
import subprocess
import time
from pathlib import Path

from tests.integration.flow_test_client import FlowTestClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlowTestServer:
    """Manages the test API server lifecycle."""
    
    def __init__(self, config_name: str = "test/test_osb_api"):
        self.config_name = config_name
        self.process = None
        self.log_file = None
        
    async def start(self):
        """Start the API server."""
        # Create log file
        log_dir = Path("/tmp/buttermilk_test/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / f"api_server_{int(time.time())}.log"
        
        # Start server
        cmd = [
            "uv", "run", "python", "-m", "buttermilk.runner.cli",
            f"--config-name={self.config_name}"
        ]
        
        logger.info(f"Starting test server: {' '.join(cmd)}")
        
        with open(self.log_file, "w") as f:
            self.process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).parent.parent.parent  # Project root
            )
        
        # Wait for server to be ready
        await self._wait_for_ready()
    
    async def _wait_for_ready(self, timeout: float = 30.0):
        """Wait for server to be ready."""
        import aiohttp
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://localhost:8000/api/session") as resp:
                        if resp.status == 200:
                            logger.info("Test server is ready")
                            return
            except:
                pass
            
            # Check if process died
            if self.process.poll() is not None:
                with open(self.log_file, "r") as f:
                    logs = f.read()
                raise RuntimeError(f"Server process died. Logs:\n{logs[-1000:]}")
            
            await asyncio.sleep(0.5)
        
        raise TimeoutError("Server failed to start within timeout")
    
    async def stop(self):
        """Stop the API server."""
        if self.process:
            logger.info("Stopping test server")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            
            # Print last few lines of log on failure
            if self.process.returncode != 0 and self.log_file and self.log_file.exists():
                with open(self.log_file, "r") as f:
                    logs = f.read()
                logger.error(f"Server logs (last 500 chars):\n{logs[-500:]}")


@pytest_asyncio.fixture
async def test_server():
    """Fixture to start/stop test server."""
    server = FlowTestServer()
    await server.start()
    yield server
    await server.stop()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_osb_hate_speech_query(test_server):
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
async def test_osb_flow_with_followup(test_server):
    """Test OSB flow with follow-up questions."""
    
    async with FlowTestClient.create() as client:
        # Start the flow
        await client.start_flow("osb", "Tell me about content moderation")
        
        # Handle initial confirmation
        prompt = await client.wait_for_prompt(timeout=60)
        await client.send_manager_response("yes")
        
        # Wait for initial results
        await client.wait_for_agent_results(
            expected_agents=["researcher"],
            timeout=120
        )
        
        # The host might ask if we want more details or have follow-up questions
        # This tests multi-turn conversation
        try:
            followup_prompt = await client.wait_for_ui_message(
                pattern="follow-up|more|another|continue",
                timeout=30
            )
            
            if followup_prompt:
                # Send a follow-up question
                await client.send_manager_response(
                    "Yes, can you explain how AI is used in content moderation?"
                )
                
                # Wait for additional agent work
                await client.wait_for_agent_results(
                    expected_agents=["researcher", "policy_analyst"],
                    timeout=120
                )
        except TimeoutError:
            # Not all flows support follow-up questions
            logger.info("No follow-up prompt received, continuing...")
        
        # Complete the flow
        all_messages = await client.wait_for_completion(timeout=300)
        
        # Verify the conversation included moderation topics
        moderation_mentioned = any(
            "moderation" in msg.content.lower() 
            for msg in client.collector.all_messages
        )
        assert moderation_mentioned, "Content moderation not discussed"


@pytest.mark.integration
@pytest.mark.asyncio  
async def test_osb_error_handling(test_server):
    """Test OSB flow error handling with invalid input."""
    
    async with FlowTestClient.create() as client:
        # Start flow with empty prompt
        await client.start_flow("osb", "")
        
        # The flow should still handle this gracefully
        prompt = await client.wait_for_prompt(timeout=60)
        
        # Send a very long response to test limits
        long_response = "x" * 10000
        await client.send_manager_response(long_response)
        
        # Flow should continue or error gracefully
        # We don't expect specific behavior, just no crashes
        try:
            await client.wait_for_completion(timeout=120)
        except TimeoutError:
            # Check if we got any error messages
            if client.collector.errors:
                logger.info(f"Got expected error: {client.collector.errors[0].data}")
            else:
                # Flow might just ignore the input
                logger.info("Flow handled invalid input without error")


if __name__ == "__main__":
    # Allow running directly for debugging
    asyncio.run(test_osb_hate_speech_query(None))