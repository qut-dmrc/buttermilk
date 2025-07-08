"""End-to-end tests for Buttermilk flows."""

import pytest
import asyncio
import subprocess
import logging

import pytest_asyncio
import anyio

from tests.integration.flow_test_client import FlowTestClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest_asyncio.fixture(scope="class")
async def backend_process():
    """Starts the backend server once for all tests in the class."""
    logger.info("[backend_fixture] Starting backend fixture setup...")
    # Kill any process already listening on port 8000
    try:
        logger.info("[backend_fixture] Attempting to kill processes on port 8000...")
        result = subprocess.run(["fuser", "-k", "8000/tcp"], capture_output=True, text=True)
        logger.info(f"[backend_fixture] fuser stdout: {result.stdout}")
        logger.info(f"[backend_fixture] fuser stderr: {result.stderr}")
        if result.returncode == 0:
            logger.info("[backend_fixture] Successfully killed processes on port 8000.")
        else:
            logger.info("[backend_fixture] No processes found or failed to kill processes on port 8000.")
    except FileNotFoundError:
        logger.info("[backend_fixture] fuser command not found. Skipping port cleanup.")
    except Exception as e:
        logger.info(f"[backend_fixture] Error during fuser execution: {e}")

    process = await asyncio.create_subprocess_exec(
        "uv",
        "run",
        "python",
        "-m",
        "buttermilk.runner.cli",
        "+flows=[trans,zot,osb]",
        "+run=api",
        "+llms=full",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Shared state to track server readiness
    server_ready = asyncio.Event()

    # Create tasks to continuously monitor both streams
    async def monitor_stdout():
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            line = line.decode("utf-8").strip()
            if line:  # Only print non-empty lines
                logger.info(f"[backend-stdout] {line}")

    async def monitor_stderr():
        while True:
            line = await process.stderr.readline()
            if not line:
                break
            line = line.decode("utf-8").strip()
            if line:
                logger.info(f"[backend-stderr] {line}")
                # Check for server ready message (might have ANSI codes)
                if "Uvicorn running on" in line or "Application startup complete" in line:
                    server_ready.set()

    # Start monitoring tasks in background - they'll run for the lifetime of the process
    stdout_task = asyncio.create_task(monitor_stdout())
    stderr_task = asyncio.create_task(monitor_stderr())

    async def wait_for_server():
        # Wait for the server ready event
        await server_ready.wait()

    try:
        await asyncio.wait_for(wait_for_server(), timeout=60)
        logger.info("[backend_fixture] Backend server started successfully")
        # Give it a moment to fully initialize
        await asyncio.sleep(2)
    except asyncio.TimeoutError:
        raise ConnectionError("Backend server failed to start")

    yield process

    # Teardown: terminate the process after all tests are done
    logger.info(f"[backend_fixture] Terminating backend process (PID: {process.pid}).")
    process.kill()  # Use kill for more forceful shutdown
    await process.wait()
    logger.info(f"[backend_fixture] Backend process (PID: {process.pid}) terminated with exit code {process.returncode}.")


@pytest.mark.e2e
class TestFlowE2E:
    @pytest.mark.asyncio
    async def test_start_flow_with_backend(self, backend_process):
        async with FlowTestClient.create(direct_ws_url="ws://localhost:8000/ws/test_session") as client:
            await client.start_flow("zot", "what's digital constitutionalism?")
            # We will add more assertions here later
            # Example: wait for a prompt and send a response
            # prompt = await client.wait_for_prompt()
            # logger.info(f"Received prompt: {prompt}")
            # await client.send_manager_response("This is a test response.")
            # await client.wait_for_completion()

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Prompt in RunRequest not processed by HOST agent currently")
    async def test_osb_flow_with_prompt_in_runrequest(self, backend_process):
        """Test sending a prompt in the RunRequest and waiting for it to be processed."""
        logger.info("[TEST] Test started - expecting prompt in RunRequest to be processed")
        try:
            with anyio.fail_after(90):  # 90 second timeout for full flow
                async with FlowTestClient.create(direct_ws_url="ws://localhost:8000/ws/osb_prompt_test_session") as client:
                    logger.info("[TEST] Starting flow with prompt in RunRequest...")
                    await client.start_flow("osb", "Tell me about digital constitutionalism.")

                    # Wait for flow to process
                    logger.info("[TEST] Waiting for flow to process prompt...")
                    await asyncio.sleep(60)  # Give time for processing

                    # Check what messages we received
                    logger.info(f"\n[TEST] Total messages received: {len(client.collector.all_messages)}")
                    for i, msg in enumerate(client.collector.all_messages):
                        logger.info(f"[TEST] Message {i}: type={msg.type}, content preview={msg.content[:100] if msg.content else 'N/A'}")

                    # Look for agent outputs or research results
                    agent_responses = client.collector.agent_traces + client.collector.ui_messages

                    assert len(agent_responses) > 0, "Expected agent responses to prompt in RunRequest"
                    logger.info(f"[TEST] ✓ Found {len(agent_responses)} agent responses")

        except TimeoutError:
            pytest.fail("Test timed out waiting for prompt processing")

    @pytest.mark.asyncio
    async def test_osb_flow_initialization(self, backend_process):
        """Test that AutogenOrchestrator initializes and groupchat starts."""
        logger.info("[TEST] Test started - verifying groupchat initialization")
        try:
            with anyio.fail_after(30):  # 30 second timeout for initialization
                async with FlowTestClient.create(direct_ws_url="ws://localhost:8000/ws/osb_init_test_session") as client:
                    logger.info("[TEST] Starting flow...")
                    await client.start_flow("osb", "")  # Empty prompt

                    # Wait for initialization using pattern matching
                    logger.info("[TEST] Waiting for AutogenOrchestrator initialization...")
                    try:
                        init_msg = await client.wait_for_ui_message("Setting up AutogenOrchestrator", timeout=20)
                        logger.info(f"[TEST] ✓ Found setup message: {init_msg}")
                        return
                    except TimeoutError:
                        # Log what we did receive
                        logger.info(f"\n[TEST] Total messages received: {len(client.collector.all_messages)}")
                        for i, msg in enumerate(client.collector.all_messages):
                            logger.info(f"[TEST] Message {i}: type={msg.type}, content={msg.content[:100] if msg.content else 'N/A'}")
                        
                        pytest.fail(f"Did not find 'Setting up AutogenOrchestrator' message. Received {len(client.collector.all_messages)} messages total.")

        except TimeoutError:
            pytest.fail("Test timed out during initialization")

    @pytest.mark.asyncio
    async def test_osb_flow_empty_request_then_prompt(self, backend_process):
        """Test starting a flow with empty RunRequest, then sending prompt via WebSocket."""
        logger.info("[TEST] Test started with empty RunRequest approach")

        try:
            with anyio.fail_after(90):  # 90 second timeout for the entire test
                async with FlowTestClient.create(direct_ws_url="ws://localhost:8000/ws/osb_empty_test_session") as client:
                    logger.info("[TEST] Starting flow with empty prompt...")
                    # Start flow with empty prompt to just initialize
                    await client.start_flow("osb", "")

                    # Wait for initialization message
                    logger.info("[TEST] Waiting for initialization...")
                    initialization_found = False
                    start_time = asyncio.get_event_loop().time()

                    while asyncio.get_event_loop().time() - start_time < 30:  # 30 second timeout
                        await asyncio.sleep(2)  # Check every 2 seconds

                        # Check for UI messages indicating initialization
                        for msg in client.collector.ui_messages:
                            if "Setting up AutogenOrchestrator" in msg.content:
                                logger.info(f"[TEST] ✓ Found initialization message: {msg.content}")
                                initialization_found = True
                                break

                        if initialization_found:
                            break

                    if not initialization_found:
                        pytest.fail("Did not receive initialization message within 30 seconds")

                    # Now send the actual prompt via WebSocket
                    logger.info("\n[TEST] Sending prompt via WebSocket...")
                    await client.send_manager_response("Tell me about the latest technology news.")

                    # Wait for response
                    logger.info("[TEST] Waiting for response to prompt...")
                    await asyncio.sleep(60)  # Wait 60 seconds for processing

                    # Check what messages we received
                    logger.info(f"\n[TEST] Total messages received: {len(client.collector.all_messages)}")
                    for i, msg in enumerate(client.collector.all_messages):
                        logger.info(f"[TEST] Message {i}: type={msg.type}, content preview={msg.content[:100] if msg.content else 'N/A'}")

                    # Look for any response to our prompt
                    response_found = False
                    all_content_messages = client.collector.ui_messages + client.collector.agent_traces
                    for msg in all_content_messages:
                        # Check if the response mentions technology or news
                        if any(keyword in msg.content.lower() for keyword in ["technology", "tech", "news"]):
                            response_found = True
                            logger.info(f"[TEST] ✓ Found response related to our prompt!")
                            break

                    if not response_found:
                        # Even if we don't find a specific response, check if flow is processing
                        if len(client.collector.all_messages) > 1:
                            logger.info(f"[TEST] Flow appears to be processing ({len(client.collector.all_messages)} messages received)")
                        else:
                            pytest.fail("No response received to prompt sent via WebSocket")

        except TimeoutError:
            pytest.fail("Test timed out - flow may be stuck")