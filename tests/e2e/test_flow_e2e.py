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

    @pytest.mark.anyio
    async def test_start_flow_with_backend(self, backend_process):
        """Basic test to verify flow can be started."""
        async with FlowTestClient.create(direct_ws_url="ws://localhost:8000/ws/test_session") as client:
            await client.start_flow("zot", "what's digital constitutionalism?")
            # Wait for orchestrator to be ready
            ready = await client.wait_for_orchestrator_ready(timeout=30)
            assert ready, "Orchestrator failed to initialize"

            # Give it some time to process
            await asyncio.sleep(10)

            # Verify we got some messages
            summary = client.get_message_summary()
            assert summary["total"] > 0, "No messages received"

    @pytest.mark.anyio
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

    @pytest.mark.anyio
    async def test_osb_flow_initialization(self, backend_process):
        """Test that AutogenOrchestrator initializes with proper state events."""
        logger.info("[TEST] Test started - verifying orchestrator initialization with state events")
        try:
            with anyio.fail_after(30):  # 30 second timeout for initialization
                async with FlowTestClient.create(direct_ws_url="ws://localhost:8000/ws/osb_init_test_session") as client:
                    logger.info("[TEST] Starting flow...")
                    await client.start_flow("osb", "")  # Empty prompt

                    # Wait for orchestrator_ready event
                    logger.info("[TEST] Waiting for orchestrator_ready event...")
                    ready = await client.wait_for_orchestrator_ready(timeout=20)
                    assert ready, "Orchestrator failed to become ready"
                    logger.info("[TEST] ✓ Orchestrator is ready")

                    # Also verify we got the setup message
                    try:
                        init_msg = await client.wait_for_ui_message("Setting up AutogenOrchestrator", timeout=5)
                        logger.info(f"[TEST] ✓ Found setup message: {init_msg}")
                    except TimeoutError:
                        # It's OK if we miss this specific message as long as orchestrator is ready
                        logger.info("[TEST] Setup message not found, but orchestrator is ready")

                    # Log message summary
                    summary = client.get_message_summary()
                    logger.info(f"[TEST] Message summary: {summary}")

                    return

        except TimeoutError:
            pytest.fail("Test timed out during initialization")

    @pytest.mark.anyio
    async def test_osb_flow_with_user_interaction(self, backend_process):
        """Test complete OSB flow with user interaction."""
        logger.info("[TEST] Starting comprehensive OSB flow test with user interaction")

        try:
            with anyio.fail_after(120):  # 2 minute timeout for the entire test
                async with FlowTestClient.create(direct_ws_url="ws://localhost:8000/ws/osb_interaction_test") as client:
                    # Step 1: Start the flow with a query
                    logger.info("[TEST] Step 1: Starting OSB flow with hate speech query")
                    await client.start_flow("osb", "What is Meta's hate speech definition?")

                    # Step 2: Wait for orchestrator to be ready
                    logger.info("[TEST] Step 2: Waiting for orchestrator to be ready...")
                    ready = await client.wait_for_orchestrator_ready(timeout=30)
                    assert ready, "Orchestrator failed to initialize"
                    logger.info("[TEST] ✓ Orchestrator is ready")

                    # Step 3: Wait for user prompt (if any)
                    logger.info("[TEST] Step 3: Checking for user prompts...")
                    try:
                        # OSB might ask for confirmation or additional input
                        prompt_msg = await client.wait_for_ui_message(pattern="(proceed|confirm|continue)", timeout=10)
                        logger.info(f"[TEST] ✓ Received prompt: {prompt_msg}")

                        # Send confirmation
                        logger.info("[TEST] Sending confirmation...")
                        await client.send_manager_response("yes")
                    except TimeoutError:
                        logger.info("[TEST] No user prompt received, flow proceeding automatically")

                    # Step 4: Wait for agent activity
                    logger.info("[TEST] Step 4: Waiting for agent responses...")
                    await asyncio.sleep(30)  # Give agents time to work

                    # Check for agent activity
                    summary = client.get_message_summary()
                    logger.info(f"[TEST] Current message summary: {summary}")

                    assert summary["agent_announcements"] > 0, "No agents announced themselves"
                    assert summary["agent_traces"] > 0 or summary["ui_messages"] > 5, "No agent activity detected"

                    # Step 5: Verify we got relevant responses
                    logger.info("[TEST] Step 5: Verifying response content...")
                    found_relevant_content = False

                    # Check all messages for hate speech related content
                    for msg in client.collector.all_messages:
                        if any(keyword in msg.content.lower() for keyword in ["hate", "speech", "meta", "facebook", "policy"]):
                            found_relevant_content = True
                            logger.info(f"[TEST] ✓ Found relevant content in {msg.type}: {msg.content[:100]}...")
                            break

                    assert found_relevant_content, "No relevant content about hate speech found in responses"

                    # Step 6: Log final state
                    logger.info("[TEST] Step 6: Test completed successfully")
                    final_summary = client.get_message_summary()
                    logger.info(f"[TEST] Final message summary: {final_summary}")
                    logger.info(f"[TEST] Active agents: {final_summary['agents_active']}")

        except TimeoutError:
            pytest.fail("Test timed out - flow may be stuck")
        except AssertionError as e:
            pytest.fail(f"Assertion failed: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error: {e}")
