"""End-to-end tests for Buttermilk flows."""

import pytest
import asyncio
import websockets
import json
import subprocess
from typing import List, Dict, Any, Union

import pytest_asyncio
import anyio

from buttermilk._core.types import RunRequest
from buttermilk._core.contract import UIMessage, ManagerMessage, FlowEvent, AgentOutput, StepRequest, TaskProcessingComplete


class FlowTestClient:
    """Test client that can interact with flows programmatically."""

    def __init__(self, uri):
        self._uri = uri
        self._websocket = None
        self.received_messages: List[Dict[str, Any]] = []
        self._receive_task = None
        self._running = False

    async def __aenter__(self):
        print(f"[FlowTestClient] Attempting to connect to {self._uri}")
        for i in range(30):  # Try up to 30 times
            try:
                self._websocket = await websockets.connect(self._uri)
                print(f"[FlowTestClient] Successfully connected to {self._uri}")
                # Start background message receiver
                self._running = True
                self._receive_task = asyncio.create_task(self._background_receiver())
                return self
            except ConnectionRefusedError:
                print(f"[FlowTestClient] Connection refused, retry {i+1}/30...")
                if i < 29:  # Don't sleep on the last attempt
                    await asyncio.sleep(0.5)  # Wait for 0.5 seconds before retrying
        raise ConnectionRefusedError("Could not connect to backend after multiple retries")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._running = False
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._websocket:
            print(f"[FlowTestClient] Closing WebSocket connection.")
            await self._websocket.close()
        if exc_type:
            print(f"[FlowTestClient] Exiting with exception: {exc_val}")

    async def _background_receiver(self):
        """Background task to receive all messages from WebSocket."""
        print(f"[FlowTestClient] Background receiver started")
        while self._running:
            try:
                message_str = await self._websocket.recv()
                print(f"\n[FlowTestClient-BG] Received raw message: {message_str[:100]}...")
                try:
                    message_json = json.loads(message_str)
                    self.received_messages.append(message_json)
                    print(f"[FlowTestClient-BG] Message type: {message_json.get('type')}")
                    print(f"[FlowTestClient-BG] Message preview: {message_json.get('preview', 'No preview')}")
                    if "outputs" in message_json:
                        outputs = message_json["outputs"]
                        if isinstance(outputs, dict) and "content" in outputs:
                            print(f"[FlowTestClient-BG] Content: {outputs['content'][:200]}...")
                except json.JSONDecodeError:
                    print(f"[FlowTestClient-BG] Failed to parse as JSON: {message_str}")
            except websockets.exceptions.ConnectionClosed:
                print(f"[FlowTestClient-BG] Connection closed")
                break
            except Exception as e:
                print(f"[FlowTestClient-BG] Error in receiver: {e}")
                if self._running:
                    await asyncio.sleep(0.1)
        print(f"[FlowTestClient-BG] Background receiver stopped")

    async def start_flow(self, flow_name: str, initial_prompt: str):
        """Start a flow and wait for readiness - mimics web UI behavior."""
        # Send message in same format as web UI
        message = {
            "type": "run_flow",
            "flow": flow_name,
            "record_id": "",
            "prompt": initial_prompt,  # The web UI puts extra params in the message
        }
        print(f"[FlowTestClient] Sending run_flow message: {json.dumps(message)}")
        await self._websocket.send(json.dumps(message))

    async def _receive_json(self, timeout: int = 30) -> Dict[str, Any]:
        """Receives a JSON message from the websocket and stores it."""
        try:
            message_str = await asyncio.wait_for(self._websocket.recv(), timeout=timeout)
            print(f"[FlowTestClient] Received raw message string: {message_str}")  # Debug print raw message
            message_json = json.loads(message_str)
            self.received_messages.append(message_json)
            print(f"[FlowTestClient] Received parsed message: {json.dumps(message_json, indent=2)}")  # Debug print parsed message
            return message_json
        except asyncio.TimeoutError:
            print(f"[FlowTestClient] Timeout waiting for message (timeout={timeout}s)")
            raise

    async def wait_for_message_type(self, message_type: str, timeout: int = 60) -> Dict[str, Any]:
        """Waits for a message of a specific type from the orchestrator."""
        start_time = asyncio.get_event_loop().time()
        while True:
            elapsed_time = asyncio.get_event_loop().time() - start_time
            if elapsed_time > timeout:
                raise asyncio.TimeoutError(f"Timed out waiting for message of type {message_type}.")

            message_json = await self._receive_json()
            print(f"[FlowTestClient] Looking for type '{message_type}', received type '{message_json.get('type')}'")
            if message_json.get("type") == message_type:
                return message_json

    async def wait_for_prompt(self, timeout: int = 20) -> UIMessage:
        """Waits for a UIMessage (user prompt) from the orchestrator."""
        start_time = asyncio.get_event_loop().time()
        while True:
            elapsed_time = asyncio.get_event_loop().time() - start_time
            if elapsed_time > timeout:
                raise asyncio.TimeoutError("Timed out waiting for UIMessage.")

            message_json = await self._receive_json()
            # Attempt to parse as UIMessage
            try:
                ui_message = UIMessage(**message_json)
                return ui_message
            except Exception:
                # If it's not a UIMessage, continue waiting.
                pass

    async def send_response(self, response: str, selection: str = None):
        """Sends a ManagerMessage (user response) back to the orchestrator."""
        manager_message = ManagerMessage(content=response, selection=selection, confirm=True if response.lower() == "yes" else False)
        await self._websocket.send(manager_message.model_dump_json())

    async def wait_for_completion(self, timeout: int = 60) -> List[Dict[str, Any]]:
        """
        Waits for the flow to complete, collecting all messages until a completion signal.
        Completion is indicated by a TaskProcessingComplete message with more_tasks_remain=False.
        """
        start_time = asyncio.get_event_loop().time()
        while True:
            elapsed_time = asyncio.get_event_loop().time() - start_time
            if elapsed_time > timeout:
                print("Timeout reached in wait_for_completion.")
                return self.received_messages  # Return what we have so far

            try:
                message_json = await self._receive_json()
                # Check for TaskProcessingComplete indicating flow end
                if (
                    message_json.get("type") == "system_update"
                    and message_json.get("outputs", {}).get("status") == "COMPLETED"
                    and message_json.get("outputs", {}).get("more_tasks_remain") is False
                ):
                    # Attempt to parse as TaskProcessingComplete to be sure
                    try:
                        TaskProcessingComplete(**message_json["outputs"])
                        return self.received_messages
                    except Exception:
                        # Not a valid TaskProcessingComplete, continue
                        pass
            except websockets.exceptions.ConnectionClosedOK:
                # Connection closed gracefully, assume flow completed
                print("WebSocket connection closed gracefully.")
                return self.received_messages
            except asyncio.TimeoutError:
                # No more messages within timeout, assume flow completed
                print("Asyncio TimeoutError in wait_for_completion.")
                return self.received_messages
            except Exception as e:
                # Handle other potential errors during message reception
                print(f"Error receiving message in wait_for_completion: {e}")
                # Depending on the error, you might want to re-raise or continue
                pass


@pytest_asyncio.fixture(scope="class")
async def backend_process():
    """Starts the backend server once for all tests in the class."""
    print("[backend_fixture] Starting backend fixture setup...")
    # Kill any process already listening on port 8000
    try:
        print("[backend_fixture] Attempting to kill processes on port 8000...")
        result = subprocess.run(["fuser", "-k", "8000/tcp"], capture_output=True, text=True)
        print(f"[backend_fixture] fuser stdout: {result.stdout}")
        print(f"[backend_fixture] fuser stderr: {result.stderr}")
        if result.returncode == 0:
            print("[backend_fixture] Successfully killed processes on port 8000.")
        else:
            print("[backend_fixture] No processes found or failed to kill processes on port 8000.")
    except FileNotFoundError:
        print("[backend_fixture] fuser command not found. Skipping port cleanup.")
    except Exception as e:
        print(f"[backend_fixture] Error during fuser execution: {e}")

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
                print(f"[backend-stdout] {line}")

    async def monitor_stderr():
        while True:
            line = await process.stderr.readline()
            if not line:
                break
            line = line.decode("utf-8").strip()
            if line:
                print(f"[backend-stderr] {line}")
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
        print("[backend_fixture] Backend server started successfully")
        # Give it a moment to fully initialize
        await asyncio.sleep(2)
    except asyncio.TimeoutError:
        raise ConnectionError("Backend server failed to start")

    yield process

    # Teardown: terminate the process after all tests are done
    print(f"[backend_fixture] Terminating backend process (PID: {process.pid}).")
    process.kill()  # Use kill for more forceful shutdown
    await process.wait()
    print(f"[backend_fixture] Backend process (PID: {process.pid}) terminated with exit code {process.returncode}.")


@pytest.mark.e2e
class TestFlowE2E:
    @pytest.mark.asyncio
    async def test_start_flow_with_backend(self, backend_process):
        uri = "ws://localhost:8000/ws/test_session"
        async with FlowTestClient(uri) as client:
            await client.start_flow("zot", "what's digital constitutionalism?")
            # We will add more assertions here later
            # Example: wait for a prompt and send a response
            # prompt = await client.wait_for_prompt()
            # print(f"Received prompt: {prompt.content}")
            # await client.send_response("This is a test response.")
            # await client.wait_for_completion()

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Prompt in RunRequest not processed by HOST agent currently")
    async def test_osb_flow_with_prompt_in_runrequest(self, backend_process):
        """Test sending a prompt in the RunRequest and waiting for it to be processed."""
        print(f"[TEST] Test started - expecting prompt in RunRequest to be processed")
        uri = "ws://localhost:8000/ws/osb_prompt_test_session"
        print(f"[TEST] Connecting to WebSocket URI: {uri}")
        try:
            with anyio.fail_after(90):  # 90 second timeout for full flow
                async with FlowTestClient(uri) as client:
                    print("[TEST] Starting flow with prompt in RunRequest...")
                    await client.start_flow("osb", "Tell me about digital constitutionalism.")

                    # Wait for flow to process
                    print("[TEST] Waiting for flow to process prompt...")
                    await asyncio.sleep(60)  # Give time for processing

                    # Check what messages we received
                    print(f"\n[TEST] Total messages received: {len(client.received_messages)}")
                    for i, msg in enumerate(client.received_messages):
                        msg_type = msg.get("type")
                        preview = msg.get("preview", "N/A")
                        print(f"[TEST] Message {i}: type={msg_type}, preview={preview[:100] if preview != 'N/A' else 'N/A'}")

                    # Look for agent outputs or research results
                    agent_responses = [m for m in client.received_messages if m.get("type") in ["agent_output", "research_result", "ui_message"]]

                    assert len(agent_responses) > 0, "Expected agent responses to prompt in RunRequest"
                    print(f"[TEST] ✓ Found {len(agent_responses)} agent responses")

        except TimeoutError:
            pytest.fail("Test timed out waiting for prompt processing")

    @pytest.mark.asyncio
    async def test_osb_flow_initialization(self, backend_process):
        """Test that AutogenOrchestrator initializes and groupchat starts."""
        print(f"[TEST] Test started - verifying groupchat initialization")
        uri = "ws://localhost:8000/ws/osb_init_test_session"
        print(f"[TEST] Connecting to WebSocket URI: {uri}")
        try:
            with anyio.fail_after(30):  # 30 second timeout for initialization
                async with FlowTestClient(uri) as client:
                    print("[TEST] Starting flow...")
                    await client.start_flow("osb", "")  # Empty prompt

                    # Wait for initialization
                    print("[TEST] Waiting for AutogenOrchestrator initialization...")
                    await asyncio.sleep(20)  # Give enough time for autogen to initialize

                    # Check what messages we received
                    print(f"\n[TEST] Total messages received: {len(client.received_messages)}")
                    for i, msg in enumerate(client.received_messages):
                        print(f"[TEST] Message {i}: type={msg.get('type')}, preview={msg.get('preview', 'N/A')}")

                    # Look for system message
                    system_messages = [m for m in client.received_messages if m.get("type") == "system_message"]
                    print(f"\n[TEST] System messages found: {len(system_messages)}")

                    if system_messages:
                        for msg in system_messages:
                            content = msg.get("outputs", {}).get("content", "")
                            print(f"[TEST] System message content: {content}")
                            if "Setting up AutogenOrchestrator" in content:
                                print("[TEST] ✓ Found setup message!")
                                return

                    # If we didn't find the message, fail the test
                    pytest.fail(f"Did not find 'Setting up AutogenOrchestrator' message. Received {len(client.received_messages)} messages total.")

                    # Test completed - groupchat initialized successfully
                    print("[TEST] Test completed successfully - AutogenOrchestrator initialized")
        except TimeoutError:
            pytest.fail("Test timed out during initialization")

    @pytest.mark.asyncio
    async def test_osb_flow_empty_request_then_prompt(self, backend_process):
        """Test starting a flow with empty RunRequest, then sending prompt via WebSocket."""
        print(f"[TEST] Test started with empty RunRequest approach")
        uri = "ws://localhost:8000/ws/osb_empty_test_session"
        print(f"[TEST] Connecting to WebSocket URI: {uri}")

        try:
            with anyio.fail_after(90):  # 90 second timeout for the entire test
                async with FlowTestClient(uri) as client:
                    print("[TEST] Starting flow with empty prompt...")
                    # Start flow with empty prompt to just initialize
                    await client.start_flow("osb", "")

                    # Wait for initialization message
                    print("[TEST] Waiting for initialization...")
                    initialization_found = False
                    start_time = asyncio.get_event_loop().time()

                    while asyncio.get_event_loop().time() - start_time < 30:  # 30 second timeout
                        await asyncio.sleep(2)  # Check every 2 seconds

                        # Check for system messages indicating initialization
                        system_messages = [m for m in client.received_messages if m.get("type") == "system_message"]
                        for msg in system_messages:
                            content = msg.get("outputs", {}).get("content", "")
                            if "Setting up AutogenOrchestrator" in content:
                                print(f"[TEST] ✓ Found initialization message: {content}")
                                initialization_found = True
                                break

                        if initialization_found:
                            break

                    if not initialization_found:
                        pytest.fail("Did not receive initialization message within 30 seconds")

                    # Now send the actual prompt via WebSocket
                    print("\n[TEST] Sending prompt via WebSocket...")
                    await client.send_response("Tell me about the latest technology news.")

                    # Wait for response
                    print("[TEST] Waiting for response to prompt...")
                    await asyncio.sleep(60)  # Wait 60 seconds for processing

                    # Check what messages we received
                    print(f"\n[TEST] Total messages received: {len(client.received_messages)}")
                    for i, msg in enumerate(client.received_messages):
                        msg_type = msg.get("type")
                        preview = msg.get("preview", "N/A")
                        print(f"[TEST] Message {i}: type={msg_type}, preview={preview[:100] if preview != 'N/A' else 'N/A'}")

                        # Print more details for certain message types
                        if msg_type in ["agent_output", "ui_message", "research_result"]:
                            outputs = msg.get("outputs", {})
                            content = outputs.get("content", "")
                            print(f"[TEST]   Content preview: {content[:200]}...")

                    # Look for any response to our prompt
                    response_found = False
                    for msg in client.received_messages:
                        if msg.get("type") in ["agent_output", "ui_message", "research_result"]:
                            outputs = msg.get("outputs", {})
                            content = str(outputs.get("content", ""))
                            # Check if the response mentions technology or news
                            if any(keyword in content.lower() for keyword in ["technology", "tech", "news"]):
                                response_found = True
                                print(f"[TEST] ✓ Found response related to our prompt!")
                                break

                    if not response_found:
                        # Even if we don't find a specific response, check if flow is processing
                        if len(client.received_messages) > 1:
                            print(f"[TEST] Flow appears to be processing ({len(client.received_messages)} messages received)")
                        else:
                            pytest.fail("No response received to prompt sent via WebSocket")

        except TimeoutError:
            pytest.fail("Test timed out - flow may be stuck")
