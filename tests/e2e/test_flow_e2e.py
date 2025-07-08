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

    async def __aenter__(self):
        print(f"[FlowTestClient] Attempting to connect to {self._uri}")
        for i in range(30):  # Try up to 30 times
            try:
                self._websocket = await websockets.connect(self._uri)
                print(f"[FlowTestClient] Successfully connected to {self._uri}")
                return self
            except ConnectionRefusedError:
                print(f"[FlowTestClient] Connection refused, retry {i+1}/30...")
                if i < 29:  # Don't sleep on the last attempt
                    await asyncio.sleep(0.5)  # Wait for 0.5 seconds before retrying
        raise ConnectionRefusedError("Could not connect to backend after multiple retries")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._websocket:
            print(f"[FlowTestClient] Closing WebSocket connection.")
            await self._websocket.close()
        if exc_type:
            print(f"[FlowTestClient] Exiting with exception: {exc_val}")

    async def start_flow(self, flow_name: str, initial_prompt: str):
        """Start a flow and wait for readiness."""
        run_request = RunRequest(
            flow=flow_name,
            prompt=initial_prompt,
            ui_type="test",
        )
        print(f"[FlowTestClient] Sending RunRequest: {run_request.model_dump_json()}")
        await self._websocket.send(run_request.model_dump_json())

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

    async def wait_for_server():
        while True:
            line = await process.stderr.readline()
            if not line:
                break
            line = line.decode("utf-8").strip()
            print(f"[backend] {line}")
            if "Uvicorn running on" in line:
                return

    try:
        await asyncio.wait_for(wait_for_server(), timeout=30)
        print("[backend_fixture] Backend server started successfully")
        # Give it a moment to fully initialize
        await asyncio.sleep(1)
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
    async def test_osb_flow_interaction(self, backend_process):
        uri = "ws://localhost:8000/ws/osb_test_session"
        print(f"[TEST] Connecting to WebSocket URI: {uri}")
        try:
            with anyio.fail_after(20):  # 20 second timeout for the entire test interaction
                async with FlowTestClient(uri) as client:
                    await client.start_flow("osb", "Tell me about the latest news.")

                    # Add a small delay to allow backend to initialize and send initial messages
                    await asyncio.sleep(1)

                    # Wait for the initial system message indicating flow setup
                    print("[TEST] Waiting for initial system message...")
                    initial_message = await client.wait_for_message_type("system_message", timeout=10)
                    print(f"[TEST] Received initial system message: {initial_message.get('outputs', {}).get('content')}")
                    assert "Setting up AutogenOrchestrator" in initial_message.get("outputs", {}).get("content")

                    # Send the actual query as a ManagerMessage
                    await client.send_response("I'm interested in technology news.")

                    # Wait for the flow to complete
                    messages = await client.wait_for_completion()
                    print(f"Flow completed. Received {len(messages)} messages.")
                    print(f"All received messages: {json.dumps(messages, indent=2)}")  # Print all messages
                    # Add assertions to check the content of the received messages
                    # Expecting a ResearchResult message with relevant content
                    found_research_result = False
                    for msg in messages:
                        if msg.get("type") == "research_result":
                            if "technology news" in json.dumps(msg.get("outputs")):
                                found_research_result = True
                                break
                        elif msg.get("type") == "system_update":
                            if msg.get("outputs", {}).get("status") == "COMPLETED":
                                print("Received TaskProcessingComplete message.")

                    assert found_research_result, "Did not find expected research result with 'technology news'"
        except TimeoutError:
            pytest.fail("Test timed out due to hanging flow.")
