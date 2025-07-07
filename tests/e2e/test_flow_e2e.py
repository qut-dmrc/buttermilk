"""End-to-end tests for Buttermilk flows."""

import pytest
import asyncio
import websockets
import json
import subprocess
from typing import List, Dict, Any, Union

import pytest_asyncio

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
        for i in range(30): # Try up to 30 times
            try:
                self._websocket = await websockets.connect(self._uri)
                print(f"[FlowTestClient] Successfully connected to {self._uri}")
                return self
            except ConnectionRefusedError:
                print(f"[FlowTestClient] Connection refused, retry {i+1}/30...")
                if i < 29: # Don't sleep on the last attempt
                    await asyncio.sleep(0.5) # Wait for 0.5 seconds before retrying
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
        message_str = await asyncio.wait_for(self._websocket.recv(), timeout=timeout)
        message_json = json.loads(message_str)
        self.received_messages.append(message_json)
        print(f"[FlowTestClient] Received message: {json.dumps(message_json, indent=2)}") # Debug print
        return message_json

    async def wait_for_prompt(self, timeout: int = 60) -> UIMessage:
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
        manager_message = ManagerMessage(
            content=response,
            selection=selection,
            confirm=True if response.lower() == "yes" else False
        )
        await self._websocket.send(manager_message.model_dump_json())

    async def wait_for_completion(self, timeout: int = 120) -> List[Dict[str, Any]]:
        """
        Waits for the flow to complete, collecting all messages until a completion signal.
        Completion is indicated by a TaskProcessingComplete message with more_tasks_remain=False.
        """
        start_time = asyncio.get_event_loop().time()
        while True:
            elapsed_time = asyncio.get_event_loop().time() - start_time
            if elapsed_time > timeout:
                print("Timeout reached in wait_for_completion.")
                return self.received_messages # Return what we have so far

            try:
                message_json = await self._receive_json()
                # Check for TaskProcessingComplete indicating flow end
                # Note: This assumes TaskProcessingComplete is sent as a top-level message
                # and contains 'status' and 'more_tasks_remain' keys directly.
                if message_json.get("type") == "system_update" and \
                   message_json.get("outputs", {}).get("status") == "COMPLETED" and \
                   not message_json.get("outputs", {}).get("more_tasks_remain"):
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
    process = await asyncio.create_subprocess_exec(
        "uv", "run", "python", "-m", "buttermilk.runner.cli", "+flows=[trans,zot,osb]", "+run=api", "+llms=full",
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
        await asyncio.wait_for(wait_for_server(), timeout=60)
    except asyncio.TimeoutError:
        raise ConnectionError("Backend server failed to start")

    yield process

    # Teardown: terminate the process after all tests are done
    process.terminate()
    await process.wait()

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
        async with FlowTestClient(uri) as client:
            await client.start_flow("osb", "Tell me about the latest news.")

            # Wait for the flow to complete
            messages = await client.wait_for_completion()
            print(f"Flow completed. Received {len(messages)} messages.")
            print(f"All received messages: {json.dumps(messages, indent=2)}") # Print all messages
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