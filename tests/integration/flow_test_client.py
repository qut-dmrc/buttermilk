"""Flow Test Client for Integration Testing.

This module provides a WebSocket client specifically designed for testing
Buttermilk flows. It respects the natural flow timing and provides
event-driven testing capabilities.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Optional, Pattern
import re

import aiohttp
from aiohttp import ClientWebSocketResponse

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Known message types in the WebSocket protocol."""
    RUN_FLOW = "run_flow"
    MANAGER_RESPONSE = "manager_response"
    UI_MESSAGE = "ui_message"
    AGENT_ANNOUNCEMENT = "agent_announcement"
    AGENT_TRACE = "agent_trace"
    FLOW_COMPLETE = "flow_complete"
    ERROR = "error"
    TASK_PROCESSING_STARTED = "task_processing_started"
    TASK_PROCESSING_COMPLETE = "task_processing_complete"


@dataclass
class CollectedMessage:
    """A message collected during flow execution."""
    type: str
    data: dict
    timestamp: datetime
    raw: str
    
    @property
    def content(self) -> str:
        """Extract content from various message formats."""
        if isinstance(self.data, dict):
            # Try common content locations
            if "content" in self.data:
                return str(self.data["content"])
            if "data" in self.data and isinstance(self.data["data"], dict):
                if "content" in self.data["data"]:
                    return str(self.data["data"]["content"])
        return ""
    
    @property
    def agent_role(self) -> Optional[str]:
        """Extract agent role if this is an agent message."""
        if self.type == MessageType.AGENT_ANNOUNCEMENT:
            agent_config = self.data.get("data", {}).get("agent_config", {})
            return agent_config.get("role")
        elif self.type == MessageType.AGENT_TRACE:
            agent_info = self.data.get("data", {}).get("agent_info", {})
            return agent_info.get("role")
        return None


class MessageCollector:
    """Collects and categorizes messages during flow execution."""
    
    def __init__(self):
        self.all_messages: list[CollectedMessage] = []
        self.ui_messages: list[CollectedMessage] = []
        self.agent_announcements: list[CollectedMessage] = []
        self.agent_traces: list[CollectedMessage] = []
        self.errors: list[CollectedMessage] = []
        
    def add_message(self, msg_type: str, data: dict, raw: str):
        """Add a message to the collection."""
        message = CollectedMessage(
            type=msg_type,
            data=data,
            timestamp=datetime.now(),
            raw=raw
        )
        
        self.all_messages.append(message)
        
        # Categorize
        if msg_type == MessageType.UI_MESSAGE:
            self.ui_messages.append(message)
        elif msg_type == MessageType.AGENT_ANNOUNCEMENT:
            self.agent_announcements.append(message)
        elif msg_type == MessageType.AGENT_TRACE:
            self.agent_traces.append(message)
        elif msg_type == MessageType.ERROR:
            self.errors.append(message)
    
    def get_agents_announced(self) -> list[str]:
        """Get list of agent roles that have announced themselves."""
        return [msg.agent_role for msg in self.agent_announcements if msg.agent_role]
    
    def get_agent_results(self, agent_role: Optional[str] = None) -> list[CollectedMessage]:
        """Get agent trace messages, optionally filtered by role."""
        if agent_role:
            return [msg for msg in self.agent_traces if msg.agent_role == agent_role]
        return self.agent_traces


class FlowEventWaiter:
    """Waits for specific events with timeout support."""
    
    def __init__(self, collector: MessageCollector):
        self.collector = collector
    
    async def wait_for_ui_message(
        self, 
        pattern: Optional[str | Pattern] = None,
        timeout: float = 30.0,
        poll_interval: float = 0.1
    ) -> CollectedMessage:
        """Wait for a UI message matching the pattern."""
        start_time = time.time()
        
        if isinstance(pattern, str):
            # Convert string pattern to regex (case-insensitive)
            pattern = re.compile(pattern, re.IGNORECASE)
        
        while time.time() - start_time < timeout:
            # Check existing messages
            for msg in self.collector.ui_messages:
                if pattern is None or pattern.search(msg.content):
                    return msg
            
            # Wait a bit before checking again
            await asyncio.sleep(poll_interval)
        
        # Timeout - provide helpful error
        raise TimeoutError(
            f"Timed out waiting for UI message matching {pattern}. "
            f"Received {len(self.collector.ui_messages)} UI messages: "
            f"{[msg.content[:50] + '...' for msg in self.collector.ui_messages[-3:]]}"
        )
    
    async def wait_for_agent_announcement(
        self,
        agent_role: str,
        timeout: float = 30.0
    ) -> CollectedMessage:
        """Wait for a specific agent to announce itself."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            announced = self.collector.get_agents_announced()
            if agent_role in announced:
                # Find the announcement
                for msg in self.collector.agent_announcements:
                    if msg.agent_role == agent_role:
                        return msg
            
            await asyncio.sleep(0.1)
        
        raise TimeoutError(
            f"Timed out waiting for {agent_role} announcement. "
            f"Announced agents: {self.collector.get_agents_announced()}"
        )
    
    async def wait_for_agents(
        self,
        expected_agents: list[str],
        timeout: float = 60.0
    ) -> dict[str, CollectedMessage]:
        """Wait for multiple agents to announce themselves."""
        results = {}
        for agent in expected_agents:
            msg = await self.wait_for_agent_announcement(agent, timeout)
            results[agent] = msg
        return results
    
    async def wait_for_completion(
        self,
        timeout: float = 300.0
    ) -> list[CollectedMessage]:
        """Wait for flow completion."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check for completion message
            for msg in self.collector.all_messages:
                if msg.type == MessageType.FLOW_COMPLETE:
                    return self.collector.all_messages
            
            # Check for errors
            if self.collector.errors:
                error = self.collector.errors[0]
                raise RuntimeError(f"Flow error: {error.data}")
            
            await asyncio.sleep(0.1)
        
        raise TimeoutError(
            f"Timed out waiting for flow completion after {timeout}s. "
            f"Received {len(self.collector.all_messages)} messages total."
        )


class FlowTestClient:
    """WebSocket client for testing Buttermilk flows."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        ws_url: str = "ws://localhost:8000/ws",
        direct_ws_url: Optional[str] = None
    ):
        self.base_url = base_url
        self.ws_url = ws_url
        self.direct_ws_url = direct_ws_url  # For direct WebSocket connection without session
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[ClientWebSocketResponse] = None
        self.session_id: Optional[str] = None
        self.collector = MessageCollector()
        self.waiter = FlowEventWaiter(self.collector)
        self._listener_task: Optional[asyncio.Task] = None
        
    @classmethod
    @asynccontextmanager
    async def create(cls, **kwargs) -> AsyncGenerator["FlowTestClient", None]:
        """Create and initialize a test client."""
        client = cls(**kwargs)
        try:
            await client.connect()
            yield client
        finally:
            await client.disconnect()
    
    async def connect(self):
        """Connect to the WebSocket with retry logic."""
        # Create HTTP session
        self.session = aiohttp.ClientSession()
        
        # If direct WebSocket URL is provided, connect directly without session
        if self.direct_ws_url:
            ws_retry_count = 0
            max_ws_retries = 30
            while ws_retry_count < max_ws_retries:
                try:
                    self.ws = await self.session.ws_connect(self.direct_ws_url)
                    logger.info(f"WebSocket connected directly to {self.direct_ws_url}")
                    break
                except aiohttp.ClientConnectionError as e:
                    ws_retry_count += 1
                    if ws_retry_count >= max_ws_retries:
                        raise ConnectionRefusedError(
                            f"Could not connect to WebSocket after {max_ws_retries} retries"
                        ) from e
                    logger.info(f"WebSocket connection refused, retry {ws_retry_count}/{max_ws_retries}...")
                    await asyncio.sleep(0.5)
        else:
            # Get session ID with retries
            session_retry_count = 0
            max_session_retries = 10
            while session_retry_count < max_session_retries:
                try:
                    async with self.session.get(f"{self.base_url}/api/session") as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            raise RuntimeError(f"Failed to get session: {resp.status} - {text}")
                        
                        data = await resp.json()
                        self.session_id = data.get("session_id")
                        logger.info(f"Got session ID: {self.session_id}")
                        break
                except aiohttp.ClientConnectionError as e:
                    session_retry_count += 1
                    if session_retry_count >= max_session_retries:
                        raise ConnectionRefusedError(
                            f"Could not get session ID after {max_session_retries} retries"
                        ) from e
                    logger.info(f"Session endpoint connection refused, retry {session_retry_count}/{max_session_retries}...")
                    await asyncio.sleep(0.5)
            
            # Connect WebSocket with retries
            ws_retry_count = 0
            max_ws_retries = 30
            while ws_retry_count < max_ws_retries:
                try:
                    self.ws = await self.session.ws_connect(f"{self.ws_url}/{self.session_id}")
                    logger.info("WebSocket connected")
                    break
                except aiohttp.ClientConnectionError as e:
                    ws_retry_count += 1
                    if ws_retry_count >= max_ws_retries:
                        raise ConnectionRefusedError(
                            f"Could not connect to WebSocket after {max_ws_retries} retries"
                        ) from e
                    logger.info(f"WebSocket connection refused, retry {ws_retry_count}/{max_ws_retries}...")
                    await asyncio.sleep(0.5)
        
        # Start message listener
        self._listener_task = asyncio.create_task(self._listen_for_messages())
    
    async def disconnect(self):
        """Disconnect and cleanup."""
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        
        if self.ws:
            await self.ws.close()
        
        if self.session:
            await self.session.close()
    
    async def _listen_for_messages(self):
        """Background task to listen for messages."""
        try:
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        msg_type = data.get("type", "unknown")
                        
                        logger.info(f"Received message type: {msg_type}, data keys: {list(data.keys())}")
                        self.collector.add_message(msg_type, data, msg.data)
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse message: {e}, raw: {msg.data[:100]}")
                        
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self.ws.exception()}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("WebSocket closed")
                    break
                    
        except asyncio.CancelledError:
            logger.debug("Message listener cancelled")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in message listener: {e}")
            raise
    
    async def start_flow(self, flow_name: str, prompt: str = "", record_id: str = ""):
        """Start a flow and wait for initialization."""
        message = {
            "type": MessageType.RUN_FLOW,
            "flow": flow_name,
            "prompt": prompt,
            "record_id": record_id  # Added for compatibility with web UI format
        }
        
        logger.info(f"Starting flow: {flow_name}")
        await self.ws.send_json(message)
        
        # Give the flow a moment to start
        await asyncio.sleep(0.5)
    
    async def send_manager_response(self, content: str, selection: Optional[str] = None):
        """Send a manager response message."""
        message = {
            "type": MessageType.MANAGER_RESPONSE,
            "content": content,
            "confirm": content.lower() == "yes" if selection is None else False
        }
        
        # Add selection if provided
        if selection is not None:
            message["selection"] = selection
        
        logger.info(f"Sending response: {content}")
        await self.ws.send_json(message)
    
    async def wait_for_ui_message(self, pattern: Optional[str] = None, timeout: float = 30.0) -> str:
        """Wait for a UI message and return its content."""
        msg = await self.waiter.wait_for_ui_message(pattern, timeout)
        return msg.content
    
    async def wait_for_prompt(self, timeout: float = 30.0) -> str:
        """Wait for a prompt from the system."""
        # Common prompt patterns
        prompt_pattern = r"(confirm|proceed|continue|yes/no|y/n|\?|:)$"
        msg = await self.waiter.wait_for_ui_message(prompt_pattern, timeout)
        return msg.content
    
    async def wait_for_agent_results(
        self,
        expected_agents: list[str],
        timeout: float = 120.0
    ) -> list[CollectedMessage]:
        """Wait for specific agents to provide results."""
        # First wait for agents to announce
        await self.waiter.wait_for_agents(expected_agents, timeout/2)
        
        # Then wait for their results
        start_time = time.time()
        while time.time() - start_time < timeout/2:
            results = []
            for agent in expected_agents:
                agent_results = self.collector.get_agent_results(agent)
                if agent_results:
                    results.extend(agent_results)
            
            if len(results) >= len(expected_agents):
                return results
            
            await asyncio.sleep(0.5)
        
        # Timeout
        actual_agents = list(set(msg.agent_role for msg in self.collector.agent_traces if msg.agent_role))
        raise TimeoutError(
            f"Timed out waiting for results from {expected_agents}. "
            f"Got results from: {actual_agents}"
        )
    
    async def wait_for_completion(self, timeout: float = 300.0) -> list[CollectedMessage]:
        """Wait for flow completion and return all messages."""
        return await self.waiter.wait_for_completion(timeout)