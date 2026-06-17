"""Debug Agent for LLM-driven debugging.

This agent provides tools for debugging Buttermilk flows,
including log reading and WebSocket flow control.
"""

import glob
import os
from pathlib import Path
from typing import Any

from buttermilk import AgentInput, logger
from buttermilk._core.agent import Agent
from buttermilk._core.contract import AgentOutput

# Tool decorator removed with MCP implementation
from buttermilk.agents.test_utils import FlowTestClient


class DebugAgent(Agent):
    """Agent providing debugging tools for Buttermilk flows."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._active_clients: dict[str, FlowTestClient] = {}
        self._puppet_client: FlowTestClient | None = None
        self._puppet_listening: bool = False

    async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput | None:
        """Process debugging requests."""
        # This agent is primarily tool-based, so _process just returns a helpful message
        return AgentOutput(
            agent_id=self.agent_id,
            outputs={
                "message": "Debug agent is ready. Use the exposed tools for debugging.",
                "available_tools": [
                    "get_latest_buttermilk_logs",
                    "list_log_files",
                    "start_websocket_client",
                    "send_websocket_message",
                    "get_websocket_messages",
                    "get_websocket_summary",
                    "stop_websocket_client",
                    "list_active_clients",
                    "start_puppet_mode",
                    "puppet_start_flow",
                    "puppet_send_response",
                    "puppet_get_messages",
                    "puppet_get_summary",
                    "stop_puppet_mode",
                ],
            },
        )

    # Log reading tools

    def get_latest_buttermilk_logs(self, lines: int = 100) -> str:
        """Get the last N lines from the most recent buttermilk log file.

        Args:
            lines: Number of lines to retrieve from the end of the log

        Returns:
            The last N lines of the most recent log file, or error message if no logs found
        """
        log_files = glob.glob("/tmp/buttermilk_*.jsonl")
        if not log_files:
            return "No buttermilk log files found in /tmp/"

        # Get the most recent log file
        latest_log = max(log_files, key=os.path.getmtime)

        try:
            with open(latest_log) as f:
                all_lines = f.readlines()
                return "".join(all_lines[-lines:])
        except Exception as e:
            return f"Error reading log file {latest_log}: {e!s}"

    def list_log_files(self) -> list[dict[str, Any]]:
        """List all buttermilk log files with their metadata.

        Returns:
            List of log files with path, size, and modification time
        """
        log_files = glob.glob("/tmp/buttermilk_*.jsonl")

        files_info: list[dict[str, Any]] = []
        for log_file in log_files:
            stat = os.stat(log_file)
            files_info.append(
                {
                    "path": log_file,
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": stat.st_mtime,
                    "modified_str": Path(log_file).stat().st_mtime,
                }
            )

        # Sort by modification time, newest first
        files_info.sort(key=lambda x: x["modified"], reverse=True)
        return files_info

    # WebSocket client tools

    async def start_websocket_client(
        self,
        flow_id: str,
        host: str = "localhost",
        port: int = 8000,
        use_direct_ws: bool = False,
    ) -> dict[str, str]:
        """Start a WebSocket client for testing a flow.

        Args:
            flow_id: Unique identifier for this client instance
            host: Host to connect to
            port: Port to connect to
            use_direct_ws: If True, connect directly to ws://host:port/ws without session

        Returns:
            Status message and connection details
        """
        if flow_id in self._active_clients:
            return {
                "status": "error",
                "message": f"Client already running for flow_id: {flow_id}",
            }

        try:
            if use_direct_ws:
                client = FlowTestClient(direct_ws_url=f"ws://{host}:{port}/ws")
            else:
                client = FlowTestClient(base_url=f"http://{host}:{port}", ws_url=f"ws://{host}:{port}/ws")

            await client.connect()
            self._active_clients[flow_id] = client

            return {
                "status": "success",
                "message": f"Started WebSocket client for {flow_id}",
                "session_id": client.session_id or "direct_connection",
            }

        except Exception as e:
            return {"status": "error", "message": f"Failed to start client: {e!s}"}

    async def send_websocket_message(
        self,
        flow_id: str,
        message_type: str,
        content: str | None = None,
        flow_name: str | None = None,
    ) -> dict[str, str]:
        """Send a message to an active WebSocket client.

        Args:
            flow_id: ID of the client to send to
            message_type: Type of message - "run_flow" or "manager_response"
            content: Message content (for manager_response)
            flow_name: Flow name to run (for run_flow)

        Returns:
            Status of the send operation
        """
        if flow_id not in self._active_clients:
            return {
                "status": "error",
                "message": f"No active client for flow_id: {flow_id}",
            }

        client = self._active_clients[flow_id]

        try:
            if message_type == "run_flow":
                if not flow_name:
                    return {
                        "status": "error",
                        "message": "flow_name required for run_flow message",
                    }
                await client.start_flow(flow_name, content or "")
                return {"status": "success", "message": f"Started flow {flow_name}"}

            if message_type == "manager_response":
                if content is None:
                    return {
                        "status": "error",
                        "message": "content required for manager_response",
                    }
                await client.send_manager_response(content)
                return {"status": "success", "message": f"Sent response: {content}"}

            return {
                "status": "error",
                "message": f"Unknown message type: {message_type}",
            }

        except Exception as e:
            return {"status": "error", "message": f"Failed to send message: {e!s}"}

    def get_websocket_messages(
        self,
        flow_id: str,
        last_n: int | None = None,
        message_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get messages from a WebSocket client.

        Args:
            flow_id: ID of the client
            last_n: Number of most recent messages to return (None for all)
            message_type: Filter by message type (None for all types)

        Returns:
            List of messages with type, timestamp, and data
        """
        if flow_id not in self._active_clients:
            return [{"error": f"No active client for flow_id: {flow_id}"}]

        client = self._active_clients[flow_id]
        collector = client.collector

        # Get messages based on type filter
        if message_type:
            if message_type == "ui_message":
                messages = collector.ui_messages
            elif message_type == "agent_announcement":
                messages = collector.agent_announcements
            elif message_type == "agent_trace":
                messages = collector.agent_traces
            elif message_type == "error":
                messages = collector.errors
            elif message_type == "flow_event":
                messages = collector.flow_events
            else:
                messages = collector.all_messages
        else:
            messages = collector.all_messages

        # Apply last_n filter
        if last_n is not None:
            messages = messages[-last_n:]

        # Convert to dict format
        return [
            {
                "type": msg.type,
                "timestamp": msg.timestamp.isoformat(),
                "content": msg.content,
                "agent_role": msg.agent_role,
                "data": msg.data,
            }
            for msg in messages
        ]

    def get_websocket_summary(self, flow_id: str) -> dict[str, Any]:
        """Get a summary of WebSocket client state and messages.

        Args:
            flow_id: ID of the client

        Returns:
            Summary of message counts and active agents
        """
        if flow_id not in self._active_clients:
            return {"error": f"No active client for flow_id: {flow_id}"}

        client = self._active_clients[flow_id]
        return client.get_message_summary()

    async def cleanup(self) -> None:
        """Cleanup all active WebSocket clients."""
        # Clean up any active clients
        for flow_id in list(self._active_clients.keys()):
            try:
                client = self._active_clients[flow_id]
                await client.disconnect()
            except Exception as e:
                logger.warning(f"Error cleaning up client {flow_id}: {e}")
        self._active_clients.clear()

        # Clean up puppet client
        if self._puppet_client:
            try:
                await self._puppet_client.disconnect()
                self._puppet_client = None
                self._puppet_listening = False
            except Exception as e:
                logger.warning(f"Error cleaning up puppet client: {e}")

        # Call parent cleanup
        await super().cleanup()

    async def stop_websocket_client(self, flow_id: str) -> dict[str, str]:
        """Stop and cleanup a WebSocket client.

        Args:
            flow_id: ID of the client to stop

        Returns:
            Status of the stop operation
        """
        if flow_id not in self._active_clients:
            return {
                "status": "error",
                "message": f"No active client for flow_id: {flow_id}",
            }

        try:
            client = self._active_clients[flow_id]
            await client.disconnect()
            del self._active_clients[flow_id]

            return {"status": "success", "message": f"Stopped client for {flow_id}"}

        except Exception as e:
            return {"status": "error", "message": f"Error stopping client: {e!s}"}

    def list_active_clients(self) -> list[str]:
        """List all active WebSocket client IDs.

        Returns:
            List of active flow_ids
        """
        return list(self._active_clients.keys())

    # Puppet mode - continuous listening for LLM control

    async def start_puppet_mode(self, host: str = "localhost", port: int = 8000) -> dict[str, Any]:
        """Start puppet mode - continuous WebSocket client that acts as UI replacement.

        Args:
            host: Host to connect to
            port: Port to connect to

        Returns:
            Status message and session details
        """
        if self._puppet_client and self._puppet_listening:
            return {"status": "error", "message": "Puppet mode already active"}

        try:
            # Stop any existing puppet client
            if self._puppet_client:
                await self._puppet_client.disconnect()

            # Create new puppet client
            self._puppet_client = FlowTestClient(base_url=f"http://{host}:{port}", ws_url=f"ws://{host}:{port}/ws")

            await self._puppet_client.connect()
            self._puppet_listening = True

            return {
                "status": "success",
                "message": "Puppet mode started - ready for LLM control",
                "session_id": self._puppet_client.session_id or "unknown",
                "host": host,
                "port": port,
            }

        except Exception as e:
            self._puppet_listening = False
            return {
                "status": "error",
                "message": f"Failed to start puppet mode: {e!s}",
            }

    async def puppet_start_flow(self, flow_name: str, prompt: str = "", record: str = "", criteria: str = "") -> dict[str, Any]:
        """Start a flow in puppet mode.

        Args:
            flow_name: Name of the flow to start
            prompt: Initial prompt/query
            record: Record ID to process
            criteria: Criteria to use

        Returns:
            Status and initial response summary
        """
        if not self._puppet_client or not self._puppet_listening:
            return {
                "status": "error",
                "message": "Puppet mode not active. Start with start_puppet_mode() first.",
            }

        try:
            await self._puppet_client.start_flow(flow_name, prompt, record, criteria)

            # Give it a moment to collect initial messages
            import asyncio

            await asyncio.sleep(2)

            # Get message summary
            summary = self._puppet_client.get_message_summary()

            return {
                "status": "success",
                "message": f"Started flow '{flow_name}' in puppet mode",
                "flow_name": flow_name,
                "prompt": prompt,
                "record": record,
                "criteria": criteria,
                "session_id": self._puppet_client.session_id,
                "initial_messages": summary["total"],
                "agents_active": str(summary["agents_active"]),
            }

        except Exception as e:
            return {"status": "error", "message": f"Failed to start flow: {e!s}"}

    async def puppet_send_response(self, content: str) -> dict[str, Any]:
        """Send a manager response in puppet mode.

        Args:
            content: Response content to send

        Returns:
            Status and response details
        """
        if not self._puppet_client or not self._puppet_listening:
            return {
                "status": "error",
                "message": "Puppet mode not active. Start with start_puppet_mode() first.",
            }

        try:
            await self._puppet_client.send_manager_response(content)

            return {
                "status": "success",
                "message": f"Sent response in puppet mode: {content[:50]}...",
                "response_content": content,
                "session_id": self._puppet_client.session_id,
            }

        except Exception as e:
            return {"status": "error", "message": f"Failed to send response: {e!s}"}

    def puppet_get_messages(self, last_n: int | None = 10, message_type: str | None = None) -> list[dict[str, Any]]:
        """Get recent messages from puppet mode client.

        Args:
            last_n: Number of most recent messages (default: 10, None for all)
            message_type: Filter by message type (None for all types)

        Returns:
            List of recent messages
        """
        if not self._puppet_client or not self._puppet_listening:
            return [{"error": "Puppet mode not active. Start with start_puppet_mode() first."}]

        collector = self._puppet_client.collector

        # Get messages based on type filter
        if message_type:
            if message_type == "ui_message":
                messages = collector.ui_messages
            elif message_type == "agent_announcement":
                messages = collector.agent_announcements
            elif message_type == "agent_trace":
                messages = collector.agent_traces
            elif message_type == "error":
                messages = collector.errors
            elif message_type == "flow_event":
                messages = collector.flow_events
            else:
                messages = collector.all_messages
        else:
            messages = collector.all_messages

        # Apply last_n filter
        if last_n is not None:
            messages = messages[-last_n:]

        # Convert to dict format
        return [
            {
                "type": msg.type,
                "timestamp": msg.timestamp.isoformat(),
                "content": msg.content,
                "agent_role": msg.agent_role,
                "data": msg.data,
            }
            for msg in messages
        ]

    def puppet_get_summary(self) -> dict[str, Any]:
        """Get summary of puppet mode client state.

        Returns:
            Summary of puppet client state and messages
        """
        if not self._puppet_client or not self._puppet_listening:
            return {"error": "Puppet mode not active. Start with start_puppet_mode() first."}

        summary = self._puppet_client.get_message_summary()
        summary["puppet_mode"] = {
            "active": self._puppet_listening,
            "session_id": self._puppet_client.session_id,
            "connection_status": "connected" if self._puppet_client.ws else "disconnected",
        }

        return summary

    async def stop_puppet_mode(self) -> dict[str, str]:
        """Stop puppet mode and cleanup.

        Returns:
            Status of the stop operation
        """
        if not self._puppet_client:
            return {"status": "info", "message": "Puppet mode was not active"}

        try:
            await self._puppet_client.disconnect()
            self._puppet_client = None
            self._puppet_listening = False

            return {
                "status": "success",
                "message": "Puppet mode stopped and cleaned up",
            }

        except Exception as e:
            self._puppet_listening = False
            return {
                "status": "error",
                "message": f"Error stopping puppet mode: {e!s}",
            }
