#!/usr/bin/env python3
"""Non-interactive WebSocket Debug CLI for Buttermilk flows.

This provides a command-line interface for debugging flows that executes
specific actions and returns results to stdout, suitable for LLM usage.
"""

import asyncio
import glob
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

# Use the flow test client from the agents test utilities
from buttermilk.agents.test_utils import FlowTestClient


class NonInteractiveDebugClient:
    """Non-interactive WebSocket debug client for single operations."""

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.client: Optional[FlowTestClient] = None
        self.console = Console()
        self.session_file = Path(tempfile.gettempdir()) / "buttermilk_debug_session.json"

    def save_session(self, session_id: str):
        """Save session ID to file for reuse."""
        session_data = {
            "session_id": session_id,
            "host": self.host,
            "port": self.port,
            "timestamp": datetime.now().isoformat()
        }
        self.session_file.write_text(json.dumps(session_data, indent=2))

    def load_session(self) -> Optional[str]:
        """Load session ID from file if it exists."""
        if self.session_file.exists():
            try:
                data = json.loads(self.session_file.read_text())
                # Check if session is for the same host/port
                if data.get("host") == self.host and data.get("port") == self.port:
                    return data.get("session_id")
            except Exception:
                pass
        return None

    async def connect(self, session_id: Optional[str] = None):
        """Connect to the WebSocket server, optionally reusing a session."""
        try:
            if session_id:
                # Direct WebSocket connection with existing session ID
                direct_ws_url = f"{self.ws_url}/{session_id}"
                self.client = FlowTestClient(
                    base_url=self.base_url,
                    ws_url=self.ws_url,
                    direct_ws_url=direct_ws_url
                )
                await self.client.connect()
                self.client.session_id = session_id  # Set the session ID manually
            else:
                # Normal connection that creates a new session
                self.client = FlowTestClient(
                    base_url=self.base_url,
                    ws_url=self.ws_url
                )
                await self.client.connect()
                if self.client.session_id:
                    self.save_session(self.client.session_id)

            return True
        except Exception as e:
            self.console.print(f"[red]Failed to connect: {e}[/red]")
            return False

    async def disconnect(self):
        """Disconnect from the WebSocket server."""
        if self.client:
            await self.client.disconnect()

    async def start_flow(self, flow_name: str, query: str, wait_time: int = 5) -> dict:
        """Start a flow and wait for initial responses."""
        if not await self.connect():
            return {"error": "Failed to connect to server"}

        try:
            await self.client.start_flow(flow_name, query)

            # Wait for initial responses
            await asyncio.sleep(wait_time)

            # Collect results
            result = {
                "session_id": self.client.session_id,
                "flow": flow_name,
                "query": query,
                "messages": []
            }

            for msg in self.client.collector.all_messages:
                result["messages"].append({
                    "timestamp": msg.timestamp.isoformat(),
                    "type": msg.type,
                    "content": msg.content,
                    "agent_role": msg.agent_role,
                    "data": msg.data
                })

            return result

        except Exception as e:
            return {"error": str(e)}
        finally:
            await self.disconnect()

    async def send_message(self, message_type: str, content: str, wait_time: int = 5, session_id: Optional[str] = None) -> dict:
        """Send a message to an existing session."""
        # Use provided session_id or load from file
        session_id = session_id or self.load_session()
        if not session_id:
            return {"error": "No session ID provided and no saved session found. Start a flow first."}

        if not await self.connect(session_id):
            return {"error": f"Failed to reconnect to session {session_id}"}

        try:
            # Send the message based on type
            if message_type == "response":
                await self.client.send_manager_response(content)
            else:
                # Generic message sending
                message = {
                    "type": message_type,
                    "content": content
                }
                await self.client.ws.send_json(message)

            # Wait for responses
            await asyncio.sleep(wait_time)

            # Collect results
            result = {
                "session_id": session_id,
                "message_sent": {
                    "type": message_type,
                    "content": content
                },
                "messages": []
            }

            for msg in self.client.collector.all_messages:
                result["messages"].append({
                    "timestamp": msg.timestamp.isoformat(),
                    "type": msg.type,
                    "content": msg.content,
                    "agent_role": msg.agent_role,
                    "data": msg.data
                })

            return result

        except Exception as e:
            return {"error": str(e)}
        finally:
            await self.disconnect()

    async def wait_for_messages(self, session_id: Optional[str] = None, wait_time: int = 5,
                               pattern: Optional[str] = None, message_type: Optional[str] = None) -> dict:
        """Connect to existing session and wait for messages."""
        # Use provided session_id or load from file
        session_id = session_id or self.load_session()
        if not session_id:
            return {"error": "No session ID provided and no saved session found. Start a flow first."}

        if not await self.connect(session_id):
            return {"error": f"Failed to reconnect to session {session_id}"}

        try:
            # Wait for messages
            await asyncio.sleep(wait_time)

            # Filter messages
            messages = self.client.collector.all_messages

            if message_type:
                messages = [msg for msg in messages if msg.type == message_type]

            if pattern:
                import re
                pattern_re = re.compile(pattern, re.IGNORECASE)
                messages = [msg for msg in messages if pattern_re.search(msg.content or "")]

            # Collect results
            result = {
                "session_id": session_id,
                "filter": {
                    "pattern": pattern,
                    "message_type": message_type
                },
                "messages": []
            }

            for msg in messages:
                result["messages"].append({
                    "timestamp": msg.timestamp.isoformat(),
                    "type": msg.type,
                    "content": msg.content,
                    "agent_role": msg.agent_role,
                    "data": msg.data
                })

            return result

        except Exception as e:
            return {"error": str(e)}
        finally:
            await self.disconnect()

    async def get_logs(self, lines: int = 50) -> dict:
        """Get recent log lines."""
        log_files = glob.glob("/tmp/buttermilk_*.log")
        if not log_files:
            return {"error": "No log files found in /tmp/"}

        # Get the most recent log file
        latest_log = max(log_files, key=os.path.getmtime)

        try:
            with open(latest_log, "r") as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:]

                return {
                    "log_file": latest_log,
                    "lines": [line.rstrip() for line in recent_lines]
                }
        except Exception as e:
            return {"error": f"Error reading log file: {e}"}

    async def list_flows(self) -> dict:
        """List available flows (requires connection to get from server)."""
        if not await self.connect():
            return {"error": "Failed to connect to server"}

        try:
            # In a real implementation, this would query the server for available flows
            # For now, return a placeholder
            return {
                "note": "Flow listing not yet implemented in server API",
                "common_flows": ["simple_flow", "test_flow", "debug_flow"]
            }
        finally:
            await self.disconnect()

    def clear_session(self):
        """Clear saved session."""
        if self.session_file.exists():
            self.session_file.unlink()
            return {"status": "Session cleared"}
        return {"status": "No session to clear"}


@click.group()
@click.option("--host", default="localhost", help="WebSocket server host")
@click.option("--port", default=8000, type=int, help="WebSocket server port")
@click.option("--json-output", is_flag=True, help="Output results as JSON")
@click.pass_context
def cli(ctx, host: str, port: int, json_output: bool):
    """Non-interactive WebSocket debug client for Buttermilk flows."""
    ctx.ensure_object(dict)
    ctx.obj["HOST"] = host
    ctx.obj["PORT"] = port
    ctx.obj["JSON_OUTPUT"] = json_output


@cli.command()
@click.argument("flow_name")
@click.argument("query", default="")
@click.option("--wait", default=5, help="Seconds to wait for responses")
@click.pass_context
def start(ctx, flow_name: str, query: str, wait: int):
    """Start a flow with an optional initial query."""
    client = NonInteractiveDebugClient(ctx.obj["HOST"], ctx.obj["PORT"])
    result = asyncio.run(client.start_flow(flow_name, query, wait))

    if ctx.obj["JSON_OUTPUT"]:
        print(json.dumps(result, indent=2))
    else:
        console = Console()
        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
        else:
            console.print(f"[green]Started flow '{flow_name}' with session: {result['session_id']}[/green]")
            console.print(f"Query: {query}")
            console.print(f"\nMessages ({len(result['messages'])}):")
            for msg in result["messages"]:
                timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M:%S")
                msg_type = msg["type"]
                content = msg["content"] or "(no content)"
                agent = msg["agent_role"] or "system"
                console.print(f"[dim]{timestamp}[/dim] [{msg_type}] {agent}: {content}")


@cli.command()
@click.argument("content")
@click.option("--type", "msg_type", default="response", help="Message type (default: response)")
@click.option("--wait", default=5, help="Seconds to wait for responses")
@click.option("--session", help="Session ID (uses saved session if not provided)")
@click.pass_context
def send(ctx, content: str, msg_type: str, wait: int, session: Optional[str]):
    """Send a message to the current session."""
    client = NonInteractiveDebugClient(ctx.obj["HOST"], ctx.obj["PORT"])
    result = asyncio.run(client.send_message(msg_type, content, wait, session))

    if ctx.obj["JSON_OUTPUT"]:
        print(json.dumps(result, indent=2))
    else:
        console = Console()
        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
        else:
            console.print(f"[green]Sent {msg_type} message to session {result['session_id']}[/green]")
            console.print(f"Content: {content}")
            console.print(f"\nNew messages ({len(result['messages'])}):")
            for msg in result["messages"]:
                timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M:%S")
                msg_type = msg["type"]
                content = msg["content"] or "(no content)"
                agent = msg["agent_role"] or "system"
                console.print(f"[dim]{timestamp}[/dim] [{msg_type}] {agent}: {content}")


@cli.command()
@click.option("--wait", default=5, help="Seconds to wait for messages")
@click.option("--pattern", help="Regex pattern to filter messages")
@click.option("--type", "msg_type", help="Filter by message type")
@click.option("--session", help="Session ID (uses saved session if not provided)")
@click.pass_context
def wait(ctx, wait: int, pattern: Optional[str], msg_type: Optional[str], session: Optional[str]):
    """Wait for messages from the current session."""
    client = NonInteractiveDebugClient(ctx.obj["HOST"], ctx.obj["PORT"])
    result = asyncio.run(client.wait_for_messages(session, wait, pattern, msg_type))

    if ctx.obj["JSON_OUTPUT"]:
        print(json.dumps(result, indent=2))
    else:
        console = Console()
        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
        else:
            console.print(f"[green]Messages from session {result['session_id']}[/green]")
            if pattern or msg_type:
                console.print(f"Filters: pattern={pattern}, type={msg_type}")
            console.print(f"\nMessages ({len(result['messages'])}):")
            for msg in result["messages"]:
                timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M:%S")
                msg_type = msg["type"]
                content = msg["content"] or "(no content)"
                agent = msg["agent_role"] or "system"
                console.print(f"[dim]{timestamp}[/dim] [{msg_type}] {agent}: {content}")


@cli.command()
@click.pass_context
def session(ctx):
    """Show current saved session information."""
    client = NonInteractiveDebugClient(ctx.obj["HOST"], ctx.obj["PORT"])
    session_id = client.load_session()

    if ctx.obj["JSON_OUTPUT"]:
        if session_id and client.session_file.exists():
            data = json.loads(client.session_file.read_text())
            print(json.dumps(data, indent=2))
        else:
            print(json.dumps({"error": "No saved session"}, indent=2))
    else:
        console = Console()
        if session_id:
            data = json.loads(client.session_file.read_text())
            console.print("[green]Saved session:[/green]")
            console.print(f"  Session ID: {data['session_id']}")
            console.print(f"  Host: {data['host']}:{data['port']}")
            console.print(f"  Created: {data['timestamp']}")
        else:
            console.print("[yellow]No saved session[/yellow]")


@cli.command()
@click.pass_context
def clear_session(ctx):
    """Clear the saved session."""
    client = NonInteractiveDebugClient(ctx.obj["HOST"], ctx.obj["PORT"])
    result = client.clear_session()

    if ctx.obj["JSON_OUTPUT"]:
        print(json.dumps(result, indent=2))
    else:
        console = Console()
        console.print(f"[green]{result['status']}[/green]")


@cli.command()
@click.option("--lines", "-n", default=50, help="Number of log lines to show")
@click.pass_context
def logs(ctx, lines: int):
    """Show recent log lines from Buttermilk log files."""
    client = NonInteractiveDebugClient(ctx.obj["HOST"], ctx.obj["PORT"])
    result = asyncio.run(client.get_logs(lines))

    if ctx.obj["JSON_OUTPUT"]:
        print(json.dumps(result, indent=2))
    else:
        console = Console()
        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
        else:
            console.print(f"[dim]Log file: {result['log_file']}[/dim]\n")
            for line in result["lines"]:
                # Color code by log level
                if "ERROR" in line:
                    console.print(f"[red]{line}[/red]")
                elif "WARNING" in line:
                    console.print(f"[yellow]{line}[/yellow]")
                elif "INFO" in line:
                    console.print(f"[green]{line}[/green]")
                else:
                    console.print(f"[dim]{line}[/dim]")


@cli.command()
@click.pass_context
def list_flows(ctx):
    """List available flows."""
    client = NonInteractiveDebugClient(ctx.obj["HOST"], ctx.obj["PORT"])
    result = asyncio.run(client.list_flows())

    if ctx.obj["JSON_OUTPUT"]:
        print(json.dumps(result, indent=2))
    else:
        console = Console()
        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
        else:
            if "note" in result:
                console.print(f"[yellow]Note: {result['note']}[/yellow]")
            if "common_flows" in result:
                console.print("\nCommon flows:")
                for flow in result["common_flows"]:
                    console.print(f"  - {flow}")


@cli.command()
@click.pass_context
def test_connection(ctx):
    """Test connection to the WebSocket server."""
    client = NonInteractiveDebugClient(ctx.obj["HOST"], ctx.obj["PORT"])

    async def test():
        if await client.connect():
            await client.disconnect()
            return {"status": "success", "url": client.ws_url}
        else:
            return {"status": "failed", "url": client.ws_url}

    result = asyncio.run(test())

    if ctx.obj["JSON_OUTPUT"]:
        print(json.dumps(result, indent=2))
    else:
        console = Console()
        if result["status"] == "success":
            console.print(f"[green]✓ Successfully connected to {result['url']}[/green]")
        else:
            console.print(f"[red]✗ Failed to connect to {result['url']}[/red]")


if __name__ == "__main__":
    cli()
