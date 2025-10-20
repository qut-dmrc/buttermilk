#!/usr/bin/env python3
"""WebSocket Debug CLI for Buttermilk flows.

Provides both CLI commands and programmatic API for debugging Buttermilk flows:

CLI Commands (optimized for LLM usage):
- start: Start a flow and capture all messages (JSON by default)
- send: Send messages to an active session
- wait: Wait for and collect messages with optional filtering
- session: Show current session info
- clear-session: Clear saved session
- logs: View structured log files
- list-logs: List recent log files
- test-connection: Test WebSocket connectivity

Programmatic API:
- NonInteractiveDebugClient: Python class for automation and testing

By default, all commands output JSON for easy parsing by LLMs and automation tools.
Use --pretty flag for human-readable console output.
"""

import asyncio
import glob
import json
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path

import click
from rich import print
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
        self.client: FlowTestClient | None = None
        self.console = Console()
        self.session_file = Path(tempfile.gettempdir()) / "buttermilk_debug_session.json"

    def save_session(self, session_id: str):
        """Save session ID to file for reuse."""
        session_data = {
            "session_id": session_id,
            "host": self.host,
            "port": self.port,
            "timestamp": datetime.now().isoformat(),
        }
        self.session_file.write_text(json.dumps(session_data, indent=2))

    def load_session(self) -> str | None:
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

    async def connect(self, session_id: str | None = None):
        """Connect to the WebSocket server, optionally reusing a session."""
        try:
            if session_id:
                # Direct WebSocket connection with existing session ID
                direct_ws_url = f"{self.ws_url}/{session_id}"
                self.client = FlowTestClient(
                    base_url=self.base_url,
                    ws_url=self.ws_url,
                    direct_ws_url=direct_ws_url,
                )
                await self.client.connect()
                self.client.session_id = session_id  # Set the session ID manually
            else:
                # Normal connection that creates a new session
                self.client = FlowTestClient(
                    base_url=self.base_url,
                    ws_url=self.ws_url,
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

    async def start_flow(self, flow_name: str, query: str, wait_time: int = 60, record: str = "", criteria: str = "") -> dict:
        """Start a flow and wait for initial responses."""
        if not await self.connect():
            return {"error": "Failed to connect to server"}

        try:
            await self.client.start_flow(flow_name, query, record, criteria)

            # Stream messages in real-time
            start_time = time.time()
            last_message_count = 0
            flow_completed = False

            print(f"Started flow '{flow_name}' with session: {self.client.session_id}")
            if record:
                print(f"Record: {record}")
            if criteria:
                print(f"Criteria: {criteria}")
            print()

            while time.time() - start_time < wait_time and not flow_completed:
                current_messages = self.client.collector.all_messages

                # Print only new messages since last check
                new_messages = current_messages[last_message_count:]
                for msg in new_messages:
                    timestamp_str = msg.timestamp.strftime("%H:%M:%S")
                    content = msg.content

                    # Truncate very long content for readability
                    if len(content) > 200:
                        content = content[:200] + "..."

                    print(f"{timestamp_str}  {msg.type}: {content}")

                    # Check for flow completion
                    if msg.type in ["flow_complete", "system_update"] and "complet" in content.lower():
                        flow_completed = True

                last_message_count = len(current_messages)
                await asyncio.sleep(0.1)  # Check every 100ms

            if flow_completed:
                print(f"\n✅ Flow completed after {time.time() - start_time:.1f}s")
            else:
                print(f"\n⏱️  Timeout reached after {wait_time}s")

            # Collect final results for return
            result = {
                "session_id": self.client.session_id,
                "flow": flow_name,
                "query": query,
                "record": record,
                "criteria": criteria,
                "messages": [],
                "completed": flow_completed,
                "duration": time.time() - start_time,
            }

            for msg in self.client.collector.all_messages:
                result["messages"].append(
                    {
                        "timestamp": msg.timestamp.isoformat(),
                        "type": msg.type,
                        "content": msg.content,
                        "agent_role": msg.agent_role,
                        "data": msg.data,
                    },
                )

            return result

        except Exception as e:
            return {"error": str(e)}
        finally:
            await self.disconnect()

    async def send_message(self, message_type: str, content: str, wait_time: int = 60, session_id: str | None = None) -> dict:
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
                    "content": content,
                }
                await self.client.ws.send_json(message)

            # Wait for responses
            await asyncio.sleep(wait_time)

            # Collect results
            result = {
                "session_id": session_id,
                "message_sent": {
                    "type": message_type,
                    "content": content,
                },
                "messages": [],
            }

            for msg in self.client.collector.all_messages:
                result["messages"].append(
                    {
                        "timestamp": msg.timestamp.isoformat(),
                        "type": msg.type,
                        "content": msg.content,
                        "agent_role": msg.agent_role,
                        "data": msg.data,
                    },
                )

            return result

        except Exception as e:
            return {"error": str(e)}
        finally:
            await self.disconnect()

    async def wait_for_messages(
        self,
        session_id: str | None = None,
        wait_time: int = 60,
        pattern: str | None = None,
        message_type: str | None = None,
    ) -> dict:
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
                    "message_type": message_type,
                },
                "messages": [],
            }

            for msg in messages:
                result["messages"].append(
                    {
                        "timestamp": msg.timestamp.isoformat(),
                        "type": msg.type,
                        "content": msg.content,
                        "agent_role": msg.agent_role,
                        "data": msg.data,
                    },
                )

            return result

        except Exception as e:
            return {"error": str(e)}
        finally:
            await self.disconnect()

    async def get_logs(self, lines: int = 50, min_level: str = "INFO", log_file: str | None = None) -> dict:
        """Get recent log lines, with an optional minimum level filter.

        Args:
            lines: Number of lines to retrieve
            min_level: Minimum log level to filter
            log_file: Specific log file path to read (optional). If not provided,
                     uses the most recent bm_*.jsonl file in /tmp/
        """
        if log_file:
            # Use specified log file
            if not os.path.exists(log_file):
                return {"error": f"Log file not found: {log_file}"}
            latest_log = log_file
        else:
            # Find Buttermilk log files with bm_ prefix
            log_files = glob.glob("/tmp/bm_*.jsonl")
            if not log_files:
                return {
                    "error": "No Buttermilk log files found in /tmp/",
                    "hint": "Log files must start with 'bm_' and end with '.jsonl' (searched with pattern: /tmp/bm_*.jsonl)"
                }

            # Get the most recent log file
            latest_log = max(log_files, key=os.path.getmtime)

        levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
        min_level_num = levels.get(min_level.upper(), 1)

        def get_line_level(line: str) -> int:
            try:
                # Try to parse as JSON first (new JSONL format)
                import json
                log_entry = json.loads(line.strip())
                level = log_entry.get("level", "").upper()
                return levels.get(level, -1)
            except (json.JSONDecodeError, AttributeError):
                # Fallback to old text format parsing
                match = re.search(r" - (DEBUG|INFO|WARNING|ERROR|CRITICAL) - ", line)
                if match:
                    return levels.get(match.group(1), -1)
                return 99

        try:
            with open(latest_log) as f:
                all_lines = f.readlines()

                # Filter by level
                if min_level_num > 0:
                    filtered_lines = [line for line in all_lines if get_line_level(line) >= min_level_num]
                else:
                    filtered_lines = all_lines

                recent_lines = filtered_lines[-lines:]

                return {
                    "log_file": latest_log,
                    "lines": [line.rstrip() for line in recent_lines],
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
                "common_flows": ["simple_flow", "test_flow", "debug_flow"],
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
@click.option("--json-output", is_flag=True, help="Output results as JSON (default, optimized for LLMs)")
@click.option("--pretty", is_flag=True, help="Pretty console output for humans (overrides --json-output)")
@click.pass_context
def cli(ctx, host: str, port: int, json_output: bool, pretty: bool):
    """WebSocket debug client for Buttermilk flows.

    By default, outputs JSON for easy parsing by LLMs and automation.
    Use --pretty for human-readable console output.
    """
    ctx.ensure_object(dict)
    ctx.obj["HOST"] = host
    ctx.obj["PORT"] = port
    # Pretty overrides json-output (pretty takes precedence)
    ctx.obj["JSON_OUTPUT"] = not pretty if pretty else (json_output or True)  # Default to JSON


@cli.command()
@click.argument("flow_name")
@click.argument("query", default="")
@click.option("--record", default="", help="Record ID to analyze")
@click.option("--criteria", default="", help="Criteria template to use")
@click.option("--wait", default=60, help="Seconds to wait for flow completion")
@click.pass_context
def start(ctx, flow_name: str, query: str, record: str, criteria: str, wait: int):
    """Start a flow and capture all messages.

    Examples:
        # Start trans flow with record and criteria (JSON output by default)
        ws_debug_cli start trans --record betoota_snape_trans --criteria tja

        # Human-readable output
        ws_debug_cli --pretty start trans --record betoota_snape_trans --criteria tja

        # With query text
        ws_debug_cli start trans "analyze this" --record betoota_snape_trans
    """
    client = NonInteractiveDebugClient(ctx.obj["HOST"], ctx.obj["PORT"])
    result = asyncio.run(client.start_flow(flow_name, query, wait, record, criteria))

    if ctx.obj["JSON_OUTPUT"]:
        print(json.dumps(result, indent=2))
    else:
        console = Console()
        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
        else:
            console.print(f"[green]Started flow '{flow_name}' - Session: {result['session_id']}[/green]")
            if query:
                console.print(f"Query: {query}")
            if record:
                console.print(f"Record: {record}")
            if criteria:
                console.print(f"Criteria: {criteria}")
            console.print(f"\n[dim]Messages ({len(result['messages'])})[/dim]:")
            for msg in result["messages"]:
                timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M:%S")
                msg_type = msg["type"]
                content = msg["content"] or "(no content)"
                # Truncate for console readability
                if len(content) > 100:
                    content = content[:100] + "..."
                agent = msg["agent_role"] or "system"
                console.print(f"[dim]{timestamp}[/dim] [{msg_type}] {agent}: {content}")


@cli.command()
@click.argument("content")
@click.option("--type", "msg_type", default="response", help="Message type (default: response)")
@click.option("--wait", default=5, help="Seconds to wait for responses")
@click.option("--session", help="Session ID (uses saved session if not provided)")
@click.pass_context
def send(ctx, content: str, msg_type: str, wait: int, session: str | None):
    """Send a message to the current session.

    Examples:
        # Send manager response to current session
        ws_debug_cli send "approved"

        # Send to specific session
        ws_debug_cli send "approved" --session abc123
    """
    client = NonInteractiveDebugClient(ctx.obj["HOST"], ctx.obj["PORT"])
    result = asyncio.run(client.send_message(msg_type, content, wait, session))

    if ctx.obj["JSON_OUTPUT"]:
        print(json.dumps(result, indent=2))
    else:
        console = Console()
        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
        else:
            console.print(f"[green]Sent {msg_type} to session {result['session_id']}[/green]")
            console.print(f"Content: {content}")
            console.print(f"\n[dim]New messages ({len(result['messages'])})[/dim]:")
            for msg in result["messages"]:
                timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M:%S")
                msg_type = msg["type"]
                content = msg["content"] or "(no content)"
                if len(content) > 100:
                    content = content[:100] + "..."
                agent = msg["agent_role"] or "system"
                console.print(f"[dim]{timestamp}[/dim] [{msg_type}] {agent}: {content}")


@cli.command()
@click.option("--wait", default=5, help="Seconds to wait for messages")
@click.option("--pattern", help="Regex pattern to filter messages")
@click.option("--type", "msg_type", help="Filter by message type")
@click.option("--session", help="Session ID (uses saved session if not provided)")
@click.pass_context
def wait(ctx, wait: int, pattern: str | None, msg_type: str | None, session: str | None):
    """Wait for and collect messages from the current session.

    Examples:
        # Wait 10 seconds and get all messages
        ws_debug_cli wait --wait 10

        # Wait for messages containing "conclusion"
        ws_debug_cli wait --pattern "conclusion" --wait 30

        # Get only ui_message type messages
        ws_debug_cli wait --type ui_message
    """
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
                console.print(f"[dim]Filters: pattern={pattern}, type={msg_type}[/dim]")
            console.print(f"\n[dim]Messages ({len(result['messages'])})[/dim]:")
            for msg in result["messages"]:
                timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M:%S")
                content = msg["content"] or "(no content)"
                if len(content) > 100:
                    content = content[:100] + "..."
                agent = msg["agent_role"] or "system"
                console.print(f"[dim]{timestamp}[/dim] [{msg['type']}] {agent}: {content}")


@cli.command()
@click.pass_context
def session(ctx):
    """Show current saved session information.

    Examples:
        # Show session info as JSON
        ws_debug_cli session

        # Pretty format
        ws_debug_cli --pretty session
    """
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
    """Clear the saved session.

    Examples:
        ws_debug_cli clear-session
    """
    client = NonInteractiveDebugClient(ctx.obj["HOST"], ctx.obj["PORT"])
    result = client.clear_session()

    if ctx.obj["JSON_OUTPUT"]:
        print(json.dumps(result, indent=2))
    else:
        console = Console()
        console.print(f"[green]{result['status']}[/green]")


@cli.command()
@click.option("--lines", "-n", default=50, help="Number of log lines to show")
@click.option(
    "--level",
    "-l",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Minimum log level to display.",
)
@click.option("--file", "-f", default=None, help="Specific log file to read (optional)")
@click.pass_context
def logs(ctx, lines: int, level: str, file: str | None):
    """Show recent log lines from Buttermilk log files."""
    client = NonInteractiveDebugClient(ctx.obj["HOST"], ctx.obj["PORT"])
    result = asyncio.run(client.get_logs(lines, min_level=level, log_file=file))

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
@click.option("--count", "-n", default=5, help="Number of recent log files to show")
@click.pass_context
def list_logs(ctx, count: int):
    """List the most recent Buttermilk log files."""
    log_files = glob.glob("/tmp/bm_*.jsonl")

    if not log_files:
        console = Console()
        console.print("[yellow]No Buttermilk log files found in /tmp/[/yellow]")
        console.print("[dim]Searched for files matching pattern: /tmp/bm_*.jsonl[/dim]")
        return

    # Sort by modification time (most recent first)
    sorted_logs = sorted(log_files, key=os.path.getmtime, reverse=True)
    recent_logs = sorted_logs[:count]

    if ctx.obj["JSON_OUTPUT"]:
        result = {
            "log_files": [
                {
                    "path": log,
                    "modified": datetime.fromtimestamp(os.path.getmtime(log)).isoformat(),
                    "size": os.path.getsize(log),
                }
                for log in recent_logs
            ]
        }
        print(json.dumps(result, indent=2))
    else:
        console = Console()
        console.print(f"[green]Most recent {len(recent_logs)} Buttermilk log files:[/green]\n")
        for log in recent_logs:
            mtime = datetime.fromtimestamp(os.path.getmtime(log))
            size = os.path.getsize(log)
            console.print(f"[dim]{mtime.strftime('%Y-%m-%d %H:%M:%S')}[/dim]  {os.path.basename(log)}  [dim]({size:,} bytes)[/dim]")
        console.print(f"\n[dim]Total Buttermilk log files in /tmp/: {len(log_files)}[/dim]")


@cli.command()
@click.pass_context
def test_connection(ctx):
    """Test connection to the WebSocket server."""
    client = NonInteractiveDebugClient(ctx.obj["HOST"], ctx.obj["PORT"])

    async def test():
        if await client.connect():
            await client.disconnect()
            return {"status": "success", "url": client.ws_url}
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
