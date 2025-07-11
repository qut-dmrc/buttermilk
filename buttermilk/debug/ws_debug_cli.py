#!/usr/bin/env python3
"""Standalone WebSocket Debug CLI for Buttermilk flows.

This provides an interactive command-line interface for debugging flows
without requiring MCP. It connects directly to the Buttermilk WebSocket
API and provides real-time interaction capabilities.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import click
import aiohttp
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
import glob
import os

# Use the existing flow test client
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.integration.flow_test_client import FlowTestClient, MessageType


class InteractiveDebugClient:
    """Interactive WebSocket debug client with rich terminal UI."""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.client: Optional[FlowTestClient] = None
        self.console = Console()
        self.running = True
        self.current_flow: Optional[str] = None
        
    async def connect(self):
        """Connect to the WebSocket server."""
        try:
            self.client = FlowTestClient(
                base_url=self.base_url,
                ws_url=self.ws_url
            )
            await self.client.connect()
            self.console.print("[green]✓ Connected to WebSocket server[/green]")
            self.console.print(f"Session ID: {self.client.session_id}")
            return True
        except Exception as e:
            self.console.print(f"[red]✗ Failed to connect: {e}[/red]")
            return False
    
    async def disconnect(self):
        """Disconnect from the WebSocket server."""
        if self.client:
            await self.client.disconnect()
            self.console.print("[yellow]Disconnected from server[/yellow]")
    
    def show_help(self):
        """Display available commands."""
        help_text = """
[bold cyan]Available Commands:[/bold cyan]

[bold]Flow Control:[/bold]
  start <flow_name> [query]  - Start a flow with optional initial query
  response <text>           - Send a response to the current flow
  status                    - Show current flow status
  
[bold]Message Inspection:[/bold]
  messages [n]              - Show last n messages (default: 10)
  messages <type>           - Show messages of specific type
  agents                    - Show active agents
  errors                    - Show error messages
  
[bold]Log Analysis:[/bold]
  logs [n]                  - Show last n lines from latest log file
  logfiles                  - List available log files
  
[bold]Session Control:[/bold]
  clear                     - Clear message history
  export <file>             - Export messages to JSON file
  help                      - Show this help
  quit/exit                 - Exit the debugger
"""
        self.console.print(Panel(help_text, title="WebSocket Debug CLI"))
    
    def format_message(self, msg) -> str:
        """Format a message for display."""
        timestamp = msg.timestamp.strftime("%H:%M:%S")
        
        if msg.type == MessageType.UI_MESSAGE:
            content = msg.content or "(empty)"
            return f"[dim]{timestamp}[/dim] [blue]UI:[/blue] {content}"
        
        elif msg.type == MessageType.AGENT_ANNOUNCEMENT:
            role = msg.agent_role or "unknown"
            content = msg.content or "(thinking...)"
            return f"[dim]{timestamp}[/dim] [green]{role}:[/green] {content}"
        
        elif msg.type == MessageType.ERROR:
            error = msg.data.get("error", "Unknown error")
            return f"[dim]{timestamp}[/dim] [red]ERROR:[/red] {error}"
        
        elif msg.type == MessageType.FLOW_COMPLETE:
            return f"[dim]{timestamp}[/dim] [yellow]Flow completed[/yellow]"
        
        else:
            return f"[dim]{timestamp}[/dim] [{msg.type}] {json.dumps(msg.data)[:100]}..."
    
    async def show_messages(self, count: int = 10, msg_type: Optional[str] = None):
        """Display recent messages."""
        if not self.client:
            self.console.print("[red]Not connected[/red]")
            return
        
        messages = self.client.collector.all_messages
        
        if msg_type:
            messages = [m for m in messages if m.type == msg_type]
            self.console.print(f"\n[bold]Messages of type '{msg_type}':[/bold]")
        else:
            self.console.print(f"\n[bold]Last {count} messages:[/bold]")
        
        # Show last n messages
        for msg in messages[-count:]:
            self.console.print(self.format_message(msg))
    
    async def show_agents(self):
        """Show active agents from announcements."""
        if not self.client:
            self.console.print("[red]Not connected[/red]")
            return
        
        agents = {}
        for msg in self.client.collector.agent_announcements:
            role = msg.agent_role
            if role:
                agent_config = msg.data.get("data", {}).get("agent_config", {})
                agents[role] = {
                    "name": agent_config.get("agent_name", "Unknown"),
                    "model": agent_config.get("model_config", {}).get("model", "Unknown"),
                    "last_seen": msg.timestamp
                }
        
        if agents:
            table = Table(title="Active Agents")
            table.add_column("Role", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Model", style="yellow")
            table.add_column("Last Seen", style="dim")
            
            for role, info in agents.items():
                table.add_row(
                    role,
                    info["name"],
                    info["model"],
                    info["last_seen"].strftime("%H:%M:%S")
                )
            
            self.console.print(table)
        else:
            self.console.print("[yellow]No agent announcements received yet[/yellow]")
    
    async def show_errors(self):
        """Show error messages."""
        if not self.client:
            self.console.print("[red]Not connected[/red]")
            return
        
        errors = self.client.collector.errors
        if errors:
            self.console.print(f"\n[red bold]Errors ({len(errors)}):[/red bold]")
            for error in errors:
                self.console.print(self.format_message(error))
                if "stack_trace" in error.data:
                    self.console.print(f"[dim]{error.data['stack_trace']}[/dim]")
        else:
            self.console.print("[green]No errors recorded[/green]")
    
    def show_logs(self, lines: int = 50):
        """Show recent log lines from Buttermilk log files."""
        log_files = glob.glob('/tmp/buttermilk_*.log')
        if not log_files:
            self.console.print("[yellow]No log files found in /tmp/[/yellow]")
            return
        
        # Get the most recent log file
        latest_log = max(log_files, key=os.path.getmtime)
        self.console.print(f"[dim]Reading from: {latest_log}[/dim]\n")
        
        try:
            with open(latest_log, 'r') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:]
                
                for line in recent_lines:
                    # Color code by log level
                    if "ERROR" in line:
                        self.console.print(f"[red]{line.rstrip()}[/red]")
                    elif "WARNING" in line:
                        self.console.print(f"[yellow]{line.rstrip()}[/yellow]")
                    elif "INFO" in line:
                        self.console.print(f"[green]{line.rstrip()}[/green]")
                    else:
                        self.console.print(f"[dim]{line.rstrip()}[/dim]")
        except Exception as e:
            self.console.print(f"[red]Error reading log file: {e}[/red]")
    
    def list_log_files(self):
        """List available log files."""
        log_files = glob.glob('/tmp/buttermilk_*.log')
        
        if not log_files:
            self.console.print("[yellow]No log files found in /tmp/[/yellow]")
            return
        
        table = Table(title="Buttermilk Log Files")
        table.add_column("File", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Modified", style="yellow")
        
        for log_file in sorted(log_files, key=os.path.getmtime, reverse=True):
            stat = os.stat(log_file)
            size_mb = stat.st_size / (1024 * 1024)
            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            table.add_row(
                Path(log_file).name,
                f"{size_mb:.2f} MB",
                modified
            )
        
        self.console.print(table)
    
    async def start_flow(self, flow_name: str, initial_query: Optional[str] = None):
        """Start a new flow."""
        if not self.client:
            self.console.print("[red]Not connected[/red]")
            return
        
        try:
            self.console.print(f"[cyan]Starting flow: {flow_name}[/cyan]")
            await self.client.start_flow(flow_name, initial_query or "")
            self.current_flow = flow_name
            self.console.print(f"[green]✓ Flow '{flow_name}' started[/green]")
            
            # Show initial messages
            await asyncio.sleep(1)  # Give time for initial messages
            await self.show_messages(5)
            
        except Exception as e:
            self.console.print(f"[red]Failed to start flow: {e}[/red]")
    
    async def send_response(self, response: str):
        """Send a response to the current flow."""
        if not self.client:
            self.console.print("[red]Not connected[/red]")
            return
        
        if not self.current_flow:
            self.console.print("[yellow]No active flow. Start a flow first.[/yellow]")
            return
        
        try:
            await self.client.send_manager_response(response)
            self.console.print(f"[green]✓ Sent response: {response}[/green]")
        except Exception as e:
            self.console.print(f"[red]Failed to send response: {e}[/red]")
    
    def show_status(self):
        """Show current session status."""
        if not self.client:
            self.console.print("[red]Not connected[/red]")
            return
        
        summary = self.client.get_message_summary()
        
        status_text = f"""
[bold]Session Status:[/bold]
Connected: {'Yes' if self.client else 'No'}
Session ID: {self.client.session_id or 'N/A'}
Current Flow: {self.current_flow or 'None'}
WebSocket URL: {self.ws_url}

[bold]Message Summary:[/bold]
Total Messages: {summary['total_messages']}
UI Messages: {summary['ui_messages']}
Agent Announcements: {summary['agent_announcements']}
Agent Traces: {summary['agent_traces']}
Errors: {summary['errors']}
"""
        
        if summary['active_agents']:
            status_text += f"\n[bold]Active Agents:[/bold] {', '.join(summary['active_agents'])}"
        
        self.console.print(Panel(status_text, title="Debug Session Status"))
    
    async def export_messages(self, filename: str):
        """Export collected messages to a JSON file."""
        if not self.client:
            self.console.print("[red]Not connected[/red]")
            return
        
        try:
            messages_data = []
            for msg in self.client.collector.all_messages:
                messages_data.append({
                    "timestamp": msg.timestamp.isoformat(),
                    "type": msg.type,
                    "data": msg.data,
                    "content": msg.content,
                    "agent_role": msg.agent_role
                })
            
            output_path = Path(filename)
            output_path.write_text(json.dumps(messages_data, indent=2))
            self.console.print(f"[green]✓ Exported {len(messages_data)} messages to {filename}[/green]")
            
        except Exception as e:
            self.console.print(f"[red]Failed to export messages: {e}[/red]")
    
    async def run_interactive(self):
        """Run the interactive debug session."""
        self.console.print("\n[bold cyan]Buttermilk WebSocket Debug CLI[/bold cyan]")
        self.console.print("Type 'help' for available commands\n")
        
        if not await self.connect():
            return
        
        try:
            while self.running:
                try:
                    # Get user input
                    command = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: input("> ")
                    )
                    
                    if not command.strip():
                        continue
                    
                    parts = command.strip().split(maxsplit=1)
                    cmd = parts[0].lower()
                    args = parts[1] if len(parts) > 1 else ""
                    
                    # Process commands
                    if cmd in ["quit", "exit"]:
                        self.running = False
                    
                    elif cmd == "help":
                        self.show_help()
                    
                    elif cmd == "clear":
                        self.client.collector = self.client.collector.__class__()
                        self.console.print("[yellow]Message history cleared[/yellow]")
                    
                    elif cmd == "start":
                        if args:
                            flow_parts = args.split(maxsplit=1)
                            flow_name = flow_parts[0]
                            query = flow_parts[1] if len(flow_parts) > 1 else None
                            await self.start_flow(flow_name, query)
                        else:
                            self.console.print("[red]Usage: start <flow_name> [query][/red]")
                    
                    elif cmd == "response":
                        if args:
                            await self.send_response(args)
                        else:
                            self.console.print("[red]Usage: response <text>[/red]")
                    
                    elif cmd == "messages":
                        if args.isdigit():
                            await self.show_messages(int(args))
                        elif args:
                            await self.show_messages(msg_type=args)
                        else:
                            await self.show_messages()
                    
                    elif cmd == "agents":
                        await self.show_agents()
                    
                    elif cmd == "errors":
                        await self.show_errors()
                    
                    elif cmd == "status":
                        self.show_status()
                    
                    elif cmd == "logs":
                        lines = int(args) if args.isdigit() else 50
                        self.show_logs(lines)
                    
                    elif cmd == "logfiles":
                        self.list_log_files()
                    
                    elif cmd == "export":
                        if args:
                            await self.export_messages(args)
                        else:
                            self.console.print("[red]Usage: export <filename>[/red]")
                    
                    else:
                        self.console.print(f"[red]Unknown command: {cmd}[/red]")
                        self.console.print("Type 'help' for available commands")
                
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Use 'quit' or 'exit' to leave[/yellow]")
                    continue
                
        finally:
            await self.disconnect()


@click.command()
@click.option('--host', default='localhost', help='WebSocket server host')
@click.option('--port', default=8000, type=int, help='WebSocket server port')
def main(host: str, port: int):
    """Interactive WebSocket debug client for Buttermilk flows."""
    client = InteractiveDebugClient(host, port)
    
    try:
        asyncio.run(client.run_interactive())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()