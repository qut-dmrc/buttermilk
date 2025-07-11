#!/usr/bin/env python3
"""Non-interactive WebSocket Debug CLI for Buttermilk flows.

This provides a command-line interface for debugging flows that executes
specific actions and returns results to stdout, suitable for LLM usage.
"""

import asyncio
import json
from typing import Optional
import click
from rich.console import Console
from datetime import datetime
import glob
import os

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
        
    async def connect(self):
        """Connect to the WebSocket server."""
        try:
            self.client = FlowTestClient(
                base_url=self.base_url,
                ws_url=self.ws_url
            )
            await self.client.connect()
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
    
    async def send_response(self, session_id: str, response: str, wait_time: int = 5) -> dict:
        """Send a response to an existing session."""
        # This would require session persistence which is not currently implemented
        # in FlowTestClient. For now, return an informative error.
        return {
            "error": "Session persistence not implemented. Each command creates a new session."
        }
    
    async def get_logs(self, lines: int = 50) -> dict:
        """Get recent log lines."""
        log_files = glob.glob('/tmp/buttermilk_*.log')
        if not log_files:
            return {"error": "No log files found in /tmp/"}
        
        # Get the most recent log file
        latest_log = max(log_files, key=os.path.getmtime)
        
        try:
            with open(latest_log, 'r') as f:
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


@click.group()
@click.option('--host', default='localhost', help='WebSocket server host')
@click.option('--port', default=8000, type=int, help='WebSocket server port')
@click.option('--json-output', is_flag=True, help='Output results as JSON')
@click.pass_context
def cli(ctx, host: str, port: int, json_output: bool):
    """Non-interactive WebSocket debug client for Buttermilk flows."""
    ctx.ensure_object(dict)
    ctx.obj['HOST'] = host
    ctx.obj['PORT'] = port
    ctx.obj['JSON_OUTPUT'] = json_output


@cli.command()
@click.argument('flow_name')
@click.argument('query', default='')
@click.option('--wait', default=5, help='Seconds to wait for responses')
@click.pass_context
def start(ctx, flow_name: str, query: str, wait: int):
    """Start a flow with an optional initial query."""
    client = NonInteractiveDebugClient(ctx.obj['HOST'], ctx.obj['PORT'])
    result = asyncio.run(client.start_flow(flow_name, query, wait))
    
    if ctx.obj['JSON_OUTPUT']:
        print(json.dumps(result, indent=2))
    else:
        console = Console()
        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
        else:
            console.print(f"[green]Started flow '{flow_name}' with session: {result['session_id']}[/green]")
            console.print(f"Query: {query}")
            console.print(f"\nMessages ({len(result['messages'])}):")
            for msg in result['messages']:
                timestamp = datetime.fromisoformat(msg['timestamp']).strftime("%H:%M:%S")
                msg_type = msg['type']
                content = msg['content'] or '(no content)'
                agent = msg['agent_role'] or 'system'
                console.print(f"[dim]{timestamp}[/dim] [{msg_type}] {agent}: {content}")


@cli.command()
@click.option('--lines', '-n', default=50, help='Number of log lines to show')
@click.pass_context
def logs(ctx, lines: int):
    """Show recent log lines from Buttermilk log files."""
    client = NonInteractiveDebugClient(ctx.obj['HOST'], ctx.obj['PORT'])
    result = asyncio.run(client.get_logs(lines))
    
    if ctx.obj['JSON_OUTPUT']:
        print(json.dumps(result, indent=2))
    else:
        console = Console()
        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
        else:
            console.print(f"[dim]Log file: {result['log_file']}[/dim]\n")
            for line in result['lines']:
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
    client = NonInteractiveDebugClient(ctx.obj['HOST'], ctx.obj['PORT'])
    result = asyncio.run(client.list_flows())
    
    if ctx.obj['JSON_OUTPUT']:
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
                for flow in result['common_flows']:
                    console.print(f"  - {flow}")


@cli.command()
@click.pass_context
def test_connection(ctx):
    """Test connection to the WebSocket server."""
    client = NonInteractiveDebugClient(ctx.obj['HOST'], ctx.obj['PORT'])
    
    async def test():
        if await client.connect():
            await client.disconnect()
            return {"status": "success", "url": client.ws_url}
        else:
            return {"status": "failed", "url": client.ws_url}
    
    result = asyncio.run(test())
    
    if ctx.obj['JSON_OUTPUT']:
        print(json.dumps(result, indent=2))
    else:
        console = Console()
        if result["status"] == "success":
            console.print(f"[green]✓ Successfully connected to {result['url']}[/green]")
        else:
            console.print(f"[red]✗ Failed to connect to {result['url']}[/red]")


if __name__ == "__main__":
    cli()