#!/usr/bin/env python3
"""
Buttermilk MCP Server
Provides MCP tools for AI assistants working on the Buttermilk project.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

# MCP Server base implementation
class ButtermilkMCPServer:
    def __init__(self):
        self.tools_dir = Path(__file__).parent / "tools"
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load the MCP server configuration."""
        config_path = Path(__file__).parent / "buttermilk-server.json"
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        return self.config.get("tools", [])
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific tool with given arguments."""
        # Find the tool definition
        tool = None
        for t in self.config.get("tools", []):
            if t["name"] == tool_name:
                tool = t
                break
        
        if not tool:
            return {
                "error": f"Tool '{tool_name}' not found",
                "success": False
            }
        
        # Map tool name to script
        script_path = self.tools_dir / f"{tool_name}.sh"
        if not script_path.exists():
            return {
                "error": f"Tool script not found: {script_path}",
                "success": False
            }
        
        # Prepare arguments based on tool
        args = [str(script_path)]
        
        # Add arguments based on tool schema
        if tool_name == "buttermilk-workflow-check":
            args.append(arguments.get("step", ""))
            if "task_description" in arguments:
                args.append(arguments["task_description"])
                
        elif tool_name == "buttermilk-logs":
            args.append(arguments.get("mode", "tail"))
            args.append(str(arguments.get("lines", 50)))
            if "pattern" in arguments:
                args.append(arguments["pattern"])
                
        elif tool_name == "buttermilk-server":
            args.append(arguments.get("action", "status"))
            if arguments.get("action") == "start":
                args.append(str(arguments.get("debug", False)).lower())
                if "flows" in arguments:
                    args.append(",".join(arguments["flows"]))
                    
        elif tool_name == "buttermilk-test-flow":
            args.append(arguments.get("flow", ""))
            args.append(arguments.get("prompt", ""))
            args.append(arguments.get("mode", "http"))
            
        elif tool_name == "buttermilk-config-validate":
            args.append(arguments.get("config_path", ""))
            args.append(str(arguments.get("check_interpolations", True)).lower())
            
        elif tool_name == "buttermilk-github-issue":
            args.append(arguments.get("action", ""))
            args.append(arguments.get("query", ""))
            if "body" in arguments:
                args.append(arguments["body"])
            if "labels" in arguments:
                args.append(",".join(arguments["labels"]))
                
        elif tool_name == "buttermilk-ws-debug":
            args.append(arguments.get("command", ""))
            args.append(arguments.get("flow_name", ""))
            args.append(arguments.get("content", ""))
            args.append(arguments.get("message_type", "manager_response"))
            if "pattern" in arguments:
                args.append(arguments["pattern"])
            args.append(str(arguments.get("wait_time", 10)))
            
        elif tool_name == "buttermilk-advanced-debug":
            # Advanced debug tool using full ws_debug_cli.py functionality
            # Requires Buttermilk installation and provides comprehensive debugging features
            command = arguments.get("command", "")
            args.append(command)
            
            # Add parameters based on command type
            if command in ["start", "start-debug"]:
                args.append(arguments.get("flow_name", ""))
                args.append(arguments.get("query", ""))
                args.append(str(arguments.get("wait_time", 5)))
                if "record" in arguments:
                    args.append(arguments["record"])
                else:
                    args.append("")
                if "criteria" in arguments:
                    args.append(arguments["criteria"])
                else:
                    args.append("")
                args.append(str(arguments.get("json_output", False)).lower())
                
            elif command == "start-server":
                args.append(arguments.get("flow_name", "trans"))
                args.append(arguments.get("criteria", "hrc"))
                args.append(arguments.get("host", "localhost"))
                args.append(str(arguments.get("port", 8000)))
                
            elif command == "send":
                args.append(arguments.get("content", ""))
                args.append(arguments.get("message_type", "response"))
                args.append(str(arguments.get("wait_time", 5)))
                if "session_id" in arguments:
                    args.append(arguments["session_id"])
                else:
                    args.append("")
                args.append(str(arguments.get("json_output", False)).lower())
                
            elif command == "wait":
                args.append(str(arguments.get("wait_time", 5)))
                if "pattern" in arguments:
                    args.append(arguments["pattern"])
                else:
                    args.append("")
                if "message_type" in arguments:
                    args.append(arguments["message_type"])
                else:
                    args.append("")
                if "session_id" in arguments:
                    args.append(arguments["session_id"])
                else:
                    args.append("")
                args.append(str(arguments.get("json_output", False)).lower())
                
            elif command in ["session", "clear-session", "test-connection", "list-flows"]:
                args.append(str(arguments.get("json_output", False)).lower())
                
            elif command == "logs":
                args.append(str(arguments.get("lines", 50)))
                args.append(str(arguments.get("json_output", False)).lower())
        
        # Execute the tool
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=tool.get("timeout", 60)  # Configurable timeout with default value
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
                "exit_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Tool execution timed out after 60 seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}"
            }

def main():
    """Main entry point for the MCP server."""
    server = ButtermilkMCPServer()
    
    # Simple command-line interface for testing
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "list":
            tools = server.list_tools()
            print(json.dumps(tools, indent=2))
            
        elif command == "execute":
            if len(sys.argv) < 3:
                print("Usage: buttermilk-mcp-server.py execute <tool_name> [json_args]")
                sys.exit(1)
                
            tool_name = sys.argv[2]
            args = {}
            if len(sys.argv) > 3:
                args = json.loads(sys.argv[3])
                
            result = server.execute_tool(tool_name, args)
            print(json.dumps(result, indent=2))
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands: list, execute")
            sys.exit(1)
    else:
        print("Buttermilk MCP Server")
        print("Usage: buttermilk-mcp-server.py <command>")
        print("Commands: list, execute")

if __name__ == "__main__":
    main()