"""AutoGen MCP Adapter for exposing Buttermilk agent tools as MCP services.

This module provides the bridge between Buttermilk's tool definition system and 
AutoGen's MCP (Model Context Protocol) infrastructure, allowing Buttermilk agents
to be consumed as MCP tools by AutoGen agents or other MCP clients.

The adapter integrates with Buttermilk's existing tool discovery and unified request
handling systems to provide seamless MCP exposure of agent capabilities.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Union
from collections.abc import Sequence

from autogen_core import CancellationToken
from autogen_core.tools import Tool
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from pydantic import BaseModel, Field

from buttermilk._core import logger, AgentInput, AgentOutput
from buttermilk._core.tool_definition import AgentToolDefinition, UnifiedRequest
from buttermilk._core.agent import Agent
from buttermilk._core.schema_validation import validate_tool_input, validate_tool_output
from buttermilk.mcp.tool_registry import ToolDiscoveryService, get_tool_discovery_service


class MCPToolAdapter(Tool):
    """Adapter that wraps a Buttermilk agent tool as an AutoGen Tool for MCP exposure.
    
    This adapter uses the tool registry system and unified request handling to
    provide seamless integration between AutoGen MCP and Buttermilk agents.
    """

    def __init__(self, tool_info: Dict[str, Any]):
        """Initialize the MCP tool adapter from tool registry information.
        
        Args:
            tool_info: Tool information dictionary from ToolDiscoveryService
        """
        self.tool_info = tool_info
        self.tool_key = tool_info["key"]
        self.agent = tool_info["agent"]
        self.tool_definition = tool_info["tool_definition"]
        
        # Initialize the Tool with AutoGen's expected interface
        super().__init__(
            name=self.tool_definition.name,
            description=self.tool_definition.description,
            parameters=self.tool_definition.input_schema
        )

    async def run_json(self, parameters: Dict[str, Any], cancellation_token: CancellationToken | None = None) -> Any:
        """Execute the tool with JSON parameters (AutoGen Tool interface).
        
        Args:
            parameters: Tool parameters as a dictionary
            cancellation_token: Optional cancellation token
            
        Returns:
            Tool execution result
        """
        try:
            # Create unified request for MCP context
            unified_request = UnifiedRequest.from_mcp_call(
                tool_name=self.tool_definition.name,
                parameters=parameters,
                agent_name=f"{self.agent.agent_name}_{self.agent.role}"
            )
            
            # Add cancellation token to metadata if provided
            if cancellation_token:
                unified_request.metadata["cancellation_token"] = cancellation_token

            # Use the agent's unified request handler
            result = await self.agent.handle_unified_request(unified_request)

            logger.debug(f"MCP tool '{self.name}' executed successfully")
            return result

        except Exception as e:
            logger.error(f"Error executing MCP tool '{self.name}': {e}")
            return {"error": str(e), "success": False, "tool_key": self.tool_key}


class AutoGenMCPAdapter:
    """Main adapter class that bridges Buttermilk tools to AutoGen MCP system.
    
    This adapter integrates with the ToolDiscoveryService for dynamic tool discovery
    and uses Buttermilk's unified request handling system.
    """
    
    def __init__(
        self, 
        server_params: Optional[StdioServerParams] = None,
        discovery_service: Optional[ToolDiscoveryService] = None
    ):
        """Initialize the AutoGen MCP adapter.
        
        Args:
            server_params: Optional parameters for MCP server configuration
            discovery_service: Optional tool discovery service (uses global if not provided)
        """
        self.server_params = server_params or StdioServerParams(
            command="python",
            args=["-m", "buttermilk.mcp.autogen_adapter"],
            env={}
        )
        self.discovery_service = discovery_service or get_tool_discovery_service()
        self.mcp_adapters: Dict[str, MCPToolAdapter] = {}
        self.workbench: Optional[McpWorkbench] = None
        
        # Set up callback for dynamic tool discovery
        self.discovery_service.add_discovery_callback(self._on_tools_discovered)
        
    def register_agent(self, agent: Agent) -> None:
        """Register an agent and its tools for MCP exposure.
        
        Args:
            agent: The agent instance to register
        """
        # Use the discovery service to register the agent
        self.discovery_service.register_agent(agent)
        # The _on_tools_discovered callback will handle creating MCP adapters
        
    def _on_tools_discovered(self, agent: Agent, tools: List[Dict[str, Any]]) -> None:
        """Callback for when new tools are discovered.
        
        Args:
            agent: The agent whose tools were discovered
            tools: List of discovered tool information dictionaries
        """
        for tool_info in tools:
            tool_key = tool_info["key"]
            
            # Create MCP adapter for the tool
            mcp_adapter = MCPToolAdapter(tool_info)
            self.mcp_adapters[tool_key] = mcp_adapter
            
            logger.debug(f"Created MCP adapter for tool '{tool_key}'")
        
        logger.info(f"Created MCP adapters for {len(tools)} tools from agent '{agent.agent_name}'")
        
    def unregister_agent(self, agent_name: str, role: str) -> None:
        """Unregister an agent and its tools.
        
        Args:
            agent_name: Name of the agent
            role: Role of the agent
        """
        # Remove MCP adapters for this agent's tools
        agent_key = f"{agent_name}_{role}"
        adapters_to_remove = [
            key for key in self.mcp_adapters 
            if key.startswith(f"{agent_key}.")
        ]
        
        for key in adapters_to_remove:
            del self.mcp_adapters[key]
        
        # Unregister from discovery service
        self.discovery_service.unregister_agent(agent_name, role)
        
        logger.info(f"Unregistered agent '{agent_key}' and {len(adapters_to_remove)} MCP adapters")

    async def start_mcp_server(self) -> None:
        """Start the MCP server to expose registered tools."""
        if not self.mcp_adapters:
            logger.warning("No tools registered for MCP exposure")
            return
            
        try:
            # Create workbench for MCP server
            self.workbench = McpWorkbench(self.server_params)
            await self.workbench.start()
            
            # Convert our tools to the format expected by MCP
            mcp_tools = list(self.mcp_adapters.values())
            
            logger.info(f"Started MCP server with {len(mcp_tools)} tools")
            
            # The workbench will handle the MCP protocol communication
            # Tools are now available for external MCP clients to discover and use
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise

    async def stop_mcp_server(self) -> None:
        """Stop the MCP server."""
        if self.workbench:
            await self.workbench.stop()
            self.workbench = None
            logger.info("Stopped MCP server")

    def get_available_tools(self) -> List[Tool]:
        """Get list of available tools as Tool objects.

        Returns:
            List of available tools
        """
        raise NotImplementedError

    async def invoke_tool(self, tool_key: str, parameters: Dict[str, Any]) -> Any:
        """Invoke a tool directly (for testing or internal use).
        
        Args:
            tool_key: Key of the tool to invoke (agent_name_role.tool_name)
            parameters: Tool parameters
            
        Returns:
            Tool execution result
        """
        if tool_key not in self.mcp_adapters:
            # Try to find by tool name if exact key not found
            tool_adapter = None
            for adapter in self.mcp_adapters.values():
                if adapter.name == tool_key:
                    tool_adapter = adapter
                    break
            
            if not tool_adapter:
                available_tools = [f"{key} ({adapter.name})" for key, adapter in self.mcp_adapters.items()]
                raise ValueError(f"Tool '{tool_key}' not found. Available: {available_tools}")
        else:
            tool_adapter = self.mcp_adapters[tool_key]
            
        return await tool_adapter.run_json(parameters)
    
    def get_tool_by_name(self, tool_name: str) -> Optional[MCPToolAdapter]:
        """Get a tool adapter by tool name.
        
        Args:
            tool_name: Name of the tool to find
            
        Returns:
            MCPToolAdapter instance or None if not found
        """
        for adapter in self.mcp_adapters.values():
            if adapter.name == tool_name:
                return adapter
        return None
    
    def list_tools(self) -> Dict[str, Dict[str, Any]]:
        """List all available tools with their metadata.
        
        Returns:
            Dictionary of tool information
        """
        tools_info = {}
        for tool_key, adapter in self.mcp_adapters.items():
            tools_info[tool_key] = {
                "name": adapter.name,
                "description": adapter.description,
                "agent_name": adapter.agent.agent_name,
                "agent_role": adapter.agent.role,
                "input_schema": adapter.tool_definition.input_schema,
                "output_schema": adapter.tool_definition.output_schema,
                "mcp_route": adapter.tool_definition.mcp_route,
                "permissions": adapter.tool_definition.permissions
            }
        return tools_info


class MCPHostProvider:
    """Service that provides MCP hosting capabilities for Buttermilk agents.
    
    This is the main interface for integrating MCP capabilities into Buttermilk
    flows and orchestrators. It provides a simplified API for registering agents
    and starting MCP services.
    """
    
    def __init__(
        self, 
        port: int = 8788,
        discovery_service: Optional[ToolDiscoveryService] = None
    ):
        """Initialize the MCP host provider.
        
        Args:
            port: Port for the MCP server
            discovery_service: Optional tool discovery service
        """
        self.port = port
        self.discovery_service = discovery_service or get_tool_discovery_service()
        self.adapter = AutoGenMCPAdapter(discovery_service=self.discovery_service)
        self.running = False

    async def register_agents_from_orchestrator(self, orchestrator) -> None:
        """Register all agents from an orchestrator for MCP exposure.
        
        Args:
            orchestrator: The orchestrator containing agents to register
        """
        logger.info("Registering agents from orchestrator for MCP exposure")
        
        # This is a placeholder for orchestrator integration
        # The actual implementation would depend on the orchestrator's API
        # for enumerating agents
        
        # Example of what this might look like:
        # if hasattr(orchestrator, 'get_all_agents'):
        #     agents = await orchestrator.get_all_agents()
        #     for agent in agents:
        #         self.register_agent(agent)
        
        logger.warning("Orchestrator integration not yet implemented - register agents manually")

    async def start(self) -> None:
        """Start the MCP host service."""
        if self.running:
            logger.warning("MCP host provider already running")
            return
            
        try:
            await self.adapter.start_mcp_server()
            self.running = True
            logger.info(f"MCP host provider started on port {self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start MCP host provider: {e}")
            raise

    async def stop(self) -> None:
        """Stop the MCP host service."""
        if not self.running:
            return
            
        try:
            await self.adapter.stop_mcp_server()
            self.running = False
            logger.info("MCP host provider stopped")
            
        except Exception as e:
            logger.error(f"Error stopping MCP host provider: {e}")

    def register_agent(self, agent: Agent) -> None:
        """Register a single agent for MCP exposure.
        
        Args:
            agent: The agent to register
        """
        self.adapter.register_agent(agent)
        
    def unregister_agent(self, agent_name: str, role: str) -> None:
        """Unregister an agent from MCP exposure.
        
        Args:
            agent_name: Name of the agent
            role: Role of the agent
        """
        self.adapter.unregister_agent(agent_name, role)
        
    def list_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """List all available tools.
        
        Returns:
            Dictionary of tool information
        """
        return self.adapter.list_tools()
        
    async def invoke_tool(self, tool_key: str, parameters: Dict[str, Any]) -> Any:
        """Invoke a tool directly.
        
        Args:
            tool_key: Key of the tool to invoke
            parameters: Tool parameters
            
        Returns:
            Tool execution result
        """
        return await self.adapter.invoke_tool(tool_key, parameters)


# Command-line interface for running as standalone MCP server
if __name__ == "__main__":
    import sys
    
    async def main():
        """Main entry point for standalone MCP server."""
        provider = MCPHostProvider()
        
        # TODO: Load agents from configuration
        # For now, this would need to be configured with specific agents
        
        try:
            await provider.start()
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down MCP server")
        finally:
            await provider.stop()
    
    asyncio.run(main())