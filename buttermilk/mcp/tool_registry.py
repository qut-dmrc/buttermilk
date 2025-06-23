"""Dynamic tool registry for MCP exposure of Buttermilk agents.

This module provides functionality to discover, register, and manage tools from
Buttermilk agents for exposure via MCP (Model Context Protocol).
"""

import asyncio
from typing import Any, Dict, List, Optional, Set
from collections.abc import AsyncGenerator

from buttermilk._core import logger
from buttermilk._core.agent import Agent
from buttermilk._core.tool_definition import AgentToolDefinition, UnifiedRequest
from buttermilk._core.contract import AgentAnnouncement


class ToolDiscoveryService:
    """Service for discovering and managing agent tools for MCP exposure."""
    
    def __init__(self):
        """Initialize the tool discovery service."""
        self.agent_registry: Dict[str, Agent] = {}
        self.tool_registry: Dict[str, Dict[str, Any]] = {}
        self._discovery_callbacks: List[callable] = []
        
    def register_agent(self, agent: Agent) -> None:
        """Register an agent and discover its tools.
        
        Args:
            agent: The agent to register
        """
        agent_key = f"{agent.agent_name}_{agent.role}"
        
        # Skip if already registered
        if agent_key in self.agent_registry:
            logger.debug(f"Agent {agent_key} already registered")
            return
            
        self.agent_registry[agent_key] = agent
        
        # Discover tools from the agent
        tools_discovered = self._discover_agent_tools(agent)
        
        logger.info(
            f"Registered agent '{agent_key}' with {len(tools_discovered)} tools: "
            f"{[tool['name'] for tool in tools_discovered]}"
        )
        
        # Notify callbacks about new tools
        self._notify_discovery_callbacks(agent, tools_discovered)
    
    def unregister_agent(self, agent_name: str, role: str) -> None:
        """Unregister an agent and its tools.
        
        Args:
            agent_name: Name of the agent
            role: Role of the agent
        """
        agent_key = f"{agent_name}_{role}"
        
        if agent_key not in self.agent_registry:
            return
            
        # Remove agent's tools from registry
        tools_to_remove = [
            tool_key for tool_key in self.tool_registry 
            if tool_key.startswith(f"{agent_key}.")
        ]
        
        for tool_key in tools_to_remove:
            del self.tool_registry[tool_key]
            
        # Remove agent
        del self.agent_registry[agent_key]
        
        logger.info(f"Unregistered agent '{agent_key}' and {len(tools_to_remove)} tools")
    
    def _discover_agent_tools(self, agent: Agent) -> List[Dict[str, Any]]:
        """Discover tools from an agent using its get_tool_definitions method.
        
        Args:
            agent: Agent to discover tools from
            
        Returns:
            List of tool information dictionaries
        """
        tools_discovered = []
        
        try:
            # Use the agent's built-in tool discovery
            tool_definitions = agent.get_tool_definitions()
            
            for tool_def in tool_definitions:
                tool_key = f"{agent.agent_name}_{agent.role}.{tool_def.name}"
                
                tool_info = {
                    "key": tool_key,
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "agent": agent,
                    "agent_name": agent.agent_name,
                    "agent_role": agent.role,
                    "tool_definition": tool_def,
                    "input_schema": tool_def.input_schema,
                    "output_schema": tool_def.output_schema,
                    "mcp_route": tool_def.mcp_route,
                    "permissions": tool_def.permissions
                }
                
                self.tool_registry[tool_key] = tool_info
                tools_discovered.append(tool_info)
                
        except Exception as e:
            logger.error(f"Error discovering tools from agent {agent.agent_name}: {e}")
            
        return tools_discovered
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all registered tools.
        
        Returns:
            List of all tool information dictionaries
        """
        return list(self.tool_registry.values())
    
    def get_tools_by_agent(self, agent_name: str, role: str) -> List[Dict[str, Any]]:
        """Get tools for a specific agent.
        
        Args:
            agent_name: Name of the agent
            role: Role of the agent
            
        Returns:
            List of tool information dictionaries for the agent
        """
        agent_key = f"{agent_name}_{role}"
        return [
            tool for tool in self.tool_registry.values()
            if tool["key"].startswith(f"{agent_key}.")
        ]
    
    def get_tool_by_name(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific tool by name (searches all agents).
        
        Args:
            tool_name: Name of the tool to find
            
        Returns:
            Tool information dictionary or None if not found
        """
        for tool in self.tool_registry.values():
            if tool["name"] == tool_name:
                return tool
        return None
    
    def get_tool_by_key(self, tool_key: str) -> Optional[Dict[str, Any]]:
        """Get a specific tool by its unique key.
        
        Args:
            tool_key: Unique key of the tool (agent_name_role.tool_name)
            
        Returns:
            Tool information dictionary or None if not found
        """
        return self.tool_registry.get(tool_key)
    
    async def invoke_tool(
        self, 
        tool_key: str, 
        inputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Invoke a tool by its key.
        
        Args:
            tool_key: Unique key of the tool to invoke
            inputs: Input parameters for the tool
            metadata: Optional metadata for the request
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool not found
            Exception: If tool execution fails
        """
        tool_info = self.get_tool_by_key(tool_key)
        if not tool_info:
            raise ValueError(f"Tool '{tool_key}' not found")
        
        agent = tool_info["agent"]
        tool_name = tool_info["name"]
        
        # Create unified request for MCP context
        unified_request = UnifiedRequest.from_mcp_call(
            tool_name=tool_name,
            parameters=inputs,
            agent_name=f"{agent.agent_name}_{agent.role}"
        )
        
        if metadata:
            unified_request.metadata.update(metadata)
        
        # Use the agent's unified request handler
        try:
            result = await agent.handle_unified_request(unified_request)
            logger.debug(f"Successfully invoked tool '{tool_key}'")
            return result
        except Exception as e:
            logger.error(f"Error invoking tool '{tool_key}': {e}")
            raise
    
    def add_discovery_callback(self, callback: callable) -> None:
        """Add a callback to be notified when new tools are discovered.
        
        Args:
            callback: Function to call when tools are discovered
                     Signature: callback(agent: Agent, tools: List[Dict[str, Any]])
        """
        self._discovery_callbacks.append(callback)
    
    def _notify_discovery_callbacks(self, agent: Agent, tools: List[Dict[str, Any]]) -> None:
        """Notify all registered callbacks about newly discovered tools.
        
        Args:
            agent: The agent whose tools were discovered
            tools: List of discovered tool information dictionaries
        """
        for callback in self._discovery_callbacks:
            try:
                callback(agent, tools)
            except Exception as e:
                logger.error(f"Error in discovery callback: {e}")


class AgentRegistryWatcher:
    """Watches agent registries for dynamic tool discovery."""
    
    def __init__(self, discovery_service: ToolDiscoveryService):
        """Initialize the registry watcher.
        
        Args:
            discovery_service: Tool discovery service to notify of changes
        """
        self.discovery_service = discovery_service
        self._watched_registries: Set[object] = set()
        self._running = False
    
    def watch_registry(self, registry: object) -> None:
        """Start watching an agent registry for changes.
        
        Args:
            registry: Agent registry object to watch
        """
        if registry in self._watched_registries:
            return
            
        self._watched_registries.add(registry)
        logger.info(f"Started watching agent registry: {type(registry).__name__}")
    
    async def start_watching(self) -> None:
        """Start the registry watching process."""
        if self._running:
            return
            
        self._running = True
        logger.info("Started agent registry watching")
        
        # This would be implemented based on specific registry types
        # For now, it's a placeholder for the pattern
        while self._running:
            await asyncio.sleep(5)  # Check every 5 seconds
            # TODO: Implement actual registry watching logic
    
    def stop_watching(self) -> None:
        """Stop watching all registries."""
        self._running = False
        logger.info("Stopped agent registry watching")


# Global instance for easy access
_global_discovery_service: Optional[ToolDiscoveryService] = None


def get_tool_discovery_service() -> ToolDiscoveryService:
    """Get the global tool discovery service instance.
    
    Returns:
        Global ToolDiscoveryService instance
    """
    global _global_discovery_service
    if _global_discovery_service is None:
        _global_discovery_service = ToolDiscoveryService()
    return _global_discovery_service


def register_agent_for_mcp(agent: Agent) -> None:
    """Convenience function to register an agent for MCP exposure.
    
    Args:
        agent: Agent to register
    """
    discovery_service = get_tool_discovery_service()
    discovery_service.register_agent(agent)