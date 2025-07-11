"""Debug flow configuration for MCP access.

This module creates a flow configuration that includes the DebugAgent,
making its tools available through the existing Buttermilk API's MCP endpoints.
"""

from omegaconf import DictConfig
from buttermilk.debug.debug_agent import DebugAgent


def create_debug_flow_config() -> DictConfig:
    """Create a flow configuration with the DebugAgent.
    
    Returns:
        Flow configuration that can be loaded by Buttermilk
    """
    return DictConfig({
        "flow": {
            "name": "debug_flow",
            "description": "Debug flow with MCP-exposed debugging tools",
            "orchestrator": {
                "_target_": "buttermilk.orchestrators.BasicOrchestrator",
                "agent_configs": [
                    {
                        "_target_": "buttermilk.debug.debug_agent.DebugAgent",
                        "agent_name": "debug_agent",
                        "role": "debugger",
                        "description": "Provides debugging tools for Buttermilk flows"
                    }
                ]
            }
        }
    })


# Example YAML configuration that could be saved as debug_flow.yaml:
DEBUG_FLOW_YAML = """
flow:
  name: debug_flow
  description: Debug flow with MCP-exposed debugging tools
  orchestrator:
    _target_: buttermilk.orchestrators.groupchat.GroupChatOrchestrator
    agent_configs:
      - _target_: buttermilk.debug.debug_agent.DebugAgent
        agent_name: debug_agent
        role: debugger
        description: Provides debugging tools for Buttermilk flows
"""